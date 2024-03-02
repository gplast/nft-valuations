#%%
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
import os

SHOW_PLOTS = False
EVALUATION_COLUMN = 'eth'
drop_cols = ['token_index', 
             'eth', 
             'usd', 
             'date', 
             'eth_usd', 
             'eth_usd_normalized', 
             'rarity_score_calculated', 
             'rarity_score', 
             'rarest_property_name', 
             'Trait Count', 
             'open']


#load data from 20230509 folder
print(f'#########################################################')
print(f'Loading data from 20230509 folder')
eth_usd_fx_rates = pd.read_csv('20230509/eth_usd_fx_rates.csv')
token_metadata = pd.read_csv('20230509/token_metadata.csv')
token_sales = pd.read_csv('20230509/token_sales.csv')
print(f'Data loaded successfully')

#drop columns with all null values
eth_usd_fx_rates = eth_usd_fx_rates.dropna(axis=1, how='all')
token_metadata = token_metadata.dropna(axis=1, how='all')
token_sales = token_sales.dropna(axis=1, how='all')

#based on the formula [Rarity Score for a Trait Value] = 1 / [Trait Rarity of that Trait Value] find the 
#rarity score for each trait value by iterate each row
print(f'\n#########################################################')
print(f'Calculating rarity score for each trait value')
drop_cols_rarity = ['token_index', 'Trait Count', 'rarest_property_name', 'rarity_score']

for col in token_metadata.columns:
    if col in drop_cols_rarity:
        continue
    #nan values are not considered in the value count
    token_metadata[col] = token_metadata[col].fillna('nan')
    #find the percentage of each trait value that contains the same value but if the value is nan then it will be 0
    token_metadata[col] = token_metadata[col].map(token_metadata[col].value_counts(normalize=True))
    #find the rarity score
    token_metadata[col] = 1 / token_metadata[col]

#calculate the rarity score to check if it matches the rarity score column
token_metadata['rarity_score_calculated'] = token_metadata.drop(columns=drop_cols_rarity).sum(axis=1)
#print average difference between the calculated rarity score and the rarity score column
print(f'Average difference between the calculated rarity score and the rarity score column: {abs(token_metadata["rarity_score_calculated"] - token_metadata["rarity_score"]).mean()}')
print(f'Rarity score calculated successfully')

print(f'\n#########################################################')
print(f'Preprocessing and merging data...')
#find the eth price in usd when the token was sold
token_sales['eth_usd'] = token_sales['usd'] / token_sales['eth']

#convert date columns to date type
eth_usd_fx_rates['date'] = pd.to_datetime(eth_usd_fx_rates['date'], dayfirst=False).dt.date

#convert timestamp to date type for token_sales
token_sales['timestamp'] = pd.to_datetime(token_sales['timestamp'], unit='s')
token_sales['date'] = token_sales['timestamp'].dt.date

#drop timestamp column
token_sales = token_sales.drop(columns='timestamp')

#group by date and get the average of the price for each token for each date
token_sales = token_sales.groupby(['date', 'token_index']).mean().reset_index()

#merge token_metadata and token_sales on token_index
token_metadata_sales = pd.merge(token_metadata, token_sales, on='token_index')

#merge token_metadata_sales and eth_usd_fx_rates on date
token_metadata_sales_fx = pd.merge(token_metadata_sales, eth_usd_fx_rates, on='date')

#normalize the price based on open price of the day
token_metadata_sales_fx['eth_usd_normalized'] = token_metadata_sales_fx['eth_usd'] / token_metadata_sales_fx['open']

# #find the percentage change of the price based on token_index
# token_metadata_sales_fx['eth_usd_pct_change'] = token_metadata_sales_fx.groupby('token_index')['eth_usd'].pct_change()
# token_metadata_sales_fx['eth_usd_pct_change'] = token_metadata_sales_fx['eth_usd_pct_change'].fillna(0)
print(f'Preprocessing and merging data completed')

#%%
#%%
print(f'\n#########################################################')
print(f'Fitting the model...')
df = token_metadata_sales_fx.copy()

#remove covid periods - use data before 2019 adn after 2021
df['date'] = pd.to_datetime(df['date'])
df = df[(df['date'].dt.year < 2019) | (df['date'].dt.year > 2021)]

cb = CatBoostRegressor(verbose=0)

#split the data into train and test
X = df.drop(columns=drop_cols)
y = df[EVALUATION_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#fit the model
cb.fit(X_train, y_train)

#predict the price
y_pred = cb.predict(X_test)

#calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'Model fitted successfully')

print(f'\n#########################################################')
print(f'Feature Importance')
#plot the feature importance
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': cb.feature_importances_})
feature_importance = feature_importance.sort_values(by='importance', ascending=False)
print(f'{feature_importance}')
if not os.path.exists('output'):
    os.makedirs('output')
feature_importance.to_csv('output/feature_importance.csv', index=False)
print(f'Feature Importance saved to output/feature_importance.csv')

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='importance', y='feature')
plt.title('Feature Importance')
plt.savefig('output/feature_importance.png')
print(f'Feature Importance plot saved to output/feature_importance.png')
if SHOW_PLOTS:
    plt.show()


#%%
print(f'\n#########################################################')
print(f'Fitting the model per year...')
df = token_metadata_sales_fx.copy()
cb = CatBoostRegressor(verbose=0)
df['date'] = pd.to_datetime(df['date'])
years = df['date'].dt.year.unique()

feature_importance = pd.DataFrame()
for year in years:
    df_year = df[df['date'].dt.year == year]
    X = df_year.drop(columns=drop_cols)
    y = df_year[EVALUATION_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    cb.fit(X_train, y_train)
    #predict the price
    y_pred = cb.predict(X_test)
    #calculate the mean squared error
    mse = mean_squared_error(y_test, y_pred)
    if year == 2020 or year == 2021:
        print(f'Mean Squared Error for COVID {year}: {mse}')
    else:
        print(f'Mean Squared Error for {year}: {mse}')
    feature_importance_year = pd.DataFrame({'feature': X.columns, 'importance': cb.feature_importances_})
    feature_importance_year['year'] = year
    feature_importance = pd.concat([feature_importance, feature_importance_year])
print(f'Model fitted per year successfully')

print(f'\n#########################################################')
print(f'Feature Importance per Year')

#print the feature importance per year
print(f'{feature_importance}')
if not os.path.exists('output'):
    os.makedirs('output')
feature_importance.to_csv('output/feature_importance_per_year.csv', index=False)
print(f'Feature Importance per Year saved to output/feature_importance_per_year.csv')

#plot the feature importance per year
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='importance', y='feature', hue='year')
plt.title('Feature Importance per Year')
plt.savefig('output/feature_importance_per_year.png')
print(f'Feature Importance per Year plot saved to output/feature_importance_per_year.png')
if SHOW_PLOTS:
    plt.show()