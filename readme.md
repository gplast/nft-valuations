# Project Title

Providing the exercise for NFTvaluation purpose.

## Installation

# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Run main.py
python main.py

## Customization

You can customize the behavior of the `main.py` script by adjusting the following variables:

- `SHOW_PLOTS`: Set this variable to `True` if you want to display plots during the evaluation process. Set it to `False` otherwise.

- `EVALUATION_COLUMN`: Specify the column name in your dataset that contains the values to be evaluated. This was used for testing purposed mostly Options are 'eth', 'usd', 'eth_usd', 'eth_usd_normalized'

