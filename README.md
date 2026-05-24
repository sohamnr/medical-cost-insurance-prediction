# Medical Insurance Cost Prediction

A polished Streamlit app and training script for estimating annual medical insurance charges from demographic and health-related inputs.

## Project Contents

- `app.py`: Streamlit interface for interactive predictions
- `main.py`: training script that rebuilds and saves `model.pkl`
- `model.pkl`: saved scikit-learn pipeline used by the app
- `dataset/insurance.csv`: original dataset
- `dataset/processed_insurance.csv`: processed dataset used for training
- `eda.ipynb` and `train.ipynb`: exploratory analysis and notebook-based experimentation

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Train the Model

Run the training script to create or refresh `model.pkl`:

```bash
python main.py
```

The script:

- loads the processed dataset
- applies preprocessing with one-hot encoding and scaling
- trains a `RandomForestRegressor`
- prints train and test metrics
- saves the trained pipeline to `model.pkl`

## Launch the App

Start the Streamlit interface with:

```bash
streamlit run app.py
```

The app:

- loads the saved pipeline from `model.pkl`
- collects the same input fields used during training
- predicts estimated annual insurance charges
- displays the user inputs alongside the prediction

## Notes

- The app is intended for educational and portfolio use.
- If `model.pkl` is missing, run `python main.py` first.
- If the app cannot load the model, verify that your installed package versions are compatible with the saved file.
