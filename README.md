# Used Car Price Prediction 🚗🚗🚗

This project is a machine learning web app for predicting the price of used cars in India. Model used to train - Random Forest Regressor

## Features

- Predicts used car prices based on user input
- Interactive Streamlit UI
- Handles categorical variables with label encoding
- Model trained using Random Forest for high accuracy
- Outputs prices formatted in Indian currency style

## How It Works

1. **Data Preprocessing:**  
   - Loads and cleans the dataset (`data/used_cars.csv`)
   - Encodes categorical features using `LabelEncoder`
   - Drops missing values for robust training

2. **Model Training:**  
   - Splits data into training and test sets
   - Trains a `RandomForestRegressor`
   - Evaluates with MAE and R² metrics
   - Saves the trained model and encoders to the `models/` directory

3. **Prediction App:**  
   - Loads the trained model and encoders
   - Accepts user input for car features (brand, model, city, mileage, year, fuel types, transmission, owners)
   - Encodes inputs and displays the predicted price

### Prerequisites

- Python 3.7+
- Required packages:  
  `pandas`, `scikit-learn`, `streamlit`

### Installation

1. Clone the repository:
   ```sh
   git clone 
   cd used cars
   ```

2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

3. Place the dataset in the `data/` folder as `used_cars.csv`.

### Training the Model

Run the training script:
```sh
python trainmodel.py
```
This will train the model and save it to the `models/` directory.

### Running the App

Start the Streamlit app:
```sh
streamlit run app.py
```

## Usage

- Select car details in the web interface.
- Click "Predict Price" to get the estimated price.
- The result is displayed in Indian currency (INR).

## Project Structure

```
used cars/
├── app.py
├── trainmodel.py
├── data/
│   └── used_cars.csv
├── models/
│   ├── used_cars_prediction.pkl
│   └── label_encoders.pkl
├── requirements.txt
└── README.md
```

**Contributors:**  
- Shajil BP