import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import pickle

# load the dataset
df = pd.read_csv('data/used_cars.csv')

# Preprocess the data
df = df.dropna()  # Handle missing values by dropping

# encode categorical variables
label_encoders = {}
for col in ['city','make','model','variant','fuel_type','color','body_type','transmission']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define features and target variable
feature_cols = ['city','make','model','mileage','make_year','fuel_type','transmission','no_of_owners']
x = df[feature_cols]
y = df['price']

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(random_state=42)
model.fit(x_train, y_train)
print("Model training completed.")

# Evaluate the model
# high r2_score and low mae indicates a good fit
# 
y_pred = model.predict(x_test)
mse = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Model Evaluation:\nMean Absolute Error: {mse}\nR^2 Score: {r2}")

with open('models/used_cars_prediction.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('models/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

print("Model and label encoders saved.")

