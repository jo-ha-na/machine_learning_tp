import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import requests
import os

# Downloading required data
url_bike_data = "https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/Data/ml-basics/daily-bike-share.csv"

# Download daily-bike-share.csv if it doesn't exist
if not os.path.exists('daily-bike-share.csv'):
    response = requests.get(url_bike_data)
    with open('daily-bike-share.csv', 'wb') as f:
        f.write(response.content)

# Load data into a pandas dataframe
bike_data = pd.read_csv('daily-bike-share.csv')

# Feature engineering: add day of the month feature
bike_data['day'] = pd.DatetimeIndex(bike_data['dteday']).day

# Separate features and labels
X = bike_data[['season', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']].values
y = bike_data['rentals'].values

# Split data 70%-30% into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# Train and evaluate multiple models
models = {
    "Linear Regression": LinearRegression(),
    "Lasso Regression": Lasso(),
    "Decision Tree Regressor": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor(),
    "Gradient Boosting Regressor": GradientBoostingRegressor()
}

for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    # Make predictions
    predictions = model.predict(X_test)
    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    # Print evaluation metrics
    print(f"{name}:")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"R2: {r2}\n")
    # Plot predictions vs actual labels
    plt.scatter(y_test, predictions)
    plt.xlabel('Actual Labels')
    plt.ylabel('Predicted Labels')
    plt.title(f'{name} - Daily Bike Share Predictions')
    # overlay the regression line
    z = np.polyfit(y_test, predictions, 1)
    p = np.poly1d(z)
    plt.plot(y_test, p(y_test), color='magenta')
    plt.show()

# Visualize the Decision Tree model
tree = export_text(models["Decision Tree Regressor"], feature_names=['season', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed'])
print(tree)