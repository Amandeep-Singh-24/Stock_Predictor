'''
-------------------ATTENTION------------------------
Please check README.md

REFER TO COMMON STEPS FOR ALL SYSTEMS, IF THOSE
WORK YOU MAY RUN THE PROGRAM (bordered section)
- Must have python3 up to date along with pip
- If vscode make sure to (in settings):
Set the default interpreter path in VSCode to `python3`

If you are without wifi, documentation includes
all outputs of the program.
--------------------------------------------------
'''
import subprocess
import sys

# Function to install required Python packages
def install(package):
    # Use subprocess to run pip and install the specified package
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# List of required packages for data analysis and visualization
required_packages = ["pandas", "numpy", "matplotlib", "scikit-learn"]

# Check if required packages are installed, install them if not
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        install(package)

# Importing necessary libraries for data analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from datetime import timedelta

# Function to suggest the best model based on RMSE, R², and future price
def suggest_best_model(rmse_values, r2_values, future_prices):
    best_model = None
    best_score = float('inf')
    best_model_reasoning = ""

    for model, rmse in rmse_values.items():
        r2 = r2_values[model]
        future_price = future_prices[model][0]  # Predicted price for 3 months later

        # Check if the future price is within a plausible range
        if future_price < 0 or future_price > 1000:  # Arbitrary threshold, can be adjusted
            continue

        # Calculate a score considering RMSE and R² (can be adjusted)
        score = rmse - r2 * 1000  # Giving more weight to R²

        if score < best_score:
            best_score = score
            best_model = model
            best_model_reasoning = (
                f"Model '{model}' is chosen as the best because it has a lower RMSE of {rmse:.2f} (indicating lower prediction errors) "
                f"and a higher R² of {r2:.2f} (indicating better fit to the data). "
                f"Additionally, its future price prediction of {future_price:.2f} USD is considered realistic."
            )

    return best_model, best_model_reasoning

# Load the Apple stock data from a CSV file
file_path = 'AAPL.csv'
apple_stock_data = pd.read_csv(file_path)

# Convert the 'Date' column to datetime format
apple_stock_data['Date'] = pd.to_datetime(apple_stock_data['Date'])

# Creating a 'Time' column for regression analysis
apple_stock_data['Time'] = np.arange(len(apple_stock_data))

# Selecting relevant columns for analysis
stock_data = apple_stock_data[['Date', 'Close', 'Time']]

# Setting 'Time' as an independent variable and 'Close' price as the dependent variable
X = stock_data['Time'].values.reshape(-1, 1)
y = stock_data['Close'].values

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Polynomial Regression Analysis
degrees = [2, 3, 4, 5]
models = []  # Store regression models for different degrees
predictions = []  # Store predictions for different degrees
rmse_values = {}  # Store RMSE values for different degrees
r2_values = {}  # Store R² values for different degrees
future_prices = {}  # Store future price predictions for different degrees

print("\nPolynomial Regression Analysis:\n")
for degree in degrees:
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)
    models.append(model)

    y_pred = model.predict(X_poly)
    predictions.append(y_pred)

    # Store RMSE and R² values
    rmse = mean_squared_error(y, y_pred, squared=False)
    r2 = r2_score(y, y_pred)
    rmse_values[f'Degree {degree}'] = rmse
    r2_values[f'Degree {degree}'] = r2

    # Model Evaluation
    train_rmse = mean_squared_error(y_train, model.predict(poly_features.transform(X_train)), squared=False)
    test_rmse = mean_squared_error(y_test, model.predict(poly_features.transform(X_test)), squared=False)
    train_r2 = r2_score(y_train, model.predict(poly_features.transform(X_train)))
    test_r2 = r2_score(y_test, model.predict(poly_features.transform(X_test)))

    print(f"Degree {degree}:")
    print(f"  - Train RMSE = {train_rmse:.2f}")
    print(f"  - Test RMSE = {test_rmse:.2f}")
    print(f"  - Train R² = {train_r2:.2f}")
    print(f"  - Test R² = {test_r2:.2f}\n")

# Plotting Polynomial Regression Results
plt.figure(figsize=(14, 8))
plt.plot(stock_data['Date'], stock_data['Close'], label='Actual Close Price', color='blue')
for i, degree in enumerate(degrees):
    plt.plot(stock_data['Date'], predictions[i], label=f'Degree {degree} Fit', linestyle='--')
plt.title('Polynomial Regression Fits to Apple Stock Closing Prices')
plt.xlabel('Date')
plt.ylabel('Closing Price (USD)')
plt.legend()
plt.grid(True)
plt.show()

# Exponential Regression
y_transformed = np.log(stock_data['Close'].values)
X_train, X_test, y_transformed_train, y_transformed_test = train_test_split(X, y_transformed, test_size=0.2, random_state=0)

exp_model = LinearRegression()
exp_model.fit(X_train, y_transformed_train)

y_transformed_pred = exp_model.predict(X)
y_exp_pred = np.exp(y_transformed_pred)

# Store RMSE and R² for Exponential Model
rmse_values['Exponential'] = mean_squared_error(y, y_exp_pred, squared=False)
r2_values['Exponential'] = r2_score(y, y_exp_pred)

exp_model_rmse = mean_squared_error(y_transformed_test, exp_model.predict(X_test), squared=False)
exp_model_r2 = r2_score(y_transformed_test, exp_model.predict(X_test))

# Plotting Exponential Regression Results
plt.figure(figsize=(14, 8))
plt.plot(stock_data['Date'], stock_data['Close'], label='Actual Close Price', color='blue')
plt.plot(stock_data['Date'], y_exp_pred, label='Exponential Model Fit', color='red', linestyle='--')
plt.title('Exponential Regression Fit to Apple Stock Closing Prices')
plt.xlabel('Date')
plt.ylabel('Closing Price (USD)')
plt.legend()
plt.grid(True)
plt.show()

# Future Stock Price Predictions
future_months = [3, 6, 9, 12]
future_dates = [stock_data['Date'].iloc[-1] + timedelta(days=30 * month) for month in future_months]
future_times = [len(stock_data) + 30 * month for month in future_months]

degree_predictions = {}
for i, degree in enumerate(degrees):
    poly_features = PolynomialFeatures(degree=degree)
    X_future_poly = poly_features.fit_transform(np.array(future_times).reshape(-1, 1))
    future_pred = models[i].predict(X_future_poly)
    degree_predictions[degree] = future_pred

svr_model = SVR(kernel='rbf', C=100, epsilon=0.1)
svr_model.fit(X_train, y_train)
svr_predictions = svr_model.predict(np.array(future_times).reshape(-1, 1))
y_svr_pred = svr_model.predict(X)

# Plotting SVR Regression Results
plt.figure(figsize=(14, 8))
plt.plot(stock_data['Date'], stock_data['Close'], label='Actual Close Price', color='blue')
plt.plot(stock_data['Date'], y_svr_pred, label='SVR Model Fit', color='green', linestyle='--')
plt.title('Support Vector Regression (SVR) Fit to Apple Stock Closing Prices')
plt.xlabel('Date')
plt.ylabel('Closing Price (USD)')
plt.legend()
plt.grid(True)
plt.show()

# Store RMSE and R² for SVR Model
rmse_values['SVR'] = mean_squared_error(y, y_svr_pred, squared=False)
r2_values['SVR'] = r2_score(y, y_svr_pred)

# Plotting Future Predictions
plt.figure(figsize=(14, 8))
plt.plot(stock_data['Date'], stock_data['Close'], label='Actual Close Price', color='blue')

# Plot for Polynomial Degrees 2 to 4
for i, model in enumerate(models):
    degree = degrees[i]
    poly_features = PolynomialFeatures(degree=degree)
    X_future_poly = poly_features.fit_transform(np.array(future_times).reshape(-1, 1))
    future_pred = model.predict(X_future_poly)
    future_prices[f'Degree {degree}'] = future_pred.tolist()
    plt.plot(future_dates, future_pred, label=f'Degree {degree} Prediction', linestyle='--')

# Plot for Exponential Model
plt.plot(future_dates, np.exp(exp_model.predict(np.array(future_times).reshape(-1, 1))), label='Exponential Model Prediction', color='red', linestyle='--')
# Future predictions for Exponential Model
exp_future_pred = np.exp(exp_model.predict(np.array(future_times).reshape(-1, 1)))
future_prices['Exponential'] = exp_future_pred.tolist()

# Plot for SVR Model
svr_future_pred = svr_model.predict(np.array(future_times).reshape(-1, 1))
plt.plot(future_dates, svr_future_pred, label='SVR Model Prediction', color='green', linestyle='--')
# Future predictions for SVR Model
svr_future_pred = svr_model.predict(np.array(future_times).reshape(-1, 1))
future_prices['SVR'] = svr_future_pred.tolist()
plt.title('Future Stock Price Predictions')
plt.xlabel('Date')
plt.ylabel('Closing Price (USD)')
plt.ylim(0, 500)  # Setting the maximum y-value to 500
plt.legend()
plt.grid(True)
plt.show()

# Printing Future Predictions
print("Future Stock Price Predictions:")
for month, date in zip(future_months, future_dates):
    print(f"\n{month} Months Later (by {date.date()}):")
    print("Polynomial:")
    for degree in degrees:
        print(f"  Degree {degree}: {degree_predictions[degree][future_months.index(month)]:.2f} USD")
    print(f"Exponential: {np.exp(exp_model.predict(np.array([len(stock_data) + 30 * month]).reshape(-1, 1)))[0]:.2f} USD")
    print(f"SVR: {svr_predictions[future_months.index(month)]:.2f} USD")

best_model, reasoning = suggest_best_model(rmse_values, r2_values, future_prices)
print(f"\nThe best model for investment is: {best_model}")
print(f"Reasoning: {reasoning}")
# Note on Investment Risks
print("\nNote: Stock market investments carry risks, and predictions based on historical data should not be the sole basis for investment decisions.")
