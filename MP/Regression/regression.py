import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# 1. Load data
temp = pd.read_csv("../csv_files/annual_temperature_data.csv")

# We will use the annual mean absolute temperature
X = temp[['Year']]
y = temp['Annual_Mean_Absolute']

# 2. Linear Regression Model
model = LinearRegression()
model.fit(X, y)

# 3. Make forecast for the next decade
last_year = temp['Year'].max()
future_years = np.arange(last_year + 1, last_year + 11).reshape(-1, 1)

future_predictions = model.predict(future_years)

# 4. Model performance

y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
rmse = mean_squared_error(y, y_pred, squared=False)

# 5. Print results
print("=== LINEAR REGRESSION RESULTS ===")
print(f"R-squared: {r2:.4f}")
print(f"RMSE: {rmse:.4f} °C\n")

print("=== 10-YEAR TEMPERATURE FORECAST ===")
for year, temp_pred in zip(future_years.flatten(), future_predictions):
    print(f"{year}: {temp_pred:.4f} °C")