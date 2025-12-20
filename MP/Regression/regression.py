import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# 1. Load data
temp = pd.read_csv("../csv_files/annual_temperature_data.csv")

# Use the annual mean absolute temperature
X = temp[['Year']].copy()
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
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)

# 5. Save model performance to CSV
performance_df = pd.DataFrame({
    "Metric": ["R-squared", "MSE", "RMSE"],
    "Value": [r2, mse, rmse]
})

performance_df.to_csv("../csv_files/model_performance.csv", index=False)

# 6. Save 10-year forecast to CSV
forecast_df = pd.DataFrame({
    "Year": future_years.flatten(),
    "Predicted_Annual_Mean_Absolute_Temperature": future_predictions
})

forecast_df.to_csv("../csv_files/temperature_forecast_10_years.csv", index=False)