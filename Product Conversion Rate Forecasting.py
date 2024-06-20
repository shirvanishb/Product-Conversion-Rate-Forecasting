!pip install shap

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import shap
from itertools import product

# Load the data
df = pd.read_excel('/content/CR3 - 10 product- test.xlsx', parse_dates=['date'], index_col='date')

# Define a function to process each product
def process_product(product_data, product_name):
    print(f"Processing product: {product_name}")
    print(f"Initial number of data points: {len(product_data)}")

    # Check stationarity and determine `d`
    adf_result = adfuller(product_data['conversion_rate'])
    if adf_result[1] > 0.05:
        product_data_diff = product_data['conversion_rate'].diff().dropna()
        d = 1
    else:
        d = 0

    # Determine seasonal differencing `D`
    s = 12  # Assuming monthly data and yearly seasonality
    product_data_seasonal_diff = product_data['conversion_rate'].diff(s).dropna()
    print(f"Seasonal differenced data points: {len(product_data_seasonal_diff)}")
    
    adf_result = adfuller(product_data_seasonal_diff)
    if (adf_result[1] > 0.05) and (d == 0):
        D = 1
    else:
        D = 0

    # Apply weights to the exogenous variables
    product_data['weighted_brand'] = product_data['brand'] * 1.0
    product_data['weighted_brand_popularity'] = product_data['brand_popularity'] * 1.0
    product_data['weighted_dollar_rate'] = product_data['dollar_rate'] * 1.0
    product_data['weighted_price'] = product_data['price'] * 0.5
    product_data['weighted_discount'] = product_data['discount'] * 0.2

    exog_vars = ['weighted_brand', 'weighted_brand_popularity', 'weighted_dollar_rate', 'weighted_price', 'weighted_discount']

    # Split the data into training and testing sets for this product
    train_size = int(len(product_data) * 0.8)
    train, test = product_data.iloc[:train_size], product_data.iloc[train_size:]
    exog_train = train[exog_vars]
    exog_test = test[exog_vars]

    print(f"Training data points: {len(train)}")
    print(f"Testing data points: {len(test)}")
    
    # Print lengths of exogenous variables in train and test sets
    for var in exog_vars:
        print(f"Exogenous variable '{var}' - Train size: {len(exog_train[var])}, Test size: {len(exog_test[var])}")

    # Define the p, q, P, Q ranges for grid search
    p_range = range(0, 3)
    q_range = range(0, 3)
    P_range = range(0, 2)
    Q_range = range(0, 2)
    param_combinations = list(product(p_range, q_range, P_range, Q_range))

    best_aic = float("inf")
    best_params = None

    # Grid search to find the best parameters
    for param in param_combinations:
        try:
            model = sm.tsa.SARIMAX(train['conversion_rate'],
                                   exog=exog_train,
                                   order=(param[0], d, param[1]),
                                   seasonal_order=(param[2], D, param[3], s),
                                   enforce_stationarity=False,
                                   enforce_invertibility=False)
            result = model.fit(disp=False)
            if result.aic < best_aic:
                best_aic = result.aic
                best_params = param
        except Exception as e:
            print(f"Error with parameters {param}: {e}")
            continue

    # Ensure best_params is not None before unpacking
    if best_params is not None:
        p, q, P, Q = best_params
    else:
        raise ValueError(f"No suitable model found for product: {product_name}")

    # Fit the SARIMAX model with the best parameters
    model = sm.tsa.SARIMAX(train['conversion_rate'],
                           exog=exog_train,
                           order=(p, d, q),
                           seasonal_order=(P, D, Q, s),
                           enforce_stationarity=False,
                           enforce_invertibility=False)
    result = model.fit()

    # Forecast
    forecast = result.get_forecast(steps=len(test), exog=exog_test)
    predicted_mean = forecast.predicted_mean
    conf_int = forecast.conf_int()

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train['conversion_rate'], label='Train')
    plt.plot(test.index, test['conversion_rate'], label='Test')
    plt.plot(test.index, predicted_mean, label='Forecast')
    plt.fill_between(test.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3)
    plt.xlabel('Date')
    plt.ylabel('Conversion Rate')
    plt.legend()
    plt.title(f'Conversion Rate Forecast for Product: {product_name}')
    plt.show()

    # Model summary
    print(result.summary())

    return result

# Process each product
product_names = df['product'].unique()
results = {}
for product_name in product_names:
    product_data = df[df['product'] == product_name]
    results[product_name] = process_product(product_data, product_name)

# Prepare data for SHAP analysis
features = df[['brand_popularity', 'conversion_rate_category', 'conversion_rate_product']]
scaler = StandardScaler()
normalized_features = scaler.fit_transform(features)
normalized_df = pd.DataFrame(normalized_features, columns=features.columns, index=df.index)

# Train a RandomForestRegressor model for SHAP analysis
X = normalized_df
y = df['conversion_rate']
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Calculate SHAP values
explainer = shap.Explainer(model)
shap_values = explainer(X)

# Plot SHAP values
shap.summary_plot(shap_values, X, plot_type="dot")