import mlflow
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

def arima_model(train_data, test_data, p,d,q, model_name='ARIMA_model'):

    with mlflow.start_run():
        model = ARIMA(train_data, order=(p,d,q))
        model_fit = model.fit()
        forecast = model_fit.get_forecast(steps=len(test_data))

        mse = mean_squared_error(test_data, forecast)
        
        mlflow.log_param('p', p)
        mlflow.log_param('d', d)
        mlflow.log_param('q', q)
        mlflow.log_metric('MSE', mse)
        mlflow.statsmodels.log_model(model_fit, model_name)

    return model_fit
 