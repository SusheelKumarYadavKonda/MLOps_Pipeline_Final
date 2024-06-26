import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from urllib.parse import urlparse
from mlflow.utils.environment import _mlflow_conda_env
import cloudpickle
import time
import sklearn


class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        return self.model.predict(model_input)

def train_sgd():
    data = pd.read_csv('data/output.csv')
    with mlflow.start_run(run_name='sgd_regressor'):
        
        X = data.drop(columns=['Quantity'],axis=1)
        y = data['Quantity']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        #Linear Regression:
        sgd_regressor = SGDRegressor(max_iter=1000, tol=1e-3, penalty='l2', alpha=0.001)
        sgd_regressor.fit(X_train, y_train)

        predictions = sgd_regressor.predict(X_test)
        mse_1 = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions) 
        
        mlflow.log_metric("MSE",mse_1)
        mlflow.log_metric("MAE",mae)
        mlflow.log_metric("r2",r2)

        wrappedModel = SklearnModelWrapper(sgd_regressor)
        signature = infer_signature(X,wrappedModel.predict(None, X_train))

        conda_env =  _mlflow_conda_env(
            additional_conda_deps=None,
            additional_pip_deps=["cloudpickle=={}".format(cloudpickle.__version__), "scikit-learn=={}".format(sklearn.__version__)],
            additional_conda_channels=None,
        )
        mlflow.pyfunc.log_model("sgd_regressor",
                            python_model=wrappedModel,
                            conda_env=conda_env,
                            signature=signature)
        #Register model
        run_id = mlflow.search_runs(filter_string='tags.mlflow.runName = "sgd_regressor"').iloc[0].run_id
        model_name = "online_sales_sgd_regressor"
        print("run_id:",run_id)
        model_version = mlflow.register_model(f"runs:/{run_id}/sgd_regressor", model_name)
        mlflow.log_param('Model', "online_sales_model")
        mlflow.sklearn.log_model(sgd_regressor, "sgd_regressor")
        # Registering the model takes a few seconds, so add a small delay
        time.sleep(15)

if __name__=="__main__":
    train_sgd()