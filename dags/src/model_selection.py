import mlflow
import mlflow.sklearn
import pandas as pd
def get_Best_run():
    # Search for runs and retrieve logged metrics

    runs = mlflow.search_runs()
    print("Number of runs:",len(runs))
    print(runs[['run_id', 'metrics.MSE']])

    # Find the run with the least MSE
    best_run = runs.loc[runs['metrics.MSE'].idxmin()]
    print("Best run:",best_run)
    best_run_id = best_run['run_id']
    print("Best run id:",best_run_id)
    best_run_name = best_run['tags.mlflow.runName']
    print("Best run name:",best_run_name)
    # Load the model from the specified run
    loaded_model = mlflow.sklearn.load_model(f"runs:/{best_run_id}/{best_run_name}")
    print(loaded_model)

    data = pd.read_csv('data/output.csv')
    # Select a random sample from the dataset
    random_sample = data.sample(n=2)  # Adjust the number of samples as needed

    # Drop the target variable from the samples
    X = random_sample.drop(columns=['Quantity'], axis=1)

    # Make predictions using the loaded model
    predictions = loaded_model.predict(X)

    print("Prediciton made:",predictions)

if __name__=="__main__":
    get_Best_run()