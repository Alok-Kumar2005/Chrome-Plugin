import json
import mlflow
import logging
from src.logger import logging
import os
import dagshub
from dotenv import load_dotenv
load_dotenv()

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

dagshub_token = os.getenv('CAPSTONE_TEST')
if not dagshub_token:
    raise ValueError("DagsHub token not found. Please set the CAPSTONE_TEST environment variable.")

os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_token
os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token

dagshub_url = 'https://dagshub.com'
repo_owner = 'vishwajitsarakpatil'
repo_name = 'Chrome-Plugin'

mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

mlflow.set_tracking_uri('https://dagshub.com/vishwajitsarakpatil/Chrome-Plugin.mlflow')
dagshub.init(repo_owner='vishwajitsarakpatil', repo_name='Chrome-Plugin', mlflow=True)


def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logging.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model info: %s', e)
        raise

def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry."""
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        
        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)
        
        # Transition the model to "Staging" stage
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"  ## keeping model to staging
        )
        
        logging.debug(f'Model {model_name} version {model_version.version} registered and transitioned to Staging.')
    except Exception as e:
        logging.error('Error during model registration: %s', e)
        raise

def main():
    try:
        model_info_path = 'reports/experiment_info.json'
        model_info = load_model_info(model_info_path)
        
        model_name = "my_model"
        register_model(model_name, model_info)
    except Exception as e:
        logging.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()