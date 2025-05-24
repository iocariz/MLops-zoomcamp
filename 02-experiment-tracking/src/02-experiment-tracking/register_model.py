import os
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List, Union

import click
import mlflow
import mlflow.sklearn
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Configuration
HPO_EXPERIMENT_NAME = "random-forest-hyperopt"
EXPERIMENT_NAME = "random-forest-best-models_v2"
RF_PARAMS = ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'random_state']
MODEL_NAME = "nyc-taxi-regressor"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_mlflow(tracking_uri: str = "http://127.0.0.1:5000") -> None:
    """Initialize MLflow configuration."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.sklearn.autolog(disable=True)


def load_pickle(filename: Union[Path, str]) -> Any:
    """Load pickle file with error handling."""
    try:
        with open(filename, "rb") as f_in:
            return pickle.load(f_in)
    except FileNotFoundError:
        logger.error(f"File not found: {filename}")
        raise
    except pickle.UnpicklingError:
        logger.error(f"Failed to unpickle file: {filename}")
        raise


def load_data(data_path: Union[Path, str]) -> Tuple[Tuple, Tuple, Tuple]:
    """Load train, validation, and test datasets."""
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path}")
    
    train_data = load_pickle(data_path / "train.pkl")
    val_data = load_pickle(data_path / "val.pkl")
    test_data = load_pickle(data_path / "test.pkl")
    
    return train_data, val_data, test_data


def prepare_params(params: Dict[str, str]) -> Dict[str, Any]:
    """Convert and validate parameters for RandomForest."""
    new_params = {}
    
    for param in RF_PARAMS:
        if param in params:
            try:
                # Handle random_state separately as it might be None
                if param == 'random_state' and params[param] == 'None':
                    new_params[param] = None
                else:
                    new_params[param] = int(params[param])
            except (ValueError, TypeError):
                logger.warning(f"Invalid parameter value for {param}: {params[param]}")
                continue
    
    return new_params


def train_and_log_model(data_path: Union[Path, str], params: Dict[str, str]) -> None:
    """Train model and log metrics to MLflow."""
    try:
        # Load data
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(data_path)
        
        with mlflow.start_run():
            # Prepare parameters
            model_params = prepare_params(params)
            logger.info(f"Training model with parameters: {model_params}")
            
            # Train model
            rf = RandomForestRegressor(**model_params)
            rf.fit(X_train, y_train)
            
            # Evaluate model
            val_predictions = rf.predict(X_val)
            test_predictions = rf.predict(X_test)
            
            # Calculate RMSE (compatible with older sklearn versions)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
            test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
            
            # Log metrics
            mlflow.log_metric("val_rmse", val_rmse)
            mlflow.log_metric("test_rmse", test_rmse)
            
            logger.info(f"Model trained - Val RMSE: {val_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
            
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise


def get_top_runs(client: MlflowClient, experiment_name: str, top_n: int) -> List:
    """Retrieve top N runs from hyperparameter optimization experiment."""
    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise ValueError(f"Experiment '{experiment_name}' not found")
        
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=top_n,
            order_by=["metrics.rmse ASC"]
        )
        
        if not runs:
            raise ValueError(f"No runs found in experiment '{experiment_name}'")
        
        logger.info(f"Retrieved {len(runs)} top runs from {experiment_name}")
        return runs
        
    except MlflowException as e:
        logger.error(f"MLflow error retrieving runs: {str(e)}")
        raise


def get_best_run(client: MlflowClient, experiment_name: str):
    """Get the best run based on test RMSE."""
    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise ValueError(f"Experiment '{experiment_name}' not found")
        
        best_runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=1,
            order_by=["metrics.test_rmse ASC"]
        )
        
        if not best_runs:
            raise ValueError(f"No runs found in experiment '{experiment_name}'")
        
        return best_runs[0]
        
    except MlflowException as e:
        logger.error(f"MLflow error retrieving best run: {str(e)}")
        raise


def register_best_model(client: MlflowClient, best_run, model_name: str) -> None:
    """Register the best model in MLflow Model Registry."""
    try:
        model_uri = f"runs:/{best_run.info.run_id}/model"
        
        result = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )
        
        logger.info(f"Model registered successfully: {model_name} (version {result.version})")
        logger.info(f"Best run ID: {best_run.info.run_id}")
        logger.info(f"Test RMSE: {best_run.data.metrics.get('test_rmse', 'N/A')}")
        
    except MlflowException as e:
        logger.error(f"Error registering model: {str(e)}")
        raise


@click.command()
@click.option(
    "--data_path",
    default="./output",
    type=click.Path(exists=True, path_type=Path),
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--top_n",
    default=5,
    type=int,
    help="Number of top models that need to be evaluated to decide which one to promote"
)
@click.option(
    "--tracking_uri",
    default="http://127.0.0.1:5000",
    help="MLflow tracking server URI"
)
@click.option(
    "--model_name",
    default=MODEL_NAME,
    help="Name for the registered model"
)
def run_register_model(
    data_path: Path, 
    top_n: int, 
    tracking_uri: str,
    model_name: str
) -> None:
    """
    Main function to retrieve top models, retrain them, and register the best one.
    
    This script:
    1. Retrieves the top N runs from the hyperparameter optimization experiment
    2. Retrains these models and logs them to a new experiment
    3. Selects the model with the lowest test RMSE
    4. Registers the best model in MLflow Model Registry
    """
    try:
        # Setup
        setup_mlflow(tracking_uri)
        client = MlflowClient()
        
        logger.info(f"Starting model registration process with top {top_n} models")
        
        # Retrieve top runs and retrain models
        top_runs = get_top_runs(client, HPO_EXPERIMENT_NAME, top_n)
        
        for i, run in enumerate(top_runs, 1):
            logger.info(f"Training model {i}/{len(top_runs)}")
            train_and_log_model(data_path=data_path, params=run.data.params)
        
        # Select and register the best model
        best_run = get_best_run(client, EXPERIMENT_NAME)
        register_best_model(client, best_run, model_name)
        
        logger.info("Model registration process completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        raise


if __name__ == '__main__':
    run_register_model()