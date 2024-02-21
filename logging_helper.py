from sklearn.metrics import (r2_score,
                             mean_absolute_error,
                             explained_variance_score,
                             mean_squared_error, PredictionErrorDisplay)
import logging.config
import logging.handlers
import pathlib
import json


logger = logging.getLogger(__name__)

logging_config = {
    "version": 1,
    "disable_existing_loggers": False,  # False enables third-party logs to be displayed
    "formatters": {  # Define the format of the logs, i should be able to add more formats for error or debug logs
        "simple": {  # simple format for displaying logs
            "format": "%(levelname)s: %(message)s"
        }
    },
    "handlers": {  # Define the handlers for the logs
        "stdout": {  # stdout handler for displaying logs in the console USE STDOUT PLEASE PER CONVENTION
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "formatter": "simple",
            "level": "INFO"
        }
    },
    "loggers": {  # Define the loggers
        "root": {  # root logger
            "handlers": ["stdout"],
            "level": "INFO"
        }
    }
}
def setup_logging(config_path: str):
    config_file = pathlib.Path(config_path)
    with open(config_file) as file:
        config = json.load(file)
    logging.config.dictConfig(config)


def print_regression_metrics(y_true, y_pred):
    logger.info(f"[bold green]Regression metrics: \n"
                f"    -> R2:  {r2_score(y_true=y_true, y_pred=y_pred)}\n"
                f"    -> MAE: {mean_absolute_error(y_true=y_true, y_pred=y_pred)}\n"
                f"    -> MSE: {mean_squared_error(y_true=y_true, y_pred=y_pred)}\n"
                f"    -> VAR: {explained_variance_score(y_true=y_true, y_pred=y_pred)}\n")