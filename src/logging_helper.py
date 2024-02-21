from sklearn.metrics import (r2_score,
                             mean_absolute_error,
                             explained_variance_score,
                             mean_squared_error)
import logging.config
import logging.handlers
import pathlib
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)


def setup_logging(config_path: str):
    config_file = pathlib.Path(config_path)
    with open(config_file) as file:
        config = json.load(file)
    logging.config.dictConfig(config)


def get_handler(handler_name: str) -> logging.handlers:
    for handler in logger.handlers:
        if handler.name == handler_name:
            return handler


def print_regression_metrics(y_true, y_pred):
    logger.info(f"[bold green]Regression metrics: \n"
                f"    -> R2:  {r2_score(y_true=y_true, y_pred=y_pred)}\n"
                f"    -> MAE: {mean_absolute_error(y_true=y_true, y_pred=y_pred)}\n"
                f"    -> MSE: {mean_squared_error(y_true=y_true, y_pred=y_pred)}\n"
                f"    -> VAR: {explained_variance_score(y_true=y_true, y_pred=y_pred)}\n")


def generate_run_directories(tag):
    """
    Create directories for the run
    Also sets the log file path for the logger

    """
    main_dir = init_dir(root_dir="../runs", tag=tag)
    plot_dir = os.path.join(main_dir, "plots")
    model_dir = os.path.join(main_dir, "models/")
    os.mkdir(plot_dir)
    os.mkdir(model_dir)
    handler = get_handler(handler_name="stdout")
    logger.baseFilename = os.path.join(main_dir, "log.log")

    return main_dir, model_dir, plot_dir


def init_dir(root_dir: str = "../runs", tag: str = "") -> str:
    """
    Create a directory for the run

    :param root_dir: Working directory
    :param tag: Giving the run a tag if None tag will be the current date and time
    """
    if not os.path.exists(root_dir):
        logger.info(f"-> Creating root dir: {root_dir}")
        os.mkdir(root_dir)
    if tag != "":
        if os.path.exists(os.path.join(root_dir, tag)):
            tag = f"{tag}_{datetime.now().strftime('%d_%m_%Y-%H:%M:%S')}"
        run_dir = os.path.join(root_dir, tag)
    else:
        run_dir = os.path.join(root_dir,
                               datetime.now().
                               strftime("pipeline_%d_%m_%Y-%H:%M:%S"))
    if not os.path.exists(run_dir):
        logger.info(f"-> Creating run directory: {run_dir}")
        os.mkdir(run_dir)
    return run_dir


if __name__ == '__main__':
    setup_logging('/home/daniel/SideProjects/ml_helper/configs/logging_config.json')
    logger.info('test')
    print(1)
