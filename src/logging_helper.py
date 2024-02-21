from sklearn.metrics import (r2_score,
                             mean_absolute_error,
                             explained_variance_score,
                             mean_squared_error, PredictionErrorDisplay)
import logging.config
import logging.handlers
import pathlib
import json
import os
from datetime import datetime
from time import sleep

logger = logging.getLogger(__name__)


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


def generate_run_directories(tag):
    """

    :return:
    """
    main_dir = init_dir(root_dir="../runs", tag=tag)
    try:
        plot_dir = os.path.join(main_dir, "plots")
        model_dir = os.path.join(main_dir, "models/")
        error_dir = os.path.join(plot_dir, "plots_error")
        os.mkdir(plot_dir)
        os.mkdir(error_dir)
        os.mkdir(model_dir)
    except FileExistsError:
        pass
    # console.log(f"[green]Starting the pipeline!")
    sleep(0.75)
    return error_dir, main_dir, model_dir


def init_dir(root_dir: str = "../runs", tag: str = "") -> str:
    if os.path.exists(root_dir):
        tag = f"{tag}_{datetime.now().strftime('%d_%m_%Y-%H:%M:%S')}"

    if not os.path.exists(root_dir):
        logger.info(f"-> Creating root dir: {root_dir}")
        os.mkdir(root_dir)
    if tag != "":
        run_dir = os.path.join(root_dir, tag)
        # str(datetime.now().
        #     strftime("pipeline_%d_%m_%Y-%H:%M:%S") + "-" + tag))
    else:
        run_dir = os.path.join(root_dir,
                               datetime.now().
                               strftime("pipeline_%d_%m_%Y-%H:%M:%S"))
    if not os.path.exists(run_dir):
        logger.info(f"-> Creating run directory: {run_dir}")
        os.mkdir(run_dir)
    return run_dir
