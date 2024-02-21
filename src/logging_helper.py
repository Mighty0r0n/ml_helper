import logging.config
import logging.handlers
import pathlib
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)


def setup_logging(config_path: str):
    """
    Sets the logging configuration for the project

    :param config_path: path to the logging configuration file
    """
    config_file = pathlib.Path(config_path)
    with open(config_file) as file:
        config = json.load(file)
    logging.config.dictConfig(config)


def get_handler(handler_name: str) -> logging.handlers:
    """
    Get the handler by name

    :param handler_name: handler name to get
    """
    for handler in logging.getLogger().handlers:
        if handler.name == handler_name:
            return handler


def change_file_handler_path(handler_name: str, new_path: str):
    """
    Change the file path of the file handler

    :param handler_name: handler name to change
    :param new_path: new file path
    """
    handler = get_handler(handler_name=handler_name)
    # Close the existing handler
    handler.close()

    # Create a new FileHandler with the desired file path
    new_handler = logging.FileHandler(new_path)
    new_handler.setLevel(handler.level)
    new_handler.setFormatter(handler.formatter)

    # Replace the old handler with the new one
    logger = logging.getLogger()
    logger.removeHandler(handler)
    logger.addHandler(new_handler)





def generate_run_directories(log_name: str, tag: str = ""):
    """
    Create directories for the run
    Also sets the log file path for the logger

    """
    main_dir = init_dir(root_dir="../runs", tag=tag)
    plot_dir = os.path.join(main_dir, "plots/")
    model_dir = os.path.join(main_dir, "models/")
    log_dir = os.path.join(main_dir, "logs/")
    os.mkdir(plot_dir)
    os.mkdir(model_dir)
    os.mkdir(log_dir)
    change_file_handler_path(handler_name="file", new_path=log_dir + f"{log_name}.log")

    return main_dir, model_dir, plot_dir, log_dir


def init_dir(tag: str, root_dir: str = "../runs") -> str:
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
