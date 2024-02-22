import logging.config
import logging.handlers
import pathlib
import json
import os
from datetime import datetime

"""
For basic usage of the logging helper, you can use the following code snippet:


    # import the logging helper
    from logging_helper import setup_logging, generate_run_directories
    from logging_helper import logger
    
    # with the logger imported you only need to call logger.LEVEL("MESSAGE") instead of print("MESSAGE")
    # Replace LEVEL for the desired log level (info, debug, warning, error, critical) (eg. logger.info("MESSAGE"))
    
    # Setup the logging
    setup_logging('YOUR/CONFIG/PATH.json')

    # Create the run directories
    
    # If you want to create multiple models in the same run, you can wrap the generate_run_directories()
    # function inside a for loop and give the tag a unique name for each model
    # the generate_run_directories() function will create a directory for each individual run and also
    # will take care, that logging files are stored in the correct directory dynamically so the user
    # won't have to redirect the logging file path for each run.
    
    # for run in pipeline:  <---- Include for loop for multi run
    
    main_dir, model_dir, plot_dir, log_dir = generate_run_directories(
        log_name=NAME_OF_LOG_FILE,  # preferably the name of the model
        tag=f"NAME_OF_SINGLE_RUN_FOLDER"  # preferably the name of the model and architecture 
                                          # or something to identify the model
    )
    
With the above code snippet, you will get a dynamically growing directory structure for your runs.
     
For further advanced usage you can add new filters, handlers, and loggers here in this file
and use them in the logging_config.json file. (For example, the StartWithFilter)
"""

# Create a logger
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
    :return handler: handler with the given name
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


def generate_run_directories(log_name: str, tag: str = "") -> tuple[str, str, str, str]:
    """
    Create directories for the run
    Also sets the log file path for the logger

    :param log_name: Name of the log file
    :param tag: Giving the run a tag if None tag will be the current date and time
                Used for Folder naming, preferably the name of the model or something
                to identify the run
    :return: main_dir, model_dir, plot_dir, log_dir paths
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
    :return: Path to the run directory
    """
    if not os.path.exists(root_dir):
        print(f"-> Creating root dir: {root_dir}")
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
        print(f"-> Creating run directory: {run_dir}")
        os.mkdir(run_dir)
    return run_dir



if __name__ == '__main__':
    setup_logging('/home/daniel/SideProjects/ml_helper/configs/logging_config.json')
    logger.info('test')
    print(1)
