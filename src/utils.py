import json
import os
import pickle
from time import sleep

import pandas as pd
from sklearn.metrics import (r2_score,
                             mean_absolute_error,
                             explained_variance_score,
                             mean_squared_error)

from logging_assistance.logging_helper import logger


def save_params(logdir, filename, params):
    with open(os.path.join(logdir, filename + "_parameters_.json"), 'w', encoding='utf8') as f:
        json.dump(params, f, indent=2)


def save_model(model, path="./model.pkl"):
    with open(path, mode="wb") as outfile:
        pickle.dump(model, outfile)


def load_model(path):
    with open(path, mode="rb") as infile:
        model = pickle.load(infile)
    return model


def load_data(path: str, decimal: str) -> pd.DataFrame:
    sleep(0.75)
    logger.info(f"Loading data: {path}")

    # Automatically determines seperator between ; and \t in file
    # to-do: should i add more seperators?
    # POSSIBLE BUG WITH THIS EXPRESSION! Since for example .gff files can contain multiple line seperators with their
    # attribute fields. This could lead to a wrong seperator detection since those files contain \t and ; as seperators
    seperator = ';' if len(open(path, 'r').readline().split(';')) > 1 else '\t'
    df = pd.read_csv(filepath_or_buffer=path,
                     header=0,
                     sep=seperator,
                     verbose=False,
                     decimal=decimal
                     )

    return df


def generate_markdown_tree(root_path: str, file_extensions: list = None, indent=0) -> str:
    """
    Generate a markdown tree of the given directory
    Only used for documentation purposes

    :param root_path: Path to the root directory
    :param file_extensions: List of file extensions to include in the tree
    :param indent: Indentation level for the tree
    :return: Markdown representation of the directory tree
    """
    tree = ''
    files = []
    directories = []

    for item in os.listdir(root_path):
        item_path = os.path.join(root_path, item)

        if os.path.isdir(item_path):
            if not item.startswith("__") and not item.startswith(".") and not item.startswith("runs"):
                directories.append(item)
        else:
            files.append(item)

    for directory in sorted(directories):
        tree += f"{'|   ' * indent}├── {directory}\\\n"
        subtree = generate_markdown_tree(os.path.join(root_path, directory), file_extensions, indent + 1)
        tree += subtree

    for file in sorted(files):
        if file_extensions is None or any(file.endswith(ext) for ext in file_extensions):
            tree += f"{'|   ' * indent}├── {file}\\\n"

    return tree


def print_regression_metrics(y_true, y_pred) -> str:
    return (f"Regression metrics: \n"
            f"    -> R2:  {r2_score(y_true=y_true, y_pred=y_pred)}\n"
            f"    -> MAE: {mean_absolute_error(y_true=y_true, y_pred=y_pred)}\n"
            f"    -> MSE: {mean_squared_error(y_true=y_true, y_pred=y_pred)}\n"
            f"    -> VAR: {explained_variance_score(y_true=y_true, y_pred=y_pred)}\n")


if __name__ == '__main__':
    # Example usage
    project_root = ".."
    markdown_tree = generate_markdown_tree(project_root)

    print(markdown_tree)
