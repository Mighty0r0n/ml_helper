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


def mc_dropout(model, X, T=100):
    """Monte Carlo Dropout for uncertainty estimation
    Model needs to be pretrained, prediction will be an ensemble decision of T forward passes
    In this ensemble dropout is still active, which isn't the norm for predicting with
    models using dropout. This is why the model is called with training=True
    
    Args:
        model (keras model): Trained model
        X (ndarray): Input data
        T (int, optional): Number of forward passes. Defaults to 100.

    Returns:
        ndarray: Mean prediction
        ndarray: Standard deviation
    """
    y_probas = np.stack([model(X, training=True) for sample in range(T)])
    y_mean = y_probas.mean(axis=0)
    y_std = y_probas.std(axis=0)
    
    return y_mean, y_std

def make_plotly_figure(y_true, y_pred, y_mean):

    # fit through mean
    reg = LinearRegression(fit_intercept=False)
    reg.fit(y_true.reshape(-1, 1), y_mean)

    # adjust plot size bc of many data
    plt.figure(figsize=(16, 14))

    # scatter first the mean pred values
    plt.scatter(x=y_true, y=y_mean, label='Actual', c="blue", s=20)
    for i in range(len(y_pred)):
        # scatter now every layer of np.stack monte carlo prediction
        # more opaque for trace like view
        plt.scatter(x=y_true, y=y_pred[i].reshape(-1,), label=f'Predictions {i+1}', c='red', alpha=0.01, s=10)

    # plot the fitted line through the origin
    x_fit = np.linspace(min(y_true), max(y_true), 100).reshape(-1, 1)
    y_fit = reg.predict(x_fit)
    plt.plot(x_fit, y_fit, color='green', label=f"Fitted Line:y = {reg.coef_[0]}x")

    # set view limits of plot axis
    plt.xlim([-3, 4])
    plt.ylim([-3, 4])
    plt.xlabel('Actual')
    plt.ylabel('Predicted')

    # big line to origin
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)

    #plt.legend()
    plt.grid(color='gray', linestyle='--', linewidth=0.5)

    plt.savefig("../test_data/plot.png")


if __name__ == '__main__':
    # Example usage
    project_root = ".."
    markdown_tree = generate_markdown_tree(project_root)

    print(markdown_tree)
