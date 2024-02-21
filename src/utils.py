import os
import json
import pickle
from time import sleep
import pandas as pd
from src.logging_helper import logger


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
    logger.info(f"[green]Loading data: {path}")

    # Automatically determines seperator between ; and \t in file
    seperator = ';' if len(open(path, 'r').readline().split(';')) > 1 else '\t'
    df = pd.read_csv(filepath_or_buffer=path,
                     header=0,
                     sep=seperator,
                     verbose=False,
                     decimal=decimal
                     )

    return df
