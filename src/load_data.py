import pandas as pd
from src.utils import load_params


def load_data(params: dict):
    data_link = params["data"]["raw_data_link"]
    dest = params["data"]["raw_dataset_path"] # chemin de sauvegarde

    df = pd.read_csv(data_link)
    df.to_csv(dest, index=False)


if __name__ == "__main__":
    params = load_params()
    load_data(params)