import pandas as pd

def load_dataset():

    df = pd.read_csv("data/diabetes_dataset.csv")

    return df
