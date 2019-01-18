import pandas as pd


def read_data(file_path, sep=','):
    return pd.read_csv(file_path, sep)


def fill_na_with_value(df, value=-1, inplace=True):
    df.fillna(-1, inplace=inplace)


def normalize(df, name):
    min_val = df[name].min()
    max_val = df[name].max()
    df[name] = df[name].map(lambda x: float(x - min_val + 1) / float(max_val - min_val + 1))
    return df
