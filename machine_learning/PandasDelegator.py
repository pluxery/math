import pandas as pd


class PandasDelegator:

    @staticmethod
    def dropna(df: pd.DataFrame):
        df.dropna(inplace=True)

    @staticmethod
    def fillna(df: pd.DataFrame, value):
        df.fillna(value, inplace=True)

    @staticmethod
    def replace(df: pd.DataFrame, value, to_replace):
        df.replace(value, to_replace, inplace=True)

    @staticmethod
    def reset_drop_index(df: pd.DataFrame):
        df.reset_index(drop=True, inplace=True)
