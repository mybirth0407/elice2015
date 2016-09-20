import os
import pandas as pd
import numpy as np
import sklearn.decomposition
import sklearn.preprocessing
import sklearn.cluster
import scipy.spatial.distance

def main():
    # 1
    stocks_df, code_to_name = load_data()

    stocks_df = calculate_fluctuations(stocks_df)
    print(stocks_df)

def calculate_fluctuations(stocks_df):
    # 2
    stocks_df.drop([col for col in stocks_df.columns.tolist() if stocks_df[col].count() < 1400], 1, inplace=True)
    stocks_df = stocks_df.pct_change().fillna(0)
    
    # 3
    return stocks_df

def load_data():
    stocks_df = pd.read_csv("./stocks.csv")
    stocks_df = stocks_df.set_index('index')
    krx_listed_companies = pd.read_csv("./krx_listed_companies.csv")

    code_to_name = {}
    for code, name in zip(krx_listed_companies['Code'].values, krx_listed_companies['Name'].values):
        z_code = '0' * (6 - len(str(code))) + str(code)
        code_to_name[z_code] = name

    return stocks_df, code_to_name

if __name__ == "__main__":
    main()
