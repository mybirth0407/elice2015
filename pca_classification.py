import sklearn.decomposition
import numpy as np
import pandas as pd
import elice_utils

def main():
    # 1
    wine_df = pd.read_csv('wine.csv')
    class_df = wine_df.pop ('class')
    feature_df = wine_df
    #feature_df = pd.DataFrame (wine_df)
    
    #feature_a1 = wine_df.pop ('a1')
    #feature_a2 = wine_df.pop ('a2')
    #feature_a3 = wine_df.pop ('a3')
    #feature_a4 = wine_df.pop ('a4')
    #feature_a5 = wine_df.pop ('a5')
    #feature_a6 = wine_df.pop ('a6')
    #feature_a7 = wine_df.pop ('a7')
    #feature_a8 = wine_df.pop ('a8')
    #feature_a9 = wine_df.pop ('a9')
    #feature_a10 = wine_df.pop ('a10')
    #feature_a11 = wine_df.pop ('a11')
    #feature_a12 = wine_df.pop ('a12')
    #feature_a13 = wine_df.pop ('a13')
    
    # 2
    pca, pca_array = run_PCA(feature_df, 2)

    # 4
    print(elice_utils.wine_graph(pca_array, class_df))


def run_PCA(dataframe, n_components):
    
    pca = sklearn.decomposition.PCA(n_components = 2)
    pca.fit(dataframe)
    pca_array = pca.transform(dataframe)

    return pca, pca_array

if __name__ == '__main__':
    main()
    