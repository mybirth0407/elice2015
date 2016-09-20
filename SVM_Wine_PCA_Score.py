import pandas as pd
import numpy as np
import sklearn.decomposition
import sklearn.preprocessing
import sklearn.cluster
import sklearn.cross_validation


def main():
    C = 1.0

    # 1
    X, y = load_data()

    X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(X, y, test_size=0.2, random_state=0)

    print("# Comp\tLinear\tRBF")
    # 6
    svc = run_linear_SVM(X_train, y_train, C)
    rbf_svc = run_rbf_SVM(X_train, y_train, C)

    print("All\t%f\t%f" % (test_svm_models(X_test, y_test, svc),
                           test_svm_models(X_test, y_test, rbf_svc)))

    # 7
    df, a = load_data ()
    df = df.transpose ()

    for num_feature in range(1, 7):
        pca_train, pca_train_arr = run_PCA(df, num_feature)

        svc = run_linear_SVM(pca_train, pca_train_arr, C)
        rbf_svc = run_rbf_SVM(pca_train, pca_train_arr, C)

        pca_test, pca_test_arr = run_PCA(df, num_feature)

        print("%d:\t%f\t%f" % (num_feature,
                                   test_svm_models(pca_train_arr, pca_test_arr, svc),
                                   test_svm_models(pca_train_arr, pca_test_arr, rbf_svc)))


def load_data():
    # 1
    wine_df = pd.read_csv('wine.csv')
    class_df = wine_df.pop ('class')
    feature_df = wine_df

    return feature_df, class_df


def run_PCA(dataframe, num_components):
    # 2
    pca = sklearn.decomposition.PCA(n_components = num_components)
    pca.fit(dataframe)
    pca_array = pca.transform(dataframe)

    return pca, pca_array


def run_linear_SVM(X, y, C):
    # 3
    svc_linear = sklearn.svm.SVC (C = C, kernel = 'linear').fit (X, y)

    return svc_linear


def run_rbf_SVM(X, y, C, gamma=0.7):
    # 4
    svc_rbf = sklearn.svm.SVC (C = C, kernel = 'rbf', gamma = gamma).fit (X, y)

    return svc_rbf


def test_svm_models(X_test, y_test, each_model):
    # 5
    score_value = each_model.score(X_test, y_test)

    return score_value


if __name__ == "__main__":
    main()
