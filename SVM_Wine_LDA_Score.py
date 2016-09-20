import pandas as pd
import numpy as np
import sklearn.decomposition
import sklearn.preprocessing
import sklearn.cluster
import sklearn.cross_validation
import sklearn.lda


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

    for num_feature in range(1, 7):
        lda_train, lda_train_arr = run_LDA(X, y, num_feature)

        svc = run_linear_SVM(lda_train, lda_train_arr, C)
        rbf_svc = run_rbf_SVM(lda_train, lda_train_arr, C)

        lda_test, lda_test_arr = run_LDA(lda_train, lda_train_arr, num_feature)

        print("%d:\t%f\t%f" % (num_feature,
                                   test_svm_models(lda_train_arr, lda_test_arr, svc),
                                   test_svm_models(lda_train_arr, lda_test_arr, rbf_svc)))


def load_data():
    # 1
    wine_df = pd.read_csv('wine.csv')
    class_df = wine_df.pop ('class')
    feature_df = wine_df

    return feature_df, class_df


def run_LDA(X, y, num_components):
    # 2
    lda = sklearn.lda.LDA(n_components = num_components)
    lda_array = lda.fit(X, y).transform(X)
    
    return lda, lda_array


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
