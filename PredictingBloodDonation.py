import pandas as pd
import numpy as np
import sklearn.decomposition
import sklearn.preprocessing
import sklearn.cluster
import sklearn.cross_validation


def main():
    C = 1.0

    X, y = load_data()

    # 2
    X = sklearn.preprocessing.scale(X)

    X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(X, y, test_size=0.2, random_state=0)

    svc_linear = run_linear_SVM(X_train, y_train, C)
    svc_poly2 = run_poly_SVM(X_train, y_train, 2, C)
    svc_poly3 = run_poly_SVM(X_train, y_train, 3, C)
    svc_rbf = run_rbf_SVM(X_train, y_train, C)

    model_names = ['Linear', 'Poly degree 2', 'Poly degree 3', 'RBF']
    for model_name, each_model in zip (model_names, [svc_linear, svc_poly2, svc_poly3, svc_rbf]):
        model_score = test_svm_models(X_test, y_test, each_model)
        print('%s score: %f' % (model_name, model_score))


def load_data():
    # 1
    blood = pd.read_csv('blood_donation.csv')
    #blood = blood.set_index('index')

    for dataset in [blood]:
        y = dataset.pop('class')
        y = np.array(y)
        X = dataset
        X = np.array(X)
    
    return X, y

def run_linear_SVM(X, y, C):
    # 3
    svc_linear = sklearn.svm.SVC (C = C, kernel = 'linear').fit (X, y)

    return svc_linear


def run_poly_SVM(X, y, degree, C):
    # 4
    svc_poly = sklearn.svm.SVC (C = C, kernel = 'poly', degree = degree).fit (X, y)

    return svc_poly


def run_rbf_SVM(X, y, C, gamma=0.7):
    # 5
    svc_rbf = sklearn.svm.SVC (C = C, kernel = 'rbf', gamma = gamma).fit (X, y)

    return svc_rbf


def test_svm_models(X_test, y_test, each_model):
    # 6
    score_value = each_model.score(X_test, y_test)

    return score_value


if __name__ == "__main__":
    main()