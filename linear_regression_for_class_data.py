import statsmodels.api
import numpy

def main():
    (N, X, Y) = read_data()

    results = do_multivariate_regression(N, X, Y)

    effective_variables = get_effective_variables(results)
    print(effective_variables)

def read_data():
    #1
    f = open ("students.dat", "r")

    X = []
    Y = []

    N = f.read().split('\n')
    
    for i in range (1, 31):

        t = (N[i].split(' '))
        Y.append (t.pop())
        X.append (t)

    for i in range (0, 30):

        for j in range (0, 5):

            X[i][j] = float(X[i][j])

    for i in range (0, 30):

        Y[i] = float(Y[i])

    N = N[0]
    
    #print (X)
    #print (Y)

    X = numpy.array(X)
    # X must be numpy.array in (30 * 5) shape
    X = X.reshape ( (30, 5))
    #print (X.shape)
    Y = numpy.array(Y)
    # Y must be 1-dimensional numpy.array.
    Y = Y.reshape ( (30, 1))
    #print (Y.shape)

    return (N, X, Y)

def do_multivariate_regression(N, X, Y):

    #X = statsmodels.api.add_constant (X)

    results = statsmodels.api.OLS (Y, X).fit()
    # 2

    return results

def get_effective_variables(results):
    eff_vars = []
	# 3
    
    for i in range (0, 5):
        
        if results.pvalues[i] < 0.05:

            eff_vars.append ('x%d' % (i + 1))

    return eff_vars

def print_students_data():
    with open("students.dat") as f:
        for line in f:
            print(line)

if __name__ == "__main__":
    main()