import numpy

def main():
    print(matrix_tutorial())

def matrix_tutorial():
    A = numpy.array([[1,4,5,8], [2,1,7,3], [5,4,5,9]])

    # 1
    B = A.reshape ((6, 2))
    # 2
    C = numpy.array ([[2, 2], [5, 3]])
    B = numpy.concatenate ((B, C), axis = 0)
    # 3
    Slice = numpy.split (B, 2, axis = 0)
    C = Slice[0]
    D = Slice[1]
    # 4
    E = numpy.concatenate ((C, D), axis = 1)
    # 5
    return E

if __name__ == "__main__":
    main()