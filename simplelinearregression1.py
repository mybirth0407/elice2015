import numpy

def main():
    (N, X, Y) = read_data()
    print(N)
    print(X)
    print(Y)

def read_data():
    # 1
	(N, X, Y) = (input (), [], [])

	for i in range (0, int(N)):

		[x, y] = [int(i) for i in input().strip().split(" ")]
		X.append (float(x))
		Y.append (float(y))

	# 2
	return (N, X, Y)

if __name__ == "__main__":
    main()