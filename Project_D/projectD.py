import numpy as np
import scipy.linalg as linear_algebra
import matplotlib.pyplot as plt


def systemmatrix(d, k=(-1000)):
    return np.array(
        [[0,0,1,0],
        [0,0,0,1],
        [k,0.5,-d, 1.0],
        [0.5, k, 1.0, -d]]
    )


def get_eigen_values(matrix):
    return (linear_algebra.eig(matrix))[0]


def plot_eigen_values(N=500, M=100):
    d_input = np.linspace(0, M, N)
    results = np.array([systemmatrix(d) for d in d_input])
    eigen_values = np.array([get_eigen_values(matrix) for matrix in results])
    X = []
    Y = []

    for values in eigen_values:
        for val in values:
            X.append(val.real)
            Y.append(val.imag)

    plt.scatter(X,Y, color='red')
    plt.xlabel('Re(z)')
    plt.ylabel('Im(z)')
    plt.title('Eigen values')
    plt.savefig('Eigen_Values.pdf')

plot_eigen_values()