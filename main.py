import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy import fftpack, special
import scipy.linalg as linear_algebra
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


#Project A main function
def projectA(file_name, save=True, title='Input Data', x_label='x axis', y_label='y axis'):

    # Clean plot memory
    plt.clf()
    #Load txt file
    x, y = np.loadtxt(file_name, skiprows=0, unpack=True)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.margins(x=0.05, y=0.05)
    plt.plot(x,y)

    #If Save parameter is passed as Truthy, then save
    if save:
        file_start = file_name.split('.')[0]
        plt.savefig(f'{file_start}.pdf')
        np.savetxt(f'{file_start}.csv', list(zip(x, y)), fmt="%0.2f", delimiter=",")

    return x, y

#Project B starts here

#Function dictionary
funcs = {
    1: special.jn,
    2: special.yn,
}

#Function name dictionary
name_map = {
    2: 'y',
    1: 'j'
}

#Function, that computes Z coordinates for plotting
def spherical_bassel(x, y, function_type, n):
    function = funcs.get(function_type)

    if (not function):
        return

    r = np.sqrt(x ** 2 + y ** 2)

    return function(n, r)


#Project B main function
def projectB(x_range, y_range, function_type, n):
    # Clean plot memory
    plt.clf()

    #Deconstruct plot range from input
    x_start, x_end = x_range
    y_start, y_end = y_range

    x = np.linspace(x_start, x_end)
    y = np.linspace(y_start, y_end)

    X,Y = np.meshgrid(x,y)
    Z = spherical_bassel(X, Y, function_type, n)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z)

    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    name = name_map[function_type]

    #Add string translation to make the plot prettier
    SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    ax.set_zlabel(f'{name}{n}(x,y)'.translate(SUB))
    ax.set_title('Spherical Bassel Function')

    plt.savefig('projectB.pdf')


#Project C starts here


def equation(y, t, params):
    x, v = y
    miu, A, omega = params

    #Separate 2 order differential equation into 2 1st order differential equations
    return [v, miu * (1 - (x ** 2)) * v - x + A*np.sin(omega * t)]


def solve_equation(miu, A, omega, t, x0, v0):
    return odeint(equation, [x0, v0], t, args=([miu, A, omega],))


def projectC(miu, A, omega, start, end, x0=1, v0=0):
    # Clean plot memory
    plt.clf()
    t = np.arange(start, end, 0.01)

    #Get solutions for differential equation
    solution = solve_equation(miu, A, omega, t, x0, v0)

    fig = plt.figure(1, figsize=(8, 8))
    ax1 = fig.add_subplot(311)
    ax1.plot(t, solution[:, 0])
    ax1.set_xlabel('time')
    ax1.set_ylabel('x(t)')
    ax1.set_title('x versus time graph')

    ax2 = fig.add_subplot(312)
    ax2.plot(t, solution[:, 1])
    ax2.set_xlabel('time')
    ax2.set_ylabel('v(t)')
    ax2.set_title('v versus time graph')

    ax3 = fig.add_subplot(313)
    ax3.plot(solution[:, 0], solution[:, 1], '.', ms=1)
    ax3.set_xlabel('x(t)')
    ax3.set_ylabel('v(t)')
    ax3.set_title('v versus x graph')

    plt.tight_layout()
    plt.savefig('projectC.pdf')

    return t, solution[:, 0]


#Project D starts here


#Matrix generation
def systemmatrix(d, k=(-1000)):
    return np.array(
        [[0,0,1,0],
        [0,0,0,1],
        [k,0.5,-d, 1.0],
        [0.5, k, 1.0, -d]]
    )


def get_eigen_values(matrix):
    return (linear_algebra.eig(matrix))[0]


#Function to flatten the array and split eigen values into real and imaginary part for the plot
def separate_number_parts(eigen_list):
    return [(val.real, val.imag) for sublist in eigen_list for val in sublist]


#Project D main function
def projectD(N=500, M=100):
    # Clean plot memory
    plt.clf()
    d_input = np.linspace(0, M, N)

    #Iterate systemmatrix function to with different d values
    results = np.array([systemmatrix(d) for d in d_input])
    # Get eigen values for different iterations
    eigen_values = np.array([get_eigen_values(matrix) for matrix in results])

    #Flatten the array and
    eigen_values = separate_number_parts(eigen_values)

    X = np.array([value[0] for value in eigen_values])
    Y = np.array([value[1] for value in eigen_values])

    plt.scatter(X,Y, color='red')
    plt.xlabel('Re(z)')
    plt.ylabel('Im(z)')
    plt.title('Eigen values')
    plt.savefig('projectD.pdf')


#Project E starts here


def interpolate_data(x, y, interval):
    return interp1d(x, y, kind='cubic')(interval)


def fit_data(fit_function, x, y):
    #Get parameters of fit curve, but ignore covariance
    params, _ = curve_fit(fit_function, x, y)

    #Apply fit_function to data, pass params as a pointer to save text space
    return fit_function(x, *params)


def projectE(x_array, y_array, title=None, x_axis=None, y_axis=None):
    # Clean plot memory
    plt.clf()

    #Generate interpolation interval, matching limits of the input, but higher density
    interpolation_interval = np.linspace(
        min(x_array), max(x_array), len(x_array) * 5
    )
    cubic_interpolation = interpolate_data(x_array, y_array, interpolation_interval)

    # #Fit function for the graph
    fit_function = lambda x, a, c, e: a * np.tan(c * x) + e

    fit_result = fit_data(fit_function, x_array, y_array)

    plt.plot(
        x_array, y_array, 'o', #Data plot
        interpolation_interval, cubic_interpolation, '--', #Interpolation plot
        x_array, fit_result, 'b-' #Fit plot
    )

    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(title)

    plt.legend(['data', 'cubic interpolation', 'fit function'], loc='best')
    plt.savefig('projectE.pdf', bbox_inches='tight')


#Project F starts here


def projectF(t,x):
    # Clean plot memory
    plt.clf()

    dt = t[1]-t[0]
    G = fftpack.fft(x)
    f = fftpack.fftfreq(x.size, d=dt)
    f = fftpack.fftshift(f)
    G = fftpack.fftshift(G)


    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6))
    ax1.plot(t, x)
    ax1.set_xlabel(r'$t$')
    ax1.set_ylabel(r'$x(t)$')

    ax2.plot(f, np.real(G), color='dodgerblue',
             label='real part')
    ax2.plot(f, np.imag(G), color='coral',
             label='imaginary part')
    ax2.legend()
    ax2.set_xlabel(r'$f$')
    ax2.set_ylabel(r'$G(f)$')
    fig.tight_layout()
    fig.savefig('projectF.pdf')


#Function examples


# #Project A function
print(projectA('edata35.dat', True, 'Project main data', 'Project x axis', 'Project Y axis'))

# #Project B function
projectB([0,5], [1,6], 1, 3)

# #Project C function
print(projectC(10, 12, 3, 2, 13, 1, 4))

# #Project D function
projectD()

#Project E function
projectE(*(projectA('edata35.dat', False)))

#Project F function
projectF(*(projectC(10, 12, 3, 2, 13, 1, 4)))
