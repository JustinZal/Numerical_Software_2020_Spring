import numpy as np
from scipy import special
import matplotlib.pyplot as plt

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



def plot_bassel_3d(x_range, y_range, function_type, n):
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

    plt.savefig('bassel.pdf')
