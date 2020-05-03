import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


#Fit function for the graph
def fit_function(x, a, c):
    return (a * (x ** 3)) + c


def plot_data(x_array, y_array, title=None, x_axis=None, y_axis=None):
    # Clean plot memory
    plt.clf()

    #Generate interpolation interval, matching limits of the input, but higher density
    interpolation_interval = np.linspace(
        min(x_array), max(x_array), len(x_array) * 5
    )

    cubic_interp = interp1d(x_array, y_array, kind='cubic')
    cubic_results = cubic_interp(interpolation_interval)

    #Get parameters of fit curve, but ignore covariance
    params, _ = curve_fit(fit_function, x_array, y_array)

    #Apply fit_function to data, pass params as a pointer to save text space
    fit_result = fit_function(x_array, *params)

    plt.plot(
        x_array, y_array, 'o', #Data plot
        interpolation_interval, cubic_results, '--', #Interpolation plot
        x_array, fit_result, 'b-' #Fit plot
    )

    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(title)

    plt.legend(['data', 'cubic interpolation', 'fit function'], loc='best')
    plt.savefig('projectE.pdf', bbox_inches='tight')
