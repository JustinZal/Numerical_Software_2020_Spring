import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


def fit_function(x, a, c):
    return (a * (x ** 3)) + c


def plot_data(x_array, y_array, title=None, x_axis=None, y_axis=None):
    plt.clf()
    interpolation_interval = np.linspace(
        min(x_array), max(x_array), len(x_array) * 5
    )

    cubic_interp = interp1d(x_array, y_array, kind='cubic')
    cubic_results = cubic_interp(interpolation_interval)

    params, _ = curve_fit(fit_function, x_array, y_array)
    fit_result = fit_function(x_array, *params)

    plt.plot(
    x_array, y_array, 'o',
    interpolation_interval, cubic_results, '--',
    x_array, fit_result, 'b-'
    )

    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(title)

    plt.legend(['data', 'cubic interpolation', 'fit function'], loc='best')
    plt.savefig('projectE.pdf', bbox_inches='tight')
