import matplotlib.pyplot as plt
import numpy as np

def file_work(file_name, title='Input Data', x_label='x axis', y_label='y axis'):
    x, y = np.loadtxt(f'{file_name}.txt', skiprows=1, unpack=True)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.margins(x=0.05, y=0.05)
    plt.plot(x,y)
    print(list(zip(x,y)))
    plt.savefig(f'{file_name}.pdf')
    np.savetxt(f'{file_name}.csv', list(zip(x, y)), fmt="%0.2f", delimiter=",")


file_work('vehicles')