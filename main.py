from projectA import file_work
from projectB import plot_bassel_3d
from projectC import solve_oscilator_equation
from projectD import plot_eigen_values
from projectE import plot_data

#Project A function
file_work('vehicles', True, 'Vehicle mileage data', 'Miles per galon', 'Wattage')

#Project B function
plot_bassel_3d([0,5], [1,6], 1, 3)

#Project C function
solve_oscilator_equation(3, 2, 5, 2, 13)

#Project D function
plot_eigen_values()

#Project E function
plot_data(*(file_work('virus', False)), 'Walking dead virus', 'Days', 'Infections per day')
