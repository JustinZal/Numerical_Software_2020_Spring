# Jacobs University Numerical Software 2020 Spring

Author: Justinas Zaliaduonis

### Setup
Before installing packages venv should be configured and activated
```
make venv
. venv/bin/activate
```
Command for installing packages
```
make packages
```
### Project A description
`file_work` function, that takes file name as an input and gives first two columns as an output.
The output format is a tuple of arrays. The function also saves the 2D plot of the values with 5% margin and converts
columns into CSV output. Folder `Project_A` contains `vehicle.txt` sample data taken from a statistical modeling course.

Funcion can be tested by running the following command:

`cd Project_A && python3 projectA.py`

files `Project_A/vehicles.pdf` and `Project_A/vehicles.csv` should appear after running the command.

**Disclaimer**: The function was written assuming, that the first line of input TXT file contains titles of the columns and 
that it atleast contains 2 columns

### Project B description

`plot_bassel_3d` takes 2 list arguments as ranges of x and y e.g. `[1,4]`, Bassel function type in the form of integer
 1 or 2 and the function order n in the form of integer and plots the required Spherical Bassel Function in 3D using 
 matplotlib and numpy and scipy libraries. The transition from 1D to 2D input is made by composing spherical
 2D vector norm functions. Plot is saved in a file named `Project_B/bassel.pdf`


### Project B description
 `solve_oscilator_equation` takes a series of numbers and solves the `van der Pol oscillator` using differential equation
 and saves the plot in the file `Project_C/oscillator.pdf`. The solution is achieved using `scipy.integrate.odeint` function,
 the plot is made using `matplotlib.pyplot` function.  
 
 ### TODO
 
 - Complete project D
 - Write main.py function, that calls all project functions
 - Remove Makefile and venv from architecture for simplicity
 
    