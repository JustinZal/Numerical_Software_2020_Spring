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

files `vehicles.pdf` and `vehicles.csv` should appear after running the command.

**Disclaimer**: The function was written assuming, that the first line of input TXT file contains titles of the columns and 
that it atleast contains 2 columns  