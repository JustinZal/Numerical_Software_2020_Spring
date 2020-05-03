import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def equation(y, t, params):
    x, v = y
    miu, A, omega = params

    return [v, miu * (1 - (x ** 2)) * v - x + A*np.sin(omega * t)]


def solve_equation(miu, A, omega, t, x0, v0):
    return odeint(equation, [x0, v0], t, args=([miu, A, omega],))


def solve_oscilator_equation(miu, A, omega, start, end, x0=1, v0=0):
    plt.clf()
    t = np.arange(start, end, 0.01)
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
    ax3.set_title('v versus t graph')

    plt.tight_layout()
    plt.savefig('oscillator.pdf')
