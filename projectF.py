import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy import fftpack


def equation(y, t, params):
    x, v = y
    miu, A, omega = params

    #Separate 2 order differential equation into 2 1st order differential equations
    return [v, miu * (1 - (x ** 2)) * v - x + A*np.sin(omega * t)]


def solve_equation(miu, A, omega, t, x0, v0):
    return odeint(equation, [x0, v0], t, args=([miu, A, omega],))


def solve_oscilator_equation(miu, A, omega, start, end, x0=1, v0=0):
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
    plt.savefig('oscillator.pdf')

    return t, solution[:, 0]


def fourier_transform(x,t):
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
    fig.savefig('fourierTransform.pdf')
    plt.show()
