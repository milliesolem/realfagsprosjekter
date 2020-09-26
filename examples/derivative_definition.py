import numpy as np
import matplotlib.pyplot as plt

class NumericalDerivative:
    """
    Functor for computing the derivative from function values
    ----------------------------------------------------------
    Derivative can be computed using a forward, backward or
    central difference scheme.

    Notation:
        x_i             # i^th value of x
        y_i  = f(x_i)   # function value of y=f(x) at x=x_i
        y'_i = f'(x_i)  # derivative of f(x) at x=x_i

    Forward Difference:
        y'_i = ( y_(i+1) - y_(i-1) ) / ( x_(i+1) - x_i )
    
    Backward Difference:
        y'_i = ( y_i - y(i-1) ) / ( x_i - x_(i-1) )
    
    Central Difference:
        y'_i = ( y_(i+1) - y_(i-1) ) / ( x_(i+1) - x_(i-1) )
    
    In the forward difference scheme, the derivative is computed
    at all x values except the final x value.
    In the backward difference scheme, the derivative is computed
    at all x values except the first x value.
    In the central difference scheme, the derivative is computed
    at all x values except both the final and first x values.
    """
    def __init__(self, difference="forward"):
        """
        Setup differentiation functor

        input:
            difference (str) : "forward", "backward" or "central"
        """
        # check input
        difference = str(difference).lower()
        if difference not in ["forward", "backward", "central"]:
            print(f"Error: difference type '{difference}' is not available, using 'forward'")
            difference = "forward"
        
        # setup differentiation function
        self.difference = difference
    
    # numerical differentiation schemes
    def forward_difference(self, x, y):
        return x[:-1], (y[1:] - y[:-1]) / (x[1:] - x[:-1])
    
    def backward_difference(self, x, y):
        return x[1:], (y[:-1] - y[1:]) / (x[:-1] - x[1:])
    
    def central_difference(self, x, y):
        return x[1:-1], (y[2:] - y[:-2]) / (x[2:] - x[:-2])
    
    def __call__(self, x, y):
        """
        Compute the numerical derivative of y=f(x) at x.

        input:
            y (array)       : function values of y=f(x)
            x (array/float) : x values corresponding to y
        
        if x is a float, then it is taken as dx.
        
        output:
            f' (array) : function values of f'(x) (derivative)
        """
        # setup constant x difference
        if isinstance(x,(float,int)):
            x = (x * np.linspace(0,1,y.size)).astype(np.float_)
        
        # compute and return derivative
        if self.difference == "forward":
            return self.forward_difference(x,y)
        if self.difference == "backward":
            return self.backward_difference(x,y)
        if self.difference == "central":
            return self.central_difference(x,y)
    

if __name__ == '__main__':
    """
    Example of how to compute the derivative of a function using function values

    Here we use 2 different differentiation schemes: forward and central.

    The relative error plot (2nd plot) shows that the central scheme provides
    a better approximation of the real derivative.

    The derivative is computed for many x-values, so the error plot provides
    errorbars as a representation of the typical variation in the error.
    """
    # generate x and function values
    def generate_values(f, x0, x1, N):
        x = np.linspace(x0, x1, N)
        return x, f(x)

    # setup differentiation functor
    D = NumericalDerivative()
    schemes = ["forward", "central"]

    # some constants
    x0 = 0.01
    x1 = 0.97
    maxpow = 1
    minpow = 5
    Npows = 8

    # setup some arrays
    x_real = np.linspace(x0,x1,int(1e5))
    DX = np.logspace(-maxpow, -minpow, Npows)

    # function to consider
    f = lambda x: np.sin(4*x)*np.exp(-x)
    g = lambda x: (4*np.cos(4*x)-np.sin(4*x))*np.exp(-x)

    # plot function and its derivative
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10,8))
    axs[0].plot(x_real, f(x_real))
    axs[1].plot(x_real, g(x_real))
    axs[0].set_title("Function to study", fontsize=20)
    axs[1].set_xlabel(r"$x$", fontsize=16)
    axs[0].set_ylabel(r"$y=\sin(4x)e^{-x}$", fontsize=16)
    axs[1].set_ylabel(r"$y'=(4\cos(4x)-\sin(4x))e^{-x}$", fontsize=16)
    plt.show()

    # setup error arrays
    error_mean = np.zeros((Npows,2))
    error_std = np.zeros((Npows,2))

    # loop over dx = 0.1, 0.01, ...
    for i,dx in enumerate(DX):

        # setup data
        N = 1 + np.floor((x1-x0)/dx)
        x, y = generate_values(f, x0, x1, N)

        # loop over differentiation schemes
        for j, scheme in enumerate(["forward", "central"]):

            # compute derivative
            D.difference = scheme
            x_d, y_d = D(x,y)

            # evaluate total error
            y_d_real = g(x_d)
            error = np.abs((y_d - y_d_real) / y_d_real)
            error_mean[i,j] = error.mean()
            error_std[i,j] = error.std()

    # plot change in error
    plt.figure(figsize=(10,8))
    for i, scheme in enumerate(["forward", "central"]):
        plt.errorbar(x=DX, y=error_mean[:,i], yerr=error_std[:,i], label=scheme, capsize=7, fmt="o")
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Relative Error of Derivative", fontsize=20)
    plt.xlabel(r"$\Delta\,x$", fontsize=16)
    plt.ylabel(r"$\epsilon$", fontsize=16)
    xmin, xmax = plt.xlim()
    plt.xlim(xmax, xmin)
    plt.show()    
    