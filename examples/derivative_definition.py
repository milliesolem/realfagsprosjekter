import numpy as np

def compute_derivative(f, x, dx):
    return (f(x+dx) - f(x)) / dx


f = lambda x: x**2 + 1
g = lambda x: 2*x

x = 2

f_d_true = g(x)

print(f"True f'({x}) = {f_d_true}")

for dx in [1, 0.1, 0.001, 0.00000001]:

    f_d = compute_derivative(f, 2, dx)

    print(f"f'({x}) = {f_d} with dx = {dx} and error = {abs(f_d - f_d_true)}")

    
