import sys
import numpy as np
sys.path.append(".")
sys.path.append("..")
from single_particle_dynamics import PointParticle

# forces
def Gravity(t, x, v):
    return -9.81

def SpringForce(t, x, v):
    return -3*x

def NormalForce(t, x, v):
    if x < 0:
        return -1e4*x
    else:
        return 0

def Drag(t, x, v):
    return -1.5*v

def SquareSpringForce(t, x, v):
    if (t % 2) < 1:
        return 1.0
    else:
        return -1.0

def ExponentialDecay(t, x, v):
    return -10*np.exp(-t)

# setup particle
particle = PointParticle(mass=2, Ndim=1)

# gravity drop
particle.simulate_path(Gravity, dt=1e-3, N=1e4, r0=1000, v0=0)
particle.plot(filename="1Dsystems/test1")

# spring system
particle.simulate_path(SpringForce, dt=1e-3, N=1e4, r0=10, v0=0)
particle.plot(filename="1Dsystems/test2")

# bouncing ball
particle.simulate_path(Gravity, NormalForce, dt=1e-3, N=1e4, r0=10, v0=0)
particle.plot(filename="1Dsystems/test3")

# object falling with air resistance
particle.simulate_path(Gravity, Drag, dt=1e-3, N=1e4, r0=1000, v0=0)
particle.plot(filename="1Dsystems/test4")

# ball shot through air from a distance
particle.simulate_path(Gravity, NormalForce, Drag, dt=1e-3, N=1e4, r0=0, v0=10)
particle.plot(filename="1Dsystems/test5")

# square pulse action
particle.simulate_path(SquareSpringForce, dt=1e-3, N=1e4, r0=0, v0=0)
particle.plot(filename="1Dsystems/test6")

# decaying force
particle.simulate_path(ExponentialDecay, dt=1e-3, N=1e4, r0=0, v0=4)
particle.plot(filename="1Dsystems/test7")