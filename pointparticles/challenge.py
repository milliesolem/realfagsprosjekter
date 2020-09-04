from single_particle_dynamics import PointParticle

"""
YOUR TASK:

    Recreate the motion shown in images "test*.png".

    You need to find the forces used and the initial conditions of the motion.

    A simulation can (in fact I have used) several functions if necessary.

    Initial conditions:
        r0 = x(t=0)
        v0 = v(t=0)

EXAMPLE: particle experiencing an increasing force and a spring force
"""

# forces
def ConstantForce(t, x, v):
    return 100*t

def SpringForce(t, x, v):
    return -4*x

# create particle  |  NOTE: Use mass=2 and Ndim=1
particle = PointParticle(mass=2, Ndim=1)

# simulate dynamics  |  NOTE: Use dt=1e-3 and N=1e4
particle.simulate_path(ConstantForce, SpringForce, dt=1e-3, N=1e4, r0=0, v0=0)

# plot motion  |  NOTE:  use filename = "motion" to save image as "motion.png"
particle.plot(filename="motion")
