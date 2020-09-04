from single_particle_dynamics import PointParticle

# forces
def Force(t, x, v):
    return -1


# create particle  |  NOTE: Use mass=2 and Ndim=1
particle = PointParticle(mass=2, Ndim=1)

# simulate dynamics  |  NOTE: Use dt=1e-3 and N=1e4
particle.simulate_path(Force, dt=1e-3, N=1e4, r0=0, v0=0)

# plot motion  |  NOTE:  use filename = "motion" to save image as "motion.png"
particle.plot(filename="motion")