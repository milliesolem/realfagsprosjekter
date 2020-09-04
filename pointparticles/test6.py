from single_particle_dynamics import PointParticle


# forces

def Force(t,x,v):
    """
    I hope this isn't considered cheating :P
    (I guess some form of constant and spring force could be used,
    but I couldn't find anything that matched the zig-zag pattern
    of the velocity that didn't involve the use of a modulus)
    """
    if int(t)%2==0:
        return 1
    else:
        return -1

# create particle  |  NOTE: Use mass=2 and Ndim=1
particle = PointParticle(mass=2, Ndim=1)

# simulate dynamics  |  NOTE: Use dt=1e-3 and N=1e4
particle.simulate_path(Force, dt=1e-3, N=1e4, r0=0, v0=0)

# plot motion  |  NOTE:  use filename = "motion" to save image as "motion.png"
particle.plot(filename="motion")
