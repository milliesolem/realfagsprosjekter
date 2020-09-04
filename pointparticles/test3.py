from single_particle_dynamics import PointParticle

#Test 3 looks like a ball bouncing, hence the naming

# forces
def Gravity(t, x, v):
	return -10
def Bounce(t, x, v):
	if x<0:
		return 10000
	return 0


# create particle  |  NOTE: Use mass=2 and Ndim=1
particle = PointParticle(mass=2, Ndim=1)

# simulate dynamics  |  NOTE: Use dt=1e-3 and N=1e4
particle.simulate_path(Gravity, Bounce, dt=1e-3, N=1e4, r0=10, v0=0)

# plot motion  |  NOTE:  use filename = "motion" to save image as "motion.png"
particle.plot(filename="motion")
