from single_particle_dynamics import PointParticle

#Test 5 looks like a ball bouncing, hence the naming

# forces
def Drag(t, x, v):
        """
        We can see on the graph that velocity changes
        nonlinearly while the ball is in the air,
        so there must be some non-constant force
        acting on the ball
        """
        return -v
def Gravity(t, x, v):
        #Gravity constantly pulls down
        return -10
def BounceForce(t, x, v):
        #When x hits the ground, we give it a big bounce
        if(x<0):
                return 1000
        return 0
# create particle  |  NOTE: Use mass=2 and Ndim=1
particle = PointParticle(mass=2, Ndim=1)

# simulate dynamics  |  NOTE: Use dt=1e-3 and N=1e4
particle.simulate_path(Gravity,BounceForce,Drag, dt=1e-3, N=1e4, r0=0, v0=10)

# plot motion  |  NOTE:  use filename = "motion" to save image as "motion.png"
particle.plot(filename="motion")
