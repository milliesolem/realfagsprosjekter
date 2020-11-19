from ParticlesInABox import Measurement

class CountCollisions (Measurement):
    def __init__(self):
        super().__init__(N_collisions = 0)
    def measure(self, particles):
        for p in particles:
            self.properties["N_collisions"]+=np.any(p.mask)