from ParticlesInABox import Particle, MeasurementTool, ParticleBox
import numpy as np

class CollisionCounter(MeasurementTool):
    """
    Wall-collision Counter
    ----------------------------------------------------
    Counts the number of (particle-wall) collisions that
    occurred during the simulation

    attributes:
        total_count (int)             : total number of collisions
        collisions  (np.ndarray[int]) : number of collisions per particle
    """
    def __init__(self):
        # setup measurement tool
        super().__init__(N_collisions = 0)
    
    def setup_process(self, **kwargs):
        # setup process
        super().setup_process(**kwargs)

        # prepare property attributes
        self.results["total_count"] = 0
        self.results["collisions"] = np.zeros(self.simsetup["N_particles"])

    def measure(self, t, particles):
        # loop over each particle
        for i, p in enumerate(particles):
            # identify collision
            hit = np.any(p.mask)

            # add collision count contribution
            self.results["total_count"] += hit
            self.results["collisions"][i] += hit
    
    @property
    def total_count(self):
        return self.results["total_count"]
    
    @property
    def collisions(self):
        return self.results["collisions"]


if __name__ == "__main__":
    # setup system and measurment tool
    B = ParticleBox(dim=2)
    B.generate_particles(15)
    C = CollisionCounter()

    # simulate particles
    B.simulate(1e-3, 1e4, C)

    # print results
    print(f"Number of particles = {B.N_particles}, dt = {B.dt}, N_steps = {B.N_steps}")
    print(f"Total number of collisions = {C.total_count}")
    print("Collisions per particle:")
    print(C.collisions)
