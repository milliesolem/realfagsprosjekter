from ParticlesInABox import Particle, MeasurementTool, ParticleBox
import numpy as np

class CollisionCounter(MeasurementTool):
    """
    Wall-collision Counter
    ----------------------------------------------------
    Counts the number of (particle-wall) collisions that
    occurred during the simulation

    property attributes:
        total_count         (int)             : total number of collisions
        count_dist_particle (np.ndarray[int]) : number of collisions for each particle
        count_dist_wall     (np.ndarray[int]) : number of collisions for each wall
        count_per_particle  (float)           : mean number of collisions per particle
        count_per_wall      (float)           : mean number of collisions per wall
        mean_count_rate     (float)           : average number of collisions per time step
    """
    def __init__(self):
        # setup measurement tool
        super().__init__(N_collisions = 0)
        self._prepared = False
    
    def setup_process(self, Box):
        # read simulation parameters
        self.simsetup.update({
            "dim"         : Box.dim,
            "N_particles" : Box.N_particles,
            "N_steps"     : Box.N_steps,
            "dt"          : Box.dt
        })

        # setup counter | 2 = min & max collisions
        self.results["counter"] = np.zeros((self.simsetup["N_particles"], 2, self.simsetup["dim"])).astype(np.int_)

        # reset property attributes
        self._prepared = True
        self._total_count = None
        self._count_dist_particle = None
        self._count_dist_wall = None
        self._mean_count_per_particle = None
        self._mean_count_per_wall = None
        self._mean_count_rate = None

    def measure(self, t, particles):
        # loop over each particle
        for i, p in enumerate(particles):
            # record counts
            self.results["counter"][i] += [p.min_mask, p.max_mask]
    
    def check_prepared(self):
        assert self._prepared, "The simulation has not been run yet."
    
    # compute and access total number of collisions
    @property
    def total_count(self):
        self.check_prepared()
        if self._total_count is None:
            self._total_count = int(np.sum(self.results["counter"]))
        return self._total_count
    
    # compute and access number of collisions for each particle
    @property
    def count_dist_particle(self):
        self.check_prepared()
        if self._count_dist_particle is None:
            self._count_dist_particle = np.sum(np.sum(self.results["counter"], axis=2), axis=1)
        return self._count_dist_particle
    
    # compute and access number of collisions for each wall
    @property
    def count_dist_wall(self):
        self.check_prepared()
        if self._count_dist_wall is None:
            self._count_dist_wall = np.sum(self.results["counter"], axis=0)
        return self._count_dist_wall
    
    # compute and access average number of collisions per particle
    @property
    def mean_count_per_particle(self):
        self.check_prepared()
        if self._mean_count_per_particle is None:
            self._mean_count_per_particle = np.mean(self.count_dist_particle.astype(np.float_))
        return self._mean_count_per_particle
    
    # compute and access average number of collisions per wall
    @property
    def mean_count_per_wall(self):
        self.check_prepared()
        if self._mean_count_per_wall is None:
            self._mean_count_per_wall = np.mean(self.count_dist_wall.astype(np.float_))
        return self._mean_count_per_wall
    
    # compute and access average collision rate
    @property
    def mean_count_rate(self):
        self.check_prepared()
        if self._mean_count_rate is None:
            self._mean_count_rate = float(self.total_count) / self.simsetup["dt"]
        return self._mean_count_rate
    
    # display all results
    def summarize(self):
        # verify simulation has been run
        self.check_prepared()

        # summarize results
        print("Particle Simulation Collision Summary")
        print("------------------------------------------------------------------")
        print(f"Number of particles         : {self.simsetup['N_particles']}")
        print(f"Number of integration steps : {self.simsetup['N_steps']}")
        print(f"Integration step length     : {self.simsetup['dt']}")
        print("------------------------------------------------------------------")
        print(f"Total number of collisions             : {self.total_count}")
        print(f"Mean number of collisions per particle : {self.mean_count_per_particle}")
        print(f"Mean number of collisions per wall     : {self.mean_count_per_wall}")
        print(f"Mean rate of collision                 : {self.mean_count_rate}")


if __name__ == "__main__":
    # setup system and measurment tool
    B = ParticleBox(dim=2)
    B.generate_particles(75)
    C = CollisionCounter()
    
    # simulate particles
    B.simulate(0.002, 5700, C)

    # summarize results
    C.summarize()
