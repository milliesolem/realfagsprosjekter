from ParticlesInABox import Particle, MeasurementDevice, ParticleBox
import numpy as np

class CollisionCounter(MeasurementDevice):
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
    # prepare _measurements dictionary
    def setup_dict(self):
        shape = (2, self.sim_params["N_particles"], self.sim_params["dim"])
        self._measurements['counter'] = np.zeros(shape).astype(np.int_)
    
    # count the number of wall collisions
    def measure(self, ti, particles):
        for i, p in enumerate(particles):
            self._measurements["counter"][0,i] += p.min_mask
            self._measurements["counter"][1,i] += p.max_mask

    # compute results
    def compute_results(self):
        # helper variables
        counter = self._measurements["counter"]
        N_walls = 2 * float(self.sim_params["dim"])
        N_particles = float(self.sim_params["N_particles"])
        t = self.sim_params["t"]
        T = float(t[-1] - t[0])

        # computations
        self._results.update({
            'total'         : counter.sum(),
            'dim_dist'      : counter.sum(axis=(0,1)),
            'wall_dist'     : counter.sum(axis=1),
            'particle_dist' : counter.sum(axis=(0,2))
        })
        self._results.update({
            'total_rate'         : float(self.total) / T,
            'dim_dist_rate'      : self.dim_dist.astype(np.float_) / T,
            'wall_dist_rate'     : self.wall_dist.astype(np.float_) / T,
            'particle_dist_rate' : self.particle_dist.astype(np.float_) / T
        })
        self._results.update({
            'mean_wall_rate'     : self.wall_dist_rate.sum() / N_walls,
            'mean_particle_rate' : self.particle_dist_rate.sum() / N_particles
        })
    
    # summarize all results in terminal
    def summarize(self):
        print("Particle Simulation - Collision Summary")
        print("------------------------------------------------------------------")
        print(f"Number of particles         : {self.sim_params['N_particles']}")
        print(f"Number of integration steps : {self.sim_params['N_steps']}")
        print(f"Integration step length     : {self.sim_params['dt']}")
        print("------------------------------------------------------------------")
        print(f"Total number of collisions           : {self.total}")
        print(f"Mean rate of total collisions        : {self.total_rate}")
        print(f"Mean rate of collisions per particle : {self.mean_particle_rate}")
        print(f"Mean rate of collisions per wall     : {self.mean_wall_rate}")
    
class PressueGauge(MeasurementDevice):
    """
    Thermodynamic Pressure Gauge
    ---------------------------------------------------
    Measures the average force per area exerted on the
    box by the particles

    property attributes:
        total_impulse (float) : total change in particle momentum
        mean_force    (float) : total impulse averaged over integration time
        pressure      (float) : mean thermodynamic pressure
    """
    def setup_dict(self):
        pass
    
    def measure(self, ti, particles):
        pass

    def compute_results(self):
        pass


if __name__ == "__main__":
    # setup system and measurment tool
    B = ParticleBox(dim=3)
    B.generate_particles(1000, beta=10)
    C = CollisionCounter()
    
    # simulate particles
    B.simulate(0.001, 1e4, C)

    # summarize results
    C.summarize()

    print(C.particle_dist_rate)
