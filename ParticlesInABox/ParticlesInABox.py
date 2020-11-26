import numpy as np

class Particle:
    """
    N-dimensional point particle
    ---------------------------------------------
    attributes:
        r (np.ndarray) : n-dim position array
        v (np.ndarray) : n-dim velocity array
    
    methods:
        __init__ : setup particle with initial pos & vel
        move     : update position and velocity
        collide  : test and perform wall collision
    """
    def __init__(self, r, v):
        self.r = np.array(r).astype(np.float_)
        self.v = np.array(v).astype(np.float_)

    def move(self, dt, acc):
        # Euler-Cromer integration scheme
        self.v += acc*dt
        self.r += self.v*dt

    def collide(self, Lmin, Lmax):
        # identify r components outside the box
        self.min_mask = Lmin >= self.r
        self.max_mask = Lmax <= self.r

        # flip velocity sign for components outside the box
        self.v[self.min_mask | self.max_mask] *= -1

class MeasurementDevice:
    """
    Parent Class: Measurement Device
    -----------------------------------------------------
    internal attributes
        device_params (dict) : parameters of measurement device
        sim_params    (dict) : simulation parameters
        _measurements (dict) : raw data
        _results      (dict) : results of the measurement
        _complete     (bool) : True if measurement is complete
    
    property attributes:
        results (dict)    : access _results dictionary
        *key*   (keytype) : access element *key* in _results dictionary
    
    methods:
        __init__            : initialize device
        prepare_device      : prepare device for the measurement process
        finish_measurement  : finish the measurement process

    placeholder methods: (to be defined in child classes)
        setup_dict          : setup _measurements dictionary
        measure             : use device to collect raw data
        compute_results     : compute results and store in _results dictionary
    """
    def __init__(self, **kwargs):
        # setup internal structure
        self.device_params = kwargs
        self.sim_params = dict()
        self._measurements = dict()
        self._results = dict()
        self._complete = False

    # extract box and simulation parameters
    def prepare_measurement(self, Box):
        params = ["dim", "N_particles", "beta", "mass", "mu", "sigma", "dt", "N_steps", "t"]
        self.sim_params.update({param : getattr(Box, param) for param in params})
        self.setup_dict()

    # finish measurement process
    def finish_measurement(self):
        self._complete = True
        self.compute_results()
    
    # access results dictionary only if the measurement has been completed
    @property
    def results(self):
        assert self._complete, "No result available as the simulation has not been run yet."
        return self._results
    
    # directly access result elements by key
    def __getattr__(self, key):
        if key in self._results:
            return self.results[key]
        else:
            return self.__getattribute__(key)
    
    # placeholders
    def setup_dict(self):
        pass
    def measure(self, time, particles):
        pass
    def compute_results(self):
        pass

class ParticleBox:
    """
    Thermodynamic Particle Box
    ----------------------------------------------------------------
    attributes:
        dim         (int)            : number of spatial dimensions
        Lmin        (np.ndarray)     : lower boundaries for box
        Lmax        (np.ndarray)     : upper boundaries for box
        N_particles (int)            : number of particles in the box
        beta        (float)          : thermodynamic temperature (1/kT)
        mass        (float)          : particle mass
        mu          (np.ndarray)     : initial mean particle velocity
        sigma       (np.ndarray)     : initial std. dev. particle velocity
        particles   (list[Particle]) : particles in the box

    methods:
        __init__           : construct box
        generate_particles : define particle type and initialize particles
        simulate           : execute particle simulation
    """
    def __init__(self, dim=2, Lmin=None, Lmax=None):
        # setup box
        self.dim = int(dim)

        # handle default box
        if Lmin is None and Lmax is None:
            self.Lmin = np.zeros(self.dim)
            self.Lmax = np.ones(self.dim)
        
        # handle box with specified Lmax
        elif Lmin is None and Lmax is not None:
            self.Lmax = np.array(Lmax).astype(np.float_)
            self.Lmin = np.zeros(self.dim) if np.all(self.Lmax > 0) else self.Lmax - 1
        
        # handle box with specified Lmin
        else:
            self.Lmin = np.array(Lmin).astype(np.float_)
            self.Lmax = np.ones(self.dim) if np.all(self.Lmin < 1) else self.Lmin + 1

        # verify box dimensions are consistent
        assert self.Lmin.size == self.dim, f"'Lmin' has dimension {self.Lmin.size}, but {self.dim} dimensions expected"
        assert self.Lmax.size == self.dim, f"'Lmax' has dimension {self.Lmax.size}, but {self.dim} dimensions expected"
        assert np.all(self.Lmin < self.Lmax), f"Lmin cannot be greater or equal to Lmax"

        # setup empty variables (to be filled)
        self.N_particles = None
        self.beta = None
        self.mass = None
        self.mu = None
        self.sigma = None
        self.dt = None
        self.N_steps = None
        self.t = None
        
    def generate_particles(self, N, beta=1, mass=1, mu=None):
        # setup parameters
        self.N_particles = int(N)
        self.beta = float(beta)
        self.mass = float(mass)
        self.mu = 0 if mu is None else np.array(mu).astype(np.float_)
        self.sigma = (self.mass*self.beta)**(-0.5)

        # verify input
        assert self.N_particles > 0, f"Expected positive number of particles 'N', got: {self.N_particles}"
        assert self.beta > 0, f"Expected positive temperature 'beta', got: {self.beta}"
        assert self.mass > 0, f"Expected positive mass 'mass', got: {self.mass}"

        # generate initial particle positions and velocities
        R0 = self.Lmin + (self.Lmax - self.Lmin) * np.random.uniform(size=(self.N_particles, self.dim))
        V0 = self.mu + self.sigma * np.random.randn(self.N_particles, self.dim)
        
        # generate particles
        self.particles = [Particle(r0, v0) for r0,v0 in zip(R0,V0)]

    def simulate(self,  dt, N, *MDevices, a=0, t0=0):
        # setup simulation parameters
        self.dt = float(dt)
        self.N_steps = int(N)
        self.t0 = float(t0)

        # verify simulation parameters
        assert self.dt > 0, f"Expected positive time step 'dt', got: {dt}"
        assert self.N_steps > 0, f"Expected positive number of time steps 'N', got: {self.N_steps}"

        # setup time array and prepare acceleration function
        self.t = self.t0 + self.dt * np.linspace(0, self.N_steps, self.N_steps + 1)
        A = a if callable(a) else lambda t: a

        # prepare measurement devices for data collection
        for M in MDevices:
            M.prepare_measurement(self)

        # main integration loop
        for t_i in self.t[:-1]:

            # compute acceleration
            acc = A(t_i)

            # update particles
            for p in self.particles:
                p.move(self.dt, acc)
                p.collide(self.Lmin, self.Lmax)
            
            # data collection
            for M in MDevices:
                M.measure(t_i, self.particles)
        
        # compute the results of the measurements
        for M in MDevices:
            M.finish_measurement()
        
if __name__ == "__main__":
    A = ParticleBox(dim = 2, Lmax = [-1,-1])
    A.generate_particles(100)

    EmptyDevice = MeasurementDevice()

    A.simulate(1e-4, 1e3, EmptyDevice)