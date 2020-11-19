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
        # identify r components "outside" the box
        self.mask = (Lmin >= self.r) | (Lmax <= self.r)
        self.v[self.mask] *= -1

class MeasurementTool:
    """
    Measurement tool
    -----------------------------------------------------
    attributes:
        params   (dict) : measurement parameters
        simsetup (dict) : simulation parameters
        results  (dict) : results of the measurement
    
    methods:
        __init__      : setup and adjust measurement parameters
        setup_process : setup measurement process
        measure       : execute measurement
    """
    def __init__(self, **kwargs):
        # setup dictionaries
        self.params = kwargs
        self.simsetup = dict()
        self.results = dict()

    def setup_process(self, **kwargs):
        # store all simulation keyword arguments
        self.simsetup.update(kwargs)

    # placeholder
    def measure(self, t, particles):
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

    def simulate(self,  dt, N, *Measurements, a=0, t0=0):
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

        # prepare measurement tools for data collection
        for M in Measurements:
            M.setup_process(N_particles=self.N_particles, N_steps=self.N_steps, t=self.t)

        # main integration loop
        for t_i in self.t[:-1]:

            # compute acceleration
            acc = A(t_i)

            # update particles
            for p in self.particles:
                p.move(self.dt, acc)
                p.collide(self.Lmin, self.Lmax)
            
            # data collection
            for M in Measurements:
                M.measure(t_i, self.particles)
        
        
if __name__ == "__main__":
    A = ParticleBox(dim = 2, Lmax = [-1,-1])
    A.generate_particles(100)

    A.simulate(1e-4, 10000)
