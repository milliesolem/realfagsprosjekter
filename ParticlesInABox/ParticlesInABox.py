import numpy as np
import random

class Particle:
    def __init__(self,r,v):
        self.r = np.array(r).astype(np.float_)
        self.v = np.array(v).astype(np.float_)
    def move(self,dt,a):
        self.v += a*dt
        self.r += self.v*dt

    def collide(self,lmin,lmax):
        self.mask = (lmin<=self.r) | (lmax>=self.r)
        self.v[self.mask] *= -1

class Measurement:
    def __init__(self, **kwargs):
        self.properties = kwargs
        self.results = dict()
    def measure(self, particles):
        pass

class Box:
    def __init__(self,dim,lmin = None, lmax = None):
        self.dim = int(dim)
        

        if lmin is None and lmax is None:
            self.lmin = np.zeroes(self.dim)
            self.lmax = np.ones(self.dim)
        elif lmin is None and lmax is not None:
            self.lmax = np.array(lmax).astype(np.float_)
            self.lmin = self.lmax - 1
        else:
            self.lmin = np.array(lmin).astype(np.float_)
            self.lmax = self.lmin + 1

        assert self.lmin.size==self.dim, f"'lmin'  has dimension {self.lmin.size}, but {self.dim} dimensions expected"
        assert self.lmax.size==self.dim, f"'lmax'  has dimension {self.lmax.size}, but {self.dim} dimensions expected"
        assert np.all(self.lmin<self.lmax), f"lmin cannot be greater or equal to lmax"

        
    def generate_particles(self, N, beta=1, mass=1, mu=0):
        self.N = int(N)
        self.beta = float(beta)
        self.mass = float(mass)
        self.mu = float(mu)
        self.sigma = (self.mass*self.beta)**(-0.5)

        dl = self.lmax - self.lmin
        self.particles = []
        for i in range(self.N):
            
            r0 = self.lmin+dl*np.random.uniform(size=self.dim)
            v0 = self.mu + self.sigma * np.random.randn(self.dim)

            p = Particle(r = r0, v = v0)
            self.particles.append(p)

    def simulate(self,  dt, n, *Measurements, a=0, t0 = 0):
        self.dt = float(dt)
        self.n = int(n)
        self.t0 = float(t0)

        t = self.t0 + self.dt*np.linspace(0,self.n,self.n+1)

        A = a if callable(a) else lambda t: a
        for i in range(n):

            for p in self.particles:
                p.move(dt, A(t[i]))
                p.collide(self.lmin,self.lmax)
                
            for m in Measurements:
                m.measure(self.particles)
        
    
        
        

class CountCollisions (Measurement):
    def __init__(self):
        super().__init__(N_collisions = 0)
    def measure(self, particles):
        for p in particles:
            self.properties["N_collisions"]+=np.any(p.mask)
        
if __name__ == "__main__":
    A = Box(dim = 2,lmax = [-1,-1])
    A.generate_particles(100)

    C = CountCollisions()
    
    A.simulate(1e-4, 10000, C)

    print(C.properties)
        
