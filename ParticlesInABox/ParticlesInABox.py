import random

class Particle:
    def __init__(self,x=0,y=0,z=0,vx=0,vy=0,vz=0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

        self.vx = float(vx)
        self.vy = float(vy)
        self.vz = float(vz)
    def move(self,dt,ax,ay,az):
        self.vx += ax*dt
        self.vy += ay*dt
        self.vz += az*dt
        
        self.x += self.vx*dt
        self.y += self.vy*dt
        self.z += self.vz*dt

    def collide(self,Lx1,Lx2,Ly1,Ly2,Lz1,Lz2):
        if self.x>=Lx1 or self.x<=Lx2:
            self.vx = -self.vx
        if self.y>=Ly1 or self.x<=Ly2:
            self.vy = -self.vy
        if self.z>=Lz1 or self.x<=Lz2:
            self.vz = -self.vz

class Box:
    def __init__(self,Lx2 = 1,Ly2=None,Lz2=None,Lx1=0,Ly1=0,Lz1=0):
        if Ly2 is None:
            Ly2 = Lx2
        if Lz2 is None:
            Lz2 = Lx2
        self.Lx2 = float(Lx2)
        self.Ly2 = float(Ly2)
        self.Lz2 = float(Lz2)

        self.Lx1 = float(Lx1)
        self.Ly1 = float(Ly1)
        self.Lz1 = float(Lz1)
    def generate_particles(self, n, mass, kT, mu=0):
        self.particles = []
        self.kT = float(kT)
        self.mass = float(mass)
        self.mu = float(mu)
        
        self.sigma = (self.kT/self.mass)**0.5
        
        for i in range(n):
            
            P = Particle(
                x = random.uniform(self.Lx1,self.Lx2),
                y = random.uniform(self.Ly1,self.Ly2),
                z = random.uniform(self.Lz1,self.Lz2),

                vx = random.gauss(self.mu,self.sigma),
                vy = random.gauss(self.mu,self.sigma),
                vz = random.gauss(self.mu,self.sigma)
            )
            self.particles.append(P)

    def simulate(dt,n):
        self.dt = float(dt)
        self.n = float(n)
        for i in range(n):
            for p in self.particles:
                p.move(dt,0,0,0)
                p.collide(
                    self.Lx1,self.Lx2,
                    self.Ly1,self.Ly2,
                    self.Lz1,self.Lz2,
                )
        
        
        
            
        
