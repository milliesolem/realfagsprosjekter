class NetSpringForce:
    def __init__(self,m,k,g=9.81,xb=1,L=1):
        self.m = m
        self.k = k
        self.g = abs(g)
        self.xb = xb
        self.L = L
        self.xeq = xb-L
    def __call__(self,t,r,v):
        G = -self.m*self.g
        S = self.k*(self.xeq-r)
        return G + S
