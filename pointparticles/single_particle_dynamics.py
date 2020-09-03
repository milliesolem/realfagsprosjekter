import numpy as np
import matplotlib.pyplot as plt

class PointParticle:
    """
    Classical Point Particle
    --------------------------------------------------
    Extensionless object in N-dimensional space with
    definite mass, position and velocity.
    """
    def __init__(self, mass, Ndim=1):
        self.mass = float(mass)  # particle mass
        self.Ndim = int(Ndim)    # dimensionality of space
        self.t = None            # time-points of simulation
        self.r = None            # simulated positions
        self.v = None            # simulated velocities

        # plotting configuration
        self.plotting_configuration = {
            'figsize'       : (10,8),
            'titlefontsize' : 20,
            'labelfontsize' : 16,
            'dotsize'       : 60
        }
    
    # pretty-print
    def __str__(self):
        return f"Point particle in {self.Ndim}-dimensional space with mass = {self.mass}"
    
    # update plotting confguration dict
    def plot_settings(self, **kwargs):
        self.plotting_configuration.update(kwargs)
    
    def simulate_path(self, *Forces, dt, N, t0=0, r0=None, v0=None):
        """
        Simulates a dynamic system consisting of a single PointParticle.
        
        Dynamics obey Newtonian mechanics: F_net = m * a.
        
        Differential equation is integrated using the semi-implicit
        Euler-Cromer scheme:
            v_(i+1) = v_i + dt * a_i
            r_(i+1) = r_i + dt * v_(i+1)
        Integration time-steps are linear.
        ----------------------------------------------------------------------
        A force is a callable (function/functor) with signature:
            def force(t_i, r_i, v_i):
                # compute force 'F'
                return F
        If Ndim=1, then F can be single number.
        Otherwise it must be an ndarray (or list/tuple).
        """
        # standardize types
        t0 = float(t0)
        dt = float(dt)
        N = int(N)
        
        # prepare net acceleration
        def acc(t_i, r_i, v_i):
            F = np.zeros(self.Ndim)
            for Force in Forces:
                F += Force(t_i, r_i, v_i)
            return F / self.mass

        # setup integration variables
        self.t = t0 + dt * np.linspace(0, N, N+1)
        self.r = np.zeros((N+1, self.Ndim), dtype=np.float_)
        self.v = np.zeros((N+1, self.Ndim), dtype=np.float_)
        self.r[0] = np.zeros(self.Ndim) if r0 is None else r0
        self.v[0] = np.zeros(self.Ndim) if v0 is None else v0

        # run simulation
        for i in range(0, N):
            self.v[i+1] = self.v[i] + dt * acc(self.t[i], self.r[i], self.v[i])
            self.r[i+1] = self.r[i] + dt * self.v[i+1]

    def plot(self, dim=0, filename=None, show=True):
        """
        Plot the position and velocity along a given dimension

        input:
            dim      (int)  : dimension to plot
            filename (str)  : relative path to save location (None = don't save)
            show     (bool) : show plot if True, return (fig,(ax1,ax2)) if False
        """
        # extract plotting settings
        figsize = self.plotting_configuration["figsize"]
        titlesize = self.plotting_configuration["titlefontsize"]
        labelsize = self.plotting_configuration["labelfontsize"]

        # setup figure
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=figsize)

        # plot position
        ax1.plot(self.t, self.r[:,dim])
        ax1.set_title("Position", fontsize=titlesize)

        # plot veloicty
        ax2.plot(self.t, self.v[:,dim])
        ax2.set_title("Velocity", fontsize=titlesize)
        ax2.set_xlabel(r"$t$", fontsize=labelsize)

        # finalize figure
        fig.tight_layout()
        if filename is not None:
            fig.savefig(filename, dpi=300)
        if show:
            plt.show()
        else:
            return fig, (ax1, ax2)
    
    def plot_2D_path(self, filename=None, show=True):
        """
        Plot the planar path of the particle

        input:
            filename (str)  : relative path to save location (None = don't save)
            show     (bool) : show plot if True, return (fig,ax) if False
        """
        # extract plotting settings
        dotsize = self.plotting_configuration["dotsize"]
        figsize = self.plotting_configuration["figsize"]
        titlesize = self.plotting_configuration["titlefontsize"]
        labelsize = self.plotting_configuration["labelfontsize"]

        # setup figure
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

        # plot path
        ax.plot(self.r[:,0], self.r[:,1])
        ax.scatter(self.r[0,0], self.r[0,1], s=dotsize, label="start")
        ax.scatter(self.r[-1,0], self.r[-1,1], s=dotsize, label="end")
        ax.legend()
        ax.set_title("Particle Path", fontsize=titlesize)
        ax.set_xlabel(r"$x(t)$", fontsize=labelsize)
        ax.set_ylabel(r"$y(t)$", fontsize=labelsize)
        
        # finalize figure
        if filename is not None:
            fig.savefig(filename, dpi=300)
        if show:
            plt.show()
        else:
            return fig, ax

if __name__ == '__main__':
    """
    Example: particle in a harmonic oscillator potential
             experiencing drag and a harmonic disturbance
    """
    # forces
    def HarmonicForce(t_i, r_i, v_i):
        return -2*r_i
    def Drag(t_i, r_i, v_i):
        return -0.1*v_i
    def Drive(t_i, r_i, v_i):
        return 0.1*np.array([np.cos(2*t_i), np.cos(2*t_i)])
    
    # setup particle
    particle = PointParticle(mass=1, Ndim=2)
    r0 = np.array([1,0])
    v0 = np.array([0,1])

    # simulate and plot movement
    particle.simulate_path(HarmonicForce, Drag, Drive, r0=r0, v0=v0, dt=1e-3, N=int(3e4))
    particle.plot(0)
    particle.plot(1)
    particle.plot_2D_path()
    