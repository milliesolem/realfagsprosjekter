import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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
    
    def plot_2D_path(self, dimx=0, dimy=1, filename=None, show=True):
        """
        Plot the planar path of the particle

        input:
            dimx     (int)  : horizontal dimension
            dimy     (int)  : vertical dimension
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
        ax.plot(self.r[:,dimx], self.r[:,dimy])
        ax.scatter(self.r[0,dimx], self.r[0,dimy], s=dotsize, label="start")
        ax.scatter(self.r[-1,dimx], self.r[-1,dimy], s=dotsize, label="end")
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
    
    def animate_2D_path(self, xlim, ylim, dimx=0, dimy=1, trace=1, trlen=0.3, intrv=20, nskip=0, show=True):
        """
        Animate the planar path of the particle along dimensions dimx and dimy

        input:
            xlim     (tuple of int) : (min, max), horizontal boundaries of animation
            ylim     (tuple of int) : (min, max), vertical boundaries of animation
            dimx     (int)          : horizontal dimension
            dimy     (int)          : vertical dimension
            trace    (int)          : draw trace (0=False, 1=fading trace, 2=full trace)
            trlen    (int)          : portion of complete path to trace
            intrv    (int)          : pause between frames (ms)
            nskip    (int)          : frames to skip
            show     (bool)         : show plot if True, return (fig,ax) if False
        
        output:
            if show is False:
                fig, ax, ani : matplotlib figure, axes and animation
        """
        # extract plotting settings
        dotsize = self.plotting_configuration["dotsize"]
        figsize = self.plotting_configuration["figsize"]
        titlesize = self.plotting_configuration["titlefontsize"]
        labelsize = self.plotting_configuration["labelfontsize"]

        # extract data
        r, x, y = self.r[::nskip+1], self.r[::nskip+1,dimx], self.r[::nskip+1,dimy]

        # auto trace length
        trlen = int(trlen*x.size)

        # setup figure
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

        # prepare animation figure
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        # init dynamic elements
        mass = ax.scatter(x[0], y[0], c='k', s=dotsize)
        if trace > 0:
            trace_ = ax.scatter(x[0], y[0], s=dotsize, cmap="binary_r")

        # setup update functions
        def update_no_trace(i):
            mass.set_offsets([[x[i],y[i]]])
            return mass,

        def update_with_trace(i):
            mass.set_offsets([[x[i],y[i]]])
            if i < trlen:
                trace_.set_offsets(r[:i])
                trace_.set_sizes(np.linspace(0,dotsize,i+1))
                trace_.set_array(np.linspace(1,0,i+1))
                trace_.set_cmap("binary_r")
            else:
                trace_.set_offsets(r[i-trlen:i])
                trace_.set_sizes(np.linspace(0,dotsize,trlen))
                trace_.set_array(np.linspace(1,0,trlen))
                trace_.set_cmap("binary_r")
            return mass, trace_,
        
        # select update function
        if trace == 0:
            update = update_no_trace
        if trace > 0:
            update = update_with_trace
        
        # prettify
        ax.set_title("Particle Path Animation", fontsize=titlesize)
        ax.set_xlabel("dim 0", fontsize=labelsize)
        ax.set_ylabel("dim 1", fontsize=labelsize)

		# produce animation
        self.ani = FuncAnimation(fig, func=update, interval=intrv, frames=np.arange(x.size))

        # finalize figure
        if show:
            plt.show()
        else:
            return fig, ax, self.ani
    
    def save_animation(self, filename, fps=60, **kwargs):
        """
        Save previous animation

        input:
            filename (str) : path to file
            fps      (int) : frames per second
            kwargs   (...) : matplotlib.Animation kwargs
        
        For more details on available kwargs, see:
        https://matplotlib.org/api/_as_gen/matplotlib.animation.Animation.save.html
        """
        filename = filename if filename.endswith(".mp4") else filename + ".mp4"
        self.ani.save(filename, fps=60)


if __name__ == '__main__':
    """
    Example: particle in a harmonic oscillator potential
             experiencing drag and a harmonic disturbance
    """
    # forces
    def HarmonicForce(t_i, r_i, v_i):
        return -2*r_i
    def Drag(t_i, r_i, v_i):
        return -0.8*v_i
    def Drive(t_i, r_i, v_i):
        return 0.1*np.array([np.cos(2*t_i), np.cos(2*t_i)])
    
    # setup particle
    particle = PointParticle(mass=1, Ndim=2)
    r0 = np.array([0.5,0])
    v0 = np.array([-0.5,0.5])

    # simulate movement
    particle.simulate_path(HarmonicForce, Drag, r0=r0, v0=v0, dt=1e-3, N=5e3)
    
    # plot x and y components
    particle.plot(0)
    particle.plot(1)

    # plot 2D path
    particle.plot_2D_path()

    # animate 2D path
    particle.animate_2D_path(xlim=(-0.4,0.5), ylim=(-0.2,0.3), dimx=0, dimy=1, trace=1, trlen=0.3, intrv=1, nskip=0)
    
    # save animation (this takes time)
    #particle.save_animation(filename="test.mp4", fps=60)
    