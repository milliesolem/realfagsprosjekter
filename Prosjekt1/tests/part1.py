import numpy as np

def test_Quadratic_Functor(Quadratic_Functor):
    # setup default functor
    default_F = Quadratic_Functor()

    # test attributes
    assert hasattr(default_F, "a"), "Quadratic_Functor doesn't have a variable called 'a'"
    assert hasattr(default_F, "b"), "Quadratic_Functor doesn't have a variable called 'b'"
    assert hasattr(default_F, "c"), "Quadratic_Functor doesn't have a variable called 'c'"

    # test default values
    assert default_F.a == 1, "Wrong default value of 'a'"
    assert default_F.b == 0, "Wrong default value of 'b'"
    assert default_F.c == 0, "Wrong default value of 'c'"

    # setup functor
    F = Quadratic_Functor(a=2, b=1, c=-3)

    # test output
    assert F(-1) == -2, "Quadratic f(x)=2x^2+x-3 does not give f(-1)=2"
    assert F(2) == 7, "Quadratic f(x)=2x^2+x-3 does not give f(-2)=7"
    assert F(5) == 52, "Quadratic f(x)=2x^2+x-3 does not give f(5)=52"

    print("Nice! Task complete.")

def test_NetSpringForce(NetSpringForce, PointParticle):
    # test construction
    try:
        A = NetSpringForce(m=1, k=1)
    except:
        raise AttributeError("unable to call NetSpringForce(m=1, k=1)")

    # test attributes
    assert hasattr(A, "m"), "NetSpringForce doesn't have a variable called 'm'"
    assert hasattr(A, "k"), "NetSpringForce doesn't have a variable called 'k'"
    assert hasattr(A, "g"), "NetSpringForce doesn't have a variable called 'g'"
    assert hasattr(A, "x0"), "NetSpringForce doesn't have a variable called 'x0'"
    assert hasattr(A, "L"), "NetSpringForce doesn't have a variable called 'L'"

    # test default values
    assert A.g == 9.81, "Wrong default value of 'g'"
    assert A.x0 == 1, "Wrong default value of 'x0'"
    assert A.L == 1, "Wrong default value of 'L'"

    # test __call__ output
    assert abs(A(0,1,0) - (-1-9.81))<1e-3, "__call__ returns the wrong number"
    assert A(0, 0, 0) != A(0, 1, 0), "__cal__ doesn't change when argument x=0 is changed to x=1, it should!"
    assert A(0, 1, 0) == A(1, 1, 1), "__call__ changes when argument t=0 is changed to t=1, it should not!"
    assert A(0, 1, 0) == A(0, 1, 1), "__call__ changes when argument v=0 is changed to v=1, it should not!"

    # check negative g is accepted
    assert NetSpringForce(1,1,g=1)(1,1,1) == NetSpringForce(1,1,g=-1)(1,1,1), "the sign of 'g' shouldnt' matter!"

    # evaluate a path
    A = NetSpringForce(m=1, k=100, g=0, x0=1, L=1)
    particle = PointParticle(mass=A.m)
    particle.simulate_path(A, dt=1e-4, N=1e4, r0=1, v0=0)
    x = np.cos(10*particle.t)
    assert np.all(np.abs(particle.r.flatten() - x) < 1e-3), "Invalid path, check __call__ to see if it's correct"

    # plot path
    print("Nice Work! Your implementation was successful!")
    particle.plot()