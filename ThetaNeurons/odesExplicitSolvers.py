# ------------------------------------------------------------------------------- #
import numpy;
# ------------------------------------------------------------------------------- #
def string2function(rhsFunction):
    """ Here is a transformation <class 'string'> to <class 'function'>. """
    if type(rhsFunction) == type(''):
        rhsFunction = eval(rhsFunction);
    return rhsFunction;
# ------------------------------------------------------------------------------- #
def odesSolverDef(method):
    """ Here is a definition of the selected solver for an ODEs system. """
    odesSolverVersion = 0;
    if type(method) == type([]):
        odesSolver = string2function(method[0]);
        if len(method) > 1:
            odesSolverVersion = int(method[1]);
    else:
        odesSolver = string2function(method);
    return odesSolver, odesSolverVersion;
# ------------------------------------------------------------------------------- #
def rhsFunctionDef(problem):
    """ Here is a definition of a right-hand side function of an ODEs system. """        
    if type(problem) == type([]):
        if len(problem) > 1:
            rhsFunction = getattr(problem[0], problem[1]);
        else:
            rhsFunction = string2function(problem[0]);
    else:
        rhsFunction = string2function(problem);
    return rhsFunction;
# ------------------------------------------------------------------------------- #
def ivpSolution(odesMethod, xSpan, numSteps, odesProblem, yInitial, *args, **kwargs):
    """
    This function numerically integrates a system of ordinary differential equations (ODEs)
    given initial values, i.e., it solves an initial value problem (IVP) for an ODEs system.
    """
    tolerance = 1e+5;
    odesSolver, odesSolverVersion = odesSolverDef(odesMethod);
    rhsFunction = rhsFunctionDef(odesProblem);
    dx = (xSpan[1] - xSpan[0]) / (numSteps[0] * numSteps[1]);
    x = numpy.linspace(xSpan[0], xSpan[1], numSteps[0] + 1);
    y = numpy.zeros([numSteps[0] + 1, len(yInitial)]);
    y[0] = yInitial.copy();
    yy, xx = y[0], x[0];
    for nx in range(0, numSteps[0]):
        for mx in range(0, numSteps[1]):
            yy, xx = odesSolver(rhsFunction, yy, xx, dx, odesSolverVersion, *args, **kwargs);
        if (yy != yy).any() or (abs(yy - y[nx]) >= tolerance).any():
            break;
        else:
            y[nx + 1] = yy;
    return y, x;
# ------------------------------------------------------------------------------- #
def ivpBVP(odesMethod, xSpan, numSteps, odesProblem, yInitial, *args, **kwargs):
    """
    This function numerically integrates an ODEs system  and returns values of dependent
    variables  at the end  of the  given  interval vs. values of  these variables at the
    starting point, i.e., it  solves an  initial  value problem (IVP) for an ODEs system.
    """
    tolerance = 1e+5;
    odesSolver, odesSolverVersion = odesSolverDef(odesMethod);
    rhsFunction = rhsFunctionDef(odesProblem);
    dx = (xSpan[1] - xSpan[0]) / numpy.float_(numSteps);
    y, x = yInitial.copy(), xSpan[0];
    for nx in range(0, numSteps):
        yy, xx = odesSolver(rhsFunction, y, x, dx, odesSolverVersion, *args, **kwargs);
        if (yy != yy).any() or (numpy.abs(yy - y) >= tolerance).any():
            break;
        else:
            y, x = yy, xx;
    if (nx < numSteps - 1):
        yBoundaries = [];        
        checkReachingBorder = False;
    else:
        checkReachingBorder = True;
        yBoundaries = numpy.append([], yInitial);
        yBoundaries = numpy.append(yBoundaries, y);
        yBoundaries = numpy.reshape(yBoundaries, (2, len(y)), 'C');
    return yBoundaries, checkReachingBorder;
# ------------------------------------------------------------------------------- #
def euler(rhsFunction, y, x, dx, btType = 0, *args, **kwargs):
    """ Forward Euler method (first-order Runge–Kutta method). """
    k1 = rhsFunction(y, x, *args, **kwargs);
    x_new = x + dx;
    y_new = y + k1 * dx;
    return y_new, x_new;
# ------------------------------------------------------------------------------- #
def rk2(rhsFunction, y, x, dx, btType = 0, *args, **kwargs):
    """
    Second-order Runge–Kutta methods, including
    (0) - Explicit midpoint method, (1) - Heun's method, (2) - Ralston's method.
    """    
    ButcherTableau = {
    # Explicit midpoint method.
    0: [[1./2., 1./2.], [1., 0., 1.]],
    # Heun's method.
    1: [[1., 1.], [1., 1./2., 1./2.]],
    # Ralston's method.
    2: [[2./3., 2./3.], [1., 1./4., 3./4.]]
    };
    c = ButcherTableau.get(btType, "Error::Invalid type of the Butcher tableau.");
    k1 = rhsFunction(y, x, *args, **kwargs);
    k2 = rhsFunction(y + c[0][1] * k1 * dx, x + c[0][0] * dx, *args, **kwargs);
    x_new = x + c[1][0] * dx;
    y_new = y + (c[1][1] * k1 + c[1][2] * k2) * dx;
    return y_new, x_new;

def rk2g(rhsFunction, y, x, dx, a, *args, **kwargs):
    """ Generic second-order Runge–Kutta method. """    
    c = [[a, a], [1., 1. - 1. / (2. * a), 1. / (2. * a)]];
    k1 = rhsFunction(y, x, *args, **kwargs);
    k2 = rhsFunction(y + c[0][1] * k1 * dx, x + c[0][0] * dx, *args, **kwargs);
    x_new = x + c[1][0] * dx;
    y_new = y + (c[1][1] * k1 + c[1][2] * k2) * dx;
    return y_new, x_new;
# ------------------------------------------------------------------------------- #
def rk3(rhsFunction, y, x, dx, btType = 0, *args, **kwargs):
    """
    Third-order Runge–Kutta method, including
    (0) - Kutta's method, (1) - Heun's method, (2) - Ralston's method,
    (3) - Strong Stability Preserving Runge-Kutta method.
    """    
    ButcherTableau = {
    # Kutta's method.
    0: [[1./2., 1./2.], [1., -1., 2.], [1., 1/6., 2./3., 1./6.]],
    # Heun's method.
    1: [[1./3., 1./3.], [2./3., 0., 2./3.], [1., 1./4., 0., 3./4.]],
    # Ralston's method.
    2: [[1./2., 1./2.], [3./4., 0., 3./4.], [1., 2./9., 1./3., 4./9.]],
    # Strong Stability Preserving Runge-Kutta method.
    3: [[1., 1.], [1./2., 1./4., 1./4.], [1., 1/6., 1./6., 2./3.]]
    };
    c = ButcherTableau.get(btType, "Error::Invalid type of the Butcher tableau.");
    k1 = rhsFunction(y, x, *args, **kwargs);
    k2 = rhsFunction(y + c[0][1] * k1 * dx, x + c[0][0] * dx, *args, **kwargs);
    k3 = rhsFunction(y + (c[1][1] * k1 + c[1][2] * k2) * dx, x + c[1][0] * dx, *args, **kwargs);
    x_new = x + c[2][0] * dx;
    y_new = y + (c[2][1] * k1 + c[2][2] * k2 + c[2][3] * k3) * dx;
    return y_new, x_new;

def rk3g(rhsFunction, y, x, dx, a, *args, **kwargs):
    """ Generic third-order Runge–Kutta method. """    
    assert(a != 0. and a != 2./3. and a !=1.), 'Error::Invalid value of the method parameter a.';
    c = [[a, a], [1., 1. + (1. - a) / (a * (3. * a - 2.)), (a - 1.) / (a * (3. * a - 2.))],
         [1., (3. * a - 2.) / (6. * a), 1. / (6. * a * (1. - a)), (2. - 3. * a) / (6. * (1. - a))]];
    k1 = rhsFunction(y, x, *args, **kwargs);
    k2 = rhsFunction(y + c[0][1] * k1 * dx, x + c[0][0] * dx, *args, **kwargs);
    k3 = rhsFunction(y + (c[1][1] * k1 + c[1][2] * k2) * dx, x + c[1][0] * dx, *args, **kwargs);
    x_new = x + c[2][0] * dx;
    y_new = y + (c[2][1] * k1 + c[2][2] * k2 + c[2][3] * k3) * dx;
    return y_new, x_new;
# ------------------------------------------------------------------------------- #
def rk4(rhsFunction, y, x, dx, btType = 0, *args, **kwargs):
    """
    Fourth-order Runge–Kutta method, including
    (0) - Classical method, (1) - 3/8-rule, (2) - Ralston's method.
    """
    ButcherTableau = {
    # Classical Runge–Kutta method.
    0: [[1./2., 1./2.], [1./2., 0., 1./2.], [1., 0., 0., 1.], [1., 1./6., 1./3., 1./3., 1./6.]],
    # 3/8-rule.
    1: [[1./3., 1./3.], [2./3., -1./3., 1.], [1., 1., -1., 1.], [1., 1./8., 3./8., 3./8., 1./8.]],
    # Ralston's method. (*) This method has minimum truncation error.
    2: [[.4, .4], [.45573725, .29697761, .15875964],
        [1., .2181004, -3.05096516, 3.83286476], [1., .17476028, -.55148066, 1.2055356, .17118478]]
    };
    c = ButcherTableau.get(btType, "Error::Invalid type of the Butcher tableau.");
    k1 = rhsFunction(y, x, *args, **kwargs);
    k2 = rhsFunction(y + c[0][1] * k1 * dx, x + c[0][0] * dx, *args, **kwargs);
    k3 = rhsFunction(y + (c[1][1] * k1 + c[1][2] * k2) * dx, x + c[1][0] * dx, *args, **kwargs);
    k4 = rhsFunction(y + (c[2][1] * k1 + c[2][2] * k2 + c[2][3] * k3) * dx, x + c[2][0] * dx, *args, **kwargs);
    x_new = x + c[3][0] * dx;
    y_new = y + (c[3][1] * k1 + c[3][2] * k2 + c[3][3] * k3 + c[3][4] * k4) * dx;
    return y_new, x_new;
# ------------------------------------------------------------------------------- #
