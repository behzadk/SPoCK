import numpy as np
import scipy as sp
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import PyDSTool as dst
import sympy as sp
import pprint
import time
from matplotlib.backends.backend_pdf import PdfPages
import csv
from PyDSTool.Toolbox import phaseplane as pp
import pandas as pd
import time


def make_jacobian(X_sub, Y_sub, A_sub, B_sub):
    pardict = init_SPoCK()[0]   #Get parameter dictionary
    icsdict = init_SPoCK()[3]   #Get initial conditions dictionary

    # unpack pardict
    D = pardict['D']
    mu_X = pardict['mu_X']
    mu_Y = pardict['mu_Y']
    N_max = pardict['N_max']
    gamma_A = pardict['gamma_A']
    gamma_B = pardict['gamma_B']
    k_A = pardict['k_A']
    f_max = pardict['f_max']
    f_min = pardict['f_min']
    A_c = pardict['A_c']
    b = pardict['b']
    w = pardict['w']

    X, Y, A, B, t = sp.symbols('X, Y, A, B, t', real =True)

    k_u =  1 - ( (X + Y) / N_max )
    fA = f_min + ( ( (f_max - f_min) * A_c**b ) / ( A**b + A_c**b) )
    omega = w * B

    fX = k_u * mu_X * X - D * X                      # Plasmid bearing strain population
    fY = k_u * mu_Y * Y - D * Y - omega * Y          # Competitor strain population
    fA = k_A * X - gamma_A * A - D * A               # AHL quorum sensing molecule concentration
    fB = (fA * X) - (gamma_B * B) - (D * B)          # Bacteriocin concentration

    F = sp.Matrix([fX, fY, fA, fB])
    pprint.pprint(F)
    J = F.jacobian([t, X, Y, A, B])
    pprint.pprint(J)



def SPoCK_scipy_equations(y, t, D, mu_X, mu_Y, N_max,
                          gamma_A, gamma_B, k_A, f_max,
                          f_min, A_c, b, w):

    # Unpack variables
    X = y[0]
    Y = y[1]
    A = y[2]
    B = y[3]

    k_u =  1 - ( (X + Y) / N_max )
    fA = f_min + ( ( (f_max - f_min) * A**b ) / ( A**b + A_c**b) )
    omega = w * B

    X = k_u * mu_X * X - D * X                      # Plasmid bearing strain population
    Y = k_u * mu_Y * Y - D * Y - omega * Y          # Competitor strain population
    A = k_A * X - gamma_A * A - D * A               # AHL quorum sensing molecule concentration
    B = (fA * X) - (gamma_B * B) - (D * B)          # Bacteriocin concentration

    solution = [X, Y, A, B]
    return solution

def SPoCK_scipy_model(parameter_csv_path, exp_num, t_max, step):
    """
    description:
        Scipy model, used for speed of ODE solver over PyDStool ODE solver. Can use this to identify potential
        parameter sets which we can then pass to the PyDStool version.

    arguments:
        parameter_csv_path - Directory to save parameter dictionary
        exp_num - Name of .pdf and .csv output

    return:
        0

    """

    pardict = init_SPoCK()[0]   #Get parameter dictionary
    icsdict = init_SPoCK()[3]   #Get initial conditions dictionary

    # Unpack initial conditions
    X_0 = 7076637.16604
    Y_0 = 666670023.855
    A_0 = 2721783.5254
    B_0 = 1.11048463257

    # unpack pardict
    D = pardict['D']
    mu_X = pardict['mu_X']
    mu_Y = pardict['mu_Y']
    N_max = pardict['N_max']
    gamma_A = pardict['gamma_A']
    gamma_B = pardict['gamma_B']
    k_A = pardict['k_A']
    f_max = pardict['f_max']
    f_min = pardict['f_min']
    A_c = pardict['A_c']
    b = pardict['b']
    w = pardict['w']

    t = np.linspace(0, t_max, step)
    print(t_max)
    print(step)
    y0 = [X_0, Y_0, A_0, B_0]
    sol = odeint(SPoCK_scipy_equations, y0, t, args=(D, mu_X, mu_Y, N_max,
                                                     gamma_A, gamma_B, k_A, f_max,
                                                     f_min, A_c, b, w), mxstep=5000000)

    final_X = sol[:, 0][step - 1]
    final_Y = sol[:, 1][step - 1]
    print(final_X)
    print(final_Y)

    plt.figure(1)
    plt.plot(sol[:, 0], sol[:, 1], 'b', label='X')
    #plt.plot(t, sol[:, 1], 'g', label='Y')
    plt.title('SPoCK')
    plt.xlabel('X')
    #plt.yscale('log')
    #plt.xscale('log')
    plt.ylabel('Y')
    plt.legend(loc=3)  # bottom left location
    plt.show()

    """Criteria for saving parameter values and plot of populations"""
    if final_X < 10**2 and final_Y > 10**2:
        csv_path = parameter_csv_path + "exp_" + exp_num + ".csv"
        csv_file = open(csv_path, 'wb')
        wrkbk = csv.DictWriter(csv_file, pardict.keys())
        wrkbk.writeheader()
        wrkbk.writerow(pardict)

        with PdfPages(parameter_csv_path + "exp_" + exp_num + ".pdf") as pdf:
            plt.figure(1)
            plt.plot(t, sol[:, 0], 'b', label='X')
            plt.plot(t, sol[:, 1], 'g', label='Y')
            plt.title('SPoCK')
            plt.xlabel('t')
            plt.yscale('log')
            plt.ylabel('Population')
            plt.legend(loc=3)  # bottom left location
            #pdf.savefig(plt.figure(1))
            plt.show()
            #plt.close()


def init_SPoCK():
    """
    Desrtiption:
        Generates parameter, variable, function and initial conditions compatible with PyDStools.

    returns:
        (pardict, fndict, vardict, icsdict)

    """
    #Cellbio by numbers, decay rate of proteins have a half-life of ~1-3 hours.
    # giving a degradation range constant of 0.69 /h to 0.23 /h.
    gamma_B = np.random.uniform(0, 1)

    # Antibiotic concentrations working concentrations can range from 5ug/ml to 200ug/ml
    # https://academic.oup.com/jac/article/70/3/811/769950/Time-kill-kinetics-of-antibiotics-active-against
    # Kill rates of antibiotics betwen 0.0 to 0.5 per hour depending on concentration. They also have sigmoidal curves, maybe we should add this?
    # Maximum we should see cvaC reaching is about 5nM. 0.2/5 = 0.04

    pardict = {
        'D': 0.3,                   #   Dilution rate
        'mu_X': 0.4,                #   Engineered strain growth rate
        'mu_Y': 0.8,                #   Competitor growth rate
        'N_max': 10**9,             #   carrying capacity
        'gamma_A': 0.11,            #   Degradation rates of AHL
        'gamma_B': 0,               #   Degradation rate for bacteriocin
        'k_A': 0.1,                 #   AHL production coefficient
        'f_min': 0,                 #   Minimal bacteriocin production level (PLtetO promoter leakiness)
        'f_max': 0.1,               #   Maximal bacteriocin production level
        'A_c': 10**4,               #   AHL concentration at which bacteriocin production is at half-maximum
        'b': 2,                     #   Hil coefficient for bacteriocin production
        'w': 10**-3                     #   Susceptibility of competitor to bacteriocin
    }

    """Functions"""
    fndict = {
        'k_u': (['X', 'Y'], '1 - ( (X + Y) / N_max ) '),
        'fA': (['A'], 'f_min + ( ( (f_max - f_min) * pow(A_c,b) ) / ( pow(A,b) + pow(A_c,b) ) )'),
        'omega': (['B'], 'w * B')
    }

    """Differential Equations"""
    vardict = {
        'X': 'k_u(X, Y) * (mu_X * X) - (D * X)',                          # Plasmid bearing strain population
        'Y': 'k_u(X, Y) * (mu_Y * Y) - (D * Y) - (omega(B) * Y)',              # Competitor strain population
        'A': '(k_A * X) - (gamma_A * A) - (D * A)',                   # AHL quorum sensing molecule concentration
        'B': '(fA(A) * X) - (gamma_B * B) - (D * B)'                     # Bacteriocin concentration
    }

    icsdict = {
        'X': 0,
        'Y': 0,
        'A': 0,
        'B': 0
    }

    return (pardict, fndict, vardict, icsdict)

def make_jac_pydstool(ode):
    jac, new_fnspecs = dst.prepJacobian(ode.funcspec._initargs['varspecs'], ['X', 'Y', 'A', 'B'], ode.funcspec._initargs['fnspecs'])

    scope = dst.copy(ode.pars)
    scope.update(new_fnspecs)
    jac_fn = dst.expr2fun(jac, ensure_args=['t'], **scope)

    return jac_fn

def SPoCK_plot_traj(alt_ics_dict, exp_num):
    pardict, fndict, vardict, icsdict = init_SPoCK()

    DSargs = dst.args()
    DSargs.pars = pardict
    DSargs.varspecs = vardict
    DSargs.fnspecs = fndict
    DSargs.ics = alt_ics_dict       #Load initial conditions given my argument

    DSargs.name = 'SPoCK'
    DSargs.tdata = [0, 600]
    DSargs.xdomain = {'X': [0, 10 ** 9],
                      'Y': [0, 10 ** 9],
                      'A': [0, 10 ** 9],
                      'B': [0, 10 ** 9],
                      }

    spock_ode = dst.Vode_ODEsystem(DSargs)

    fixedpoints_csv_path = "/Users/behzakarkaria/Documents/UCL/Barnes Lab/PhD Project/research_code/SPoCK_model/parameter_csv/fixedponts.csv"
    plot_out_path = "/Users/behzakarkaria/Documents/UCL/Barnes Lab/PhD Project/research_code/SPoCK_model/parameter_csv/fixedpoint_plots/"

    traj = spock_ode.compute('traj_1')
    pts = traj.sample()

    with PdfPages(plot_out_path + "exp_" + str(exp_num) + ".pdf") as pdf:
        plt.figure(1)
        plt.plot(pts['t'], pts['X'], label="X")
        plt.plot(pts['t'], pts['Y'], label="Y")
        plt.xlabel("t")
        plt.ylabel("population")
        plt.yscale('log')
        plt.legend(loc=4)  # bottom left location
        pdf.savefig(plt.figure(1))
        plt.show()
        plt.close()

def SPoCK_find_fixedpoints():
    pardict, fndict, vardict, icsdict = init_SPoCK()

    DSargs = dst.args()
    DSargs.pars = pardict
    DSargs.varspecs = vardict
    DSargs.fnspecs = fndict
    DSargs.ics = icsdict

    DSargs.name = 'SPoCK'
    DSargs.tdata = [0, 50000]
    DSargs.xdomain = {'X': [0, 10 ** 9],
                      'Y': [0, 10 ** 9],
                      'A': [0, 10 ** 9],
                      'B': [0, 10 ** 9],
                      }

    spock_ode = dst.Vode_ODEsystem(DSargs)
    jac_fun = make_jac_pydstool(spock_ode)

    fp_coords = pp.find_fixedpoints(spock_ode, n=10, eps=1e-50, jac=jac_fun, subdomain={'X': [0, 10**9], 'Y':[0, 10**9], 'A':[0, 10**9], 'B':[0, 10**9] } )
    fixedpoints_pdf_path = "/Users/behzakarkaria/Documents/UCL/Barnes Lab/PhD Project/research_code/SPoCK_model/parameter_csv/"

    plt.figure(1)
    fig, ax = plt.subplots(figsize=(18, 12.5))
    ax.set_xscale('symlog', basex=10)
    ax.set_yscale('symlog', basey=10)
    ax.set_ylim(0, 10**9)
    ax.set_xlim(0, 10**9)
    good_coords = []
    for fp in fp_coords:
        try:
            fp_obj = pp.fixedpoint_nD(spock_ode, dst.Point(fp), coords=fp, jac=jac_fun, eps=1e-20)        #Does he tolerance here matter when we find the points above with a good tolerance?
            good_coords.append(fp)
        except:
            continue

        if fp_obj.stability == 'u':
            style = 'wo'
        elif fp_obj.stability == 'c':
            style = 'co'
        else: # 's'
            style = 'ko'

        print("")
        print(fp_obj.stability)
        print('X:', fp_obj.fp_coords['X'])
        print('Y:', fp_obj.fp_coords['Y'])
        print("")

        try:
            ax.plot(fp_obj.fp_coords['X'], fp_obj.fp_coords['Y'], style)

        except ValueError:
            continue

    plt.xlabel("X")
    plt.ylabel("Y")
    #plt.title('Fixed points \n w = 1*10**-2, D = 0.2 ')

    pdf_page_obj = PdfPages(fixedpoints_pdf_path + "fixedpoints_XY" + ".pdf")
    pdf_page_obj.savefig(ax.get_figure())
    pdf_page_obj.close()


    fixedpoints_csv_path = "/Users/behzakarkaria/Documents/UCL/Barnes Lab/PhD Project/research_code/SPoCK_model/parameter_csv/fixedponts.csv"
    fieldnames = ('index', 'X', 'Y', 'A', 'B')

    index = 0
    for fp_dict in good_coords:
        fp_dict['index'] = index
        index = index + 1



    data_frame = pd.DataFrame.from_records(good_coords)
    data_frame.to_csv(fixedpoints_csv_path)

    # for fp_dict in fp_coords:
    #     print(type(fp_dict))
    #     print(fp_dict)
    #     pd.DataFrame(fp_dict).to_csv(fixedpoints_csv_path)

def SPoCK_phase_plane(fixed_points_dict):
    pardict, fndict, vardict, icsdict = init_SPoCK()

    DSargs = dst.args()
    DSargs.pars = pardict
    DSargs.varspecs = vardict
    DSargs.fnspecs = fndict
    DSargs.ics = icsdict

    DSargs.name = 'SPoCK'
    DSargs.tdata = [0, 1500]
    DSargs.xdomain = {'X': [0, 10 ** 9],
                      'Y': [0, 10 ** 9],
                      'A': [0, 10 ** 9],
                      'B': [0, 10 ** 9],
                      }

    spock_ode = dst.Vode_ODEsystem(DSargs)
    print("loaded ode system")
    #pp.plot_PP_vf(spock_ode, 'X', 'Y', scale_exp=0, N=50, subdomain={'X':[X_lower ,X_Upper], 'Y':[Y_lower,Y_Upper]} )

    #plot vector field

    #ax.quiver(X_quiv, Y_quiv, dxs, dys, angles='xy', pivot='middle', units='inches', )

    plt.figure(1)
    fig, ax = plt.subplots(figsize=(18, 12.5))
    ax.set_xscale('log', basex=10)
    ax.set_yscale('log', basey=10)
    ax.set_xlim(10 ** 0, 10 ** 8)
    ax.set_ylim(10 ** 0, 10 ** 9)

    ax = custom_PP_vf_2(spock_ode, 'X', 'Y', ax, scale_exp=-1, N=150, subdomain={'X': [0, 9], 'Y': [0, 9]})
    #ax = custom_PP_vf_2(spock_ode, 'X', 'Y', ax, scale_exp=-1, N=50, subdomain={'X': [2, 4], 'Y': [2, 3]})
    print("finished plotting vf")
    # subdomain = {'X': [0, 0.7e9], 'A': [0, 0.8e9]
    count = 0
    for dict in fixed_points_dict:
        DSargs = dst.args()
        DSargs.pars = pardict
        DSargs.varspecs = vardict
        DSargs.fnspecs = fndict
        DSargs.ics = dict

        DSargs.name = 'SPoCK'
        DSargs.tdata = [0, 600]
        DSargs.xdomain = {'X': [0, 10 ** 9],
                          'Y': [0, 10 ** 9],
                          'A': [0, 10 ** 9],
                          'B': [0, 10 ** 9],
                          }

        spock_ode = dst.Vode_ODEsystem(DSargs)

        traj = spock_ode.compute('traj_1')
        try:
            pts = traj.sample()

        except AttributeError:
            print("No solution found")
            continue


        ax.plot(pts['X'], pts['Y'])
        #ax.plot(pts['t'], pts['B'])

        plt.xlabel('X')
        plt.ylabel('Y')
        print(count)

        count = count + 1

    phase_plot_path = "/Users/behzakarkaria/Documents/UCL/Barnes Lab/PhD Project/research_code/SPoCK_model/parameter_csv/"

    pdf_page_obj = PdfPages(phase_plot_path + "phase_plots_5_fp" + ".pdf")
    pdf_page_obj.savefig(ax.get_figure())
    pdf_page_obj.close()

    plt.show()



def SPoCK_bifurcation(alt_ics_dict):
    pardict, fndict, vardict, icsdict = init_SPoCK()

    DSargs = dst.args()
    DSargs.pars = pardict
    DSargs.varspecs = vardict
    DSargs.fnspecs = fndict
    DSargs.ics = alt_ics_dict

    DSargs.name = 'SPoCK'
    DSargs.tdata = [0, 10000]
    DSargs.xdomain = {'X': [0, 10 ** 9],
                      'Y': [0, 10 ** 9],
                      'A': [0, 10 ** 9],
                      'B': [0, 10 ** 9],
                      }

    spock_ode = dst.Vode_ODEsystem(DSargs)
    #spock_ode.set(pars = {'w': 1} )                    # Lower bound of the control parameter 'i'
    #traj = spock_ode.compute('test_trj')  # integrate ODE

    # setup continuation class
    PC = dst.ContClass(spock_ode)

    PCargs = dst.args(name='EQ1', type='EP-C')
    PCargs.freepars = ['A_c']
    PCargs.StepSize = 1
    PCargs.MaxNumPoints = 10000
    PCargs.MaxStepSize = 2
    PCargs.MinStepSize = 0.5
    PCargs.MaxTestIters = 10000
    PCargs.LocBifPoints = 'all'
    PCargs.SaveEigen = True
    PCargs.verbosity = 2
    PC.newCurve(PCargs)

    PC['EQ1'].backward()

    plot_out_path = "/Users/behzakarkaria/Documents/UCL/Barnes Lab/PhD Project/research_code/SPoCK_model/parameter_csv/fixedpoint_plots/"


    with PdfPages(plot_out_path + "bifurcation_Ac_X" + ".pdf") as pdf:
        plt.figure(1)
        fig, ax = plt.subplots(figsize=(18,12.5))
        ax.set_xlim(10**0, 10**9)
        ax.set_ylim(10**0, 10**9)
        ax.set_xscale('symlog', basex=10)
        ax.set_yscale('symlog', basey=10)
        PC.display(('A_c', 'X'), stability=True, figure=1)
        pdf.savefig(ax.get_figure())

    with PdfPages(plot_out_path + "bifurcation_Ac_Y" + ".pdf") as pdf:
        plt.figure(2)
        fig, ax2 = plt.subplots(figsize=(18,12.5))
        ax2.set_xlim(0, 10**9)
        ax2.set_ylim(0, 10**9)
        ax2.set_xscale('symlog', basex=10)
        ax2.set_yscale('symlog', basey=10)
        PC.display(('A_c', 'Y'),  stability=True, figure=2)
        pdf.savefig(ax2.get_figure())


    with PdfPages(plot_out_path + "bifurcation_X_Y" + ".pdf") as pdf:
        plt.figure(3)
        fig, ax3 = plt.subplots(figsize=(18, 12.5))
        ax3.set_xlim(0, 10**9)
        ax3.set_ylim(0, 10**9)
        ax3.set_xscale('symlog', basex=10)
        ax3.set_yscale('symlog', basey=10)
        PC.display(('X', 'Y'), stability=True, figure=3)
        pdf.savefig(ax3.get_figure())

def SPoCK_pydstool_model():
    pass
    #PC.display(('D', 'X'), stability=True, figure=1)
    #plt.xlim(-1, 5)
    #PC.display(('D', 'Y'), stability=True, figure=2)
    #plt.xlim(-1, 5)

    #plt.show()



    #nulls_X, nulls_Y = pp.find_nullclines(spock_ode, 'X', 'Y', n=3, eps=1e-8, max_step=0.1,fps=fp_coords)
    #plt.plot(nulls_X[:,0], nulls_X[:,1], 'b')
    #plt.plot(nulls_Y[:,0], nulls_Y[:,1], 'g')

    ##plt.axis('tight')
    #plt.axis([0,15,0,15])
    #plt.title('Phase plane')
    #plt.xlabel('X')
    #plt.ylabel('Y')
    #plt.show()
    #print("fixed points: ", fp_coords)
    """Draw 2D list of fixed points (singletons allowed), must be
    fixedpoint_2D objects.

    Optional do_evecs (default False) draws eigenvectors around each f.p.

    Requires matplotlib
    """
    if isinstance(fps, fixedpoint_2D):
        # singleton passed
        fps = [fps]

    x, y = fps[0].fp_coords
    for fp in fps:
        # When matplotlib implements half-full circle markers
        #if fp.classification == 'saddle':
            # half-open circle
        if fp.stability == 'u':
            style = 'wo'
        elif fp.stability == 'c':
            style = 'co'
        else: # 's'
            style = 'ko'
        plt.plot(fp.point[x], fp.point[y], style, markersize=markersize, mew=2)

def custom_PP_vf_2(gen, xname, yname, ax, N=20, subdomain=None, scale_exp=0):
    """Draw 2D vector field in (xname, yname) coordinates of given Generator,
    sampling on a uniform grid of n by n points.

    Optional subdomain dictionary specifies axes limits in each variable,
    otherwise Generator's xdomain attribute will be used.

    For systems of dimension > 2, the non-phase plane variables will be held
      constant at their initial condition values set in the Generator.

    Optional scale_exp is an exponent (domain is all reals) which rescales
      size of arrows in case of disparate scales in the vector field. Larger
      values of scale magnify the arrow sizes. For stiff vector fields, values
      from -3 to 3 may be necessary to resolve arrows in certain regions.

    Requires matplotlib 0.99 or later
    """
    assert N > 1
    xdom = gen.xdomain[xname]
    ydom = gen.xdomain[yname]
    if subdomain is not None:
        try:
            xdom = subdomain[xname]
        except KeyError:
            pass
        try:
            ydom = subdomain[yname]
        except KeyError:
            pass
    assert all(pp.isfinite(xdom)), "Must specify a finite domain for x direction"
    assert all(pp.isfinite(ydom)), "Must specify a finite domain for y direction"
    w = xdom[1]-xdom[0]
    h = ydom[1]-ydom[0]

    xdict = gen.initialconditions.copy()

    xix = gen.funcspec.vars.index(xname)
    yix = gen.funcspec.vars.index(yname)

    xs = np.logspace(xdom[0], xdom[1], N)
    ys = np.logspace(ydom[0], ydom[1], N)

    X, Y = np.meshgrid(xs, ys)
    dxs, dys = np.meshgrid(xs, ys)

##    dx_big = 0
##    dy_big = 0
    dz_big = 0
    vec_dict = {}

#    dxs = array((n,), float)
#    dys = array((n,), float)
    for xi, x in enumerate(xs):
        for yi, y in enumerate(ys):
            xdict.update({xname: x, yname: y})
            dx, dy = gen.Rhs(0, xdict)[[xix, yix]]
            # note order of indices
            dxs[yi,xi] = dx
            dys[yi,xi] = dy
            dz = np.linalg.norm((dx,dy))
##            vec_dict[ (x,y) ] = (dx, dy, dz)
##            if dx > dx_big:
##                dx_big = dx
##            if dy > dy_big:
##                dy_big = dy
            if dz > dz_big:
                dz_big = dz

    ax.quiver(X, Y, dxs, dys, angles='xy', pivot='middle', units='inches',
               scale=dz_big*max(h,w)/(10*pp.exp(2*scale_exp)), lw=0.01/pp.exp(scale_exp-1),
               headwidth=max(2,1.5/(pp.exp(scale_exp-1))),
               #headlength=2*max(2,1.5/(exp(scale_exp-1))),
               width=0.001*max(h,w), minshaft=2, minlength=0.001)

    return ax

##    # Use 95% of interval size
##    longest_x = w*0.95/(n-1)
##    longest_y = h*0.95/(n-1)
##    longest = min(longest_x, longest_y)
##
##    scaling_x = longest_x/dx_big
##    scaling_y = longest_y/dy_big
##    scaling = min(scaling_x, scaling_y)

    #ax = plt.gca()
##    hw = longest/10
##    hl = hw*2
##    for x in xs:
##        for y in ys:
##            dx, dy, dz = vec_dict[ (x,y) ]
##            plt.arrow(x, y, scaling*dx, yscale*scaling*dy,
##                      head_length=hl, head_width=hw, length_includes_head=True)
    #ax.set_xlim(xdom)
    #ax.set_ylim(ydom)
    #plt.draw()

def custom_PP_vf(gen, xname, yname, N=20, subdomain=None, scale_exp=0):
    """Draw 2D vector field in (xname, yname) coordinates of given Generator,
        sampling on a uniform grid of n by n points.

        Optional subdomain dictionary specifies axes limits in each variable,
        otherwise Generator's xdomain attribute will be used.

        For systems of dimension > 2, the non-phase plane variables will be held
          constant at their initial condition values set in the Generator.

        Optional scale_exp is an exponent (domain is all reals) which rescales
          size of arrows in case of disparate scales in the vector field. Larger
          values of scale magnify the arrow sizes. For stiff vector fields, values
          from -3 to 3 may be necessary to resolve arrows in certain regions.

        Requires matplotlib 0.99 or later
        """
    assert N > 1
    xdom = gen.xdomain[xname]
    ydom = gen.xdomain[yname]
    if subdomain is not None:
        try:
            xdom = subdomain[xname]
        except KeyError:
            pass
        try:
            ydom = subdomain[yname]
        except KeyError:
            pass
    assert all(pp.isfinite(xdom)), "Must specify a finite domain for x direction"
    assert all(pp.isfinite(ydom)), "Must specify a finite domain for y direction"
    w = xdom[1] - xdom[0]
    h = ydom[1] - ydom[0]

    xdict = gen.initialconditions.copy()

    xix = gen.funcspec.vars.index(xname)
    yix = gen.funcspec.vars.index(yname)

    xs = np.logspace(xdom[0], xdom[1], N)
    ys = np.logspace(ydom[0], ydom[1], N)

    X, Y = np.meshgrid(xs, ys)
    dxs, dys = np.meshgrid(xs, ys)

    ##    dx_big = 0
    ##    dy_big = 0
    dz_big = 0
    vec_dict = {}

    #    dxs = array((n,), float)
    #    dys = array((n,), float)
    for xi, x in enumerate(xs):
        for yi, y in enumerate(ys):
            xdict.update({xname: x, yname: y})
            dx, dy = gen.Rhs(0, xdict)[[xix, yix]]
            # note order of indices
            dxs[yi, xi] = dx
            dys[yi, xi] = dy
            dz = np.linalg.norm((dx, dy))
            ##            vec_dict[ (x,y) ] = (dx, dy, dz)
            ##            if dx > dx_big:
            ##                dx_big = dx
            ##            if dy > dy_big:
            ##                dy_big = dy
            if dz > dz_big:
                dz_big = dz
    return (X, Y, dxs, dys)

    """
    plt.quiver(X, Y, dxs, dys, angles='xy', pivot='middle', units='inches')
               # headlength=2*max(2,1.5/(exp(scale_exp-1))),

    ##    # Use 95% of interval size
    ##    longest_x = w*0.95/(n-1)
    ##    longest_y = h*0.95/(n-1)
    ##    longest = min(longest_x, longest_y)
    ##
    ##    scaling_x = longest_x/dx_big
    ##    scaling_y = longest_y/dy_big
    ##    scaling = min(scaling_x, scaling_y)

    #ax = plt.gca()
    ##    hw = longest/10
    ##    hl = hw*2
    ##    for x in xs:
    ##        for y in ys:
    ##            dx, dy, dz = vec_dict[ (x,y) ]
    ##            plt.arrow(x, y, scaling*dx, yscale*scaling*dy,
    ##                      head_length=hl, head_width=hw, length_includes_head=True)
    #ax.set_xscale('linear')
    #ax.set_yscale('linear')
    #ax.set_xlim(xdom)
    #ax.set_ylim(ydom)

    plt.draw()
    """
def main():
    #make_jacobian(0, 0, 0, 0)
    fixedpoints_csv_path = "/Users/behzakarkaria/Documents/UCL/Barnes Lab/PhD Project/research_code/SPoCK_model/parameter_csv/fixedponts_keep.csv"

    #SPoCK_find_fixedpoints()
    df = pd.read_csv(fixedpoints_csv_path, index_col=0)
    df = df.drop('index', axis=1)
    fixedpoints_dict_list = df.to_dict('records')

    exp_num = 0
    #SPoCK_plot_traj(fixedpoints_dict_list[3], 2)
    #exit()

    #SPoCK_plot_traj(fixedpoints_dict_list[3], 1)
    #SPoCK_phase_plane([fixedpoints_dict_list[3]])
    SPoCK_bifurcation(fixedpoints_dict_list[10])

    #SPoCK_scipy_model(fixedpoints_csv_path, 1, 1000, 100000)
    #SPoCK_bifurcation(fixedpoints_dict_list[1])

    # for fixedpoint_set in fixedpoints_dict_list:
    #     SPoCK_plot_traj(fixedpoint_set, exp_num)
    #     exp_num = exp_num + 1
    #     print(exp_num)



    #SPoCK_find_fixedpoints()
    #SPoCK_bifurcation()
    #plot_B_curve()
    exit()
    """
    #Path to output directory
    parameter_csv_path = "/Users/behzakarkaria/Documents/UCL/Barnes Lab/PhD Project/research_code/SPoCK_model/parameter_csv/"

    t_max = 100
    step = 10000

    #Identify highest exp_num in directory, used for naming output
    try:
        exp_num = get_initial_exp_number(parameter_csv_path)

    #ValueError thrown if no file currently exists, set exp_num to 1
    except ValueError:
        exp_num = 1

    z = 0

    while z < 5000:
        SPoCK_scipy_model(parameter_csv_path, exp_num, t_max, step)
        z = z + 1
        exp_num = str(int(exp_num) + 1)
    """

if __name__ == "__main__":
    main()



