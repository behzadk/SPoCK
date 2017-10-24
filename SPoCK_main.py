import numpy as np
import scipy as sp
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import PyDSTool as dst
import pprint
import time
from matplotlib.backends.backend_pdf import PdfPages
import re
import os
import glob
import csv


def SPoCK_scipy_equations(y, t, D, mu_X, mu_Y, N_max,
                          gamma_A, gamma_B, k_A, f_max,
                          f_min, A_c, b, w):


    # Unpack variables
    X = y[0]
    Y = y[1]
    A = y[2]
    B = y[3]


    k_u =  1 - ( (X + Y) / N_max )
    fA = f_min + ( ( (f_max - f_min) * A_c**b ) / ( A**b + A_c**b) )
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
    X_0 = icsdict['X']
    Y_0 = icsdict['Y']
    A_0 = icsdict['A']
    B_0 = icsdict['B']

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
    y0 = [X_0, Y_0, A_0, B_0]
    sol = odeint(SPoCK_scipy_equations, y0, t, args=(D, mu_X, mu_Y, N_max,
                                                     gamma_A, gamma_B, k_A, f_max,
                                                     f_min, A_c, b, w), mxstep=5000000)

    final_X = sol[:, 0][step - 1]
    final_Y = sol[:, 1][step - 1]


    """Criteria for saving parameter values and plot of populations"""
    if final_X > 10**2 and final_Y > 10**2:
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
            pdf.savefig(plt.figure(1))
            plt.close()


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
    w = np.random.uniform(0, 1) / 10**np.random.randint(0, 3)

    pardict = {
        'D': 0.3,                   #   Dilution rate
        'mu_X': 0.4,                #   Engineered strain growth rate
        'mu_Y': 0.8,                #   Competitor growth rate
        'N_max': 10**9,             #   carrying capacity
        'gamma_A': 0.11,            #   Degradation rates of AHL
        'gamma_B': gamma_B,         #   Degradation rate for bacteriocin
        'k_A': 0.1,                 #   AHL production coefficient
        'f_min': 0,                 #   Minimal bacteriocin production level (PLtetO promoter leakiness)
        'f_max': 0.1,               #   Maximal bacteriocin production level
        'A_c': 10,                  #   AHL concentration at which bacteriocin production is at half-maximum
        'b': 2,                     #   Hil coefficient for bacteriocin production
        'w': w                      #   Susceptibility of competitor to bacteriocin
    }

    """Functions"""
    fndict = {
        'k_u': (['X', 'Y'], '1 - ( (X + Y) / N_max ) '),
        'fA': (['A'], 'f_min + ( ( (f_max - f_min) * A_c**b ) / ( A**b + A_c**b) )'),
        'omega': (['B'], 'w * B')
    }

    """Differential Equations"""
    vardict = {
        'X': 'k_u(X, Y) * mu_X * X - D * X',                          # Plasmid bearing strain population
        'Y': 'k_u(X, Y) * mu_Y * Y - D * Y - omega(B) * Y',              # Competitor strain population
        'A': 'k_A * X - gamma_A * A - D * A',                   # AHL quorum sensing molecule concentration
        'B': 'fA(A) * X - gamma_B * B - D * B'                     # Bacteriocin concentration
    }

    X_init = np.random.randint(1, 4)
    Y_init = np.random.randint(1, 4)

    icsdict = {
        'X': 10**X_init,
        'Y': 10**Y_init,
        'A': 0,
        'B': 0
    }

    return (pardict, fndict, vardict, icsdict)

def SPoCK_pydstool_model():
    pardict, fndict, vardict, icsdict = init_SPoCK()

    DSargs = dst.args()
    DSargs.pars = pardict
    DSargs.varspecs = vardict
    DSargs.fnspecs = fndict
    DSargs.ics = icsdict

    DSargs.name = 'SPoCK'
    DSargs.tdata = [0, 5]

    lv = dst.Vode_ODEsystem(DSargs)
    traj = lv.compute('SPoCK')
    pts = traj.sample()

    """Plot solution"""
    plt.figure(1)
    plt.plot(pts['t'], pts['X'], 'g', label='X')
    plt.plot(pts['t'], pts['Y'], 'b', label='Y')
    plt.title('SPoCK')
    plt.xlabel('t')
    plt.ylabel('Population')
    plt.legend(loc=3)  # bottom left location
    plt.show()

def get_initial_exp_number(parameter_csv_path):
    """
    description:
        Returns the highest experiment number in the directory passed as argument,
        so as not to overwrite existing files

    arguments:
        Path to parameter csv files

    returns:
        String of initial exp number
    """
    parameter_csv_path = parameter_csv_path + "*.csv"
    all_files = max(glob.glob(parameter_csv_path))
    file_name = os.path.basename(all_files)
    file_name_noext = file_name.split('.')[0]
    max_exp_num = int(file_name_noext.split('_')[1])
    initial_exp_num = str(max_exp_num + 1)

    return(initial_exp_num)

def main():
    #SPoCK_pydstool_model()

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

if __name__ == "__main__":
    main()