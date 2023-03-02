import pymc3 as pm
import arviz as az
# import autograd.numpy as np
# from autograd import grad
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import scipy.stats as stats
import scipy.optimize as optimize
from scipy.integrate import odeint
from pymc3.ode import DifferentialEquation
import numpy as np


mu_max = 0.039
m_G = 69.2
a_1 = 3.2
a_2 = 2.1


def step(x, t, p):
    mu = mu_max * x[1] / (p[0] + x[1]) * x[2] / (p[1] + x[2]) * p[2] / (p[2] + x[3])
    m_N = a_1 * x[2] / (a_2 + x[2])
    dX = mu * x[0]
    dG = (mu/p[3] + m_G) * x[0]
    dN = - p[5] * (mu/p[4] + m_N) * x[0]
    dL = p[5] * (mu/p[3] + m_G) * x[0]
    dP = p[6] * (1 - mu / mu_max) * x[0]
    dI = p[7] * mu / mu_max * x[0]
    return [dX, dG, dN, dL, dP, dI]

if __name__ == '__main__':

    y0 = [0.1, 30, 5, 0, 0, 0]
    noise = 0.1
    KG, KN, KIL, Y_XG, Y_XN, Y_LG, Q_P, Q_I = 1, 0.047, 43, 0.357, 0.974, 0.7, 1.51, 1
    times = np.arange(0, 360, 1)
    y = odeint(step, t=times, y0=y0, args=tuple([[KG, KN, KIL, Y_XG, Y_XN, Y_LG, Q_P, Q_I]]))
    yobs = np.random.normal(y, 0.5)
    plt.plot(yobs[0])
    np.random.seed(20394)
    # times = np.arange(0, 72, 0.5).reshape(72 * 2)
    ode_model = DifferentialEquation(func=step, times=times, n_states=6, n_theta=8, t0=0)

    with pm.Model() as model:
        # Specify prior distributions for some of our model parameters
        KG = pm.Uniform('KG', 0, 2)  # Prior for our cooling coefficient
        KN = pm.Uniform('KN', 0, 0.1)
        KIL = pm.Uniform('KIL', 20, 100)
        Y_XG = pm.Uniform('Y_XG', 0, 0.5)
        Y_XN = pm.Uniform('Y_XN', 0.5, 1)
        Y_LG = pm.Uniform('Y_LG', 0.5, 1)
        Q_P = pm.Uniform('Q_P', 1, 2)
        Q_I = pm.Uniform('Q_I', 0, 2)
        # noise = pm.Uniform('noise', 0, 0.2)
        # Y_XG = pm.HalfNormal('Y_XG', sigma=1)    # prior for our estimated standard deviation of the error
        sigma = pm.HalfNormal('sigma', sigma=0.7)
        # If we know one of the parameter values, we can simply pass the value.
        ode_solution = ode_model(y0=y0, theta=[KG, KN, KIL, Y_XG, Y_XN, Y_LG, Q_P, Q_I])
        # The ode_solution has a shape of (n_times, n_states)
        Y = pm.Normal("Y", mu=ode_solution, sigma=sigma, observed=yobs)
        prior = pm.sample_prior_predictive()
        trace = pm.sample(2000, tune=1000, cores=14)
        posterior_predictive = pm.sample_posterior_predictive(trace)
        data = az.from_pymc3(trace=trace, prior=prior, posterior_predictive=posterior_predictive)


    ## simplify model
    ode_model = DifferentialEquation(func=step, times=times, n_states=6, n_theta=8, t0=0)

    with pm.Model() as model:
        # Specify prior distributions for some of our model parameters
        KG = pm.Uniform('KG', 0.5, 2)  # Prior for our cooling coefficient
        KN = pm.Uniform('KN', 0.01, 0.08)
        # KIL = pm.Uniform('KIL', 20, 100)
        Y_XG = pm.Uniform('Y_XG', 0.1, 0.5)
        Y_XN = pm.Uniform('Y_XN', 0.5, 1)
        # Y_LG = pm.Uniform('Y_LG', 0.5, 1)
        Q_P = pm.Uniform('Q_P', 1, 2)
        Q_I = pm.Uniform('Q_I', 0.5, 2)
        # noise = pm.Uniform('noise', 0, 0.2)
        # Y_XG = pm.HalfNormal('Y_XG', sigma=1)    # prior for our estimated standard deviation of the error
        sigma = pm.HalfNormal('sigma', sigma=0.7)
        # If we know one of the parameter values, we can simply pass the value.
        ode_solution = ode_model(y0=y0, theta=[KG, KN, 43., Y_XG, Y_XN, 0.7, Q_P, Q_I])
        # The ode_solution has a shape of (n_times, n_states)
        Y = pm.Normal("Y", mu=ode_solution, sigma=sigma, observed=yobs)
        prior = pm.sample_prior_predictive()
        trace = pm.sample(2000, tune=1000, cores=14)
        posterior_predictive = pm.sample_posterior_predictive(trace)
        data = az.from_pymc3(trace=trace, prior=prior, posterior_predictive=posterior_predictive)

