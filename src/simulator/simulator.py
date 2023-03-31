import numpy as np


class cho_cell_culture_simulator:
    def __init__(self, initial_state, delta_t, num_action, noise_level=2000, harvest_time=360):
        self.a_2 = None
        self.a_1 = None
        self.m_G = None
        self.mu_max = None
        self.noise_level = noise_level
        self.set_param()
        self.KGlc = 1  # [mM]
        self.KGln = 0.047  # [mM]
        self.KIL = 43  # [mM]
        self.YXGlc = 10.57  # [1e8 cell / mmol]
        self.YXGln = 9.74  # [1e8 cell / mmol]
        self.YLacGlc = 0.7  # [mol / mol]
        self.YNH4Gln = 0.6287  # [mol / mol]
        self.q_mab = 0.00151
        self.q_I = 0.01
        self.kd = 0.004
        self.kDlac = 45.8
        self.Glcin = 50
        self.Glnin = 10
        self.Lacin = 0
        self.p = [self.KGlc, self.KGln, self.KIL, self.YXGlc, self.YXGln, self.YLacGlc, self.q_mab, self.q_I]
        self.delta_t = delta_t  # the process measurement time interval
        self.dt = 0.1  # numerical time interval
        self.num_step = int(self.delta_t / self.dt)
        self.harvest_time = harvest_time
        self.initial_state = initial_state
        self.num_action = num_action
        self.num_state = len(initial_state)
        self.label = ['Xv', 'Glucose', 'Glutamine', 'Lactate', 'Product', 'Impurity', 'Volume']

    def set_param(self):
        self.mu_max = 0.039
        self.m_G = 69.2 * 1e-4
        self.a_1 = 3.2 * 1e-4
        self.a_2 = 2.1 * 1e-4

    def grad(self, x, u, p):
        F_evp = 0.001
        mu = self.mu_max * x[1] / (p[0] + x[1]) * x[2] / (p[1] + x[2]) * p[2] / (p[2] + x[3])
        mu_d = self.kd * x[3] / (x[3] + self.kDlac)
        m_N = self.a_1 * x[2] / (self.a_2 + x[2])

        dX = (mu - mu_d) * x[0] - (u - F_evp) / x[6] * x[0]
        dG = - ((mu - mu_d) / p[3] + self.m_G) * x[0] + (u - F_evp) / x[6] * (self.Glcin - x[1])
        dN = - ((mu - mu_d) / p[4] + m_N) * x[0] + (u - F_evp) / x[6] * (self.Glnin - x[2])
        dL = p[5] * ((mu - mu_d) / p[3] + self.m_G) * x[0] + u / x[6] * (self.Lacin - x[3])
        dP = p[6] * (1 - mu / self.mu_max) * x[0] - u / x[6] * x[4]
        dI = p[7] * mu / self.mu_max * x[0] - u / x[6] * x[5]
        dV = u - F_evp
        return np.array([dX, dG, dN, dL, dP, dI, dV])

    def step(self, x, u, p):
        grad = self.grad(x, u, p)
        return grad * self.dt

    def simulate(self, state, action):
        for _ in range(self.num_step):
            state += self.step(state, action, self.p)
            state = np.clip(state, 0, None)
        return np.clip(np.random.multivariate_normal(state, np.diag(state) / self.noise_level), 0, None)