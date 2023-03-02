import numpy as np

from src.simulator.Util import get_purity, get_reward
import logging

logger = logging.getLogger()
logger.setLevel(20)


class process_env:
    ''' fermentation and chromatography simulation environment (virtual laboratory)
    '''

    def __init__(self,
                 upstream,
                 downstream,
                 upstream_variable_cost,
                 downstream_variable_cost,
                 product_price,
                 failure_cost,
                 product_requirement,
                 purity_requirement,
                 yield_penalty_cost):
        self.done = False
        self.upstream_done = False
        self.initial_state = np.array(upstream.initial_state)
        self.state = np.array(upstream.initial_state)
        self.upstream = upstream  # simulation model
        self.downstream = downstream
        self.num_state = upstream.num_state
        self.num_action = upstream.num_action
        self.purity_requirement = purity_requirement
        self.product_requirement = product_requirement
        self.t = 0
        self.upstream_variable_cost = upstream_variable_cost
        self.downstream_variable_cost = downstream_variable_cost
        self.failure_cost = failure_cost
        self.product_price = product_price
        self.yield_penalty_cost = yield_penalty_cost

        self.upstream_termination_time = self.upstream.harvest_time / self.upstream.delta_t
        self.process_end_time = self.upstream_termination_time + self.downstream.horizon

    def reset(self):
        self.done = False
        self.upstream_done = False
        self.state = np.random.multivariate_normal(self.initial_state,
                                                   self.initial_state / 5 * np.identity(self.num_state))
        self.t = 0
        return self.state

    def step(self, action):  # predict one step lookaahead
        if self.t < self.upstream_termination_time:
            n_state = self.upstream.simulate(self.state, action)
            r = - action * self.upstream_variable_cost  # variable cost
            self.t += self.upstream.delta_t
        else:
            n_state = self.downstream.simulate(self.state, action, int(self.t - self.upstream_termination_time))
            r = - action * self.downstream_variable_cost  # variable cost
            self.t += 1
        self.done = True if self.process_end_time == self.t else False

        if self.done:
            purity = get_purity(n_state)
            r += get_reward(purity, n_state[-2], self.product_requirement, self.purity_requirement,
                            self.failure_cost, self.product_price, self.yield_penalty_cost)

        if self.t == self.upstream_termination_time:
            self.upstream_done = True
            if len(self.state) > 2:
                self.state = n_state[4:6] * n_state[-1]
                logging.info(
                    'upstream harvest: protein {:.2f} g and impurity {:.2f} g'.format(self.state[0], self.state[1]))
        else:
            self.state = n_state
        return n_state, r, self.done, self.upstream_done


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
        self.q_I = 0.003
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
        dV = 0  # u - F_evp
        return np.array([dX, dG, dN, dL, dP, dI, dV])

    def step(self, x, u, p):
        grad = self.grad(x, u, p)
        return grad * self.dt

    def simulate(self, state, action):
        for _ in range(self.num_step):
            state += self.step(state, action, self.p)
            state = np.clip(state, 0, None)
        return np.clip(np.random.multivariate_normal(state, np.diag(state) / self.noise_level), 0, None)


class chromatography:
    def __init__(self,
                 noise_level=0.05,
                 horizon=3):
        # define parameters in bioreactor model
        purity_coef = [0.48, 0.28]
        self.chrom1_protein = [self.get_alpha_beta(mu=purity, sigma=noise_level * purity_coef[0]) for purity in
                               [(1 + pool / 12) * purity_coef[0] for pool in range(10)]]
        self.chrom1_impurity = [self.get_alpha_beta(mu=purity, sigma=noise_level * purity_coef[1]) for purity in
                                [(1 + pool / 12) * purity_coef[1] for pool in range(10)]]

        # self.chrom1_protein = [[purity - 0.1 * purity_coef[0], purity + 0.1 * purity_coef[0]] for purity in [(1 + pool / 11) * purity_coef[0] for pool in range(10)]]
        # self.chrom1_impurity = [[purity - 0.1 * purity_coef[1], purity + 0.1 * purity_coef[1]] for purity in [(1 + pool / 11) * purity_coef[1] for pool in range(10)]]

        purity_coef = [0.5, 0.25]
        self.chrom2_protein = [self.get_alpha_beta(mu=purity, sigma=noise_level * purity_coef[0]) for purity in
                               [(1 + pool / 12) * purity_coef[0] for pool in range(10)]]
        self.chrom2_impurity = [self.get_alpha_beta(mu=purity, sigma=noise_level * purity_coef[1]) for purity in
                                [(1 + pool / 12) * purity_coef[1] for pool in range(10)]]

        purity_coef = [0.5, 0.22]
        self.chrom3_protein = [self.get_alpha_beta(mu=purity, sigma=noise_level * purity_coef[0]) for purity in
                               [(1 + pool / 12) * purity_coef[0] for pool in range(10)]]
        self.chrom3_impurity = [self.get_alpha_beta(mu=purity, sigma=noise_level * purity_coef[1]) for purity in
                                [(1 + pool / 12) * purity_coef[1] for pool in range(10)]]

        self.true_model_params = {1: {'protein': self.chrom1_protein, 'impurity': self.chrom1_impurity},
                                  2: {'protein': self.chrom2_protein, 'impurity': self.chrom2_impurity},
                                  3: {'protein': self.chrom3_protein, 'impurity': self.chrom3_impurity}}

        self.sim_size = 10
        self.horizon = horizon

    def get_alpha_beta(self, alpha=None, beta=None, mu=None, sigma=None):
        if (alpha is not None) and (beta is not None):
            pass
        elif (mu is not None) and (sigma is not None):
            kappa = mu * (1 - mu) / sigma ** 2 - 1
            alpha = mu * kappa
            beta = (1 - mu) * kappa
        else:
            raise ValueError('Incompatible parameterization. Either use alpha '
                             'and beta, or mu and sigma to specify distribution.')
        return alpha, beta

    def simulate(self, initial_state, window, step, rand_seed=None):
        if rand_seed == None:
            np.random.seed(
                int(str(window) + str(step) + str(int(np.sum(initial_state)*100))))
        else:
            np.random.seed(rand_seed)

        protein_param = self.true_model_params[step + 1]['protein'][window]
        impurity_param = self.true_model_params[step + 1]['impurity'][window]
        removal_rate_protein = np.random.beta(protein_param[0], protein_param[1])
        removal_rate_impurity = np.random.beta(impurity_param[0], impurity_param[1])

        protein = initial_state[0] * removal_rate_protein
        impurity = initial_state[1] * removal_rate_impurity
        logging.info('protein removal rate: {:.2f}; impurity removal rate: {:.2f}'.format(removal_rate_protein,
                                                                                          removal_rate_impurity))
        return [protein, impurity]


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    initial_state = [0.4, 10, 5, 0, 0., 0, 5]  # [3.4, 40, 5, 1.5]
    simulator = cho_cell_culture_simulator(initial_state, delta_t=1, num_action=1, noise_level=2500)
    chrom = chromatography()
    env = process_env(simulator, chrom,
                      upstream_variable_cost=0.001,
                      downstream_variable_cost=10,
                      product_price=50,
                      failure_cost=48,
                      product_requirement=10,
                      purity_requirement=0.85,
                      yield_penalty_cost=6, )
    action = 0.05  # L/h action space [0, 0.2]
    done, upstream_done = False, False
    state_buffer = []
    next_state_buffer = []
    action_buffer = []
    reward_buffer = []
    while not done:
        cur_state = env.state
        next_state, reward, done, upstream_done = env.step(action)
        if upstream_done:
            action = 4
        state_buffer.append(cur_state)
        next_state_buffer.append(next_state)
        action_buffer.append(action)
        reward_buffer.append(reward)

    print('upstream: ', next_state_buffer[359], 'downstream: ', np.array(next_state_buffer[360:]))
    plt.plot(next_state_buffer[:360], label=simulator.label)
    plt.legend()
    plt.show()
