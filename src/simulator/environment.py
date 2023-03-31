import numpy as np
import logging
from BNStructureAnalysis.src.simulator.Util import *


# Create ENV
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

        self.upstream_termination_time = self.upstream.harvest_time
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
            r = - action * self.upstream_variable_cost * self.upstream.delta_t  # variable cost
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