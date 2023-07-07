import logging
import random

import numpy as np

from chromatography import chromatography
from constants import actions_downstream, actions
from environment import process_env
from simulator import cho_cell_culture_simulator

logging.getLogger().setLevel(logging.ERROR)

def test():
    initial_state = [0.4, 10, 5, 0, 0., 0, 5]  # [3.4, 40, 5, 1.5]
    simulator = cho_cell_culture_simulator(initial_state, delta_t=24, num_action=1, noise_level=2500)
    chrom = chromatography()
    env = process_env(simulator, chrom,
                      upstream_variable_cost=2,  # sensitive hyperparameters
                      downstream_variable_cost=10,
                      product_price=30,
                      failure_cost=200,
                      product_requirement=50,  # sensitive hyperparameters 20 -60
                      purity_requirement=0.93,  # sensitive hyperparameters 0.85 - 0.93
                      yield_penalty_cost=50,  # sensitive hyperparameters
                      )
    done, upstream_done = False, False
    reward_buffer = []
    t = 0
    while not done:
        if not upstream_done:
            action_test = random.choice(actions)
        else:
            action_test = int(random.choice(actions_downstream))
        next_state, reward, done, upstream_done = env.step(action_test)
        reward_buffer.append(reward)
        t += 1

    print(np.sum(reward_buffer))

if __name__ == '__main__':
    for i in range(10):
        test()