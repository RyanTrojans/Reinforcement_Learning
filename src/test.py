import numpy as np

from constants import number_steps, product_r, purity_r
from chromatography import chromatography
from simulator import cho_cell_culture_simulator
from environment import process_env
from train import  Env

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

    initial_state = [0.4, 10, 5, 0, 0., 0, 5]  # [3.4, 40, 5, 1.5]
    simulator = cho_cell_culture_simulator(initial_state, delta_t=int(360 / number_steps), num_action=1,
                                           noise_level=2500)
    chrom = chromatography()
    env = process_env(simulator, chrom,
                           upstream_variable_cost=2,  # sensitive hyperparameters
                           downstream_variable_cost=10,
                           product_price=30,
                           failure_cost=200,
                           product_requirement=product_r,  # sensitive hyperparameters 20 -60
                           purity_requirement=purity_r,  # sensitive hyperparameters 0.85 - 0.93
                           yield_penalty_cost=50,  # sensitive hyperparameters
                           )

    action = 0.04
    done, upstream_done = False, False
    rewards = []
    t = 0
    while not done:
        next_state, reward, done, upstream_done = env.step(action)
        print(f'step {t}/{env.t}, action: {action}, state:', env.state)
        if t == 14:
            print()
        if upstream_done:
            action = 2  # [6]
        rewards.append(reward)
        t += 1
    print(np.sum(rewards), rewards)

    action = 0.04
    rewards = []
    env = Env()
    env.reset()
    while True:
        _, r, done, _ = env.step(action)
        rewards.append(r)
        if env.env.upstream_done:
            action = 2
        if done:
            break
    print(np.sum(rewards), rewards)


if __name__ == '__main__':
    test()