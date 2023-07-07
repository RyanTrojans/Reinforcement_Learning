import logging
from typing import Tuple

import gym
import numpy as np

import torch as th
from torch import nn

from stable_baselines3 import PPO, SAC, DDPG, TD3
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.sac.policies import SACPolicy

from constants import number_steps, product_r, purity_r, products, impuritys, cell_densitys, actions, actions_downstream
from chromatography import chromatography
from simulator import cho_cell_culture_simulator
from environment import process_env

print('products:', products)
print('impuritys:', impuritys)
print('cell_densitys:', cell_densitys)
print('actions:', actions)
print('actions_downstream:', actions_downstream)

logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)

def make_env(initial_state):
    simulator = cho_cell_culture_simulator(initial_state, delta_t=int(360 / number_steps), num_action=1,
                                           noise_level=2500)
    chrom = chromatography()
    return process_env(simulator, chrom,
                           upstream_variable_cost=2,  # sensitive hyperparameters
                           downstream_variable_cost=10,
                           product_price=30,
                           failure_cost=200,
                           product_requirement=50,  # sensitive hyperparameters 20 -60
                           purity_requirement=0.93,  # sensitive hyperparameters 0.85 - 0.93
                           yield_penalty_cost=50,  # sensitive hyperparameters
                           )

state_dim = 8
initial_state = [0.4, 10, 5, 0, 0., 0, 5]  # [3.4, 40, 5, 1.5]


class Env(gym.Env):
    num_steps = 0

    def __init__(self):
        self.reward_range = -1e4, 1e4
        self.action_space = gym.spaces.Discrete(n=max(len(actions), len(actions_downstream)))
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(state_dim,), dtype=float)

    def reset(self, state=None, time=None):
        self.num_steps = 0

        if state is None:
            state = initial_state

        self.env = make_env(state)

        if time is not None:
            if time >= number_steps:
                dui = {0: 360, 1: 361, 2: 362}
                self.env.t = dui[time - number_steps]
            else:
                self.env.t = time * 24

            self.env.upstream_done = time >= number_steps

        state = make_state(self.env)
        return state

    def step(self, action):
        # print('action:', action)
        self.num_steps += 1

        if self.env.upstream_done:
            action = min(action, len(actions_downstream) - 1)

        if not self.env.upstream_done:
            # print('action:', action)
            action = actions[action]
        else:
            action = actions_downstream[action]
        # print('action:', action)

        # if not self.env.upstream_done:
        #     action = 0.04
        # else:
        #     action = 2

        state, reward, done, upstream_done = self.env.step(action)

        # if not done:
        #     reward = 0

        state = make_state(self.env)

        assert abs(reward) < 1e6
        return state, reward, done, {'action': action}

def make_state(env, norm=False):
    state = env.state
    if isinstance(state, np.ndarray):
        state = state.tolist()

    # p, i = state[-2:]
    # print(check_region(p, i))

    if not env.upstream_done:
        state.append(0)
    else:
        state.extend([0, 0, 0, 0, 0, 1])

    if norm:
        state = [_ / 100 for _ in state]

    assert len(state) == state_dim, (state, len(state), state_dim, env.t, env.upstream_done)
    return state


def check_region(p, i):
    eta_t = p / (p + i + 1e-6)
    eta_d = purity_r
    p_d = 50
    p_t = p

    if eta_t >= eta_d:
        return '1,2'
    return '-'


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    # def _build_mlp_extractor(self) -> None:

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        actions, values, log_prob = super().forward(obs, deterministic)
        # print(actions.shape, actions.min(dim=1, keepdim=True)[0].shape)
        # actions = actions - actions.min(dim=1, keepdim=True)[0]
        return actions, values, log_prob


def train():
    env = Env()

    num_units = 32
    # num_units *= 2

    if 1:
        policy_kwargs = dict(net_arch=[dict(pi=[num_units, num_units], vf=[num_units, num_units])],
                             activation_fn=nn.Mish, squash_output=True)
        model = PPO(ActorCriticPolicy, env, policy_kwargs=policy_kwargs, n_steps=256, verbose=1, device='cpu')
        # model = PPO(CustomActorCriticPolicy, env, n_steps=256, verbose=1, device='cpu')
    else:
        policy_kwargs = dict(net_arch=[num_units, num_units],
                             activation_fn=nn.Mish)
        model = SAC(SACPolicy, env, policy_kwargs=policy_kwargs, verbose=1,
                    train_freq=2, batch_size=1024, learning_starts=1024,
                    buffer_size=100_000, tau=0.1)

    # model.learn(total_timesteps=200_000, )
    model.learn(total_timesteps=5_000, log_interval=10)

    env = Env()
    state = env.reset()

    while True:
        action = model.predict(state, deterministic=True)[0]
        state, reward, done, info = env.step(action)

        p, i = env.env.state[-2:]

        print('action:', info['action'], reward, check_region(p, i))
        if done:
            break


if __name__ == '__main__':
    train()
"""
学习率：learning_rate = 0.0003
每隔多少步训练：n_steps = 2048
训练批量大小：batch_size = 64
每次训练epochs：n_epochs = 10
gamma = 0.99
GAE（advantage计算）lambda参数：gae_lambda = 0.95
ratio clip参数：clip_range = 0.2
value函数clip参数：clip_range_vf = None
是否normalize advantage：normalize_advantage = True
policy分布的entropy loss系数：ent_coef = 0.0
value loss的系数：vf_coef = 0.5
训练时梯度clipping：max_grad_norm = 0.5
"""
