import logging
from typing import Tuple

import gym
import numpy as np

import torch as th
from torch import nn

from stable_baselines3 import PPO, SAC, DDPG, TD3
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.sac.policies import SACPolicy

from chromatography import chromatography
from constants import number_steps, product_r, purity_r, products, impuritys, cell_densitys, actions, actions_downstream
from simulator import cho_cell_culture_simulator
from environment import process_env

print('products:', products)
print('impuritys:', impuritys)
print('cell_densitys:', cell_densitys)
print('actions:', actions)
print('actions_downstream:', actions_downstream)

logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)

class Env(gym.Env):
    state_dim = 8
    num_steps = 0

    def __init__(self):
        # self.action_space = gym.spaces.Box(low=np.array([products[0], impuritys[0], cell_densitys[0]]),
        #                                    high=np.array([products[-1], impuritys[-1], cell_densitys[-1]]), shape=(3,),
        #                                    dtype=float)
        # self.action_space = gym.spaces.Tuple([gym.spaces.Box(low=actions[0], high=actions[-1], shape=(1,), dtype=float),
        #                                       gym.spaces.Discrete(n=1)])
        # self.action_space = gym.spaces.Dict({'upstream':gym.spaces.Box(low=actions[0], high=actions[-1], shape=(1,), dtype=float),
        #                                      'downstream':gym.spaces.Discrete(n=1)})
        # self.action_space = gym.spaces.MultiDiscrete(nvec=[len(actions), len(actions_downstream)])
        self.action_space = gym.spaces.Discrete(n=max(len(actions), len(actions_downstream)))
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(self.state_dim,), dtype=float)

    def reset(self):
        self.num_steps = 0
        initial_state = [0.4, 10, 5, 0, 0., 0, 5]  # [3.4, 40, 5, 1.5]
        simulator = cho_cell_culture_simulator(initial_state, delta_t=int(360 / number_steps), num_action=1,
                                               noise_level=2500)
        chrom = chromatography()
        self.env = process_env(simulator, chrom,
                               upstream_variable_cost=2,  # sensitive hyperparameters
                               downstream_variable_cost=10,
                               product_price=30,
                               failure_cost=200,
                               product_requirement=product_r,  # sensitive hyperparameters 20 -60
                               purity_requirement=purity_r,  # sensitive hyperparameters 0.85 - 0.93
                               yield_penalty_cost=50,  # sensitive hyperparameters
                               )
        state = self.get_state()
        return state

    def step(self, action):
        # print('action:', action)
        self.num_steps += 1

        # state, reward, done, upstream_done = self.env.step(action[0] if not self.env.upstream_done else action[1])

        if self.env.upstream_done:
            if action >= len(actions_downstream):
                action = len(actions_downstream) - 1

        state, reward, done, upstream_done = self.env.step(action)

        state = self.get_state()
        return state, reward, done, {}

    def get_state(self):
        state = self.env.state
        if isinstance(state, np.ndarray):
            state = state.tolist()

        if not self.env.upstream_done:
            assert len(state) == self.state_dim - 1
            state.append(0)
        else:
            state.extend([0, 0, 0, 0, 0, 1])

        assert len(state) == self.state_dim, (state, len(state), self.state_dim, self.env.t, self.env.upstream_done)
        return state


env = Env()

num_units = 32
# num_units *= 2


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    # def _build_mlp_extractor(self) -> None:

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        actions, values, log_prob = super().forward(obs, deterministic)
        # print(actions.shape, actions.min(dim=1, keepdim=True)[0].shape)
        # actions = actions - actions.min(dim=1, keepdim=True)[0]
        return actions, values, log_prob


if 1:
    policy_kwargs = dict(net_arch=[dict(pi=[num_units, num_units], vf=[num_units, num_units])],
                         activation_fn=nn.Mish, squash_output=True)
    # model = PPO(ActorCriticPolicy, env, policy_kwargs=policy_kwargs, n_steps=256, verbose=1)
    model = PPO(CustomActorCriticPolicy, env, n_steps=2048, gamma=0.90, verbose=1, device='cpu')
else:
    policy_kwargs = dict(net_arch=[num_units, num_units],
                         activation_fn=nn.Mish)
    model = SAC(SACPolicy, env, policy_kwargs=policy_kwargs, verbose=1,
                train_freq=2, batch_size=1024, learning_starts=1024,
                buffer_size=100_000, tau=0.1)

# model.learn(total_timesteps=200_000, )
model.learn(total_timesteps=500_000, )

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
