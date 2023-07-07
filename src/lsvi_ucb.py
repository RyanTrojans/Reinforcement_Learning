import numpy as np

from gym.spaces import Discrete

from rlberry.agents import AgentWithSimplePolicy
from rlberry.utils.jit_setup import numba_jit

import rlberry

from constants import number_steps, actions_downstream

logger = rlberry.logger


@numba_jit
def run_lsvi_jit(
    dim,
    horizon,
    bonus_factor,
    lambda_mat_inv,
    reward_hist,
    gamma,
    feat_hist,
    n_actions,
    feat_ns_all_actions,
    total_time_steps,
):
    q_w = np.zeros((horizon + 1, dim))
    for hh in range(horizon - 1, -1, -1):
        T = total_time_steps
        b = np.zeros(dim)
        # for tt in range(T):
        for ep in range(total_time_steps // horizon):
            tt = ep * horizon + hh
            # compute q function at next state, q_ns
            q_ns = np.zeros(n_actions)
            for aa in range(n_actions):
                #
                feat_ns_aa = feat_ns_all_actions[tt, aa, :]
                inverse_counts = feat_ns_aa.dot(lambda_mat_inv[hh].T.dot(feat_ns_aa))
                bonus = bonus_factor * np.sqrt(
                    inverse_counts
                )
                #
                q_ns[aa] = feat_ns_aa.dot(q_w[hh + 1, :]) + bonus
                q_ns[aa] = min(q_ns[aa], horizon)

            # compute regretion targets
            target = reward_hist[tt] + gamma * q_ns.max()
            feat = feat_hist[tt, :]
            b = b + target * feat

        # solve M x = b, where x = q_w, and M = self.lambda_mat
        q_w[hh, :] = lambda_mat_inv[hh].T @ b
    return q_w


class LSVIUCBAgent(object):

    def __init__(
        self,
        env,
        horizon,
        feature_map_fn,
        feature_map_kwargs=None,
        gamma=0.99,
        bonus_scale_factor=1.0,
        reg_factor=0.1,
    ):
        self.env = env
        self.n_episodes = None
        self.horizon = horizon
        self.gamma = gamma
        self.bonus_scale_factor = bonus_scale_factor
        self.reg_factor = reg_factor
        feature_map_kwargs = feature_map_kwargs or {}
        self.feature_map = feature_map_fn(self.env, **feature_map_kwargs)

        #
        if self.bonus_scale_factor == 0.0:
            self.name = "LSVI-Random-Expl"

        #
        assert isinstance(
            self.env.action_space, Discrete
        ), "LSVI-UCB requires discrete actions."

        #
        assert len(self.feature_map.shape) == 1
        self.dim = self.feature_map.shape[0]

        # attributes initialized in reset()
        self.episode = None
        self.lambda_mat = None  # lambda matrix
        self.lambda_mat_inv = None  # inverse of lambda matrix
        self.w_vec = None  # vector representation of Q
        self.w_policy = None  # representation of Q for final policy
        self.reward_hist = None  # reward history
        self.state_hist = None  # state history
        self.action_hist = None  # action history
        self.nstate_hist = None  # next state history

        self.feat_hist = None  # feature history
        self.feat_ns_all_actions = None  # next state features for all actions
        #

        # aux variables (init in reset() too)
        self._rewards = None

    def reset(self):
        self.episode = 0
        self.total_time_steps = 0
        self.lambda_mat = [self.reg_factor * np.eye(self.dim) for _ in range(self.horizon)]
        self.lambda_mat_inv = [(1.0 / self.reg_factor) * np.eye(self.dim) for _ in range(self.horizon)]
        self.w_vec = np.zeros((self.horizon + 1, self.dim))
        self.reward_hist = np.zeros(self.n_episodes * self.horizon)
        self.state_hist = []
        self.action_hist = []
        self.nstate_hist = []
        # episode rewards
        self._rewards = np.zeros(self.n_episodes)
        #
        self.feat_hist = np.zeros((self.n_episodes * self.horizon, self.dim))
        self.feat_ns_all_actions = np.zeros(
            (self.n_episodes * self.horizon, self.env.action_space.n, self.dim)
        )
        #
        self.w_policy = None

    def fit(self, budget, **kwargs):
        """
        Train the agent using the provided environment.

        Parameters
        ----------
        budget: int
            number of episodes. Each episode runs for self.horizon unless it
            enconters a terminal state in which case it stops early.
            Warning: Calling fit() more than once will reset the algorithm
            (to realocate memory according to the number of episodes)
        """
        del kwargs

        # Allocate memory according to budget.
        # TODO: avoid the need to reset() the algorithm if fit() is called again.
        if self.n_episodes is not None:
            logger.warning(
                "[LSVI-UCB]: Calling fit() more than once will reset the algorithm"
                + " (to realocate memory according to the number of episodes)."
            )
        self.n_episodes = budget
        self.reset()

        all_episodes_reward = []
        for ep in range(self.n_episodes):
            episode_reward = self.run_episode()
            all_episodes_reward.append(episode_reward)
            if self.bonus_scale_factor > 0.0 or ep == self.n_episodes - 1:
                # update Q function representation
                self.w_vec = self._run_lsvi(self.bonus_scale_factor)
        self.plot(all_episodes_reward)
        # self.w_policy = self._run_lsvi(bonus_factor=0.0)[0, :]

    def plot(self, y):
        import sys
        import matplotlib.pyplot as plt

        def press_key(event):
            if event.key == 'escape':
                plt.close('all')
                sys.exit(0)

        plt.gcf().canvas.mpl_connect('key_press_event', press_key)

        plt.title('Fig')

        ax = plt.subplot()

        t = range(0, len(y))
        ax.plot(t, y, '-', label='Reward')

        plt.xlabel('Time')
        # plt.ylabel('Cumulative Reward')

        plt.legend()
        plt.show()

    def policy(self, observation):
        q_w = self.w_policy
        assert q_w is not None
        #
        q_vec = self._compute_q_vec(q_w, observation, 0.0)
        return q_vec.argmax()

    def _optimistic_policy(self, observation, hh):
        q_w = self.w_vec[hh, :]
        q_vec = self._compute_q_vec(q_w, observation, self.bonus_scale_factor, hh)
        if hh >= number_steps:
            return q_vec[:len(actions_downstream)].argmax()
        return q_vec.argmax()

    def run_episode(self):
        state = self.env.reset()
        episode_rewards = 0
        for hh in range(self.horizon):
            if self.bonus_scale_factor == 0.0:
                action = self.env.action_space.sample()
            else:
                action = self._optimistic_policy(state, hh)

            next_state, reward, is_terminal, _ = self.env.step(action)

            feat = self.feature_map.map(state, action)
            outer_prod = np.outer(feat, feat)
            inv = self.lambda_mat_inv[hh]

            #
            self.lambda_mat[hh] += np.outer(feat, feat)
            # update inverse
            self.lambda_mat_inv[hh] -= (inv @ outer_prod @ inv) / (1 + feat @ inv.T @ feat)
            # self.lambda_mat_inv[hh] = np.linalg.inv(self.lambda_mat[hh])

            # update history
            self.reward_hist[self.total_time_steps] = reward
            self.state_hist.append(state)
            self.action_hist.append(action)
            self.nstate_hist.append(next_state)

            #
            tt = self.total_time_steps
            self.feat_hist[tt, :] = self.feature_map.map(state, action)
            for aa in range(self.env.action_space.n):
                self.feat_ns_all_actions[tt, aa, :] = self.feature_map.map(
                    next_state, aa
                )

            # increments
            self.total_time_steps += 1
            episode_rewards += reward

            #
            state = next_state
            if is_terminal:
                break

        # store data
        self._rewards[self.episode] = episode_rewards

        # update ep
        self.episode += 1

        print("episode_rewards:", episode_rewards, self.episode, hh)

        return episode_rewards

    def _compute_q(self, q_w, state, action, bonus_factor):
        """q_w is the vector representation of the Q function."""
        feat = self.feature_map.map(state, action)
        inverse_counts = feat @ (self.lambda_mat_inv.T @ feat)
        bonus = bonus_factor * np.sqrt(inverse_counts)
        q = feat.dot(q_w) + bonus
        return q

    def _compute_q_vec(self, q_w, state, bonus_factor, hh):
        A = self.env.action_space.n
        q_vec = np.zeros(A)
        for aa in range(A):
            # q_vec[aa] = self._compute_q(q_w, state, aa, bonus_factor)
            feat = self.feature_map.map(state, aa)
            inverse_counts = feat @ (self.lambda_mat_inv[hh].T @ feat)
            bonus = bonus_factor * np.sqrt(
                inverse_counts
            )
            q_vec[aa] = feat.dot(q_w) + bonus
            q_vec[aa] = min(q_vec[aa], self.horizon)
        return q_vec

    def _run_lsvi(self, bonus_factor):
        # run value iteration
        q_w = run_lsvi_jit(
            self.dim,
            self.horizon,
            bonus_factor,
            self.lambda_mat_inv,
            self.reward_hist,
            self.gamma,
            self.feat_hist,
            self.env.action_space.n,
            self.feat_ns_all_actions,
            self.total_time_steps,
        )
        return q_w
