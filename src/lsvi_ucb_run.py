import numpy as np

from lsvi_ucb import LSVIUCBAgent

from constants import H, actions, actions_downstream
from train import Env


class Featuriser(object):
    def __init__(self, env):
        self.env = env
        # self.shape = (self.env.observation_space.shape[0] + self.env.action_space.n,)
        self.shape = (self.env.observation_space.shape[0] + len(actions) + len(actions_downstream),)

    def map(self, s, a):
        if s == 0:
            return np.zeros(self.shape)

        # print('f:', s, a)
        a1 = [0] * len(actions)
        a2 = [0] * len(actions_downstream)
        if s[-1] == 0:
            a1[a] = 1
        else:
            a2[min(a, len(actions_downstream) - 1)] = 1
        s = [_ / 100 for _ in s]
        return np.asarray(s + a1 + a2)


def run():
    algo = LSVIUCBAgent(Env(),
                        horizon=H,
                        feature_map_fn=Featuriser,
                        gamma=0.99,
                        # gamma=1.0,
                        bonus_scale_factor=1.0,
                        # reg_factor=0,
                        )
    algo.fit(200)
    np.save('w_vec', algo.w_vec)
    np.save('lambda_mat_inv', algo.lambda_mat_inv)
    print(algo.w_vec.shape)


if __name__ == '__main__':
    run()
