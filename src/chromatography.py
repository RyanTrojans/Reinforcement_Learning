import pandas as pd
import numpy as np
import logging
from BNStructureAnalysis.src.Util import *

logger = logging.getLogger()
logger.setLevel(20)


class chromatography:
    def __init__(self,
                 noise_level=0.05,
                 horizon=3):
        # define parameters in bioreactor model
        chrom_dat = pd.read_csv(r'aa.csv')
        chrom_dat = select_action(chrom_dat)
        # print(chrom_dat)
        self.chrom1_protein = [self.get_alpha_beta(mu=q_p, sigma=noise_level * q_p) for q_p, q_i in chrom_dat]
        self.chrom1_impurity = [self.get_alpha_beta(mu=q_i, sigma=noise_level * q_i) for q_p, q_i in chrom_dat]
        self.chrom2_protein = [self.get_alpha_beta(mu=q_p, sigma=noise_level * q_p) for q_p, q_i in chrom_dat]
        self.chrom2_impurity = [self.get_alpha_beta(mu=q_i, sigma=noise_level * q_i) for q_p, q_i in chrom_dat]
        self.chrom3_protein = [self.get_alpha_beta(mu=q_p, sigma=noise_level * q_p) for q_p, q_i in chrom_dat]
        self.chrom3_impurity = [self.get_alpha_beta(mu=q_i, sigma=noise_level * q_i) for q_p, q_i in chrom_dat]

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
                int(str(1) + str(step) + str(int(np.sum(initial_state)*100))))
        else:
            np.random.seed(rand_seed)

        protein_param = self.true_model_params[step + 1]['protein'][window]
        impurity_param = self.true_model_params[step + 1]['impurity'][window]
        removal_rate_protein = np.random.beta(protein_param[0], protein_param[1])
        removal_rate_impurity = np.random.beta(impurity_param[0], impurity_param[1])

        protein = initial_state[0] * removal_rate_protein
        impurity = initial_state[1] * removal_rate_impurity
        # logging.info('protein removal rate: {:.2f}; impurity removal rate: {:.2f}'.format(removal_rate_protein,
        #                                                                                   removal_rate_impurity))
        return [protein, impurity]