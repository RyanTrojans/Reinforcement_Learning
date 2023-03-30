import numpy as np
import pandas as pd
import time
from BNStructureAnalysis.src.simulator.Util import *
import logging
import copy
import math
import random
from .constants import (
    number_steps,
    actions,
    products,
    impuritys,
    cell_densitys,
    actions_downstream,
    vector_cartesian,
    product_r,
)


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


class TreeNode:

    def __init__(self,p:float,i:float,x:float,action_set:list,dicsubnode:dict,state:list,time:int,reward):
        self.V = 0
        self.N = 1
        self.action = None
        self.p = p
        self.i = i
        self.x = x
        self.state = state
        self.time = time
        self.parent = None
        self.reward = reward
        self.dicsubnode = dicsubnode
        self.action_set = action_set

    def addsubnode(self,node,key):
        self.dicsubnode[key] = node


class chromatography:
    def __init__(self,
                 noise_level=0.05,
                 horizon=3):
        # define parameters in bioreactor model
        chrom_dat = pd.read_csv('ChromatographyData', sep='\t')
        chrom_dat = select_action(chrom_dat)
        print(chrom_dat)
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


def UcbTreePolicy(root, c):
    node = root
    if len(node.action_set):  # not fully expanded
        simulator = cho_cell_culture_simulator(node.state, delta_t=int(360 / number_steps), num_action=1,
                                               noise_level=2500)
        chrom = chromatography()
        env = process_env(simulator, chrom,
                          upstream_variable_cost=2,  # sensitive hyperparameters
                          downstream_variable_cost=10,
                          product_price=30,
                          failure_cost=200,
                          product_requirement=50,  # sensitive hyperparameters 20 -60
                          purity_requirement=0.85,  # sensitive hyperparameters 0.85 - 0.93
                          yield_penalty_cost=50, )  # sensitive hyperparameters

        if node.time >= number_steps:
            dui = {0: 360, 1: 361, 2: 361}
            env.t = dui[node.time]
            random.shuffle(node.action_set)
            action_value_selected = node.action_set.pop()
            #                 print(node.time,node.action_set)
            #                 print('action_value_selected',action_value_selected,node.time)
            action_lable = action_dic_downstream[action_value_selected]
            next_state, reward, done, upstream_done = env.step(int(action_value_selected))
            newnode_p = next_state[0]
            newnode_i = next_state[1]
            newnode_x = 0
            this = copy.deepcopy(actions_downstream)
            newnode = TreeNode(newnode_p, newnode_i, newnode_x, this, {}, next_state, node.time + 1, reward)
            newnode.action = action_value_selected
            node.addsubnode(newnode, action_lable)
            newnode.parent = node
            return newnode, reward, True
        else:
            random.shuffle(node.action_set)
            action_value_selected = node.action_set.pop()
            action_lable = action_dic[action_value_selected]
            # print(action_value_selected)
            next_state, reward, done, upstream_done = env.step(action_value_selected)
            newnode_p = next_state[4]
            newnode_i = next_state[5]
            newnode_x = next_state[0]
            if node.time + 1 != number_steps:
                this = copy.deepcopy(actions)
            else:
                this = copy.deepcopy(actions_downstream)
                temp_state = copy.deepcopy(next_state)
                next_state = [temp_state[4] * temp_state[6], temp_state[5] * temp_state[6]]
            newnode = TreeNode(newnode_p, newnode_i, newnode_x, this, {}, next_state, node.time + 1, reward)
            newnode.action = action_value_selected
            node.addsubnode(newnode, action_lable)
            newnode.parent = node

            return newnode, reward, True
    else:
        node = node.dicsubnode[BestChild(node, c)]
        if node.time == (number_steps + 3):
            return node, node.reward, True
        else:
            return node, node.reward, False


def BestChild(node, c):
    maxm = float('-inf')
    re = None
    if node.dicsubnode:
        for key,subnode in node.dicsubnode.items():
            UCBvalue = subnode.V/subnode.N + c*math.sqrt(2*math.log(node.N)/subnode.N)
            if UCBvalue > maxm:
                maxm = UCBvalue
                re = key
    return re


def linearMDP(K, H, initial_state, actions, actions_downstream, lambuda, d, vector_cartesian, beta, maxmium_value,
              product_r, purity_r):
    ttt = time.time()
    store_A = [np.zeros((d, d))] * H  # store sum of product of vector(the mapping of state and best_action)
    store_w = [np.zeros(d)] * H  # store weight
    store_inverse_A = [np.zeros(
        (d, d))] * H  # store inverse matrix about the sum of product of vector(the mapping of state and action)
    store_state_action_map = []  # store vector(the mapping of state and best_action), k episodes and each episode has H vectors
    V = []  # store a value which risk-free state value that has a fixed action
    r = []  # store reward in each time step of each episode
    mean_r = []
    for episod in range(K):
        for inv_t in range(H - 1, -1, -1):
            if episod == 0:  # to first episode, we dont have history, so just calculate lambda
                store_A[inv_t] = lambuda * np.diag(np.full(d, 1))

                store_inverse_A[inv_t] = np.linalg.inv(store_A[
                                                           inv_t])  # we shouold store this inverse matrix of lambda, because we use Sherman-Morrison formula
            else:
                temp_A_old_1 = np.array(store_inverse_A[inv_t], copy=True)
                temp_A_old_2 = np.array(store_inverse_A[inv_t], copy=True)
                zheng_A = np.array(store_A[inv_t], copy=True)
                T = store_vector_map_k[inv_t].reshape(1, d)

                inverse_A = temp_A_old_1 - (temp_A_old_2 @ store_vector_map_k[inv_t] @ T @ temp_A_old_2) / (
                            1 + T @ temp_A_old_2 @ store_vector_map_k[inv_t])
                store_A[inv_t] = zheng_A + store_vector_map_k[inv_t] @ T
                # print(store_A[inv_t])

                temp_w = np.zeros(d).reshape(d, 1)  # for store w in this time step of this episode
                for j in range(episod):
                    cw = np.array(temp_w, copy=True)
                    temp_w = cw + (inverse_A @ store_state_action_map[j * H + inv_t] * (
                                r[j * H + inv_t] + V[j * (H + 1) + inv_t + 1]))
                store_w[inv_t] = temp_w.reshape(d,
                                                1)  # store w in this time step of this episode into Corresponding time step's position
                store_inverse_A[
                    inv_t] = inverse_A  # store inverse matrix of lambda in this time step of this episode into Corresponding time step's position

        action_k = []  # store action value in each time step  of this episode
        # store initial state about product impurity cell density
        vector_map = Createmapvector(d, initial_state[4], initial_state[5], initial_state[0], True)
        store_vector_map_k = []
        r_k = []
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
        done, upstream_done = False, False

        state = initial_state
        for t in range(H):
            Q_t = float('-inf')  # set inital V value
            if not upstream_done:
                vector_map = Createmapvector(d, state[4], state[5], state[0], upstream_done)
                best_action = None
                for action in actions:
                    vector_map = mapvector_action(vector_map, action)
                    TT = store_w[t].reshape(1, d)
                    map_TT = vector_map.reshape(1, d)
                    Q_action = TT @ vector_map + beta * np.sqrt(map_TT @ store_inverse_A[t] @ vector_map)
                    if Q_action > Q_t:
                        Q_t = Q_action
                        best_action = action
                    vector_map = mapvector_reduce_action(vector_map, action)

                action_k.append(best_action)  # store best_action in this time step of this episode
                Q_t_min = np.minimum(maxmium_value, Q_t)  # use limitation
                V.append(Q_t_min)  # store this best state value whose action is fixed
                # simulator
                vector_map = mapvector_action(vector_map, best_action)
                store_state_action_map.append(np.array(vector_map, copy=True))
                store_vector_map_k.append(np.array(vector_map, copy=True))
                next_state, reward, done, upstream_done = env.step(best_action)
                r.append(reward)
                r_k.append(reward)
                state = next_state
            else:
                vector_map = Createmapvector(d, state[0], state[1], state[0], upstream_done)
                best_action = None
                for action in actions_downstream:
                    vector_map = mapvector_action_down(vector_map, action)
                    # print(np.sum(vector_map))
                    TT = store_w[t].reshape(1, d)
                    map_TT = vector_map.reshape(1, d)
                    # print(TT@vector_map)
                    # print(beta*np.sqrt(map_TT@store_inverse_A[t]@vector_map))
                    Q_action = TT @ vector_map + beta * np.sqrt(map_TT @ store_inverse_A[t] @ vector_map)
                    # print(Q_action,action)
                    if Q_action > Q_t:
                        Q_t = Q_action
                        best_action = action
                # print(best_action)
                vector_map = mapvector_reduce_action_down(vector_map, action)
                action_k.append(best_action)  # store best_action in this time step of this episode
                Q_t_min = np.minimum(maxmium_value, Q_t)  # use limitation
                V.append(Q_t_min)  # store this best state value whose action is fixed
                # simulator
                vector_map = mapvector_action_down(vector_map, best_action)
                store_state_action_map.append(np.array(vector_map, copy=True))
                store_vector_map_k.append(np.array(vector_map, copy=True))
                print(best_action)
                next_state, reward, done, upstream_done = env.step(int(best_action))
                r.append(reward)
                r_k.append(reward)
                state = next_state
        # print(action_k)
        eee = time.time()
        L_time = eee - ttt
        mean_r.append(np.sum(r_k))
        V.append(0)
    return store_A, store_w, mean_r, action_k, V, L_time


def MCTS(protein, impurity, x, c, action_set, cf, w, state, UPB, real_time):
    start1 = time.time()
    a = []
    if real_time >= number_steps:
        thisnodeaction_set = copy.deepcopy(actions_downstream)
    else:
        thisnodeaction_set = copy.deepcopy(actions)
    root = TreeNode(protein, impurity, x, thisnodeaction_set, {}, state, copy.deepcopy(real_time), 0)
    sig_time = []
    for episode in range(50):
        start = time.time()
        cur_node = root
        cur_r = 0
        planning = number_steps + 3
        for j in range(real_time, planning):
            node, cum_r_temp, whether_continue = UcbTreePolicy(cur_node, c)
            cur_r += cum_r_temp
            if whether_continue:
                break
            else:
                cur_node = node

        if node.time < planning:
            thistime_w = np.array(w[node.time - 1], copy=True)
            thistime_w = thistime_w.reshape(1, len(w[0]))
            simulation_reward = cur_r + np.dot(thistime_w, Createmapvector_MCTS(len(w[0]), node.parent.p, node.parent.i,
                                                                                node.parent.x, node.action, node.time))
        else:
            simulation_reward = node.p * UPB + cur_r
        Backup(node, simulation_reward)
        end = time.time()
        sig_time.append(end - start)

    find(root, c, a)
    if real_time < number_steps:
        re = inverse_action_dic[a[0]]
    else:
        re = inverse_action_dic_downstream[a[0]]
    end1 = time.time()
    all_time.append(end1 - start1)
    return re, sig_time


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    for i in range(30):
        initial_state = [0.4, 10, 5, 0, 0., 0, 5]  # [3.4, 40, 5, 1.5]
        simulator = cho_cell_culture_simulator(initial_state, delta_t=int(360 / number_steps), num_action=1,
                                               noise_level=2500)
        chrom = chromatography()
        env = process_env(simulator, chrom,
                          upstream_variable_cost=2,  # sensitive hyperparameters
                          downstream_variable_cost=10,
                          product_price=30,
                          failure_cost=200,
                          product_requirement=50,  # sensitive hyperparameters 20 -60
                          purity_requirement=0.85,  # sensitive hyperparameters 0.85 - 0.93
                          yield_penalty_cost=50,  # sensitive hyperparameters
                          )
        action = 0.05  # L/h action space [0, 0.05]
        done, upstream_done = False, False
        state_buffer = []
        next_state_buffer = []
        action_buffer = []
        reward_buffer = []
        real_time = 0
        simulator_actions_upstream = list(np.around(np.linspace(0.01, 0.1, 100), decimals=4))
        simulator_actions_downstream = list(np.around(np.linspace(1, 4, 4), decimals=0))
        cur_state = env.state
        action, sig_time = MCTS(cur_state[4], cur_state[5], cur_state[0], c, action_set, cf, w, cur_state, UPB,
                                real_time)
        all_episode_time.append(sig_time)
    # print(np.sum(reward_buffer))
    #
    # print('upstream: ', next_state_buffer[-4], 'downstream: ', np.array(next_state_buffer[-4:]))
    # print('purity: ', next_state_buffer[-1][-2] / np.sum(next_state_buffer[-1]))
    # plt.plot(next_state_buffer[:360], label=simulator.label)
    # plt.legend()
    # plt.show()
