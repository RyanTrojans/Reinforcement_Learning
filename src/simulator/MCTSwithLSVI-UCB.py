import numpy as np
import pandas as pd
import time
from .environment import process_env
from .simulator import cho_cell_culture_simulator
from .chromatography import chromatography
from BNStructureAnalysis.src.simulator.Util import *
import matplotlib.pyplot as plt
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


def Backup(node,simulation_reward):
    node.V += simulation_reward
    node.N += 1
    if node.parent:
        Backup(node.parent,simulation_reward)


def mapvector_reduce_action(vector_map,action):
    vector_map[actions.index(action)+len(products)-1+len(impuritys)-1+len(cell_densitys)-1] = 0
    return vector_map


def Createmapvector(d,product,impurity,cell_density,upstream_done):
    vector_map = np.zeros(d).reshape(d,1)
    for i in range(1,len(products)):
        if product > products[i]:
            continue
        else:
            vector_map[i-1] = 1
            break
    for j in range(1,len(impuritys)):
        if impurity > impuritys[j]:
            continue
        else:
            vector_map[j-1+len(products)-1] = 1
            break
    if not upstream_done:
        for n in range(len(cell_densitys)):
            if cell_density > cell_densitys[n]:
                continue
            else:
                vector_map[n-1+len(products)-1+len(impuritys)-1] = 1
                break
    #vector_map[actions.index(action)+len(products)-1+len(impuritys)-1+len(cell_densitys)-1] = 1
    return vector_map


def mapvector_action(vector_map,action):
    vector_map[actions.index(action)+len(products)-1+len(impuritys)-1+len(cell_densitys)-1] = 1
    return vector_map


def mapvector_action_down(vector_map,action):
    vector_map[actions_downstream .index(action)+len(products)-1+len(impuritys)-1+len(cell_densitys)-1+len(actions)] = 1
    return vector_map


def mapvector_reduce_action_down(vector_map,action):
    vector_map[actions_downstream .index(action)+len(products)-1+len(impuritys)-1+len(cell_densitys)-1+len(actions)] = 0
    return vector_map


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


def find(root,c,a):
    if root.dicsubnode:
        action_thistime = BestChild(root,c)
        a.append(action_thistime)
        find(root.dicsubnode[action_thistime],c,a)


if __name__ == '__main__':
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
