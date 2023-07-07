import numpy as np
import pandas as pd
import time
from BNStructureAnalysis.src.environment import process_env
from BNStructureAnalysis.src.simulator import cho_cell_culture_simulator
from BNStructureAnalysis.src.chromatography import chromatography
from BNStructureAnalysis.src.treeNode import TreeNode
import matplotlib.pyplot as plt
import logging
import copy
import math
import random
import os
from BNStructureAnalysis.src.constants import (
    number_steps,
    actions,
    products,
    impuritys,
    cell_densitys,
    actions_downstream,
    vector_cartesian,
    product_r,
    c,
    cf,
    UPB,
    purity_r,
    action_set,
    action_dic,
    action_dic_downstream,
    inverse_action_dic,
    inverse_action_dic_downstream,
    boolvalue,
    limitation_P,
    maxmium_value,
    d,
    H,
    K,
    beta,
    lambuda,
)


# class UCBTreePolicy:
#     def __init__(self, node, c):
#         self.node = node
#         self.actions_list = node.action_set
#         self.c = c
#         self.action_count = np.zeros(len(node.action_set))
#         self.timestep = 0
#         self.simulator = cho_cell_culture_simulator(node.state, delta_t=int(360 / number_steps), num_action=1,
#                                                noise_level=2500)
#         self.chrom = chromatography()
#         self.env = process_env(simulator, chrom,
#                           upstream_variable_cost=2,  # sensitive hyperparameters
#                           downstream_variable_cost=10,
#                           product_price=30,
#                           failure_cost=200,
#                           product_requirement=50,  # sensitive hyperparameters 20 -60
#                           purity_requirement=0.85,  # sensitive hyperparameters 0.85 - 0.93
#                           yield_penalty_cost=50, )  # sensitive hyperparameters
#         self.cumulative_reward = np.zeros(len(node.action_set))
#
#     def select_action(self):
#         ucb_values = np.zeros(len(self.actions_list))
#         for i in range(len(self.actions_list)):
#             if self.action_count[i] == 0:
#                 next_state, reward, done, upstream_done = env.step(int(self.actions_list[i]))
#                 ucb_values[i] = reward
#             else:
#                 reward_mean = self.cumulative_reward[i] / self.action_count[i]
#                 exploration_bonus = self.c * np.sqrt(np.log(self.timestep) / self.action_count[i])
#                 ucb_values[i] = reward_mean + exploration_bonus
#         best_action_index = np.argmax(ucb_values)
#         self.action_count[best_action_index] += 1
#         self.timestep += 1
#         next_state, reward, done, upstream_done = env.step(self.actions_list[best_action_index])
#         newnode_p = next_state[4]
#         newnode_i = next_state[5]
#         newnode_x = next_state[0]
#         if self.node.time + 1 != number_steps:
#             this = copy.deepcopy(actions)
#         else:
#             this = copy.deepcopy(actions_downstream)
#             temp_state = copy.deepcopy(next_state)
#             next_state = [temp_state[4] * temp_state[6], temp_state[5] * temp_state[6]]
#         newnode = TreeNode(newnode_p, newnode_i, newnode_x, this, {}, next_state, self.node.time + 1, reward)
#         # print('sdfsdfsdfsdf',newnode.time,newnode.action_set)
#         newnode.action = self.actions_list[best_action_index]
#         action_lable = action_dic[self.actions_list[best_action_index]]
#         self.node.addsubnode(newnode, action_lable)
#         newnode.parent = self.node
#
#         return newnode, reward, True
#
#     def update(self, action, reward):
#         self.cumulative_reward[action] += reward

def Backup(node,simulation_reward):
    node.V += simulation_reward
    node.N += 1
    if node.parent:
        Backup(node.parent,simulation_reward)


def BestChild(node,c):
    maxm = float('-inf')
    re = None
    if node.dicsubnode:
        for key,subnode in node.dicsubnode.items():
            UCBvalue = subnode.V/subnode.N + c*math.sqrt(2*math.log(node.N)/subnode.N)
            if UCBvalue > maxm:
                maxm = UCBvalue
                re = key
    return re


def find(root,c,a):
    if root.dicsubnode:
        action_thistime = BestChild(root,c)
        a.append(action_thistime)
        find(root.dicsubnode[action_thistime],c,a)


def UCBTreePolicy(root, c):
    node = root
    if len(node.action_set):  # not fully expanded
    # while len(node.action_set) != 0:
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
        # logging.info(
        #     'node.time {:.2f} s and number_steps {:.2f} s'.format(node.time, number_steps))

        if node.time >= number_steps:
            dui = {0: 360, 1: 361, 2: 361}
            env.t = dui[node.time - number_steps]
            random.shuffle(node.action_set)
            action_value_selected = node.action_set.pop()
            #                 print(node.time,node.action_set)
            #                 print('action_value_selected',action_value_selected,node.time)
            action_lable = action_dic_downstream[action_value_selected]
            logging.info(
                'action_value_selected {:.2f} s'.format(int(action_value_selected)))
            next_state, reward, done, upstream_done = env.step(int(action_value_selected))
            newnode_p = next_state[0]
            newnode_i = next_state[1]
            newnode_x = 0
            this = copy.deepcopy(actions_downstream)
            # current_time = node.time + 1
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
            logging.info(
                'node.time {:.2f} s and number_steps {:.2f} s'.format(node.time, number_steps))
            newnode = TreeNode(newnode_p, newnode_i, newnode_x, this, {}, next_state, node.time + 1, reward)
            # print('sdfsdfsdfsdf',newnode.time,newnode.action_set)
            newnode.action = action_value_selected
            node.addsubnode(newnode, action_lable)
            newnode.parent = node

            return newnode, reward, True
    node = node.dicsubnode[BestChild(node, c)]
    if node.time == (number_steps + 3):
        return node, node.reward, True
    else:
        return node, node.reward, False


def Probability_base_on_Projection_of_Euclidean(nowlocation,boundary):
    vec1 = np.array(nowlocation)
    vec2 = np.array(boundary)
    Euclidean = np.linalg.norm(vec1 - vec2)
    projection_value = 0.5#f(Euclidean)I do not know this function
    probability = 1 - projection_value
    return probability


def Createmapvector_MCTS(d,product,impurity,cell_density,action,time):
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
    if time < number_steps:
        for n in range(len(cell_densitys)):
            if cell_density > cell_densitys[n]:
                continue
            else:
                vector_map[n-1+len(products)-1+len(impuritys)-1] = 1
                break
        vector_map[actions.index(action)+len(products)-1+len(impuritys)-1+len(cell_densitys)-1] = 1
    elif time == number_steps:
        vector_map[actions.index(action)+len(products)-1+len(impuritys)-1+len(cell_densitys)-1] = 1
    else:
        vector_map[actions_downstream.index(action)+len(products)-1+len(impuritys)-1+len(cell_densitys)-1+len(actions)] = 1
    return vector_map

def mapvector_reduce_action(vector_map,action):
    vector_map[actions.index(action)+len(products)-1+len(impuritys)-1+len(cell_densitys)-1] = 0
    return vector_map

def mapvector_reduce_action_down(vector_map,action):
    vector_map[actions_downstream .index(action)+len(products)-1+len(impuritys)-1+len(cell_densitys)-1+len(actions)] = 0
    return vector_map

def mapvector_action_down(vector_map,action):
    vector_map[actions_downstream .index(action)+len(products)-1+len(impuritys)-1+len(cell_densitys)-1+len(actions)] = 1
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
    #print(actions.index(action)+len(products)-1+len(impuritys)-1+len(cell_densitys)-1)
    return vector_map


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

# Question: what's the meaning of the parameter x(cell density)
def MCTS(p, i, x, c, action_set, cf, w, state, UPB, real_time):
    a = []
    if real_time >= number_steps:
        thisnodeaction_set = copy.deepcopy(actions_downstream)
    else:
        thisnodeaction_set = copy.deepcopy(actions)
    logging.info(
        'actions {} s'.format(thisnodeaction_set))
    root = TreeNode(p, i, x, thisnodeaction_set, {}, state, copy.deepcopy(real_time), 0)
    for episode in range(50):
        cur_node = root
        cur_r = 0
        ucb_tree_policy = UCBTreePolicy(cur_node, c)
        planning = number_steps + 3
        for j in range(real_time, planning):
            # find the best next step
            # for k in range(300):
            #     node, cum_r_temp, whether_continue = ucb_tree_policy.select_action()
            #     # reward_best = env.step(action_best)
            #     ucb_tree_policy.update(action, cum_r_temp)
            node, cum_r_temp, whether_continue = UCBTreePolicy(cur_node, c)
            cur_r += cum_r_temp
            if whether_continue:
                break
            bound_i = (1 - purity_r) / purity_r * node.p
            boundary = []
            boundary.append(50)
            boundary.append(bound_i)
            if node.p <= limitation_P and node.p >= 50 and node.i < bound_i:
                find(root, c, a)
                if real_time < number_steps:
                    re = inverse_action_dic[a[0]]
                else:
                    re = inverse_action_dic_downstream[a[0]]
                return re
            elif node.p < 50 and node.i < bound_i:
                find(root, c, a)
                if real_time < number_steps:
                    re = inverse_action_dic[a[0]]
                else:
                    re = inverse_action_dic_downstream[a[0]]
                return re
            elif node.i > bound_i:
                nowlocation = [node.p, node.i]
                probability_continue = Probability_base_on_Projection_of_Euclidean(nowlocation, boundary)
                if not np.random.choice(boolvalue, 1, p=[probability_continue, 1 - probability_continue])[0]:
                    find(root, c, a)
                    if real_time < number_steps:
                        re = inverse_action_dic[a[0]]
                    else:
                        re = inverse_action_dic_downstream[a[0]]
                    return re
                else:
                    cur_node = node
        bound_i = (1 - purity_r) / purity_r * node.p
        boundary = []
        boundary.append(50)
        boundary.append(bound_i)
        if node.p <= limitation_P and node.p >= 50 and node.i < bound_i:
            find(root, c, a)
            if real_time < number_steps:
                re = inverse_action_dic[a[0]]
            else:
                re = inverse_action_dic_downstream[a[0]]
            return re
        elif node.p < 50 and node.i < bound_i:
            find(root, c, a)
            if real_time < number_steps:
                re = inverse_action_dic[a[0]]
            else:
                re = inverse_action_dic_downstream[a[0]]
            return re
        elif node.i > bound_i:
            nowlocation = [node.p, node.i]
            probability_continue = Probability_base_on_Projection_of_Euclidean(nowlocation, boundary)
            if not np.random.choice(boolvalue, 1, p=[probability_continue, 1 - probability_continue])[0]:
                find(root, c, a)
                if real_time < number_steps:
                    re = inverse_action_dic[a[0]]
                else:
                    re = inverse_action_dic_downstream[a[0]]
                return re
        if node.time < planning:
            thistime_w = np.array(w[node.time - 1], copy=True)
            thistime_w = thistime_w.reshape(1, len(w[0]))
            # consider one tree or one starting points to get starting.
            simulation_reward = cur_r + np.dot(thistime_w, Createmapvector_MCTS(len(w[0]), node.parent.p, node.parent.i,
                                                                                node.parent.x, node.action, node.time))
        else:
            simulation_reward = node.p * UPB + cur_r
        Backup(node, simulation_reward)

    find(root, c, a)
    if real_time < number_steps:
        re = inverse_action_dic[a[0]]
    else:
        re = inverse_action_dic_downstream[a[0]]
    return re


if __name__ == '__main__':
    all_time = []
    all_episode_time = []
    initial_state = [0.4, 10, 5, 0, 0., 0, 5]
    store_A, w, mean_r, action_k, V, L_time = linearMDP(K, H, initial_state, actions, actions_downstream, lambuda, d,
                                                        vector_cartesian, beta, maxmium_value, product_r, purity_r)
    for i in range(30):
        simulator = cho_cell_culture_simulator(initial_state, delta_t=int(360/number_steps), num_action=1, noise_level=2500)
        chrom = chromatography()
        env = process_env(simulator, chrom,
                          upstream_variable_cost=2,  # sensitive hyperparameters
                          downstream_variable_cost=10,
                          product_price=30,
                          failure_cost=200,
                          product_requirement=50,  # sensitive hyperparameters 20 -60
                          purity_requirement=0.85,  # sensitive hyperparameters 0.85 - 0.93
                          yield_penalty_cost=50,  # sensitive hyperparameters computitaional and algorithm
                          )
        action = 0.05  # L/h action space [0, 0.05]
        done, upstream_done = False, False
        state_buffer = []
        next_state_buffer = []
        action_buffer = []
        reward_buffer = []
        # consacrfice converge the optimal still converge and compatible
        real_time = 0
        simulator_actions_upstream = list(np.around(np.linspace(0.01,0.1,100),decimals=4))
        simulator_actions_downstream = list(np.around(np.linspace(1,4,4),decimals=0))
        cur_state = env.state
        a = time.time()
        logging.info(
            'cur_state[4] {:.2f} s and cur_state[5] {:.2f} s and cur_state[0] {:.2f} s'.format(cur_state[4], cur_state[5], cur_state[0]))
        action = MCTS(cur_state[4],cur_state[5],cur_state[0],c,action_set,cf,w,cur_state,UPB,real_time)
        b = time.time()
        all_time.append(b-a)
    plt.figure()
    x = []
    for j in range(30):
        x.append(j)
    plt.plot(x, all_time)
    plt.xlabel('rounds')
    plt.ylabel('time')
    plt.show()

    # os.chdir(r'/Users/ranyide/Desktop/reinforcementProject/BNStructureAnalysis/src/simulator/computational_time_compare/SPMCTS_33')
    #
    # np.save('L_time_new.npy', L_time)
    # np.save('all_time_new.npy', all_time)
    # # np.save('all_episode_time_18.npy',all_episode_time)
    # np.save('w_new.npy_18', w)