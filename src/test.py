import copy
import math
import random
import logging

import numpy as np

from BNStructureAnalysis.src.MCTSwithLSVI_UCB_andStructuralProperty import Probability_base_on_Projection_of_Euclidean, \
    BestChild, Createmapvector_MCTS
# from BNStructureAnalysis.src.chromatography import chromatography
from BNStructureAnalysis.src.constants import number_steps, actions_downstream, actions, purity_r, limitation_P, \
    action_dic_downstream, action_dic, c, UPB, H, cf, limitation_I
from BNStructureAnalysis.src.simulator import process_env, chromatography, cho_cell_culture_simulator
# from BNStructureAnalysis.src.environment import process_env
# from BNStructureAnalysis.src.simulator import cho_cell_culture_simulator
from BNStructureAnalysis.src.treeNode import TreeNode

w = np.load('w.npy')
total_node_reward = []

def find_risk_boundary(node):
    from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, minimize

    def f(x):
        thistime_w = np.array(w[node.time - 1], copy=True)
        # print('find_risk_boundary ', len(w[0]))
        thistime_w = thistime_w.reshape(1, len(w[0]))
        # current state features 300 sizes vector
        features = Createmapvector_MCTS(len(w[0]), node.parent.p if node.parent else 0, x[0],
                                        node.parent.x if node.parent else 0, node.action,
                                        node.time)
        # print('featuresfeatures ', features)
        # current state's Value
        # 没有
        y = np.dot(thistime_w.ravel(), features.ravel()).item()
        return -y

    # [-inf -cf]
    nonlinear_constraint = NonlinearConstraint(f, [-np.inf], [-cf])

    bounds = Bounds([0], [limitation_I])
    x0 = np.array([0])
    res = minimize(f, x0, method='trust-constr',
                   bounds=bounds, constraints=[nonlinear_constraint],
                   # options={'verbose': 0},
                   )
    # output an I which will let the V value less or equal to -cf
    return res.x[0]


def MCTS_LSVI_UCB(state, real_time, max_iters=100):
    if len(state) > 3:
        p, i, x = state[4], state[5], state[0]
    else:
        p, i, x = state[0], state[1], 0

    if real_time >= number_steps:
        thisnodeaction_set = copy.deepcopy(actions_downstream)
    else:
        thisnodeaction_set = copy.deepcopy(actions)
    root = TreeNode(p, i, x, thisnodeaction_set, {}, state, copy.deepcopy(real_time), 0)

    cur_r = 0
    for i in range(max_iters):
        node = TreePolicy(root)

        cur_r += node.reward
        # cur_r = node.reward

        n = node
        cur_r = 0
        while True:
            cur_r += n.reward
            n = n.parent
            if not n: break

        # backup with (15)
        planning = H
        # logging.info(
        #     'weight linear {} s'.format(w.shape))
        if node.time < planning and node.parent:
            thistime_w = np.array(w[node.time - 1], copy=True)
            thistime_w = thistime_w.reshape(1, len(w[0]))
            # consider one tree or one starting points to get starting.
            simulation_reward = cur_r + np.dot(thistime_w, Createmapvector_MCTS(len(w[0]), node.parent.p, node.parent.i,
                                                                                node.parent.x, node.action,
                                                                                node.time)).item()
        else:
            simulation_reward = node.p * UPB + cur_r

        # simulation_reward = node.reward
        # simulation_reward = cur_r
        # simulation_reward = RolloutPolicy(node)

        Backup(node, simulation_reward)

    return root, node.reward


def TreePolicy(node: TreeNode):
    t0 = node.time
    while ContinuePlanning(node, t0):
        if node.action_set:
            return Expand(node)
        else:
            node = node.dicsubnode[BestChild(node, c)]
    return node


def Expand(node: TreeNode, action=None):
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
        if action:
            action_value_selected = action
        else:
            random.shuffle(node.action_set)
            action_value_selected = node.action_set.pop()
        #                 print(node.time,node.action_set)
        #                 print('action_value_selected',action_value_selected,node.time)
        print('action_dic_downstream ', action_dic_downstream)
        action_lable = action_dic_downstream[action_value_selected]
        print('action_dic_downstream action_lable ', action_lable)
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
    else:
        if action:
            action_value_selected = action
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
        # print('sdfsdfsdfsdf',newnode.time,newnode.action_set)
        newnode.action = action_value_selected
        node.addsubnode(newnode, action_lable)
        newnode.parent = node

    return newnode


def RolloutPolicy(v: TreeNode):
    r_cum = 0
    v = TreeNode(v.p, v.i, v.x, v.action_set_bak, {}, v.state, v.time, v.reward)
    while ContinuePlanning(v):
        a = random.choice(v.action_set_bak)
        v = Expand(v, a)
        r_cum += v.reward
    return v.reward
    return r_cum


def Backup(v: TreeNode, c):
    assert not isinstance(c, np.ndarray)
    while v:
        v.N += 1
        v.V += c
        v = v.parent


def Probability_base_on_Projection_of_Euclidean(nowlocation, boundary):
    vec1 = np.array(nowlocation)
    vec2 = np.array(boundary)
    Euclidean = np.linalg.norm(vec1 - vec2)
    probability = 1 / (1 + math.exp(-Euclidean))
    probability = 1 - probability
    return probability

# V function maximize of action  w的最大值

def ContinuePlanning(node: TreeNode, t0=None, d_p=3):
    t = node.time

    if t0 is not None:
        if t - t0 >= d_p:
            return False

    if t >= H:
        return False

    eta_t = node.p / (node.p + node.i + 1e-6)
    eta_d = purity_r
    p_d = 50
    p_t = node.p
    # risk_bound_i = 0
    # if node.action:
    #     risk_bound_i = find_risk_boundary(node)
    # bound_i = (1 - purity_r) / purity_r * node.p
    #
    # if node.p <= limitation_P and node.p >= 50 and node.i < bound_i:
    #     return False
    # elif node.p < 50 and node.i < bound_i:
    #     return False
    # elif node.i > bound_i and node.i < risk_bound_i:
    #     return True
    # elif node.i > risk_bound_i:
    #     return random.random() < 0.5

    # if node.i < bound_i:
    # if eta_t >= eta_d:
    #     if p_t >= p_d:  # <= limitation_P:
    #         # R1
    #         return False
    #     else:
    #         return False
    return True
    # else:
        # nowlocation = [node.p, node.i]
        #
        # # bound_i = (1 - purity_r) / purity_r * node.p
        # # boundary = [p_d, bound_i]
        #
        # bound_i = (1 - purity_r) / purity_r * node.p
        # # if node not the root
        # if node.action:
        #     bound_i = find_risk_boundary(node)
        #     # print("bound_i:", bound_i)
        # boundary = [p_d, bound_i]
        #
        # # probability_continue = Probability_base_on_Projection_of_Euclidean(nowlocation, boundary)
        # # return random.random() < probability_continue
        # vec1 = np.array(nowlocation)
        # vec2 = np.array(boundary)
        # Euclidean = np.linalg.norm(vec1 - vec2)
        # prob = 1 / (1 + math.exp(-Euclidean))
        # return random.random() < prob




def BestChild(node, c, get_scores=False):
    # for a, _ in node.dicsubnode.items():
    #     s = _.V.item() / _.N + c * math.sqrt(2 * math.log(node.N) / _.N)
    scores = [(a, _.V / _.N + c * math.sqrt(2 * math.log(node.N) / _.N))
              for a, _ in node.dicsubnode.items()]
    # print('scores:', scores)
    if get_scores:
        return scores
    i = np.argmax([_ for i, _ in scores])
    return scores[i][0]


def main():
    s = [0.4, 10, 5, 0, 0., 0, 5]
    t = 0
    cums = []
    P, I, X = [], [], []

    while True:
        node, reward = MCTS_LSVI_UCB(s, t, 100)
        if ContinuePlanning(node):
            a = BestChild(node, 0)
            print(f'step {t}, action: {a}, scores:', BestChild(node, 0, True))
            node = node.dicsubnode[a]
            print(f'reward: {node.reward}')
            total_node_reward.append(node.reward)
            cums.append((cums[-1] if cums else 0) + node.reward)
            s = copy.deepcopy(node.state)
            t = node.time

            state = s
            if len(s) > 3:
                p, i, x = state[4], state[5], state[0]
            else:
                p, i, x = state[0], state[1], 0

            # if t < number_steps:
            P.append(p)
            I.append(i)
            X.append(x)
        else:
            break
    print(total_node_reward)
    plot(cums, P, I, X)


def plot(y, p, i, x):
    import sys
    import matplotlib.pyplot as plt

    def press_key(event):
        if event.key == 'escape':
            plt.close('all')
            sys.exit(0)

    plt.gcf().canvas.mpl_connect('key_press_event', press_key)

    plt.title('Fig')

    ax = plt.subplot()

    t = range(1, len(y) + 1)
    ax.plot(t, y, '-', label='Cumulative Reward')
    ax.plot(t, p, '-', label='Product')
    ax.plot(t, i, '-', label='Impurity')
    ax.plot(t, x, '-', label='Cell Density')

    plt.xlabel('Time')
    # plt.ylabel('Cumulative Reward')

    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
