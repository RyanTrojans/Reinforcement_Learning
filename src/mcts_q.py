import copy
import itertools
import math
import random

import numpy as np

from chromatography import chromatography
from constants import number_steps, actions_downstream, actions, purity_r, limitation_P, \
    action_dic_downstream, action_dic, c, UPB, H, cf, limitation_I, cell_densitys, impuritys, products
from environment import process_env
from simulator import cho_cell_culture_simulator
from treeNode import TreeNode
from train import Env, make_env, make_state

q_weights = np.load('w_vec.npy')
lambda_mat_inv = np.load('lambda_mat_inv.npy')
print('q_weights:', q_weights.shape)
# q_weights = q_weights.squeeze(1)

def find_risk_boundary(node):
    print('find_risk_boundary:', node.time)
    def V(state):
        return np.dot(q_weights, state).item()

    next_time = node.time + 1
    upstream_done = next_time >= number_steps
    next_actions = actions if not upstream_done else actions_downstream

    # find the i with supreme of V_t+1 - c(a) <= -cf
    candidates = []
    all_candidates = []
    for i in impuritys:
        for a, _ in enumerate(next_actions):
            env = Env()
            env.reset(state=node.state, time=node.time)
            next_state, *_ = env.step(a)
            next_state = [_ / 100 for _ in next_state]
            v = V(next_state)
            z = v - env.env.action_cost
            if z <= -cf:
                candidates.append((z, i))
            all_candidates.append((z, i))

    print('done')
    if candidates:
        candidates = sorted(candidates)
        # return the largest
        return candidates[-1][1]
    candidates = sorted(all_candidates)
    # return the smallest
    return candidates[0][1]

def Q(t, s, a):
    a1 = [0] * len(actions)
    a2 = [0] * len(actions_downstream)
    if s[-1] == 0:
        a1[a] = 1
    else:
        a2[min(a, len(actions_downstream) - 1)] = 1

    s = [_ / 100 for _ in s]
    feat = s + a1 + a2
    if 1:
        q_w = q_weights[t]
        # bonus_factor = 1.0
        # bonus_factor = 0
        # inverse_counts = feat @ (lambda_mat_inv[t].T @ feat)
        # bonus = bonus_factor * np.sqrt(
        #     inverse_counts
        # )
        bonus = 0
        q = np.dot(feat, q_w) + bonus
        q = np.minimum(q, H)
        return q.item()

    q = np.dot(q_weights[t], feat).item()
    return q


def MCTS(state, real_time, max_iters=100):
    if real_time < number_steps:
        assert len(state) == 7
        p, i, x = state[4], state[5], state[0]
    else:
        # assert len(state) == 2
        p, i, x = state[0], state[1], 0

    if real_time >= number_steps:
        thisnodeaction_set = copy.deepcopy(actions_downstream)
    else:
        thisnodeaction_set = copy.deepcopy(actions)
    root = TreeNode(p, i, x, thisnodeaction_set, {}, state, copy.deepcopy(real_time), 0)

    for i in range(max_iters):
        node, expanded = TreePolicy(root)

        n = node
        cur_r = 0
        while True:
            cur_r += n.reward
            n = n.parent
            if n is None: break

        # cur_r -= node.reward
        # cur_r = 0

        # backup with (15)
        # if node.time < H: # and node.parent is not None and hasattr(node.parent, 'state_'):
        if node.time < H and hasattr(node, 'state_'):
            # action_set = actions if node.time <= number_steps else actions_downstream
            # assert node.action in action_set, (node.action, actions, actions_downstream)
            # simulation_reward = cur_r + Q(node.time, node.parent.state_, action_set.index(node.action))
            # simulation_reward = cur_r + Q(node.time, node.state_, action_set.index(node.action))

            if 1:
                action_set = actions if node.time < number_steps else actions_downstream
                q_vec = []
                for a in action_set:
                    q = Q(node.time, node.state_, action_set.index(a))
                    q_vec.append(q)
                simulation_reward = cur_r + max(q_vec)
                # simulation_reward = max(q_vec)
        else:
            # cur_r = node.reward
            # simulation_reward = node.p * UPB + cur_r
            simulation_reward = node.reward

        # simulation_reward = node.reward
        # simulation_reward = cur_r
        # simulation_reward = RolloutPolicy(node)

        Backup(node, simulation_reward)

    return root, node.reward


def TreePolicy(node: TreeNode):
    t0 = node.time
    while ContinuePlanning(node, t0):
        if node.action_set:
            return Expand(node), True
        else:
            # node = node.dicsubnode[BestChild(node, c, get_key=True)]
            node = BestChild(node, c)
    return node, False


def Expand(node: TreeNode, action=None):
    env = make_env(node.state)

    if node.time >= number_steps:
        dui = {0: 360, 1: 361, 2: 362}
        env.t = dui[node.time - number_steps]
        env.upstream_done = True
        if action is not None:
            action_value_selected = action
        else:
            random.shuffle(node.action_set)
            action_value_selected = node.action_set.pop()
        # action_value_selected = 2
        #                 print(node.time,node.action_set)
        #                 print('action_value_selected',action_value_selected,node.time)
        action_lable = action_dic_downstream[action_value_selected]
        next_state, reward, done, upstream_done = env.step(int(action_value_selected))
        newnode_p = next_state[0]
        newnode_i = next_state[1]
        newnode_x = 0
        this = copy.deepcopy(actions_downstream)
        # current_time = node.time + 1
        newnode = TreeNode(newnode_p, newnode_i, newnode_x, this, {}, next_state, node.time + 1, reward)
        newnode.state_ = make_state(env, norm=False)
        # if newnode.time < H:
        #     if not ContinuePlanning(newnode):
        #         newnode.reward = -1000
        newnode.action = action_value_selected
        node.addsubnode(newnode, action_lable)
        newnode.parent = node
    else:
        env.t = node.time * 24
        if action is not None:
            action_value_selected = action
        else:
            random.shuffle(node.action_set)
            action_value_selected = node.action_set.pop()
        # action_value_selected = 0.04
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
        newnode.state_ = make_state(env, norm=False)
        # if not ContinuePlanning(newnode):
        #     newnode.reward = -1000
        # print('sdfsdfsdfsdf',newnode.time,newnode.action_set)
        newnode.action = action_value_selected
        node.addsubnode(newnode, action_lable)
        newnode.parent = node

    if action is not None:
        return newnode, env
    return newnode


def RolloutPolicy(v: TreeNode):
    r_cum = 0
    v = TreeNode(v.p, v.i, v.x, v.action_set_bak, {}, v.state, v.time, v.reward)
    while ContinuePlanning(v):
        a = random.choice(v.action_set_bak)
        v = Expand(v, a)[0]
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

def ContinuePlanning(node: TreeNode, t0=None, d_p=4):
    t = node.time

    if t0 is not None:
        if t - t0 >= d_p:
            return False

    if t >= H:
        return False

    if t == H-1:
        return True

    # return True

    # bound_i = (1 - purity_r) / purity_r * node.p
    # risk_bound_i = 0
    # risk_bound_i = 100000
    # # risk_bound_i = (1 - purity_r) / purity_r * node.p
    # # if node.action:
    # #     risk_bound_i = find_risk_boundary(node)
    # # risk_bound = 0
    # # if node.action:
    # #     risk_bound = find_risk_boundary(node)
    # if node.p <= limitation_P and node.p >= 50 and node.i < bound_i:
    #     return False
    # elif node.p < 50 and node.i < bound_i:
    #     return False
    # elif risk_bound_i > node.i > bound_i:
    #     return True
    # elif risk_bound_i < node.i:
    #     return False
    return True

def ContinuePlanning_(node: TreeNode, t0=None, d_p=H*2):
    t = node.time

    if t0 is not None:
        if t - t0 >= d_p:
            return False

    if t >= H:
        return False

    # return True

    eta_t = node.p / (node.p + node.i + 1e-6)
    eta_d = purity_r
    p_d = 50
    p_t = node.p

    # if node.i < bound_i:
    if eta_t >= eta_d:
        if p_t >= p_d:  # <= limitation_P:
            # R1
            # return True
            return False
        else:
            # return True
            return False
    else:
        bound_i = (1 - purity_r) / purity_r * node.p
        if node.action:
            bound_i = find_risk_boundary(node)
            # print("bound_i:", bound_i)

        return node.i >= bound_i

        boundary = [p_d, bound_i]
        nowlocation = [node.p, node.i]

        # probability_continue = Probability_base_on_Projection_of_Euclidean(nowlocation, boundary)
        # return random.random() < probability_continue
        # dist = math.dist(nowlocation, boundary)
        vec1 = np.array(nowlocation)
        vec2 = np.array(boundary)
        dist = np.linalg.norm(vec1 - vec2)
        prob = 1 / (1 + math.exp(-dist))
        return random.random() < prob


def BestChild(node, c, get_scores=False):
    # for a, _ in node.dicsubnode.items():
    #     s = _.V.item() / _.N + c * math.sqrt(2 * math.log(node.N) / _.N)
    scores = [(key, _, _.V / _.N + c * math.sqrt(2 * math.log(node.N) / _.N))
              for key, _ in node.dicsubnode.items()]
    # print('scores:', scores)
    if get_scores:
        return [(k, s) for k, _, s in scores]
    i = np.argmax([s for k, _, s in scores])
    return scores[i][1]

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
    action = round(random.uniform(0.001, 0.05), 4)  # L/h action space [0, 0.05]
    done, upstream_done = False, False
    reward_buffer = []
    t = 0
    while not done:
        next_state, reward, done, upstream_done = env.step(action)
        print(f'step {t}/{env.t}, action: {action}, state:', env.state)
        if t == 13:
            print()
        if upstream_done:
            action = random.randint(1, 5)  # [6]
        reward_buffer.append(reward)
        t += 1
    print(np.sum(reward_buffer), reward_buffer)
    return np.sum(reward_buffer)


def main():
    random.seed(0)
    np.random.seed(0)
    test_30reward = []
    mcts_30reward = []
    for mcts_try in range(1):
        test_reward = test()
        test_30reward.append(test_reward)
        # return

        print()
        print('starting MCTS')
        env = None
        s = [0.4, 10, 5, 0, 0., 0, 5]
        t = 0
        cums = []
        P, I, X = [], [], []
        while True:
            node, reward = MCTS(s, t, 500)
            # node, reward = MCTS(s, t, 500)
            if ContinuePlanning(node):
                a = BestChild(node, 0).action
                # if t < number_steps:
                #     a = 0.04
                # else:
                #     a = 2

                if env:
                    assert np.all(node.state == env.state), (node.state, env.state)
                    node.state = env.state

                # print(f'step {t}, action: {a}, scores:', BestChild(node, 0, True))

                # node = node.dicsubnode[a]
                assert a is not None
                node, env = Expand(node, a)
                assert np.all(node.state == env.state), (node.state, env.state)
                print(f'step {t}/{env.t}, action: {a}, state:', env.state.tolist() if isinstance(env.state, np.ndarray) else env.state)
                print(f'  -> reward: {node.reward}')
                if t == 13:
                    print()

                cums.append((cums[-1] if cums else 0) + node.reward)
                s = copy.deepcopy(node.state)
                t = node.time

                state = s
                if t < number_steps:
                    assert len(state) == 7
                    p, i, x = state[4], state[5], state[0]
                else:
                    # assert len(state) == 2
                    p, i, x = state[0], state[1], 0
                    # p, i, x = state[0], state[1], state[0]

                # if t < number_steps:
                P.append(p)
                I.append(i)
                X.append(x)
            else:
                break
        mcts_30reward.append(cums[-1])
    plot_score(mcts_30reward, test_30reward)
        # print(cums[-1], cums)
        # print(P)
        # print(I)
        # print(X)
        # plot(cums, P, I, X)


def plot_score(y, p):
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
    ax.plot(t, y, '-', label='MCTS')
    ax.plot(t, p, '-', label='Test')

    plt.xlabel('Time')
    # plt.ylabel('Cumulative Reward')

    plt.legend()
    plt.show()


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
