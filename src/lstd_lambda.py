import numpy as np

# http://incompleteideas.net/papers/boyanLSTDlambda.pdf
class LSVI(object):
    def __init__(self, env, policy, VFA, featurize, alpha, batchSize = 100,
        lamda = 0, gamma = 1, eps = 1, horizon = 1000, verbosity = 0):
        # Inputs:
        #   -env: openAI gym environment object
        #   -policy: object containing a policy from which to sample actions
        #   -VFA: object containing the value function approximator
        #   -featurize: object which featurizes states
        #   -alpha: step size parameter
        #   -batchSize: number of episodes of experience before policy evaluation
        #   -lamda: trace discount paramater
        #   -gamma: discount-rate parameter
        #   -eps: minimum difference in a weight update for methods that require
        #       convergence
        #   -horizon: finite horizon steps
        #   -verbosity: if TRUE, prints to screen additional information

        self.env = env
        self.policy = policy
        self.featurize = featurize
        self.VFA = VFA
        self.alpha = alpha
        self.batchSize = batchSize
        self.lamda = lamda
        self.gamma = gamma
        self.eps = eps
        self.horizon = horizon
        self.verbosity = verbosity

        self.nS = env.observation_space.shape[0]   # Number of states
        self.nA = env.action_space.n    # Number of actions
        self.policy.setNActions(self.nA)
        self.featurize.set_nSnA(self.nS, self.nA)
        self.featDim = featurize.featureStateAction(0,0).shape # Dimensions of the
                                                               # feature vector
        self.VFA.setUpWeights(self.featDim) # Initialize weights for the VFA
        self.learn = 0 # Initially prevent agent from learning

        self.batch_i = 0 # To keep track of the number of stored experience episodes
        self.sequence =  [] # Array to store episode sequences

    def setUpTrace(self):
        self.E = np.zeros(self.featDim)

    # Computes a single episode.
    # Returns the episode reward return.
    def episode(self):
        episodeReward = 0
        self.setUpTrace()

        # Initialize S, A
        state = self.env.reset()
        action = self.policy.getAction(self.VFA, self.featurize, state)

        # Repeat for each episode
        for t in range(self.horizon):
            # Take action A, observe R, S'
            state, action, reward, done = self.step(state, action)

            # Update the total episode return
            episodeReward += reward

            # Finish the loop if S' is a terminal state
            if done: break

        # Update the policy if the agent is learning and the amount of required
        # experience is met.
        updated = False
        if self.learn:
             self.batch_i += 1
             if (self.batch_i+1) % self.batchSize == 0:
                 updated = self.batchUpdate()

        return episodeReward, updated

    def step(self, state, action):
        # Take A, observe R and S'
        state_prime, reward, done, info = self.env.step(action)

        # Choose A' using a policy derived from S'
        action_prime = self.policy.getAction(self.VFA, self.featurize, state_prime)

        # Store experience
        if self.learn:
            # If traces are being used, update them
            if self.lamda != 0:
                features = self.featurize.featureStateAction(state, action)
                self.E = (self.gamma * self.lamda * self.E) + self.VFA.getGradient(features)

            # Store experience
            self.sequence.append((state, action, reward, state_prime, action_prime, self.E))

        return state_prime, action_prime, reward, done

    def batchUpdate(self):
        dim = self.nS * self.nA
        dim = self.nS + self.nA
        dim = self.nS
        A = np.zeros((dim, dim))
        b = np.zeros((dim, 1))
        for di, dn in enumerate(self.sequence):
            # Get data from array
            state, action, reward, state_prime, action_prime, E = dn
            reward /= 1000

            # Compute the pertinent feature vectors
            features = self.featurize.featureStateAction(state, action)
            features_prime = self.featurize.featureStateAction(state_prime, action_prime)

            A_delta = np.matmul(E, (features - self.gamma * features_prime).T)
            assert A_delta.shape == A.shape
            A += A_delta

            b_delta = reward * E
            b += b_delta

        det_A = np.linalg.det(A)
        # print('det_A:', det_A)
        if det_A != 0:
            self.VFA.updateWeightsMatrix(A, b)
            return True
        return False

class EGreedyPolicyVFA:
    def __init__(self, epsilon, decay = 1):
        self.epsilon = epsilon
        self.decay = decay

    def setNActions(self, nA):
        self.nA = nA

    def getAction(self, VFA, featurize, state):
        # VFA is the value function approximator
        if np.random.random() > self.epsilon:
            # Take a greedy action
            return self.greedyAction(VFA, featurize, state)
        # Take an exploratory action
        else: return self.randomAction()

    # Returns a random action
    def randomAction(self):
        return np.random.randint(self.nA)

    # Returns a greedy action
    def greedyAction(self, VFA, featurize, state):
        maxima_index = [] # Actions with maximum value
        maxVal = None # Value of the current best actions

        for action in range(self.nA):
             # Get the value of the state action pair from VFA
            features = featurize.featureStateAction(state, action)
            value = VFA.getValue(features)

            if maxVal is None: # For the fist (s,a), intialize 'maxVal'
                maxVal = value
            if value > maxVal: # If the action is better than previus ones, update
                maxima_index = [action]
                maxVal = value
            elif value == maxVal: # If the action is equally good, add it
                maxima_index.append(action)

        # Randomly choose one of the best actions
        return np.random.choice(maxima_index)

    # Returns an array containing the action with the highest value for every state
    def getDetArray(self, VFA, featurize, nS):
        detActions = np.zeros((nS, 1))
        actionVals = np.zeros((self.nA, 1)) # Stores the values for all actions
                                            # in a given state
        for state in range(nS):
            for action in range(self.nA):
                features = featurize.featureStateAction(state, action)
                actionVals[action] = VFA.getValue(features)
            detActions[state] = np.argmax(actionVals) # Choose highest value
        return detActions

    def epsilonDecay(self):
        self.epsilon *= self.decay

    # The policy update consists only on epsilon decay
    def episodeUpdate(self):
        self.epsilonDecay()
class LinearVFA:
    # Intialize the weights vector to a fixed  value
    def setUpWeights(self, dimensions, value = 1):
        self.weights = np.ones(dimensions) * value

    def returnWeights(self, dimensions, value = 1):
        return np.ones(dimensions) * value

    def getValue(self, features):
        return np.dot(features.T, self.weights)

    def getGradient(self, features):
        return features

    def updateWeightsDelta(self, delta_weight):
        self.weights += delta_weight

    def updateWeightsMatrix(self, A, b):
        self.weights = np.matmul(np.linalg.inv(A), b)

    def getWeights(self):
        return self.weights

    def setWeights(self, weights):
        self.weights = weights

class Featurize():
    def set_nSnA(self, nS, nA):
        self.nS = nS
        self.nA = nA

    def featureState(self, state):
        if state == 0:
            return np.zeros((self.nS, 1))
        assert isinstance(state, list)
        state = [_ / 100 for _ in state]
        return np.asarray(state).reshape((-1, 1))


    def featureStateAction(self, state, action):
        return self.featureState(state)
        if state == 0:
            return np.zeros((self.nS+self.nA, 1))
        av = [0] * self.nA
        av[action] = 1
        assert isinstance(state, list)
        state = [_ / 100 for _ in state]
        return np.asarray(state + av).reshape((-1, 1))

def plot(y):
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
    ax.plot(t, y, '-', label='Diff')

    plt.xlabel('Time')
    # plt.ylabel('Cumulative Reward')

    plt.legend()
    plt.show()

def run():
    policy = EGreedyPolicyVFA(0.1)
    vfa = LinearVFA()
    featurize = Featurize()

    from train import Env
    env = Env()
    lsvi = LSVI(env, policy, vfa, featurize, alpha=0.001, batchSize=128, lamda=0.2)
    lsvi.learn = 1

    weights = vfa.weights.copy()
    min_diff = float('inf')
    total_diff = [];

    while True:
        r, updated = lsvi.episode()

        if updated:
            diff = np.max(np.abs(weights - vfa.weights))
            print('diff:', diff, r)
            total_diff.append(diff)

            if diff < min_diff:
                np.save('weights', vfa.weights)
                min_diff = diff

            weights = vfa.weights.copy()
            if diff < 0.3:
                break;
    plot(total_diff)


if __name__ == '__main__':
    run()