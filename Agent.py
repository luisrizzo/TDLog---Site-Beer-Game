import numpy as np
import pdb
import time
from tools import *

'''
import tensorflow as tf

from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor
'''

from pathlib import Path

import LeadTime as ld
import pdb

# HYPER PARAMETERS
# EPSILON (for exlore/exploit dillema)
# LAMBDA (from TD(lambda) algorithms)
GAMMA = 0.9
ALPHA = 0.1

#dict_z is a table with the relation between Service Level and z
#it is based on Normal Gaussian courves and is really common in Supply Chain world
#The idea is to calculate how much of security should we add according to the integral from - infinity to Z to get the service level
dict_z = {  '0.50': 0,
            '0.60': 0.254,
            '0.70' : 0.525,
            '0.8' : 0.842,
            '0.85' : 1.037,
            '0.9' : 1.282,
            '0.95': 1.645,
            '0.96' : 1.751,
            '0.97': 1.880,
            '0.98': 2.055,
            '0.99': 2.325,
            '0.999':3.100}   

class Agent:
    """
        Mother class Agent 
    """
    def __init__(self, index = 0,  lead_time = None, label="agent"):
        self.i = index
        self.label = label
        self.is_AI_agent = False
        if lead_time:
            self.lead_time = lead_time

        else:
            self.lead_time = ld.Uniform_LeadTime(2,2,1)

        
    def act(self, state = None, train = True):
        pass

    def reinforce(self, s, n_s, a, n_a, r):
        pass

    def state_dict2str(self, s_dict):
        """ 
            Converts the state dictionnary into a string 
        """
        assert(isinstance(s_dict, dict))
        
        state = ""
        state += str(int(s_dict['IL'])) + "_"
        state += str(int(s_dict['BO'])) + "_"
        state += str(int(s_dict['d'])) + "_"
        state += str(int(s_dict['RS'])) + "_"
        state += str(int(s_dict['OO'][0]))

        return state

    def state_str2dict(self, s):
        """ 
            Converts the state string into a dictionnary 
        """
        assert(isinstance(s, str))
        keys = ["IL","BO", "d", "RS", "OO"]
        vals = s.split("_")
        state = {}
        for i, key in enumerate(keys):
            state[key] = int(vals[i])

        state['OO'] = [state['OO']]
        return state
    
    def state_dict2arr(self, s_dict):
        """
            Converts the state dict to an array
        """
        assert(isinstance(s_dict, dict))
        state = []
        for key, value in s_dict.items():
            state = np.append(state, value)
        return state

    def set_lead_time(self, lead_time):
        self.lead_time = lead_time

class RND_Agent(Agent):
    """ Agent using random policy for replenishement"""
    def __init__(self, index, lead_time = ld.Uniform_LeadTime(2,2,1), RND_possible_actions = [-5, 0, 5], label = "RND"):
        Agent.__init__(self, index = index, lead_time = lead_time, label = label)
        self.RND_possible_actions = RND_possible_actions
        
    def act(self, state = None, train = None):
        return self.RND_possible_actions[np.random.randint(0,len(self.RND_possible_actions))]

    def copy(self):
        return RND_Agent(self.i, label = self.label, lead_time = self.lead_time, BS_level = self.BS_level)

class BS_Agent(Agent):
    """ 
        Agent using base stock policy 
        Orders as much as he gets in demand
    """
    def __init__(self, index, label = "BS", lead_time = ld.Uniform_LeadTime(2,2,1), BS_level = 20):
        Agent.__init__(self, index = index, lead_time = lead_time, label = label+"("+str(BS_level)+")")
        self.BS_level = BS_level
        
    def act(self, state = None, train = None):
        """
        Using Base stock policy for replenishement
        Args : 
            state :
            train :  
        """

        """
        J'ai enlevé la demande de la formule.
        Je pense que la demande etait deja inclus dans le IL du période 
        A verifier
        """
        return self.BS_level - (np.sum(state['OO'][-1]) + state['IL'][-1]) # - state['d'][-1]) #+ state['BO'][-1]   

    def copy(self):
        return BS_Agent(self.i, label = self.label, lead_time = self.lead_time, BS_level = self.BS_level)     

class BS_Dynamique_Agent(Agent):
    """ 
        Agent using base stock policy 
        Orders as much as he gets in demand
        Updates BS every X periods with coefficients of revision

        Not ready to use yet!!
        Need to receive historical data for N periods to revise teh BS policy with a frequency F
        N and F are inputs needed from user at the moment of creation
    """
    def __init__(self, index, vision,freq, mu, sigma, label = "BS", lead_time = ld.Uniform_LeadTime(2,2,1), BS_level = 20):
        Agent.__init__(self, index = index, lead_time = lead_time, label = label+"("+str(BS_level)+")")
        self.BS_level = BS_level
        self.vision = vision
        self.freq = freq
        self.lastupdate = 0
        self.average = mu
        
    def act(self, state = None, train = None):
        """Using Base stock policy for replenishement"""
        #State must include period
        #currently not the case
        if state['t'] - self.lastupdate == self.freq:
            self.BS_level = (state ['d'][:]) * self.BS_level / self.average
            self.lastupdate=state['t']
        return self.BS_level - (np.sum(state['OO'][-1]) + state['IL'][-1]) # + state['BO'][0]   

    def copy(self):
        return BS_Agent(self.i, label = self.label, lead_time = self.lead_time, BS_level = self.BS_level)     
        
class BS_Agent_Gauss(Agent):
    """ 
        Author : Daniel
        Agent using base stock policy 
        Orders as much as he gets in demand
        Base stock policy built from security stock + lead time * demand
    """
    def __init__(self, index, sigma, TS, mu, label = "BS_Gauss", lead_time = ld.Constant_LeadTime(1)):
        """
        Args:
            index : position agent dans la chaine
            sigma : sigma de la demande gaussienne à lequel l'agent va faire face
            TS : souhaité par le client
            mu : mu de la demande gaussienne à lequel l'agent va faire face
            label : 
            lead_time :
        """
        global dict_z
        z = dict_z[str(TS)]
        self.BS_level = int(np.sqrt(lead_time.Mean) * sigma * z + (lead_time.Mean * mu)) + 1
        #Base stock = point de commande de stock (lead_time * mu) + stock de securité
        Agent.__init__(self, index = index, lead_time = lead_time, label = label+"("+str(self.BS_level)+")")
        
    def act(self, state = None, train = None):
        """
        Using Base stock policy for replenishement
        Args : 
            state : 
            train : 
        """        
        return self.BS_level - (np.sum(state['OO'][-1]) + state['IL'][-1]) #+ state['BO'][-1] USED ONLY WHEN BACKORDERS

class BS_Agent_Seas(Agent):
    """
        Author : Daniel
        Agent that uses base stock policy
        Dynamique BS levels with different averages for different periods of the year
        Base stocks change during the year but the policy but decision making process is still to replenish the base stock level at each period
    """
    def __init__ (self, index, sigma, TS, demand, label = "BS_Seas",lead_time = ld.Uniform_LeadTime(2,2,1)):
        """
        Args:
            index : agent's index 
            sigma : standard deviation
            TS : Taux de service
            demand : end client seasonal demand instance
            nb_periodes : number of periods that will have the demand. Used to create the list of averages per period
            label : agent's label
            lead_time : lead time of the agent
        """
        Agent.__init__(self, index = index, lead_time = lead_time, label = label)
        self.BS_level_list = []
        demand_avg = demand.demand_avg
        self.size = len(demand_avg)
        global dict_z
        z = dict_z[str(TS)]
        for t in range(self.size):
            self.BS_level_list.append(int(np.sqrt(lead_time.Mean) * sigma * z + demand_avg[t]*lead_time.Mean))

    def act (self, state = None, train = None): 
        self.Update_BS(state)
        return self.BS_level - (state['OO'][-1][0] + state['IL'][-1]) # + state['BO'][-1]

    def Update_BS(self,state = None):
        self.BS_level = self.BS_level_list[int(state['t'][0])]

class SS_Agent_Gauss(Agent):
    """ 
        Author : Daniel
        Agent using base stock policy 
        Orders as much as he gets in demand
        Base stock policy built from security stock + lead time * demand
    """
    def __init__(self, index, sigma, TS, mu, label = "SS_Gauss", lead_time = ld.Uniform_LeadTime(2,2,1)):
        """
            index : position agent dans la chaine
            sigma : sigma de la demande gaussienne à lequel l'agent va faire face
            TS : souhaité par le client
            mu : mu de la demande gaussienne à lequel l'agent va faire face
            label : 
            lead_time :
        """
        Agent.__init__(self, index = index, lead_time = lead_time, label = label)
        self.mu = mu
        self.leadtime=lead_time
        global dict_z
        z = dict_z[str(TS)]
        #Base stock = point de commande de stock (lead_time * mu) + stock de securité
        self.BS_level = int(np.sqrt(self.leadtime.Mean) * sigma * z + (self.leadtime.Mean) * self.mu)+1
        
    def act(self, state = None, train = None):
        """
        Using Base stock policy for replenishement
        '''+ state['BO'][-1]'''
        """        
        if ((np.sum(state['OO'][-1]) + state['IL'][-1]-state ['d'][0])  < self.BS_level) : return ((lead_time.Moyenne()) * self.mu)
        else : return 0

class SS_Agent_Seas(Agent):
    """
        Author : Daniel
        Agent that uses base stock policy
        Dynamique BS levels with different averages for different periods of the year
        Base stocks change during the year but the policy but decision making process is still to replenish the base stock level at each period

    """
    def __init__ (self, index, sigma, TS, demand , label = "SS_Seas", lead_time = ld.Uniform_LeadTime(2,2,1), BS_level = 20):
        """
        Args:
            index : agent's index 
            sigma : standard deviation
            TS : Taux de service
            demand : end client seasonal demand instance
            nb_periodes : number of periods that will have the demand. Used to create the list of averages per period
            label : agent's label
            lead_time : lead time of the agent
        """
        self.BS_level_list = []
        demand_avg = demand.demand_avg
        self.size = len(demand_avg)
        global dict_z
        z = dict_z[str(TS)]
        for t in range(self.size):
            self.BS_level_list.append(int(np.sqrt(lead_time.Mean) * sigma * z + demand_avg[t]*lead_time.Mean))


    def act (self, state = None, train = None): 
        self.Update_BS(state) 
        """
        '''+ state['BO'][-1]''' 
        """
        if ((np.sum(state['OO'][-1]) + state['IL'][-1] -state ['d'][0]) < self.BS_level) : return ((lead_time.Moyenne()) * self.mu)
        else : return 0

    def Update_BS(self,state):
        self.BS_level = self.BS_level_list[int(state['t'][0])]

class New_agents_generation(Agent):
    """
    Not ready to use!!
    Logic code for when minimun lots are added to actions/orders to the previous agents in the chain
    """
    def __init__(self):
        self.agent_type = "BS_SS"
        self.stock_securite = 30
        self.lot_min = 20

    def act(self,state,train):
        #commande pour ne rien commander si stocks sont pas au dessous du Stock de securité
        
        if state ['IL'][0] - state['d'][0] > self.stock_securite + mu * lead_time:
            return (0)

        elif self.agent_type == "lot minimun":
            return max(self.BS_level - (state['OO'][-1][0] + state['IL'][-1]) + state['BO'][-1],self.lot_min)
'''

#################################
##      Intelligent Agents     ##
#################################

class RL_Agent(Agent):
    """
        Mother class of Reinforcement Learning sub classes
    """
    def __init__(self,  index = 0, AI_possible_actions = [-5, 0, 5], epsilon = 0.1, lead_time = ld.Uniform_LeadTime(2,2,1), label = "RL agent"):
        Agent.__init__(self, index = index, lead_time = lead_time, label = label)
        self.AI_possible_actions = AI_possible_actions
        self.epsilon = epsilon
        self.is_AI_agent = True

    def set_epsilon(self,e):
        self.epsilon = e

    def act(self, s = None, train = True):
        pass

    def learned_act(self,s):
        """ 
            Act via the policy of the agent, 
            from a given state s it proposes an action a
        """
        pass

    def argmax_Q(self, s):
        """ 
            Returns the action that has the best Q-value
        """
        pass

    def reinforce(self, s, n_s, a, n_a, r):
        """ This function is the core of the learning algorithm. Its goal is to learn a policy.
        s : current state (as a dictionarry)
        a : chosen action for state s
        n_s : next state after action a (as a dictionarry)
        n_a : chosen action for state n_s
        r : reward for moving from state s to n_s
        """
        pass

    def save(self):
        """ This function allows to save the policy"""
        pass

    def load(self):
        """ This function allows to restore a policy"""
        pass

class RL_No_Approx_Agent(RL_Agent):
    # OBSOLETE !! 
    """ 
        Agent using Reinforcement Learning algorithms to learn best replenishement policy
        Creates a list of Q-values and updates it after each iteration using Q-Learning algorithm
    """
    def __init__(self,  index = 0, AI_possible_actions = [-5, 0, 5], epsilon = 0.1, lead_time = ld.Uniform_LeadTime(2,2,1), label = "No Approx"):
        RL_Agent.__init__(self, index, AI_possible_actions, epsilon, lead_time, label)
        self.Q = {} # Initialize Q as a dictionarry
        
    def act(self, s_dict = None, train = True):
        """ Greedy action """
        s_str = self.state_dict2str(s_dict)
        explore =  np.random.rand() < self.epsilon or s_str not in self.Q.keys()

        if train and explore:
            return self.AI_possible_actions[np.random.randint(0,len(self.AI_possible_actions))]

        return self.learned_act(s_str)

        
    def learned_act(self,s_str):
        return self.argmax_Q(s_str)

    def reinforce(self, s_dict, n_s_dict, a, n_a, r):
        s = self.state_dict2str(s_dict)
        n_s = self.state_dict2str(n_s_dict)
        
        Q = self.Q
        # Initialize Q-value to zero if state action has never been visited
        if s not in Q.keys() :
            Q[s] = {}
            Q[s][a] = 0
        
        elif a not in Q[s].keys():
            Q[s][a] = 0
        
        if n_s not in Q.keys():
            Q[n_s] = {}
            Q[n_s][n_a] = 0
            
        elif n_a not in Q[n_s].keys():
            Q[n_s][n_a] = 0
        
        # Reinforce !!
        Q[s][a] += ALPHA*(r + GAMMA*Q[n_s][self.argmax_Q(n_s)] -  Q[s][a]) # Q-Learning
        
    def argmax_Q(self, s_str):
        max_Q = -10e6
        max_a = None
        
        if s_str in self.Q.keys():
            for a, q in self.Q[s_str].items():
                if q > max_Q:
                    max_a = a
                    max_Q = q
                    
        return max_a

class RL_Lin_Approx_Agent(RL_Agent):
    # OBSOLETE !! 
    """ 
        Agent using Reinforcement Learning algorithms to learn best replenishement policy
        A Linear model is used to predict the Q-value
    """
    def __init__(self, index = 0, AI_possible_actions = [-5, 0, 5], epsilon = 0.1, lead_time = ld.Uniform_LeadTime(2,2,1), samples = [], label = "Lin Approx"):
        RL_Agent.__init__(self, index, AI_possible_actions, epsilon, lead_time, label)
        # Q is approximated by a Linear model
        self.Q = Linear_Model(samples, AI_possible_actions)
        
    def act(self, s_dict = None, train = True):
        """ Greedy action """
        explore =  np.random.rand() < self.epsilon
        if train and explore:
            return self.AI_possible_actions[np.random.randint(0,len(self.AI_possible_actions))]
        return self.learned_act(s_dict)

    def learned_act(self, s_dict):
        return self.argmax_Q(s_dict)

    def reinforce(self, s, n_s, a, n_a, r):
        # we will update Q(s,a) AS we experience the episode
        self.Q.theta[a] += ALPHA*(r/100 + GAMMA*self.Q.predict(n_s, n_a) - self.Q.predict(s, a))*self.Q.grad(s)
        
    def argmax_Q(self, s_dict):
        max_Q = None
        max_a = None
        for a in self.AI_possible_actions:
            Q_prediction = self.Q.predict(s_dict, a)
            if not max_Q or Q_prediction > max_Q:
                max_a = a
                max_Q = Q_prediction
        return max_a

class RL_RBF_Approx_Agent(RL_Agent):
    """ 
        Agent using Reinforcement Learning algorithms to learn best replenishement policy
        An RBF model is used to predict the Q-value
    """
    def __init__(self, index = 0, AI_possible_actions = [-5, 0, 5], epsilon = 0.1, lead_time = ld.Uniform_LeadTime(2,2,1), label = "RBF Approximation", samples = [], learning_rate = 0.01):
        RL_Agent.__init__(self, index, AI_possible_actions, epsilon, lead_time, label)
        # Q is approximated by a RBF model
        self.Q = RBF_Model(samples, AI_possible_actions, learning_rate)

    def reinforce(self, s_dict, n_s_dict, a, n_a, r):
        s = self.state_dict2arr(s_dict)
        n_s = self.state_dict2arr(n_s_dict)

        G = r + GAMMA* self.Q.predict(n_s, n_a)
        
        self.Q.update(s, a, G)

    def act(self, s_dict = None, train = True):
        """ Greedy action """
        explore =  np.random.rand() < self.epsilon

        if train and explore:
            return self.AI_possible_actions[np.random.randint(0,len(self.AI_possible_actions))]

        return self.learned_act(s_dict)

        
    def learned_act(self,s_dict):
        s_arr = self.state_dict2arr(s_dict)
        return self.argmax_Q(s_arr)

    def argmax_Q(self, s_arr):
        max_Q = None
        max_a = None
        
        for a in self.AI_possible_actions:
            Q_prediction = self.Q.predict(s_arr,a)
            if not max_Q or Q_prediction > max_Q:
                max_a = a
                max_Q = Q_prediction    
     
        return max_a  

class RL_PG_Agent(RL_Agent):
    def __init__(self,  index = 0, AI_possible_actions = [-5, 0, 5], epsilon = 0.1, lead_time = ld.Uniform_LeadTime(2,2,1), label = "Policy Gradients", samples = []):
        RL_Agent.__init__(self, index, AI_possible_actions, epsilon, lead_time, label)

        D = 5 # dimension of state
        K = len(AI_possible_actions)

        self.graph = tf.Graph()
        with self.graph.as_default() as g:
            self.pmodel = PolicyModel(D, K, [3],samples, AI_possible_actions)
            self.vmodel = ValueModel(D, [5],samples)
            self.init = tf.global_variables_initializer()
            self.session = tf.InteractiveSession()
            self.session.run(self.init)
        
        self.pmodel.set_session(self.session)
        self.vmodel.set_session(self.session)

    def act(self, s_dict = None, train = True):
        """ Greedy action """
        s_arr = self.state_dict2arr(s_dict)
        return self.learned_act(s_arr)

    def learned_act(self,s_arr):
        return self.argmax_Q(s_arr)

    def argmax_Q(self, s_arr):
        # probability of doing each action
        with self.graph.as_default() as g:
            p = self.pmodel.predict(s_arr)[0]
            return np.random.choice(self.AI_possible_actions, p = p) 

    def reinforce(self, s_dict, n_s_dict, a, n_a, r):
        s = self.state_dict2arr(s_dict)
        n_s = self.state_dict2arr(n_s_dict)

        with self.graph.as_default() as g:
            V_next = self.vmodel.predict(n_s)
            G = r + GAMMA*np.max(V_next)
            advantage = G - self.vmodel.predict(s)

            self.pmodel.partial_fit(s, a, advantage)
            self.vmodel.partial_fit(s, G)

class RL_DQN_Agent(RL_Agent):
    def __init__(self,  index = 0, AI_possible_actions = [-5, 0, 5], epsilon = 0.1, lead_time = ld.Uniform_LeadTime(2,2,1), label = "DQN", samples = [], graph = None, session = None):
        RL_Agent.__init__(self, index, AI_possible_actions, epsilon, lead_time, label)

        self.normalizer = MinMaxScaler()
        if len(samples) > 0:
            self.normalizer.fit(samples)
        
        self.samples = samples

        gamma = 0.9
        self.copy_period = 50

        D = len(samples[0]) # dimension of state
        K = len(AI_possible_actions)

        if not graph:
            self.graph = tf.Graph()
        else :
            self.graph = graph

        self.NN_size = [10,10]
        
        with self.graph.as_default() as g:
            self.model = DQN_Model(D, K, self.NN_size, gamma, samples = samples, AI_possible_actions = AI_possible_actions)
            self.tmodel = DQN_Model(D, K, self.NN_size, gamma, samples = samples, AI_possible_actions = AI_possible_actions)
        
            self.init = tf.global_variables_initializer()
            if not session:
                self.session = tf.InteractiveSession()
            else:
                self.session = session
            self.session.run(self.init)
        
        self.model.set_session(self.session)
        self.tmodel.set_session(self.session)

        # create replay memory
        self.experience = {'s': [], 'a': [], 'r': [], 'n_s': []}
        self.max_experiences = 10000
        self.min_experiences = 100
        self.batch_sz = 32
        self.gamma = gamma

        self.iters = 0

    def act(self, s_dict = None, train = True):
        """ Greedy action """
        s_arr = self.state_dict2arr(s_dict)
        explore =  np.random.rand() < self.epsilon

        if train and explore:
           
            a = self.AI_possible_actions[np.random.randint(0,len(self.AI_possible_actions))]
            return a

        return self.learned_act(s_arr)

    def learned_act(self,s_arr):
        return self.argmax_Q(s_arr)

    def argmax_Q(self, s_arr):
        s_arr = self.normalizer.transform([s_arr])[0]
        with self.graph.as_default() as g:
            return self.AI_possible_actions[np.argmax(self.model.predict(s_arr)[0])]

    def reinforce(self, s_dict, n_s_dict, a, n_a, r):
        s = self.state_dict2arr(s_dict)
        n_s = self.state_dict2arr(n_s_dict)

        t1 = time.time()
        s = self.normalizer.transform([s])[0]
        n_s = self.normalizer.transform([n_s])[0]
        
        t2 = time.time()
        #self.time_eval['norm'].append(t2 - t1)
        
        # Adding experience 
        if len(self.experience['s']) >= self.max_experiences:
            self.experience['s'].pop(0)
            self.experience['a'].pop(0)
            self.experience['r'].pop(0)
            self.experience['n_s'].pop(0)

        self.experience['s'].append(s)
        self.experience['a'].append(a)
        self.experience['r'].append(r)
        self.experience['n_s'].append(n_s)

        
        t3 = time.time()
        #self.time_eval['exp'].append(t3 - t2)
        with self.graph.as_default() as g:
            self.model.train(self.tmodel, self.experience)
            #self.model.train(self.model, self.experience)
            t4 = time.time()
            #self.time_eval['train'].append(t4 - t3)

            self.iters += 1
            if self.iters % self.copy_period == 0:
                self.tmodel.copy_from(self.model)
            
            t5 = time.time()
            #self.time_eval['copy'].append(t5 - t4)

    def save_model(self, path):
        with self.graph.as_default() as g:
            self.model.save(path)

    def load_model(self, path):
        with self.graph.as_default() as g:
            self.model.load(path)

    def copy_models(self, agent):
        with self.graph.as_default() as g:
            self.model.copy_from(agent.model)
            self.tmodel.copy_from(agent.tmodel)


    def copy(self):
        agent_copy = RL_DQN_Agent(index = self.i, AI_possible_actions = self.AI_possible_actions, epsilon = self.epsilon, samples = self.samples, graph = self.graph, session = self.session)
        with self.graph.as_default() as g:
            agent_copy.model.copy_from(self.model)
            agent_copy.tmodel.copy_from(self.tmodel)

        return agent_copy

############################
##      Define models     ##
############################

class Linear_Model:
    def __init__(self, samples, AI_possible_actions):
        self.theta = {}

        for a in AI_possible_actions:
            self.theta[a] = np.random.randn(21) / np.sqrt(21)
        
        self.normalizer = MinMaxScaler()
        self.normalizer.fit(samples)

    def set_features(self, s_dict):
        x = Agent().state_dict2arr(s_dict)

        X = self.normalizer.transform([x])[0]
        return np.array([1, X[0],X[1],X[2],X[3],X[4],
                        X[0]*X[0], X[1]*X[1], X[2]*X[2], X[3]*X[3], X[4]*X[4],
                        X[0]*X[1], X[0]*X[2], X[0]*X[3], X[0]*X[4], X[1]*X[2], 
                        X[1]*X[3], X[1]*X[4], X[2]*X[3], X[2]*X[4], X[3]*X[4]])

    def predict(self, s_dict, a):
        x = self.set_features(s_dict)
        return self.theta[a].dot(x)

    def grad(self, s_dict):
        return self.set_features(s_dict)

class RBF_Model:
    def __init__(self, samples, AI_possible_actions, learning_rate, n_components=5):

        self.models = {}

        self.normalizer = StandardScaler()
        self.normalizer.fit(samples)

        self.featurizer = FeatureUnion([
                ("rbf1", RBFSampler(gamma=0.5, n_components=n_components)),
                ("rbf2", RBFSampler(gamma=1, n_components=n_components))
                ])

        self.featurizer.fit_transform(self.normalizer.transform(samples))
       
        for a in AI_possible_actions:
            model = SGDRegressor()
            x = self.featurizer.transform(self.normalizer.transform([samples[0]]))
            model.partial_fit(x, [0])
            self.models[a] = model

    def predict(self, s_arr, a):
        X = self.transform([s_arr])
        return self.models[a].predict(X)

    def update(self, s_arr, a, G):
        X = self.transform([s_arr])
        self.models[a].partial_fit(X, [G])

    def transform(self, s_arr):
        normalized = self.normalizer.transform(s_arr)
        return self.featurizer.transform(normalized)

## Neural network models

class HiddenLayer:
    def __init__(self, M1, M2, f=tf.nn.tanh, use_bias=True):
        self.W = tf.Variable(tf.random_normal(shape=(M1, M2)))
        self.params = [self.W]
        self.use_bias = use_bias
        if use_bias:
            self.b = tf.Variable(np.zeros(M2).astype(np.float32))
            self.params.append(self.b)
        self.f = f  

    def forward(self, X):
        if self.use_bias:
            a = tf.matmul(X, self.W) + self.b
        else:
            a = tf.matmul(X, self.W)
        return self.f(a)

class PolicyModel:
    def __init__(self, D, K, hidden_layer_sizes, samples, AI_possible_actions):

        self.normalizer = MinMaxScaler()
        self.normalizer.fit(samples)
        self.AI_possible_actions = AI_possible_actions

        self.layers = []



        M1 = D
        for M2 in hidden_layer_sizes:
            layer = HiddenLayer(M1, M2)
            self.layers.append(layer)
            M1 = M2

        # final layer
        layer = HiddenLayer(M1, K, tf.nn.softmax, use_bias=False)
        self.layers.append(layer)

        # inputs and targets
        self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
        self.actions = tf.placeholder(tf.int32, shape=(None,), name='actions')
        self.advantages = tf.placeholder(tf.float32, shape=(None,), name='advantages')

        # calculate output and cost
        Z = self.X
        for layer in self.layers:
            Z = layer.forward(Z)
        p_a_given_s = Z
        self.predict_op = p_a_given_s

        selected_probs = tf.log(
          tf.reduce_sum(
            p_a_given_s * tf.one_hot(self.actions, K),
            reduction_indices=[1]
          )
        )

        cost = -tf.reduce_sum(self.advantages * selected_probs)
        self.train_op = tf.train.AdagradOptimizer(1e-1).minimize(cost)

    def set_session(self, session):
        self.session = session

    def partial_fit(self, X, actions, advantages):
        X = self.normalizer.transform([X])[0]
        X = np.atleast_2d(X)
        actions = np.atleast_1d(np.where(self.AI_possible_actions == actions)[0][0])
        advantages = np.atleast_1d(advantages)
        self.session.run(
          self.train_op,
          feed_dict={
            self.X: X,
            self.actions: actions,
            self.advantages: advantages,
          }
        )

    def predict(self, X):
        X = self.normalizer.transform([X])[0]
        X = np.atleast_2d(X)
    
        return self.session.run(self.predict_op, feed_dict={self.X: X})

class ValueModel:
    def __init__(self, D, hidden_layer_sizes, samples):
        self.normalizer = MinMaxScaler()
        self.normalizer.fit(samples)

        # create the graph
        self.layers = []
        M1 = D
        for M2 in hidden_layer_sizes:
            layer = HiddenLayer(M1, M2)
            self.layers.append(layer)
            M1 = M2

        # final layer
        layer = HiddenLayer(M1, 1, lambda x: x)
        self.layers.append(layer)

        # inputs and targets
        self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
        self.Y = tf.placeholder(tf.float32, shape=(None,), name='Y')

        # calculate output and cost
        Z = self.X
        for layer in self.layers:
            Z = layer.forward(Z)
        Y_hat = tf.reshape(Z, [-1]) # the output
        self.predict_op = Y_hat

        cost = tf.reduce_sum(tf.square(self.Y - Y_hat))
        self.train_op = tf.train.GradientDescentOptimizer(1e-4).minimize(cost)

    def set_session(self, session):
        self.session = session

    def partial_fit(self, X, Y):
        X = self.normalizer.transform([X])[0]

        X = np.atleast_2d(X)
        Y = np.atleast_1d(Y)

        self.session.run(self.train_op, feed_dict={self.X: X, self.Y: Y})

    def predict(self, X):
        X = self.normalizer.transform([X])[0]
        X = np.atleast_2d(X)
        return self.session.run(self.predict_op, feed_dict={self.X: X})


# DQN Model
class DQN_Model:
    def __init__(self, D, K, hidden_layer_sizes, gamma = 0.9, samples = [], AI_possible_actions =[-5,0,5], max_experiences=10000, min_experiences=100, batch_sz=32):
        """
        Initialize DQN Model instance
        Args :
            D :
            K :
            hidden_layer_sizes :
            gamma :
            samples : 
            AI_possible_actions :
            max_experiences :
            min_experiences :
            batch_sz :

        """
        self.normalizer = MinMaxScaler()
        if len(samples) > 0:
            self.normalizer.fit(samples)
        else:
            print(color.RED,"Warning : samples is empty", color.END)

        self.AI_possible_actions = np.array(AI_possible_actions)

        self.K = K
        self.D = D

        self.hidden_layer_sizes = hidden_layer_sizes
        # create the graph
        self.layers = []

        M1 = D
        for M2 in hidden_layer_sizes:
            layer = HiddenLayer(M1, M2)
            self.layers.append(layer)
            M1 = M2

        # final layer
        layer = HiddenLayer(M1, K, lambda x: x)
        self.layers.append(layer)

        # collect params for copy
        self.params = []
        for layer in self.layers:
            self.params += layer.params

        # inputs and targets
        self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
        self.G = tf.placeholder(tf.float32, shape=(None,), name='G')
        self.actions = tf.placeholder(tf.int32, shape=(None,), name='actions')

        # calculate output and cost
        Z = self.X
        for layer in self.layers:
            Z = layer.forward(Z)
        Y_hat = Z
        self.predict_op = Y_hat

        selected_action_values = tf.reduce_sum(
          Y_hat * tf.one_hot(self.actions, K),
          reduction_indices=[1]
        )

        cost = tf.reduce_sum(tf.square(self.G - selected_action_values))
        self.train_op = tf.train.AdamOptimizer(1e-2).minimize(cost)

        self.copy_ops = []
        self.to_copy = tf.placeholder(dtype=tf.float32)
        for param in self.params:
            self.copy_ops.append(param.assign(self.to_copy))
            
        # create replay memory
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences
        self.batch_sz = batch_sz
        self.gamma = gamma


    def set_session(self, session):
        self.session = session

    def predict(self, X):
        X = np.atleast_2d(X)
        return self.session.run(self.predict_op, feed_dict={self.X: X})

    def train(self, target_network, experience):
        # sample a random batch from buffer, do an iteration of GD
        if len(experience['s']) < self.min_experiences:
        # don't do anything if we don't have enough experience
            return

        idx = np.random.choice(len(experience['s']), size=self.batch_sz, replace=False)
        states = [experience['s'][i] for i in idx]
        actions = [experience['a'][i] for i in idx]
        rewards = [experience['r'][i] for i in idx]
        next_states = [experience['n_s'][i] for i in idx]
        next_Q = np.max(target_network.predict(next_states), axis=1)
        
        targets = [r + self.gamma*next_q for r, next_q in zip(rewards, next_Q)]

        actions_indices = actions
        for i, action in enumerate(actions):
            actions_indices[i] = np.where(self.AI_possible_actions==action)[0][0]

        # call optimizer
        self.session.run(
          self.train_op,
          feed_dict={
            self.X: states,
            self.G: targets,
            self.actions: actions_indices
          }
        )

    def copy_from(self, other):
        my_params = self.params
        other_params = other.params

        for op, q in zip(self.copy_ops, other_params):
            actual = self.session.run(q)
            self.session.run(op, feed_dict={self.to_copy : actual})

    def save(self, path):
        print("Saving DQN Model... ",end = "")
        saver = tf.train.Saver()
        saver.save(self.session, path)
        print("Done")


    def load(self, path):
        print("Loading DQN Model... ",end = "")
        saver = tf.train.Saver()
        saver.restore(self.session, os.fspath(path))
        print("Done")

    def copy(self):
        print("Copying DQN Model... ",end = "")
        model_copy = DQN_Model(D = self.D, K = self.K, hidden_layer_size = self.hidden_layer_sizes, gamma = self.gamma, samples = self.samples, AI_possible_actions = self.AI_possible_actions, max_experiences=self.max_experiences, min_experiences=self.min_experiences, batch_sz=self.batch_sz)
        model_copy.copy_from(self)
        print("Done")

        return model_copy
'''