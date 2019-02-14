import numpy as np 
import pdb
import Agent as Agent
import re
import json

from CONSTANTS import *

class Environment:
    """
    The Environment class simulates the beergame environement
    Args:
        T : number of periods
        NA : number of agents
        m : number of  

        RS : Received Shipement from agent i+1
        IL : Inventory Level
        d : demand received from agent i-1
        OO : On-Order items, i.e. the items that have been ordered from agent i+1 but not received yet
        a : action taken by agent i (corresponds to the order launched to agent i+1)
        BO : Back-order items, (corresponds to the demand that could not be satisfied due to a shortage)
        
        HC : Holding Costs
        SC : Shortage Costs
        CC : Cumulative costs
        r : reward = -(HC + SC)
    """
    def __init__(self, agents, game_params):
        """
        Initialize Environment
        Args :
            agents : list of agents
            game_params : dictionnary of game parameters 
        """
        self.NA = len(agents) # number_agents

        self.max_lead_time = 0
        self.update_max_lead_time(agents)

        self.agents = agents

        self.params = {}

        for key in DEFAULT_PARAMS.keys():
            if key in game_params.keys(): self.params[key] = game_params[key]
            else: self.params[key] = DEFAULT_PARAMS[key]

        self.T = self.params['number_periods'] 
        
        self.update_nb_features()

    def act(self, agent, action):
        """
        Updates the environement after an action is made by an agent
        Args : 
            agent : the agent
            action : the action that the agent will make
        """
        t = self.t
        i = agent.i
        H = self.history
        H['a'][t,i] = action

    def update_agent_state(self, agent):
        """ 
        Updates an agents state in current period 
        ** WARNING !! Note that agent i-1 should be UPDATED BEFORE agent i in period t
        Args :
            agent : the agent 
        """
        t = self.t
        i = agent.i
        H = self.history
        
        H['d'][t,i] = max(0, H['d'][t,i-1] + H['a'][t,i-1])  # order quantity = d + x
        
        # In period t=0 RS, OO, IL and BO are equal to 0
        if t > 0:
            # Compute RS
            H['RS'][t,i] = H['OO'][t-1,i][0] # RS(t,i) = OO(t-1,i)[à]
            
            # Compute IL and BO
            if self.params['use_backorders']:
                H['IL'][t,i] =  max(0, H['IL'][t-1,i] + H['RS'][t,i] - H['d'][t-1,i] - H['BO'][t-1,i])
                H['BO'][t,i] =  max(0, -(H['IL'][t-1,i] + H['RS'][t,i] - H['d'][t-1,i] - H['BO'][t-1,i]))
            
            else:
                H['IL'][t,i] =  max(0, H['IL'][t-1,i] + H['RS'][t,i] - H['d'][t-1,i])
                H['BO'][t,i] =  max(0, -(H['IL'][t-1,i] + H['RS'][t,i]- H['d'][t-1,i]))
            
            # Compute OO
            # The external manufacturer supplier never gets shortage
            if i == self.NA: 
                OO = max(0, H['d'][t-1,i] + H['a'][t-1,i]) # d + a 

            # The on order computations takes into consideration whether agent has a shortage or not
            else:  
                if self.params['use_backorders']:
                    OO = min(H['d'][t-1, i+1] + H['BO'][t-1, i+1], H['IL'][t-1, i+1] + H['OO'][t-1, i+1][0])
                else:
                    OO = min(H['d'][t-1, i+1], H['IL'][t-1, i+1] + H['OO'][t-1, i+1][0])
            
            #H['OO'][t,i] = np.append(np.delete(H['OO'][t-1,i], [0], axis = None), OO)
            H['OO'][t,i] = np.append(np.delete(H['OO'][t-1,i], [0], axis = None), 0)
            
            l = int(H['ld'][t, i+1]) # get the lead time of the next agent in that period
            #pdb.set_trace()
            H['OO'][t,i,l-1] += OO
            #H['OO'][t,i,l-2] += OO
        

        # compute parameters for gaussian demand Mu and Sigma
        if "Mu" in self.params['state_features'] and "Sigma" in self.params['state_features']:
            if t > 0:
                #pdb.set_trace()
                H['Mu'][t, i] = (t/(t+1))*H['Mu'][t-1, i] + (1/(t+1)) * H['d'][t,i]
                H['Variance'][t,i] = (t/(t+1))*H['Variance'][t-1, i] + (1/t)*np.power(H['d'][t,i] - H['Mu'][t, i], 2)
                H['Sigma'][t,i] = np.sqrt(H['Variance'][t,i])
            else:
                H['Mu'][t, i] =  np.mean(H['d'][:t+1,i])
                H['Sigma'][t, i] = np.std(H['d'][:t+1,i])
            #H['Mu0'][t, i] = np.mean(H['d'][:t+1,i])
            #H['Sigma0'][t, i] = np.std(H['d'][:t+1,i])

        # Update costs HC and SC and rewards r
        H['HC'][t, i] = H['IL'][t, i] * self.params['holding_cost']
        H['SC'][t, i] = H['BO'][t, i] * self.params['shortage_cost']
        H['r'][t, i] = -(H['HC'][t, i] + H['SC'][t, i]) # Le reward est negatif, il represente un coût
        
        # Cumulative costs CC
        if t > 1 :
            H['CC'][t, i] =  H['CC'][t-1, i] + H['HC'][t, i] + H['SC'][t, i] 
        else :
            H['CC'][t, i] =  H['HC'][t, i] + H['SC'][t, i] 
        
    def game_over(self):
        """ 
        Returns False while the game is not over 
        """
        return self.t == self.T
    
    def get_state_history(self, agent):
        """ 
        Returns current state (including all parameters) of an agent as dictionnary
        Args:
            agent : agent that we'll take history for 
        """
        t = self.t
        i = agent.i
        H = self.history
        
        state = {}
        for key, value in H.items():
            state[key] = value[t, i]
        
        return state 

    def get_state_features(self, agent, m = None, t = None):
        """ 
        Returns m states (with only selected features) of an agent as dictionnary
        Args :
            agent : agent that we'll take features for
            m : number of periods that we'll take into consideration
            t : the period from which we're taking the features
        """
        if t == None : 
            t = self.t

        i = agent.i
        H = self.history
        if m == None : 
            m = self.params['m']
        
        state = {}
        for key, value in H.items():
            if key in self.params['state_features']:
                state[key] = value[max(0,t-m+1):t+1, i]

        # If we don't have enough history
        if t-m+1 < 0:
            for key, value in H.items():
                if key in self.params['state_features']:
                    if key == 'IL' or key == 'Mu' or key == "Sigma":
                        state[key] = np.append(np.ones(-(t-m+1))*state[key][0],state[key])
                    elif key == 'OO':
                        state[key] = np.append(np.zeros((-(t-m+1), len(state[key][0]))),state[key], axis = 0)
                    else:
                        state[key] = np.append(np.zeros(-(t-m+1)),state[key])
        return state
            
    def reset(self, demand = None):
        """ 
        Resets the environment 
        Args :
            demand : the client demand to initialize the environement with
        """
        T = self.T
        NA = self.NA
        self.history = {}
        H = self.history

        # We add 2 hypothetical agents to save end-client demand and external suppliers shipments        
        # general parameters
        H['IL'] = np.ones((T, NA + 2)) * self.params['initial_inventory'] 
        H['BO'] = np.zeros((T, NA + 2)) 
        H['d'] = np.zeros((T, NA + 2)) 
        H['RS'] = np.zeros((T, NA + 2)) 
        #H['OO'] = np.zeros((T, NA + 2, self.max_lead_time - 1)) 

        H['OO'] = np.zeros((T, NA + 2, self.max_lead_time)) 

        # periods
        H['t'] = np.multiply(np.ones((T, NA + 2)),np.arange(T).reshape((T,1)))

        # actions taken = Order - Demand
        H['a'] = np.zeros((T, NA + 2)) 
        """Initialize randomly the cliend-end demand for all periods"""
        if not demand:
            H['a'][:,0] = self.params['client_demand'].generate(T)
        else:
            H['a'][:,0] = demand


        # parameters for Gaussian demand
        H['Mu'] = np.zeros((T, NA + 2)) 
        H['Sigma'] = np.zeros((T, NA + 2)) 
        H['Variance'] = np.zeros((T, NA + 2)) 
        #H['Mu0'] = np.zeros((T, NA + 2)) 
        #H['Sigma0'] = np.zeros((T, NA + 2)) 

        # costs and rewards history
        H['HC'] = np.zeros((T, NA + 2)) 
        H['SC'] = np.zeros((T, NA + 2))
        H['CC'] = np.zeros((T, NA + 2))
        H['r'] = np.zeros((T, NA + 2))

        # lead times
        H['ld'] = np.zeros((T, NA + 2))
        H['ld'][:,-1] = np.ones(T) * 1 # The last supplier has always a leadtime of 1
        #H['ld'][:,-1] = np.ones(T) * 2 # The last supplier has always a leadtime of 1
        for agent in self.agents:
            H['ld'][:, agent.i] = agent.lead_time.generate(T)
                
        self.t = 0
        
        #self.init_client_demand()

    def update_nb_features(self):
        """ 
        Updates the number of features taking into consideration the parameter m 
        """
        self.nb_state_features = len(self.params['state_features'])
        if "OO" in self.params['state_features'] : 
            self.nb_state_features += self.max_lead_time - 1 
            #self.nb_state_features += self.max_lead_time - 2 
        self.nb_state_features *= self.params['m'] 

    def update_max_lead_time(self, agents):
        """ 
        When the lead time is varient, 
        we update max lead time in order to set On-order list maximum length 
        Args:
            agents : list of agents from where to look for the max of lead times
        """
        self.max_lead_time = 0
        
        for agent in agents:
            if self.max_lead_time < agent.lead_time.Max:
                self.max_lead_time = agent.lead_time.Max

    #def init_client_demand(self):
    #    """Initialize randomly the cliend-end demand for all periods"""
    #    self.history['a'][:,0] = self.params['client_demand'].generate(self.T)
        

    ##########        Results analysis functions        ########

    def get_agent_results(self, agent):
        """
        Returns a dictionnary of an agent's performances till the period t
        Args : 
            agent : the agent
        """
        H = self.history
        t = self.t
        i = agent.i
        results = {}

        results["coverage_rate"] = np.mean(H['IL'][:t,i]) / np.mean(H['d'][:t,i]) # Taux de couverture
        results["breakdown_rate"] = np.sum(H['BO'][:t,i]) / np.sum(H['d'][:t,i]) # Taux de rupture
        results["taux_rotation"] = np.sum(H['IL'][:t,i]) / np.mean(H['IL'][:t,i]) # Taux de rotation
        results["duree_stock"] = self.T * np.mean(H['IL'][:t,i]) / np.sum(H['IL'][:t,i]) # Durée moyenne des stocks

        return results

    def get_coverage_rate(self, agent):
        """
            Calcul le taux de couverture jusqu'a l'intant t de l'agent i
        """
        H = self.history
        t = self.t
        i = agent.i

        return np.mean(H['IL'][:t,i]) / np.mean(H['d'][:t,i]) # Taux de couverture

    def get_service_rate(self,agent):
        """
            Calcul le taux de service jusqu'a l'intant t de l'agent i
        """
        H = self.history
        t = self.t
        i = agent.i
        
        return np.mean((H['IL'][:t,i] - H['d'][:t,i] < 0).astype(int))
        

    def get_breakdown_rate(self, agent):
        """
            Calcul le taux de rupture jusqu'a l'intant t de l'agent i
            WARNING !! Si on utilise les backorders, la formule est erroné 
        """

        if self.params['use_backorders']:
            print("WARNING : since backorders are used, the breakdown rate is insignificant !")

        H = self.history
        t = self.t
        i = agent.i

        return np.sum(H['BO'][:t,i]) / np.sum(H['d'][:t,i]) # Taux de rupture

    def save_json(self, path):
        # DEPRECATED !!
        no_save_list = ['agents', 'client_demand', 'history']
        json_obj = {}

        for key, value in self.__dict__.items():
            if key not in no_save_list:
                if isinstance(self.__dict__[key],np.ndarray):
                    json_obj[key] = self.__dict__[key].tolist()
                else:
                    json_obj[key] = self.__dict__[key]


        json_obj['history'] = {}
        for key, value in self.history.items():
            json_obj['history'][key] = self.history[key].tolist()

        with open(path+'/env.json', 'w') as outfile:
            json.dump(json_obj, outfile)

        self.params['client_demand'].save_json(path)

        