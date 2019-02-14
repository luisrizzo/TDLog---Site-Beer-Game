import numpy as np
import re
import Environment as Env
import BeerGame as BG
import Comparator as Comp
from tools import *
from CONSTANTS import *
import pdb 
import json


class Trainer:
    """

    """
    
    def __init__(self, agents, game_params, AI_index = None):
        """
        Initialize The trainer instance 
        Args : 
            agents : list of agents instances
            game_params : dictionnary of game parameters
            AI_index : AI index in the agents list (can be unspecified)
        """
        
        # Looking automatically for AI index
        if not AI_index:
            self.AI_index = get_AI_index(agents)
        else: self.AI_index = AI_index

        self.AI_Agent = agents[self.AI_index]

        #Initializations
        self.train_iter = 0 #number_iterations

        # initalize parameters
        self.params = game_params
        self.beerGame = BG.BeerGame()
        self.agents = agents
        self.env = Env.Environment(self.agents, self.params)            
        print_game_params(agents = self.agents)
        self.time_per_iteration = -1

        # Create the Agent to compare to depending on the demand type 
        self.demand = game_params['client_demand']
        
        self.train_costs = {
            'AI_costs' : np.array([]),
            'AI_smooth' : np.array([]),
            'AI_best_quartil' : np.array([]),
            'AI_worst_quartil' : np.array([]),
            'iterations' : np.array([])
        }

        self.test_costs = {
            'AI' : np.array([]),
            'BS' : np.array([]),
            'iterations' : np.array([])
        }
        
        self.AI_agent_costs = np.array([])
        self.AI_agent_smooth_costs = np.array([])
        self.AI_agent_best_quartil_costs = np.array([])
        self.AI_agent_worst_quartil_costs = np.array([])
        self.BS_agent_costs = np.array([])

        self.figure_displayer = None
        self.best_AI_agent = self.agents[self.AI_index].copy()

        self.comparator = Comp.ComparatorBis(agents, game_params)
    

    def train2(self, N = 3000, max_epsilon = 0.1, decaying_epsilon = True, comp_interval = 50):
        """ 
        Trains the AI
        Args:
            N : number of iterations
            max_epsilon : value btw 0 and 1 
            decaying_epsilon (Bool):
            comp_interval : a comparison will be made each comp_interval iterations

        """
        print("\n--------------------------------------------------------------------------------")
        print(color.BOLD+"Training AI (",N + self.train_iter,"iterations )")
        print("--------------------------------------------------------------------------------")
        
        # set AI agent epsilon greedy
        self.agents[self.AI_index].set_epsilon(max_epsilon)

        # initialize figures
        self.figure_displayer = FiguresDisplayer(self.env)
        self.figure_displayer.init_train_fig2(self.train_iter + N)
        
        # initialize costs dictionnary
        self.init_costs(N, comp_interval)
        
        self.min_cost = 10e10

        print_game_params(self.env, self.agents, self.params['AI_possible_actions'])
        
        # helps computing cp times
        cp_times = []
        total_training = N + self.train_iter
        
        nb_smooth_iter = 50

        # Play game N times
        print("Running game",N,"times..", end = "\n")
        for i in range(N):
            t1 = time.time()

            # Play without training in order to see the real AI costs and compare it to base stock
            if i%comp_interval == 0 or i == N-2:
                #set the same demand to make the same comparison
                
                AI_perf, BS_perf = self.comparator.launch_comparison(5, verbose = False)
                self.update_test_costs(AI_perf['costs'],comp_interval, agent_label = "AI")
                self.update_test_costs(BS_perf['costs'],comp_interval, agent_label = "BS")

            # Train our agent
            self.env.reset() # reset the environment 
            self.beerGame.play(self.env, self.agents, train = True) # play with training
            
            # Update Figure data
            self.update_train_costs()
            
            # update the best AI Agent so far
            if self.min_cost >  self.train_costs["AI_smooth"][self.train_iter]:
                self.min_cost = self.train_costs["AI_smooth"][self.train_iter]
                self.best_AI_agent.copy_models(self.agents[self.AI_index])
            
            self.train_iter += 1

            if i%2== 0:
                self.figure_displayer.update_train_fig2(self.train_iter, self.train_costs, self.test_costs, comp_interval)

            if i%2 == 1:
                print_progression(self.train_iter, total_training, cp_time=np.mean(np.array(cp_times[max(-i, -nb_smooth_iter):])), costs = self.train_costs["AI_smooth"])
            
            cp_times.append(time.time() - t1)
            
        self.figure_displayer.update_train_fig2(self.train_iter - 1, self.train_costs, self.test_costs, comp_interval)
        print_progression(self.train_iter, total_training, cp_time=np.mean(np.array(cp_times[max(-i, -nb_smooth_iter):])), costs = self.train_costs["AI_smooth"])
        
        # Resume of time spent on traning
        self.time_per_iteration = np.mean(np.array(cp_times[max(-i, -nb_smooth_iter):]))
        print("\n"+color.BOLD+"Computations time : "+color.END,str(round(np.sum(np.array(cp_times)),1))+"s")


    def init_costs(self, N, comp_interval):
        """
        Initialize dictionnaries of training and testing costs with zeros in proper dimensions
        Args:
            N : Number of training iterations
            comp_interval : interval in which a comparison test is made
        """
         # init train costs
        self.train_costs["AI_costs"] = np.append(self.train_costs["AI_costs"], np.zeros(N))
        self.train_costs["AI_smooth"] = np.append(self.train_costs["AI_smooth"], np.zeros(N))
        self.train_costs["AI_best_quartil"] = np.append(self.train_costs["AI_best_quartil"], np.zeros(N))
        self.train_costs["AI_worst_quartil"] = np.append(self.train_costs["AI_worst_quartil"], np.zeros(N))

        nb_tr_iter = len(self.train_costs["iterations"])
        self.train_costs["iterations"] = np.append(self.train_costs["iterations"], np.arange(nb_tr_iter + 1, nb_tr_iter + N + 1))

        # init test costs
        self.test_costs['AI'] = np.append(self.test_costs['AI'], np.zeros(N//comp_interval))
        self.test_costs['BS'] = np.append(self.test_costs['BS'], np.zeros(N//comp_interval))
        self.test_costs['iterations'] = np.append(self.test_costs['iterations'], np.arange(nb_tr_iter + 1, nb_tr_iter + N + 1, comp_interval))


    def update_train_costs(self, nb_smooth_iter = 50):
        """
        Updates the training costs for the last made iteration
        Warning : Costs are updated with last env history data
        Args :
            nb_smooth_iter : number of iterations taken into consideration to compute smooth costs
        """
        self.train_costs["AI_costs"][self.train_iter] = self.env.history['CC'][-1][self.AI_index + 1]
        self.train_costs["AI_smooth"][self.train_iter] = np.mean(self.train_costs["AI_costs"][max(0, self.train_iter-nb_smooth_iter):self.train_iter+1])
        self.train_costs["AI_best_quartil"][self.train_iter] = np.mean(np.sort(self.train_costs["AI_costs"][max(0, self.train_iter-nb_smooth_iter):self.train_iter+1])[-25:])
        self.train_costs["AI_worst_quartil"][self.train_iter] = np.mean(np.sort(self.train_costs["AI_costs"][max(0, self.train_iter-nb_smooth_iter):self.train_iter+1])[:25])
    

    def update_test_costs(self, cost, comp_interval, agent_label = "AI"):
        """
        Updates the test costs for the last made iteration of an agent
        Args :
            cost : ...
            comp_interval : interval in which a comparison test is made
            agent_label : specify the agent label as it is in the test costs dictionnary
        """
        self.test_costs[agent_label][self.train_iter//comp_interval] = cost 


    def train(self, N = 3000, comparator = None, epsilon = 0.1):
        print("\n--------------------------------------------------------------------------------")
        print(color.BOLD+"Training AI (",N + self.train_iter,"iterations )")
        print("--------------------------------------------------------------------------------")
        self.agents[self.AI_index].set_epsilon(epsilon)

        self.figure_displayer = FiguresDisplayer(self.env)
        
        if comparator:
            best_BS = comparator.best_BS()
            worst_BS = comparator.worst_BS()
        else:
            best_BS = ("undefined", 0 ,0)
            worst_BS = ("undefined", 0 ,0)

        self.figure_displayer.init_train_fig(self.train_iter + N, best_BS[0], worst_BS[0])
        
        self.AI_agent_costs = np.append(self.AI_agent_costs, np.zeros(N))
        self.AI_agent_smooth_costs = np.append(self.AI_agent_smooth_costs, np.zeros(N))
        self.AI_agent_best_quartil_costs = np.append(self.AI_agent_best_quartil_costs, np.zeros(N))
        self.AI_agent_worst_quartil_costs = np.append(self.AI_agent_worst_quartil_costs, np.zeros(N))
        
        self.min_cost = 10e10

        print_game_params(self.env, self.agents, self.params['AI_possible_actions'])
        cp_times = []
        total_training = N + self.train_iter
        
        # Play game N times
        print("Running game",N,"times..", end = "\n")
        for i in range(N):
            t1 = time.time()
            
            self.env.reset() # reset the environment 
            self.beerGame.play(self.env, self.agents, train = True)
            # Replay without training in order to see the real AI costs
            self.env.reset() # reset the environment 
            self.beerGame.play(self.env, self.agents, train = False)
            
            # Update Figure data
            nb_smooth_iter = 50
            self.AI_agent_costs[self.train_iter] = self.env.history['CC'][-1][self.AI_index + 1]
            self.AI_agent_smooth_costs[self.train_iter] = np.mean(self.AI_agent_costs[max(0, self.train_iter-nb_smooth_iter):self.train_iter+1])
            self.AI_agent_worst_quartil_costs[self.train_iter] = np.mean(np.sort(self.AI_agent_costs[max(0, self.train_iter-nb_smooth_iter):self.train_iter+1])[-25:])
            self.AI_agent_best_quartil_costs[self.train_iter] = np.mean(np.sort(self.AI_agent_costs[max(0, self.train_iter-nb_smooth_iter):self.train_iter+1])[:25])
            if self.min_cost >  self.AI_agent_smooth_costs[self.train_iter]:
                self.min_cost = self.AI_agent_smooth_costs[self.train_iter]
                self.best_AI_agent.copy_models(self.agents[self.AI_index])
            self.train_iter += 1

            if i%2== 0:
                self.figure_displayer.update_train_fig(self.train_iter, self.AI_agent_costs, self.AI_agent_smooth_costs, self.AI_agent_worst_quartil_costs, self.AI_agent_best_quartil_costs, best_BS[2], worst_BS[2])

            if i%2 == 1:
                print_progression(self.train_iter, total_training, cp_time=np.mean(np.array(cp_times[max(-i, -nb_smooth_iter):])), costs = self.AI_agent_smooth_costs)
            
            cp_times.append(time.time() - t1)
            
        self.figure_displayer.update_train_fig(self.train_iter - 1, self.AI_agent_costs, self.AI_agent_smooth_costs, self.AI_agent_worst_quartil_costs, self.AI_agent_best_quartil_costs, best_BS[2], worst_BS[2])
        print_progression(self.train_iter, total_training, cp_time=np.mean(np.array(cp_times[max(-i, -nb_smooth_iter):])), costs = self.AI_agent_smooth_costs)
        
        self.time_per_iteration = np.mean(np.array(cp_times[max(-i, -nb_smooth_iter):]))
        print("\n"+color.BOLD+"Computations time : "+color.END,str(round(np.sum(np.array(cp_times)),1))+"s")
    
    def get_AI_agent(self):
        return self.agents[self.AI_index]

    def generate_comparator(self, min_BS_level = 1, max_BS_level = 15):
        return Comp.Comparator(
            clone_agents_list(self.agents),
            self.params,
            min_BS_level = min_BS_level, 
            max_BS_level = max_BS_level
            )

    def get_agents_labels(self):
        labels = "["
        for i, agent in enumerate(self.agents):
            print(agent.label, end = "")
            labels += agent.label
            if i < len(self.agents) - 1: labels += " - "
        labels += "]"
        return labels

    def save_json(self, path):
        # DEPRECATED !!
        json_obj = {}
        no_save_list = ['agents', 'beerGame','env', 'best_AI_agent', 'client_demand', 'lead_times', 'figure_displayer']
        save_lists = ['']
        for key, value in self.__dict__.items():
            if key not in no_save_list:
                if isinstance(self.__dict__[key],np.ndarray):
                    json_obj[key] = self.__dict__[key].tolist()
                else:
                    json_obj[key] = self.__dict__[key]
        with open(path+'/trainer.json', 'w') as outfile:
            json.dump(json_obj, outfile)
        self.env.save_json(path)

   
