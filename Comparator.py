import numpy as np
import Environment as Env
import BeerGame as BG
import Agent as Agent
import matplotlib.pyplot as plt
import re 
from tools import *
import pdb

class ComparatorBis:

    def __init__(self,  agents, game_params):
        """
        Initialize Comparator object
        Args :
            agents : list of agents objects
            game_params : dictionnary of game parameters
            n_demands : number of demands we'll generate to make comparison on
        """

        self.AI_index = get_AI_index(agents) # Select automatically the position of the AI that will be compared
        self.AI_pos =  self.AI_index + 1 # The position in the chain begins from 1
        self.AI_Agent = agents[self.AI_index]

        # Printing comparison scheme
        print(color.BOLD+"Comparison scheme: ["+color.END, end = "")
        for i, agent in enumerate(agents):
            if i != self.AI_index: print(agent.label, end = "")
            else: print(color.RED+color.BOLD+"Agent to compare"+color.END,  end = "")
            if i < len(agents) - 1: print(end=" - ")
        print("]")

        
        # Init the environement and the game simulator
        self.env = Env.Environment(agents, game_params)
        self.beerGame = BG.BeerGame()
        self.game_params = game_params
        self.agents = clone_agents_list(agents)

        # Create the Agent to compare to depending on the demand type 
        self.demand = game_params['client_demand']
        
        self.CP_agent = game_params['comparison_agent']
        

        self.AI_performance = None
        self.CP_performance = None

    def generate_demands(self, n_demands = 50):
        """
        Generates a demand from the demand object
        This demand will be used for the comparing tests
        Args:
            n_demands : number of sets of demands that the AI and BS will be evaluated on
        """
        demands = []
        for i in range(n_demands):
            demands.append(self.demand.generate(self.game_params['number_periods']))
        return demands

    def launch_comparison(self, n_demands, verbose = True):
        """
        Launches performance evaluation for both AI and other BS agents 
        Args : 
            n_demand : number of sets of demands that the AI and BS will be evaluated on
        """
        if verbose:
            print("Launching Comparison...",end="")
        # Generate the demands

        self.all_demands = self.generate_demands(n_demands)

        self.AI_performance = self.evaluate_agent_performance(self.AI_Agent)
        self.CP_performance = self.evaluate_agent_performance(self.CP_agent)
        
        if verbose:
            print("Done")

        return self.AI_performance, self.CP_performance

    def evaluate_agent_performance(self,agent):
        """
        Evaluates an agent performance upon multiple demands and returns a dictionnary of performance
        Args :
            agent : agent that will be evaluated
        """

        assert agent.i == self.AI_pos, "verify the position of the agent to evaluate"
        
        self.agents[self.AI_index] = agent

        avg_costs = 0 
        avg_br_rate = 0
        avg_cvr_rate = 0
        avg_srv_rate = 0
        avg_sum_demand = 0

        N = len(self.all_demands)

        for i in range(N):
            self.env.reset(self.all_demands[i]) # reset the environment
            self.beerGame.play(self.env, self.agents, display = False, train = False)

            avg_costs += self.env.history['CC'][-1][self.AI_pos] / N
            avg_br_rate += self.env.get_breakdown_rate(agent) / N
            avg_cvr_rate += self.env.get_coverage_rate(agent) / N
            avg_srv_rate += self.env.get_service_rate(agent) / N
            avg_sum_demand += np.sum(self.env.history['d'][:,self.AI_pos]) / N
        
        perf = {
            'agent' : agent,
            'coverage_rate' : round(avg_cvr_rate,2),
            'breakdown_rate' : round(avg_br_rate,2),
            'service_rate' : round(avg_srv_rate,2),
            'costs' : round(avg_costs,2),
            'sum_demand' : avg_sum_demand,
            'evaluated' : True,
        }
        
        return perf

    def show_histogram(self):
        """
        Plots comparison histograms
        """
        nb_methods = 2
        
        x = np.arange(nb_methods)
        
        costs, cvr_rates, br_rates, labels = [], [], [], []
        
        
        costs = [self.AI_performance['costs'], self.CP_performance['costs']]
        cvr_rates = [self.AI_performance['coverage_rate'], self.CP_performance['coverage_rate']]
        br_rates = [self.AI_performance['breakdown_rate'], self.CP_performance['breakdown_rate']]
        labels = [self.AI_performance['agent'].label, self.CP_performance['agent'].label]
        
        histogram_fig = plt.figure(13)
        histogram_fig.clear()
        
        ax = histogram_fig.add_subplot(211)
        barlist = ax.bar(x, costs)
        barlist[0].set_color('orange')
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation = 45, ha="right", fontsize = 8)
        
        ax.set_ylabel("cost", fontsize = 10)
        ax.set_title("Comparing replenishment methods costs (mean of 50 games)", fontsize = 10)
        
        # set legend
        ax.bar([0], [0], label = "Artificial Intelligence", color = 'orange')
        ax.bar([0], [0], label = "Base Stock", color = 'C0')
        
        ax.legend(prop={'size': 8})

        bx = histogram_fig.add_subplot(212)
        bx.bar(x, cvr_rates, label='coverage rate')
        bx.set_title("Comparing replenishment methods coverage rates (mean of 50 games)", fontsize = 10)
        bx.bar(x, br_rates, label='breakdown rate', color='xkcd:cyan')
        bx.set_xticks(x)
        bx.set_xticklabels(labels, rotation = 45, ha="right", fontsize = 8)
        bx.set_xlabel("Methods", fontsize = 10)
        bx.set_ylabel("rate", fontsize = 10)
        bx.legend(prop={'size': 8})
        
        
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.5)
        plt.show()


class Comparator:
    """
    Compares an AI with a list of Base Stock Agents with different Levels
    Args:
        env : Environement
        pos : position of the agent to compare
        min_BS_level : minimum base stock level value in an agent to compare
        max_BS_level : maximum base stock level value in an agent to compare
    """
    def __init__(self, agents, game_params, min_BS_level = 0, max_BS_level = 15):

        self.AI_index = get_AI_index(agents)
        
        self.pos_agt_to_cpr = self.AI_index + 1        

        print(color.BOLD+"Comparison scheme: ["+color.END, end = "")
        for i, agent in enumerate(agents):
            if i != self.AI_index: print(agent.label, end = "")
            else: print(color.RED+color.BOLD+"Agent to compare"+color.END,  end = "")
            if i < len(agents) - 1: print(end=" - ")
        print("]")

        # Initialize agents who will play the game
        self.agents = agents
        self.agents[self.AI_index] = Agent.BS_Agent(index = self.AI_index+1, lead_time = game_params['lead_times'][self.AI_index])
        self.env = Env.Environment(agents, game_params)
        self.agents[self.AI_index] = None
        self.min_BS_level = min_BS_level
        self.max_BS_level = max_BS_level
        self.beerGame = BG.BeerGame()
        
        self.demand = game_params['client_demand']

        # Initialize base stock agents that will be compared
        self.BS_Agents_dict = {}
        for i in range(min_BS_level, max_BS_level + 1):
            new_agent = Agent.BS_Agent(index = self.pos_agt_to_cpr, BS_level = i)
            self.BS_Agents_dict[new_agent.label] = {
            'agent' : new_agent,
            'evaluated' : False
            }

        print(color.BOLD+"BS agents :"+color.END,i+1,"BS agents from",min_BS_level,'to', max_BS_level)
        
        self.game_params = game_params

        # Initialize AI Agents list
        self.AI_Agents_dict ={}
        
        self.histogram_fig = None
        self.game_fig = None

        #self.evaluate_all()
    
    def generate_demands(self, n_demands = 50):
        """
        Generates a demand from the demand object
        This demand will be used for the comparing tests
        Args:
            n_demands : number of sets of demands that the AI and BS will be evaluated on
        """
        demands = []
        for i in range(n_demands):
            demands.append(self.demand.generate(self.game_params['number_periods']))
        return demands
    
    def launch_comparison(self, n_demands = 50):
        print("Launching Comparison...",end="")


        self.all_demands = self.generate_demands(n_demands)

        for key, value in self.BS_Agents_dict.items():
            if not self.BS_Agents_dict[key]['evaluated']:
                self.BS_Agents_dict[key] = self.evaluate_agent(value['agent'])


        for key, value in self.AI_Agents_dict.items():
            if not self.AI_Agents_dict[key]['evaluated']:
                self.AI_Agents_dict[key] = self.evaluate_agent(value['agent'])

        print("done")

    def update_AI_Agents(self, AI_agents = []):
        # Initialize AI Agents list
        self.AI_Agents_dict ={}
        for AI_Agent in AI_agents:
            self.AI_Agents_dict[AI_Agent.label] = {
            'agent' : AI_Agent,
            'evaluated' : False
            }
            
    def evaluate_agent(self, agent, N = 50):
        assert agent.i == self.pos_agt_to_cpr, "verify the position of the agent to evaluate"
        
        self.agents[self.pos_agt_to_cpr-1] = agent
        costs = 0 
        br_rate = 0
        cvr_rate = 0
        sum_demand = 0
        #best_BS_cost = self.best_BS_cost()

        for i in range(N):
            self.env.reset(self.all_demands[i])
            #self.env.reset() # reset the environment
            self.beerGame.play(self.env, self.agents, display = False, train = False)

            costs += self.env.history['CC'][-1][self.pos_agt_to_cpr] / N
            br_rate += self.env.get_breakdown_rate(agent) / N
            cvr_rate += self.env.get_coverage_rate(agent) / N
            sum_demand += np.sum(self.env.history['d'][:,self.pos_agt_to_cpr]) / N
        
        agent_dict = {
            'agent' : agent,
            'coverage_rate' : round(cvr_rate,2),
            'breakdown_rate' : round(br_rate,2),
            'costs' : round(costs,2),
            'sum_demand' : sum_demand,
            'evaluated' : True,
            #'relative_perf' : costs/best_BS_cost
        }
        
        return agent_dict
        
    def histograms(self):
        nb_methods = len(self.BS_Agents_dict.keys()) + len(self.AI_Agents_dict.keys())
        
        x = np.arange(nb_methods)
        
        costs, cvr_rates, br_rates, labels = [], [], [], []
        
        best_BS = self.best_BS()
        worst_BS = self.worst_BS()
        
        for key, value in {**self.AI_Agents_dict, **self.BS_Agents_dict}.items():
            costs.append(value['costs'])
            cvr_rates.append(value['coverage_rate'])
            br_rates.append(value['breakdown_rate'])
            labels.append(value['agent'].label)
        
        self.histogram_fig = plt.figure(13)
        self.histogram_fig.clear()
        
        ax = self.histogram_fig.add_subplot(211)
        barlist = ax.bar(x, costs)
        for i in range(len(self.AI_Agents_dict.keys())):
            barlist[i].set_color('orange')
            
        barlist[labels.index(best_BS[0])].set_color('C2')
        barlist[labels.index(worst_BS[0])].set_color('C3')
        
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation = 45, ha="right", fontsize = 8)
        
        ax.set_ylabel("cost", fontsize = 10)
        ax.set_title("Comparing replenishment methods costs (mean of 50 games)", fontsize = 10)
        
        ax.bar([0], [0], label = "Artificial Intelligence", color = 'orange')
        ax.bar([0], [0], label = "Base Stock", color = 'C0')
        ax.bar([0], [0], label = "Best Base Stock", color = 'C2')
        ax.bar([0], [0], label = "Worst Base Stock", color = 'C3')
        
        ax.legend(prop={'size': 8})

        bx = self.histogram_fig.add_subplot(212)
        bx.bar(x, cvr_rates, label='coverage rate')
        bx.set_title("Comparing replenishment methods coverage rates (mean of 50 games)", fontsize = 10)
        bx.bar(x, br_rates, label='breakdown rate', color='xkcd:cyan')
        bx.set_xticks(x)
        bx.set_xticklabels(labels, rotation = 45, ha="right", fontsize = 8)
        bx.set_xlabel("Methods", fontsize = 10)
        bx.set_ylabel("rate", fontsize = 10)
        bx.legend(prop={'size': 8})
        
        
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.5)
        plt.show()
    
    def one_game_results(self, AI_agents = []):
        self.game_fig = plt.figure(15)
        self.game_fig.clear()
        
        ax = self.game_fig.add_subplot(311)
        bx = self.game_fig.add_subplot(312)
        cx = self.game_fig.add_subplot(313)
        
        demand = self.env.params['client_demand']
        unique_demand = demand.generate(self.env.T)
        
        best_BS = self.best_BS()
        worst_BS = self.worst_BS()
        
        compared_agents = {
            best_BS[0] : self.BS_Agents_dict[best_BS[0]],
            worst_BS[0] : self.BS_Agents_dict[worst_BS[0]]
        }
        
        for AI_agent in AI_agents:
            compared_agents[AI_agent.label] = {'agent':AI_agent}
        
        colors = ['C2', 'C3', 'C1', 'C0','C4','C5','C6']
        i = 0
        
        for key, value in compared_agents.items():
            self.env.reset(unique_demand) # reset the environment
            self.agents[self.pos_agt_to_cpr-1] = value['agent']
            
            self.beerGame.play(self.env, self.agents, display = False, train = False)
            H = self.env.history

            ax.plot(H['CC'][:,self.pos_agt_to_cpr], label = key, color=colors[i], linewidth=1)
            
            
            bx.plot(H['IL'][:,self.pos_agt_to_cpr], label = key, color=colors[i], linewidth=1)
            bx.plot(H['BO'][:,self.pos_agt_to_cpr], linestyle='dotted', color=colors[i], linewidth=1)
            
            
            #cx.plot(H["d"][:,self.pos], color='magenta', linestyle='--', linewidth=1)

            orders = H["a"][:,self.pos_agt_to_cpr] + H["d"][:,self.pos_agt_to_cpr]
            orders[orders < 0] = 0
            cx.plot(orders, label = key, color=colors[i], linewidth=1)
            
            i += 1
        
        cx.plot(H["d"][:,self.pos_agt_to_cpr], label = "Demand", color='magenta', linestyle='--', linewidth=1)

        ax.grid(linestyle='dotted')
        bx.grid(linestyle='dotted')
        cx.grid(linestyle='dotted')
        
        ax.set_xlabel("period", fontsize = 8)
        ax.set_ylabel("cost", fontsize = 8)
        
        bx.set_xlabel("period", fontsize = 8)
        bx.set_ylabel("quantity", fontsize = 8)
        
        cx.set_xlabel("period", fontsize = 8)
        cx.set_ylabel("order made", fontsize = 8)
        
        bx.plot(0, label = "backorders/shortage",linestyle='dotted', color = 'black', linewidth=1)
        #cx.plot(0, label = "Demand", linestyle='--', color = "magenta", linewidth=1)
        
        ax.set_title("Cumulated costs of each method", fontsize = 10) 
        bx.set_title("Inventory Leavel and backorders for each method", fontsize = 10)
        cx.set_title("Received demand and made order in each period", fontsize = 10)
        
        ax.legend(prop={'size': 8})
        bx.legend(prop={'size': 8})
        cx.legend(prop={'size': 8})
        
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.5)
        plt.show()
        
    def best_BS(self):
        best_costs = 10e6
        label = None
        stock_level = 0
        for key, value in self.BS_Agents_dict.items(): 
            if value['costs'] < best_costs:
                stock_level = value['agent'].BS_level
                best_costs = value['costs']
                label = key
                
        return label, stock_level, best_costs
    
    def best_BS_cost(self):
        best_costs = 10e6
        label = None
        stock_level = 0
        for key, value in self.BS_Agents_dict.items(): 
            if value['costs'] < best_costs:
                stock_level = value['agent'].BS_level
                best_costs = value['costs']
                label = key
                
        return best_costs

    def worst_BS(self):
        worst_costs = 0
        label = None
        stock_level = 0
        for key, value in self.BS_Agents_dict.items(): 
            if value['costs'] > worst_costs:
                stock_level = value['agent'].BS_level
                worst_costs = value['costs']
                label = key
                
        return label, stock_level, worst_costs

