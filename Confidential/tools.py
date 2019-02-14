import numpy as np
import os
import time
import matplotlib.pyplot as plt
import Agent as Agent
import BeerGame as BG
import Environment as Env
import re
import pdb
import LeadTime as ld


class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

class Figure:
    """
        n : index of the figure
        x_lim : a tuple (x1, x2) that defines the limit on x axis
        y_lim : a tuple (y1, y2) that defines the limit on y axis
        title : title of the figure
        x_label : label of axis x
        y_label ; label of axis y
    """
    def __init__(self, n = 1, title = "", label_sz = 10, title_sz = 10 , legend_sz = 8 ):
        """
            n : number of the figure
        """
        # Figure creation
        self.fig = plt.figure(n)
        self.fig.clear()
        
        # dictionaries with keys as position of the plots
        self.sub_plots = {}
        self.graph_limits = {}
        self.graphs = {}

        self.label_sz = label_sz
        self.title_sz = title_sz
        self.legend_sz = legend_sz
        
    def add_subplot(self, pos = 111 , x_lim = None, y_lim = None, title = "", xlabel = "", ylabel = ""):
        """
        Args :
            pos : position of the plot in the figure, exemple : 111
            x_lim : 
            y_lim : 
            title : 
            xlabel : 
            y_label :
        """
        ax = self.fig.add_subplot(pos)
        ax.grid(linestyle='dotted')

        # Set axis limits
        if x_lim : ax.set_xlim(x_lim[0], x_lim[1])
        if y_lim : ax.set_ylim(y_lim[0], y_lim[1])
        
        # Set title and labels
        ax.set_title(title)
        ax.set_xlabel(xlabel, fontsize = self.label_sz)
        ax.set_ylabel(ylabel, fontsize = self.label_sz)
        
        # List of graphs on the figure
        self.sub_plots[pos] = ax
        self.graph_limits[pos] = {'x_lim': [0,0], 'y_lim':[0,0]}
        self.graphs[pos] = []
        
    def add_graph(self, pos, x_data, y_data, label="", color=None, linestyle=None, linewidth=1):
        """ 
            Add a new graph to the graphs list 
            pos : position of the plot in the figure, exemple : 111
        """
        graph, = self.sub_plots[pos].plot(x_data, y_data, label=label, color=color, linestyle=linestyle, linewidth=linewidth)
        
        self.graphs[pos].append(graph)
        self.sub_plots[pos].legend(prop={'size': self.legend_sz})

        # update figure y limits
        self.update_figure_limits(pos,x_data, y_data)
        
    def update_figure_limits(self, pos, x_data, y_data):
        """
            pos : position of the plot in the figure, exemple : 111
        """
        y_min = min(y_data)
        y_max = max(y_data)
        if(y_min != y_max):
            if y_min < self.graph_limits[pos]['y_lim'][0]:
                self.graph_limits[pos]['y_lim'][0] = y_min
                #self.y_lim[0] = y_min

            if y_max > self.graph_limits[pos]['y_lim'][1]:
                self.graph_limits[pos]['y_lim'][1] = y_max
                #self.y_lim[1] = y_max

            self.sub_plots[pos].set_ylim(self.graph_limits[pos]['y_lim'][0], self.graph_limits[pos]['y_lim'][1])    
            
    def update_graph(self, pos, index = 0, x_data = None, y_data = None):
        """
        Updates a graph values with new data
        Args :
            pos : 
            index : index of the graph to update
            x_data : 
            y_data : 
        """
        self.graphs[pos][index].set_xdata(x_data)
        self.graphs[pos][index].set_ydata(y_data)
        
        self.update_figure_limits(pos, x_data, y_data)
    
    def fill_between(self, pos, x_data, y_data1, y_data2):
        """
        Fills a tunnel btw two graphs
        Args:
            pos : position of the figure
            x_data : data along axis x
            y_data1 : first data along axis 1
            y_data2 : first data along axis 2
        """
        self.sub_plots[pos].fill_between(x_data, y_data1, y_data2, color = "#fc690f20")

    def draw(self):
        """
        Draws all the figure graphs (Used after data updates)
        """
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
                
class FiguresDisplayer:
    def __init__(self, env):
        self.train_fig = None 
        self.one_agent_game_fig = None
        self.all_agents_game_fig = None
        self.env = env
        
        self.label_font_size = 10
        self.title_font_size = 10
        self.legend_font_size = 8
        
    def one_agent(self, agent, n = 100):
        """
        Plots the resluts of one agent
        Args : 
            agent : the agent instance we want to 
            n : 
        """
        H = self.env.history
        i = agent.i
        
        self.one_agent_game_fig = plt.figure(100)
        self.one_agent_game_fig.clear()
        ax = self.one_agent_game_fig.add_subplot(221)
        ax.plot(np.arange(1, self.env.T+1), H["IL"][:,i], label="Inventory Level", color='tab:orange', linewidth = 1)
        ax.plot(np.arange(1, self.env.T+1), H["BO"][:,i], label="Back Order", color='tab:orange', linestyle='dotted', linewidth = 1)
        ax.set_title("Inventory Level and Backorders in each period", fontsize = self.title_font_size)
        ax.set_xlabel("period", fontsize = self.label_font_size)
        ax.set_ylabel("quantity", fontsize = self.label_font_size)
        ax.legend(prop={'size': self.legend_font_size})
        ax.grid(linestyle='dotted')
        
        bx = self.one_agent_game_fig.add_subplot(222)
        bx.plot(np.arange(1, self.env.T+1), H["d"][:,i], label="received demand",color='tab:brown', linewidth = 1)
        orders = H["a"][:,i] + H["d"][:,i]
        orders[orders < 0] = 0
        bx.plot(np.arange(1, self.env.T+1), orders, label="order made",color='tab:olive', linewidth = 1)
        bx.set_title("Received demand and made order in each period", fontsize = self.title_font_size)
        bx.set_xlabel("period", fontsize = self.label_font_size)
        bx.set_ylabel("quantity", fontsize = self.label_font_size)
        bx.legend(prop={'size': self.legend_font_size})
        bx.grid(linestyle='dotted')

        cx = self.one_agent_game_fig.add_subplot(223)
        cx.plot(np.arange(1, self.env.T+1), H['HC'][:,i], color='tab:orange', label="Holding Costs", linewidth = 1)
        cx.plot(np.arange(1, self.env.T+1), H['SC'][:,i], color='tab:orange', label="Shortage Costs", linewidth = 1, linestyle='dotted')
        cx.set_title("holding and shortage costs in each period", fontsize = self.title_font_size)
        cx.set_xlabel("period", fontsize = self.label_font_size)
        cx.set_ylabel("cost", fontsize = self.label_font_size)
        cx.legend(prop={'size': self.legend_font_size})
        cx.grid(linestyle='dotted')

        dx = self.one_agent_game_fig.add_subplot(224)
        dx.plot(np.arange(1, self.env.T+1), H['CC'][:,i], label="Cumulated Costs",color='tab:orange', linewidth = 1)
        dx.set_title("Cumulated costs after each period", fontsize = self.title_font_size)
        dx.set_xlabel("period", fontsize = self.label_font_size)
        dx.set_ylabel("cost", fontsize = self.label_font_size)
        dx.legend(prop={'size': self.legend_font_size})
        dx.grid(linestyle='dotted')

        plt.tight_layout()
        plt.suptitle("Results of "+agent.label+" Agent after a game of "+str(self.env.T)+" periods")
        plt.subplots_adjust(left=None, bottom=None, right=None, top=0.9, wspace=0.3, hspace=0.4)
        plt.show()

    def all_agents(self, agents, n = 200):
        """
        Plots the results of all the agent
        Args : 
            agents : List of agents 
            n : 
        """
        H = self.env.history
        
        self.all_agents_game_fig = plt.figure(200)   
        self.all_agents_game_fig.clear()

        ax = self.all_agents_game_fig.add_subplot(311)
        for agent in agents:
            i = agent.i
            ax.plot(np.arange(1, self.env.T+1), H["IL"][:,i], label=agent.label+" agent", color='C'+str(i), linewidth = 1)
            #ax.plot(np.arange(1, env.T+1), H["BO"][:,i], label="Back Order", color='C'+str(i), linestyle='dotted', linewidth = 1)
            ax.plot(np.arange(1, self.env.T+1), H["BO"][:,i], color='C'+str(i), linestyle='dotted', linewidth = 1)


        ax.set_title("Inventory Level and Backorders for each agent in each period", fontsize = self.title_font_size)
        ax.set_xlabel("period", fontsize = self.label_font_size)
        ax.set_ylabel("quantity", fontsize = self.label_font_size)
        ax.legend(prop={'size': self.legend_font_size})
        ax.grid(linestyle='dotted')

        cx = self.all_agents_game_fig.add_subplot(312)
        for agent in agents:
            i = agent.i
            cx.plot(np.arange(1, self.env.T+1), H["HC"][:,i], label="HC", color='C'+str(i), linewidth = 1)
            cx.plot(np.arange(1, self.env.T+1), H["SC"][:,i], label="SC", color='C'+str(i), linestyle='dotted', linewidth = 1)

        cx.set_title("Holding and shortage costs for each agent in each period", fontsize = self.title_font_size)
        cx.set_xlabel("period", fontsize = self.label_font_size)
        cx.set_ylabel("cost", fontsize = self.label_font_size)
        cx.legend(prop={'size': self.legend_font_size})
        cx.grid(linestyle='dotted')

        bx = self.all_agents_game_fig.add_subplot(313)
        for agent in agents:
            i = agent.i
            bx.plot(np.arange(1, self.env.T+1), H["CC"][:,i], label="CC "+agent.label, color='C'+str(i), linewidth = 1)

        bx.set_title("Cumulated costs for each agent in each period", fontsize = self.title_font_size)
        bx.set_xlabel("period", fontsize = self.label_font_size)
        bx.set_ylabel("cost", fontsize = self.label_font_size)
        bx.legend(prop={'size': self.legend_font_size})
        bx.grid(linestyle='dotted')

        plt.tight_layout()
        plt.suptitle("Results of all Agents after a game of "+str(self.env.T)+" periods")
        plt.subplots_adjust(left=None, bottom=None, right=None, top=0.9,
                    wspace=0.3, hspace=0.4)
        plt.show()
    
    def init_train_fig2(self, N):
        """
        Initialize training figure
        Args : 
            N : 
        """
        self.N = N
        # Initialize figure to see evolution of total costs in each iteration
        self.train_fig = Figure( n = 20)

        # Initialize figure to see evolution of total costs in each iteration
        self.train_fig.add_subplot(111, x_lim = (0, N), title = "Evolution of Intelligent Agent costs while training", xlabel = "iteration", ylabel = "Total cost")
        self.train_fig.add_graph(111, np.arange(N),np.zeros(N), label = "Training AI costs", linewidth = 1, linestyle = 'solid', color = "#fc690f40")
        self.train_fig.add_graph(111, np.arange(N),np.zeros(N), label = "Smooth AI costs", linewidth = 1, color = "C1")
        
        # Add best and worst quartil
        self.train_fig.add_graph(111, np.arange(N),np.zeros(N), label = "best quartil", linewidth = 0.5, color = "C1", linestyle = "dotted")
        self.train_fig.add_graph(111, np.arange(N),np.zeros(N), label = "worst quartil", linewidth = 0.5, color = "C1", linestyle = "dotted")

        # Add other BS costs
        self.train_fig.add_graph(111, np.arange(N),np.zeros(N), label = "Optimal BS costs", linewidth = 1, color = "C3", linestyle = "solid")
        self.train_fig.add_graph(111, np.arange(N),np.zeros(N), label = "Optimal AI costs", linewidth = 1, color = "C0", linestyle = "solid")


    def update_train_fig2(self, i, train_costs, test_costs, comp_interval):
        """
        Updates graphs of training figure
        Args : 
            i : 
            train_costs :
            test_costs :
            comp_interval :
        """
        assert i <= self.N, "Error in size : "+str(i)

        #pdb.set_trace()

        # update AI agent costs and smooth costs
        self.train_fig.update_graph(111, 0, x_data = train_costs["iterations"][0:i], y_data = train_costs["AI_costs"][0:i])
        self.train_fig.update_graph(111, 1, x_data = train_costs["iterations"][0:i], y_data = train_costs["AI_smooth"][0:i])

        # update worst and best quartil
        self.train_fig.update_graph(111, 2, x_data = train_costs["iterations"][0:i], y_data = train_costs["AI_best_quartil"][0:i])
        self.train_fig.update_graph(111, 3, x_data = train_costs["iterations"][0:i], y_data = train_costs["AI_worst_quartil"][0:i])

        #self.train_fig.fill_between(111, x_data = train_costs["iterations"][0:i], y_data1 = train_costs["AI_best_quartil"][0:i], y_data2 = train_costs["AI_worst_quartil"][0:i])
        self.train_fig.fill_between(111, x_data = train_costs["iterations"][max(0,i-3):i], y_data1 = train_costs["AI_best_quartil"][max(0,i-3):i], y_data2 = train_costs["AI_worst_quartil"][max(0,i-3):i])
        # update bs agent costs
        #if i//comp_interval > 0:
        self.train_fig.update_graph(111, 4, x_data = test_costs["iterations"][0:i//comp_interval + 1], y_data = test_costs["BS"][0:i//comp_interval + 1])
        self.train_fig.update_graph(111, 5, x_data = test_costs["iterations"][0:i//comp_interval + 1], y_data = test_costs["AI"][0:i//comp_interval + 1])
        
        # Draw figure
        self.train_fig.draw()


    def init_train_fig(self, N, best_BS, worst_BS):
        """
        Initialize training figure
        Args : 
            N : 
            best_BS :
            worst_BS :
        """
        self.N = N
        # Initialize figure to see evolution of total costs in each iteration
        self.train_fig = Figure( n = 20)

        # Initialize figure to see evolution of total costs in each iteration
        self.train_fig.add_subplot(111, x_lim = (0, N), title = "Evolution of Intelligent Agent costs while training", xlabel = "iteration", ylabel = "Total cost")
        self.train_fig.add_graph(111, np.arange(N),np.zeros(N), label = "AI costs", linewidth = 1, linestyle = 'solid', color = "#fc690f40")
        self.train_fig.add_graph(111, np.arange(N),np.zeros(N), label = "Smooth AI costs", linewidth = 1, color = "C1")
        
        self.train_fig.add_graph(111, np.arange(N),np.zeros(N), label = "best quartil", linewidth = 1, color = "C1", linestyle = "dotted")
        self.train_fig.add_graph(111, np.arange(N),np.zeros(N), label = "worst quartil", linewidth = 1, color = "C1", linestyle = "dotted")


        self.train_fig.add_graph(111, np.arange(N),np.zeros(N), label = best_BS+" costs", linewidth = 1, color = "C2")
        self.train_fig.add_graph(111, np.arange(N),np.zeros(N), label = worst_BS+" costs", linewidth = 1, color = "C3")
        agent_costs = np.zeros(N)
        agent_smooth_costs = np.zeros(N)
    
    def update_train_fig(self, i, agent_costs, sm_agent_costs, agent_sm_worst, sm_agent_best,  best_BS_costs, worst_BS_costs):
        """
        Updates graphs of training figure
        Args : 
            i : 
            agent_costs :
            sm_agent_costs :
            agent_sm_worst :
            sm_agent_best :
            best_BS_costs :
            worst_BS_costs :
        """
        assert i <= self.N, "Error in size : "+str(i)
        N = self.N
          
        self.train_fig.update_graph(111, 0, x_data =  np.arange(1,i+1), y_data = agent_costs[0:i])
        self.train_fig.update_graph(111, 1, x_data =  np.arange(1,i+1), y_data = sm_agent_costs[0:i])

        self.train_fig.update_graph(111, 2, x_data =  np.arange(1,i+1), y_data = agent_sm_worst[0:i])
        self.train_fig.update_graph(111, 3, x_data =  np.arange(1,i+1), y_data = sm_agent_best[0:i])

        self.train_fig.update_graph(111, 4, x_data =  np.arange(1,N+1), y_data = np.ones(N)*best_BS_costs)
        self.train_fig.update_graph(111, 5, x_data =  np.arange(1,N+1), y_data = np.ones(N)*worst_BS_costs)
        self.train_fig.draw()



def display_period(env, period):
    """
    Displays One period states 
    Args :
        env : the game Environment
        period : the period t we want to idsplay
    """
    H = env.history
    t = period
    
    env_state = "\n================================== period : "+str(period+1)+"/"+str(env.T)+" =======================================\n\n"
    for i in range(1, env.NA+1) : 
        env_state += "         ____________"

    env_state += "\n"
    for i in range(1, env.NA+1) : 
        IL = three_digits_string(H['IL'][t,i])
        env_state += "         | IL = "+IL+" |"

    env_state += "\n"
    for i in range(1, env.NA+1) : 
        BO = three_digits_string(H['BO'][t,i])
        d = three_digits_string(H['d'][t,i])
        env_state += "  d = "+d+"| BO = "+BO+" |"

    env_state += "\n"
    for i in range(1, env.NA+1) :
        RS = three_digits_string(H['RS'][t,i])
        env_state += "-------->| RS = "+RS+" |"
    env_state += "\n"
    for i in range(1, env.NA+1) : 
        a = three_digits_string(H['a'][t,i])
        if int(a) >= 0:
            env_state += "         | a = "+a+"  |"
        else:
            env_state += "         | a = "+a+" |"
    env_state += "\n"
    
    OOs = H['OO'][t]

    for j in range(len(OOs[0])):

        # for each agent
        for i in range(1, env.NA+1) : 
            OO_str = three_digits_string(OOs[i,j])
            env_state += "         | OO"+str(j)+" = "+OO_str+"|"

        env_state += "\n"
  

    for i in range(1, env.NA+1) : 
        env_state += "         ____________"

    env_state += "\n"

    for i in range(1, env.NA+1) : 
        i = three_digits_string(i)
        env_state += "           agent "+i+" "

    env_state += "\n"

    return env_state

def print_progression(i, N, length =20, cp_time = None, costs = []):
    """
    Prints the progression of the training
    Args : 
        i :
        N :
        length :
        cp_time :
        costs :
    """
    assert i<=N, "Error in size : "+str(i)

    ratio = int((i/N) * length)
    s = "[" 
    for j in range(ratio):
        s += "="
    s+=">"
    for j in range(ratio, length):
        s+=" "
    s+="] "
    s+= str(int((i/N)*100)) + "%  (" +str(i)+"/"+str(N)+") " 
    
    if cp_time:
        s += "time remaining.. "+ str(int((N-i) * cp_time))+"s"

    if len(costs) > 0:
        s += " (min : " + str(round(np.min(costs[:i]), 1)) + ", mean : "+str(round(costs[i-1],1))+")"
    print(s, end='\r', flush=True)
    
def print_game_params(env=None, agents=[], actions=[]):
    """
    Prints game parameters 
    Args : 
        env :
        agents :
        actions : 
    """
    if len(actions) > 0:
        print(color.BOLD+"Number possible states :"+color.END, nb_possible_states(actions, env.params['client_demand'].get_possible_demands(), T = env.T, IL_START = env.params['initial_inventory']))
    if env:
        print(color.BOLD+"Demand :"+color.END, env.params['client_demand'].display())
        print(color.BOLD+"Number periods in a game :"+color.END, env.T)
    if len(agents) > 0:
        #print(color.BOLD+"Agents :"+color.END)
        #for agent in agents:
        #    print("\t"+"agent",agent.i,":",agent.label)

        print(color.BOLD+"Agents : "+color.END, end = "[")
        for i, agent in enumerate(agents):
            if i < len(agents) - 1:
                print(agent.label, end="  -  ")
            else : print(agent.label, end="")

        print("]")

  
def compute_possible_ILs(ILs, actions, demands, periods_left):
    """
    Generates a list of possible Inventory Levels 
    Args : 
        ILs : 
        actions : 
        demands :
        periods_left :
    Return :
        A list of possible inventory levels
    """
    if periods_left == 0:
        return np.unique(np.array(ILs))
    else:
        new_ILs = []
        for IL in ILs:
            for a in actions:
                new_IL = IL + a
                if new_IL >= 0 and new_IL not in ILs and new_IL not in new_ILs:
                    new_ILs.append(new_IL)
                    
            for d in demands:
                new_IL = IL - d
                if new_IL >= 0 and new_IL not in ILs and new_IL not in new_ILs:
                    new_ILs.append(new_IL)
        
        ILs = ILs + new_ILs
        return compute_possible_ILs(ILs, actions, demands, periods_left - 1)

def nb_possible_states(actions, demands, T = 40, IL_START = 20):
    """
    Computes number of possible states
    Args : 
        actions :
        demands : 
        T :
        IL_START :
    """
    RSs = []
    for a in actions:
        for d in demands:
            RSs.append(a+d)

    RSs = np.unique(np.array(RSs))
    RSs = RSs[np.unique(RSs)>=0]

    OOs = RSs

    ILs = compute_possible_ILs([IL_START], actions, demands, T)
    BOs = ILs
    return len(actions) * len(demands) * len(RSs) * len(OOs) * len(ILs)* len(BOs)

def three_digits_string(x):
    """
    Converts a digit into three digits three_digits_string
    Args : 
        x : the digit that will be converted 
    """
    if x // 100 > 0:
        return str(int(x))
    elif x // 10 > 0:
        return str(int(x))+" "
    else:
        return str(int(x))+"  "

def generate_agents(agents_labels=[], game_params = {}):
    """
    Generates a list of Agent instances from a list of labels
    Args : 
        agents_labels : list of strings
        games_params : a dictionnary of game parameters
    
    Return :
        agents : a list of Agent instances
    """
    agents = []
    if ("DQN" or "PG" or "Lin") in agents_labels:
        samples = generate_samples(game_params)
    for i, agent_label in enumerate(agents_labels):

        if len(re.findall("BS[0-9]+", agent_label)) > 0 :
            BS_level = int(re.findall("[0-9]+", agent_label)[0])
            agents.append(Agent.BS_Agent(index = i+1, BS_level = BS_level, lead_time = game_params['lead_times'][i]))

        elif agent_label == "BS" :
            agents.append(Agent.BS_Agent(index = i+1, lead_time = game_params['lead_times'][i]))

        elif agent_label == "RND" :
            agents.append(Agent.RND_Agent(index = i+1, RND_possible_actions = game_params['AI_possible_actions'], lead_time = game_params['lead_times'][i]))

        elif agent_label == "Lin":
            agents.append(Agent.RL_Lin_Approx_Agent(index = i+1, AI_possible_actions = game_params['AI_possible_actions'], lead_time = game_params['lead_times'][i], samples = samples))
        
        elif agent_label == "PG":
            agents.append(Agent.RL_PG_Agent(index = i+1, AI_possible_actions = game_params['AI_possible_actions'], lead_time = game_params['lead_times'][i], samples = samples))
        
        elif agent_label == "DQN":
            agents.append(Agent.RL_DQN_Agent(index = i+1, AI_possible_actions = game_params['AI_possible_actions'], lead_time = game_params['lead_times'][i], samples = samples))
        elif agent_label == "BNC":
            agents.append(game_params['client_demand'].bench_agent(i+1,game_params['TS'],game_params['number_periods']))
        else:
            assert False, "Error ! Can't create agent "+ agent_label

    return agents

def generate_samples(game_params, number_iterations = 200):
    """
    Generates samples of states based on multiple simulations (used for creating AI models)
    Args :
        game_params : a dictionnary of game parameters
        number_iterations : the number of simulations from where we're taking the samples
    """
    # initialize Random agents
    print("Generating samples from",number_iterations,"game iterations..",end=" ")
    agents_RND = generate_agents(['RND', 'RND', 'RND', 'RND'], game_params)
    env = Env.Environment(agents_RND, game_params)            
    beerGame = BG.BeerGame()
    
    X_tr = np.empty((0, env.nb_state_features))
    N = number_iterations

    for i in range(N):
        env.reset()
        X_tr_game = beerGame.play(env, agents_RND, train = False, display = False, get_data=True)
        X_tr = np.append(X_tr, X_tr_game, axis = 0)

    print("Done")
    samples = np.unique(X_tr,axis=0)
    print(samples.shape[0], "Samples generated")
    return samples

def clone_agents_list(agents):
    """
    Cloning an agents list
    Args : 
        agents : list of agents instances

    Return : a copy of the list of agents instances
    """
    agents_copy = []
    for agent in agents:
        agents_copy.append(agent.copy())

    return agents_copy

def get_AI_index(agents):
    """
    Looks for the AI agent in a list of agents instances
    Args :
        agents : list of agents intances
    """
    AI_index = 0
    while AI_index  < len(agents) and not agents[AI_index].is_AI_agent:
        AI_index += 1
    assert AI_index < len(agents), "No AI agent in list"
    return AI_index

def set_default_params(params):
    """
    Sets default parameters when they are not specified in the params dictionnary
    Args : 
        params : dictionnary of some parameters specified by the user
    Return :
        full_params : a dictionnary of complete parameters (completed by default values)
    """
    full_params = {}
    for key in DEFAULT_PARAMS.keys():
        if key in params.keys():
            full_params[key] = params[key]
        else:
            full_params[key] = DEFAULT_PARAMS[key]
    return full_params


from scipy.optimize import minimize
from scipy import stats

def fct_min(x_bs, Mu=10, Sigma=2, lead_time = 1, SC = 100):
    """
    Objective function to minimize in order to get optimal base over_stock_costs for a specified shortage cost
    Args :
        x_bs : the variable x as the base stock Level
        Mu : the mean of gaussian distribution
        Sigma ; the standard deviation of the faussian distribution
        SC : the shortage cost
    """
    Mu *= lead_time
    Sigma *= np.sqrt(lead_time)

    demand = np.arange(Mu - 3*Sigma, Mu + 3*Sigma + 1,0.01)
    probabilities = stats.norm.pdf(demand, Mu, Sigma) # list of probabilities

    over_stock_costs = (x_bs - demand)
    over_stock_costs[np.where(over_stock_costs < 0)] = 0

    shortage_costs = (demand - x_bs)*SC
    shortage_costs[np.where(shortage_costs < 0)] = 0
        
    return np.dot((shortage_costs + over_stock_costs),probabilities)


def get_optimal_gaussian_bs(Mu, Sigma,lead_time, SC):
    """
    Returns optimal BAse stock value for a given gaussian distribution and shortage cost
    Args :
        Mu : the mean of gaussian distribution
        Sigma ; the standard deviation of the faussian distribution
        SC : the shortage cost
    """
    res = minimize(fct_min, 1.2, args=(Mu,Sigma,lead_time,SC))
    return res.x , stats.norm.cdf(res.x, Mu *lead_time , Sigma*np.sqrt(lead_time))

def get_optimal_gaussian_SC(TS, Mu = 10, Sigma = 2,lead_time = 1, precision = 0.0001, verbose = False):
    """
    Returns optimal Shortage cost  for a given gaussian distribution and Taux de service
    Args :
        TS : TAux de service
        Mu : the mean of gaussian distribution
        Sigma ; the standard deviation of the faussian distribution
        precision : the precision of the result as it is computed using dichotomy
    """
    print("Computing Optimal cost for TS =",TS,"...",end="")
    min_SC = 1
    max_SC = 3000
    
    optimal_SC = min_SC + (max_SC - min_SC)/2
    
    res = minimize(fct_min, Mu, args=(Mu,Sigma,lead_time,optimal_SC))
    result_TS = stats.norm.cdf(res.x, Mu *lead_time , Sigma*np.sqrt(lead_time))
    
    while abs(result_TS - TS) > precision and (max_SC != min_SC):
        
        if result_TS > TS :
            max_SC = optimal_SC 
            optimal_SC = min_SC + (max_SC - min_SC)/2
            
        else:
            min_SC = optimal_SC
            optimal_SC = max_SC - (max_SC - min_SC)/2
        
        res = minimize(fct_min, Mu, args=(Mu,Sigma,lead_time,optimal_SC))
        result_TS = stats.norm.cdf(res.x, Mu *lead_time , Sigma*np.sqrt(lead_time))
        if verbose:
            print("[",min_SC,",",max_SC,"]",optimal_SC,result_TS,abs(result_TS - TS))
    print("Done ( SC = ",int(optimal_SC),")")
    
    return optimal_SC
