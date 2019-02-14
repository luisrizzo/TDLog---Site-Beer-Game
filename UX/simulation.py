import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import re

import sys
sys.path.insert(0, '../Confidential/')

import BeerGame as BG
import Agent as Agent
import Environment as Env
import Demand as Demand
import Comparator as Comp
import Trainer as Tr
import Saver as Savers
import LeadTime as ld
from tools import *
import pickle

def get_history(demand_type,TS,lt, agent_pos,AI_choice):
    Mu = 10
    Sigma = 5
    periods = 40
    '''
    Je ne vois pas comment est-ce qu'on peut éviter de faire le if comme tu avais proposé Thierry.
    Dans un moment ou l'autre il va falloir lire la valeur dans demand_type et créer l'instance de la bonne classe
    Donc soit on doit faire ce lecture ici dans la création des classes soit dans une fonction dans le fichier des Demandes
    De toute le façon on aura un if avec tous les cas et ensuite la création de la demande avec les bon paramêtres
    '''
    if demand_type == "Seasonal":
        demand = Demand.Seasonal_Demand(15, 5, 0, 1.5, 0, Mu - 2, Sigma)
    elif demand_type == "Growing":
        demand = Demand.Growing_Demand(0,(2*Mu/periods), 0, Sigma)
    elif demand_type == "Sporadic":
        demand = Demand.Sporadic_Demand (Mu,0.2,5)
    elif demand_type == "Gaussian":
        demand= Demand.Gaussian_Demand(Mu, Sigma, min_value = 0, max_value = 100)
    elif demand_type =="Uniform":
        demand = Demand.Uniform_Demand(Mu,Mu+1,Step = 1)
    elif demand_type == "Growingseasonal":
        demand = Demand.Growing_Seasonal_Demand(1,[Mu*0.5,Mu* 0.8,Mu*0.7,Mu*0.9,Mu,Mu,Mu*0.9,Mu*1.2,Mu,Mu*1.1,Mu*1.5,Mu*2], Sigma)
    elif demand_type == "Mixedseasonal":
        demand = Demand.Mixed_Saisonnalities_Demand(Mu, [1,2,2,2,3,4,4,2,1,1,4,3,2,3,4,2,1,2,3,2,3,2,3,2,3],[0.6,0.8,0.7,0.9], Sigma)
    elif demand_type == "Growthstable":
        demand = Demand.Growth_Stable_Demand(0, 1, Mu + 5, Sigma)
    game_params = {
        'client_demand': demand,
        'lead_times':[ld.Constant_LeadTime(lt), ld.Constant_LeadTime(lt), 
                  ld.Constant_LeadTime(lt), ld.Constant_LeadTime(lt)],
        'AI_possible_actions': np.arange(-10,11),
        'm' :1,
        'shortage_cost':get_optimal_gaussian_SC(TS, Mu = Mu, Sigma= Sigma, lead_time=lt),
        'holding_cost':1,
        'initial_inventory': Mu*lt,
        'number_periods':periods,
        'use_backorders':1,
        'state_features':["IL" ,"d", "BO", "RS", "OO","t"],
        'AI_DN':[10,10],   # Not implemented yet
        'TS':TS
    }
    list_agents = ['BS20','BS20', 'BS20' , 'BS20']
    list_agents[agent_pos-1] = 'BNC'
    test_agents = generate_agents(list_agents, game_params)
    bg = BG.BeerGame()
    env = Env.Environment(test_agents, game_params)
    env.reset()
    bg.play(env, test_agents, train = False, display = False , display_all=False)

    # History
    #print("History keys : ", env.history.keys())
    #print(env.history['IL'])
    # IMPORTANT NOTE!! 
    # There are 4 agent 
    # They're indices are 1, 2 ,3 ,4
    #print("Agent's 1 stock history : ", env.history['IL'][:,1]) 
    #print("Agent's 2 stock history : ", env.history['IL'][:,2]) 
    #print("Agent's 3 stock history : ", env.history['IL'][:,3]) 
    #print("Agent's 4 stock history : ", env.history['IL'][:,4]) 
    return env.history

if __name__ == "__main__":
    print("Hi")