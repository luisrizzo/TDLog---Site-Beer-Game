#for dtype in ('Gaussian_Demand','Variant_Gaussian_Demand','Uniform_Demand','Seasonal_Demand','Growing_Demand','Sporadic_Demand'):
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import re

import sys

import BeerGame as BG
import Agent as Agent
import Environment as Env
import Demand as Demand
import Comparator as Comp
import Trainer as Tr
import Saver as Saver
import LeadTime as ld
from tools import *

import pdb
import importlib

def reload_all():
    #%run tools.py
    importlib.reload(Agent)
    importlib.reload(Env)
    importlib.reload(Demand)
    importlib.reload(BG)
    importlib.reload(Comp)
    importlib.reload(Tr)
    importlib.reload(Saver)
    importlib.reload(ld)

def Use_Dicts(global_variables,demand_variables):
	for i in range(len(global_variables)):
		for j in range(len(demand_variables)):
			reload_all()

			#%matplotlib qt5

			# RE DEFINE PATH TO THE RESULTS FOLDER
			path = 'C:/Users/danie/Dropbox/BeerGame/'
			TS = global_variables[i]['TS']
			Mu = global_variables[i]['mu']
			Sigma = global_variables[i]['sigma']
			constant_ld = global_variables[i]['ltavg']
			periods = 40
			'''
			old way of using all demand types
			#il faut choisir le type de demand et les actions possibles
			#ensuite lancer avec les variations des autres variables avec le dictionnaire crée
			'''
			demand_type = global_variables[i]['demand_type']
			if demand_type == "Seasonal":
				demand = Demand.Seasonal_Demand(15, 5, 0, 1.5, 0, Mu - 2, Sigma)
			elif demand_type == "Growing":
				demand = Demand.Growing_Demand(0,(2*Mu/periods), 0, Sigma)
			elif demand_type == "Sporadic":
				demand = Demand.Sporadic_Demand(Mu,0.2,5)
				#demand.generate(periods)
				#bench_agent = Agent.BS_Agent_Gauss(1, Sigma, TS, Mu)
			elif demand_type == "Gaussian":
				demand= Demand.Gaussian_Demand(Mu, Sigma, min_value = 0, max_value = 100)
				#demand = Demand.Gaussian_Demand(global_variables[i]['Mu'],global_variables[i]['Sigma'],global_variables[i]['Min'],global_variables[i]['Max'])
			elif demand_type =="Uniform":
				demand = Demand.Uniform_Demand(Mu ,Mu,Step = 1)
			elif demand_type == "Growingseasonal":
				demand = Demand.Growing_Seasonal_Demand(1,[Mu*0.5,Mu* 0.8,Mu*0.7,Mu*0.9,Mu,Mu,Mu * 0.9,Mu*1.2,Mu,Mu*1.1,Mu*1.5,Mu*2], Sigma)
			elif demand_type == "Mixedseasonal":
				demand = Demand.Mixed_Saisonnalities_Demand(Mu, [1,1,2,2,2,3,4,4,2,1,1,4],[0.6,0.8,0.7,0.9], Sigma)
			elif demand_type == "Growthstable": 
				demand = Demand.Growth_Stable_Demand(0, 1, Mu + 5, Sigma)
			else:
				print("Did not recognize demand type")
				break
			bench_agent = demand.bench_agent(global_variables[i]['pos'],global_variables[i]['TS'],periods)
			game_params = {
			    'client_demand':demand,
			    'lead_times':[ld.Constant_LeadTime(global_variables[i]['lt'][0]), ld.Constant_LeadTime(global_variables[i]['lt'][1]), 
			              ld.Constant_LeadTime(global_variables[i]['lt'][2]), ld.Constant_LeadTime(global_variables[i]['lt'][3])],
			    'AI_possible_actions': np.arange(-10,10),
			    'm' : global_variables[i]['m'],
			    'shortage_cost':get_optimal_gaussian_SC(TS, Mu = Mu, Sigma= Sigma, lead_time=constant_ld),
			    'TS' : TS,
			    'holding_cost':1,
			    'initial_inventory':constant_ld * Mu + 2* Sigma,
			    'number_periods':periods,
			    'use_backorders':0,
			    'state_features':["IL" ,"d", "BO", "RS", "OO","t"],
			    'AI_DN':[10,10],   # Not implemented yet
			    'comparison_agent' : bench_agent
			}
			'''
						Alternatives to above to be more flexible
							{
						    'client_demand': demand,
						    'lead_times':(global_variables[i]['leadtimes'], global_variables[i]['leadtimes'], 
						              global_variables[i]['leadtimes'], global_variables[i]['leadtimes']),
						    'initial_inventory':global_variables[i]['leadtimes'].Mean*10,
							}
							Need to make changes to functions that creates the dictionary list
							'''
			list_agents = ['BS20','BS20', 'BS20' , 'BS20']
			list_agents[global_variables[i]['pos']] = 'DQN'
			agents = generate_agents(list_agents, game_params)
			trainer = Tr.Trainer(agents, game_params)
			comparator = trainer.generate_comparator(min_BS_level = 5, max_BS_level = 20)

			trainer.train2(400)

			AI_Agent = trainer.best_AI_agent #.get_AI_agent()
			AI_Agent.label = 'best DQN'

			comparator.update_AI_Agents([trainer.best_AI_agent, trainer.get_AI_agent()])
			comparator.launch_comparison()

			comparator.histograms()
			comparator.one_game_results([trainer.get_AI_agent()])

			importlib.reload(Saver)
			saver = Saver.Saver(path)
			saver.clean_results_folder()
			saver.save(trainer)

def dicts_TDLog():
	demand_variables = []
	global_variables = []
	for lead_time in (1,3,6):
		for position in (1,2,3,4):
			for demand_type in ("Gaussian",):
				for TS in ('80%','95%','99%'):
					i = 0

def dicts_main():
	global_variables = []
	for lead_time in (1,3,6):
		for lead_time_distri in ((1,1,2,1),(2,1,1,1),(1,3,1,1),(1,1,1,3)):
			for position in (0,1,2,3):
				for sigmamu in ((10,1),(10,2),(10,5)):
					for m in (1,2,3):
						for demand_type in ("Seasonal","Growing","Sporadic","Gaussian","Growingseasonal","Mixedseasonal","Growthstable"):
							temp_dict = {
								'ltavg':lead_time,
								'lt' : [x * lead_time for x in lead_time_distri],
								'pos' : position,
								'm': m,
								'mu':sigmamu[0],
								'sigma':sigmamu[1],
								'demand_type':demand_type,
								'TS': 0.95
							}
							global_variables.append(temp_dict)
	demand_variables = []
	demand_variables.append(5)
	return global_variables,demand_variables

						
