import numpy as np
from tools import *
from IPython.display import clear_output

import Agent as Agent
import Environment as Env
import Demand as Demand
import pdb

class BeerGame:
	def __init__(self):
		pass

	def play(self, env, agents, train = True, display = False, get_data = False, display_all = False):
		"""
		Launches the Beergame simulation
		Args :
			env : 
			agents : 
			train : 
			display :
			get_data :
			display_all :
		Returns :
			X_train : 
		"""
		# Initialize figure
		if display:
			fig = Figure( n = 1)
			fig.add_subplot(111,x_lim = (1, env.T),
				title = "Evolution of costs (from all agents)",
						xlabel = "period",
						ylabel = "cost" )
			fig.add_graph(111, np.arange(env.T), np.sum(env.history['HC'], axis = 1), label = "Holding Costs (HC)")
			fig.add_graph(111, np.arange(env.T), np.sum(env.history['SC'], axis = 1), label = "Shortage Costs (SC)")
		
		# Game Loop
		s_ = []
		a_ = []
		for i, agent in enumerate(agents):
			env.update_agent_state(agent) 
			s_.append(env.get_state_features(agent))
			a_.append(agent.act(s_[i], train))
			env.act(agent, a_[i])
		
		if display:
			if not display_all:
				clear_output(wait=True)
			print(display_period(env,env.t))
			time.sleep(0.1)
		
		env.t += 1
		while not env.game_over():

			for i, agent in enumerate(agents):
				env.update_agent_state(agent)
				n_s = env.get_state_features(agent)
				r = env.get_state_history(agent)['r']
				n_a = agent.act(n_s, train)
				if train:
					agent.reinforce(s_[i], n_s, a_[i], n_a, r)
				
				env.act(agent, n_a)
				
				s_[i] = n_s
				a_[i] = n_a
			
			# Display results of one period
			if display:
				t = env.t
				if not display_all:
					clear_output(wait=True)
				print(display_period(env,t))
				time.sleep(0.1)

				# Update Figure data
				fig.update_graph(111, 0, x_data =  np.arange(1,t+2), y_data = np.sum(env.history['HC'][:t+1], axis = 1))
				fig.update_graph(111, 1, x_data =  np.arange(1,t+2), y_data = np.sum(env.history['SC'][:t+1], axis = 1))

				# Update Figure display
				fig.draw()
			env.t += 1
		
		# Save history as training data
		if get_data:
			X_train = []
			H = env.history

			for t in range(env.T):
				for agent in agents:
					X_train.append(agent.state_dict2arr(env.get_state_features(agent, t = t)))
			return np.array(X_train)
