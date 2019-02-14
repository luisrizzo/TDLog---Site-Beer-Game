import numpy as np
import json
import pdb
import Agent as Agent
class Demand:
	def __init__(self, label = ""):
		self.label = label
	
	def generate(self, size):
		pass
	
	def display(self):
		pass

	def get_possible_demands(self):
		pass

	def save_json(self, path):
		with open(path+'/demand.json', 'w') as outfile:
			json.dump(self.__dict__, outfile)

	def load_json(self, path):
		pass

class Uniform_Demand_From_List(Demand):
	def __init__(self, possible_demands = [], label = "Uniform"):
		self.label = label
		self.possible_demands = possible_demands
		self.Min = np.min(possible_demands)
		self.Max = np.max(possible_demands)
		self.Mu = (self.Min + self.Max)/2

	def generate(self, size):
		return np.random.choice(self.possible_demands, size)

	def display(self):
		return "Possible demands "+str(self.possible_demands)

	def get_possible_demands(self):
		return self.possible_demands

	def bench_agent(self,agent_pos, TS,periods):
		agent_temp = Agent.BS_Agent_Gauss(agent_pos, 0, TS,self.Mu)

class Gaussian_Demand(Demand):
	def __init__(self, Mu, Sigma, min_value, max_value = 10, label = "Gaussian"):
		"""
		Initialize Variant Gaussian demand
		Args : 
			Mu : mean
			Sigma : standard deviation
			min_value : minimal value that could be generated
			max_value : maximal value that could be generated
			label : label of the demand
		"""
		Demand.__init__(self, label)
		assert (Mu >= min_value and Mu <= max_value)
		self.Mu = Mu
		self.Sigma = Sigma
		self.Max = max_value
		self.Min = min_value
		self.Step = 1


	def generate(self, size):
		"""
		Generates a list containing a demand for each period
		Args :
			size : size of the list that will be generated
		"""
		demands = []
	
		for i in range(size):
			demand = np.round(np.random.normal(self.Mu, self.Sigma))
			while demand < self.Min or demand > self.Max:
				demand = np.round(np.random.normal(self.Mu, self.Sigma))
			demands.append(demand)

		return demands
	
	def display(self):
		return "Gaussian demand (Mu = "+ str(self.Mu)+", Sigma = "+ str(self.Sigma)+")"
	

	def get_possible_demands(self):
		return np.arange(self.Min, self.Max, self.Step)
		
	def bench_agent(self,agent_pos, TS,periods):
		agent_temp = Agent.BS_Agent_Gauss(agent_pos, self.Sigma, TS,self.Mu)
		return agent_temp

class Variant_Gaussian_Demand(Demand):
	def __init__(self, Mu_list, Sigma_list, min_value = 0, max_value = 10, label = "Variant_Gaussian"):
		"""
		Initialize Variant Gaussian demand
		Args : 
			Mu_list : list of means
			Sigma_list : list of standard deviations
			min_value : minimal value that could be generated
			max_value : maximal value that could be generated
			label : label of the demand
		"""
		Demand.__init__(self, label)

		self.Mu_list = Mu_list
		self.Sigma_list = Sigma_list
		self.Max = max_value
		self.Min = min_value
		self.Step = 1
		

	def generate(self, size):
		"""
		Generates a list containing a demand for each period
		Args :
			size : size of the list that will be generated
		"""
		Mu = np.random.choice(self.Mu_list)
		Sigma = np.random.choice(self.Sigma_list)

		gaussian_demand = Gaussian_Demand(Mu, Sigma, self.Min, Mu + self.Max)
		return gaussian_demand.generate(size)

	def display(self):
		return "Variant Gaussian demand"
	

	def get_possible_demands(self):
		return np.arange(self.Min, self.Max, self.Step)

	def bench_agent(self,agent_pos, TS,periods):
		agent_temp = Agent.BS_Agent_Gauss(agent_pos, np.mean(self.Sigma_list), TS,np.mean(self.Mu_list))
		return agent_temp

class Uniform_Demand(Demand):
	def __init__(self, Min, Max, Step = 1, label = "Uniform"):
		"""
		Initialize uniform demand
		Args : 
			Min : minimal value that could be generated
			Max : maximal value that could be generated
			Step : step between two consecutive possible values 
			label : label of the demand
		"""
		Demand.__init__(self, label)
		self.Min = Min
		self.Max = Max
		self.Step = Step
		self.Mu = (Min + Max)/2
	def generate(self, size):
		"""
		Generates a list containing a demand for each period
		Args :
			size : size of the list that will be generated
		"""
		possible_demands = np.arange(self.Min, self.Max, self.Step)
		n = len(possible_demands)
		return possible_demands[np.random.randint(0,n, size)]
	
	def display(self):
		return "Uniform demand (min = "+ str(self.Min)+", max = "+ str(self.Max)+", step = "+str(self.Step)+")"

	def get_possible_demands(self):
		return np.arange(self.Min, self.Max, self.Step)

	def bench_agent(self,agent_pos, TS,periods):
		agent_temp = Agent.BS_Agent_Gauss(agent_pos, 0, TS,self.Mu)
		return agent_temp

class Seasonal_Demand(Demand):
	def __init__(self, peak_period_0,peak_duration,transition_periods, multiplier_peak, multiplier_transition, demand_base,Sigma, label = "Seasonal"):
		"""
		Initialize seasonal demand
		Args : 
			peak_period_0 : The period where the peak of the seasonal demand begins
			peak_duration : The legnth of the peak period of the seasonal demand
			transition_periods : Number of periods during the transition of the seasonality. Before and after the peak season
			multiplier_peak : multiple of the demand base in the peak periods
			multiplier_transition : multiple of the demand base in the transition periods
			demand_base : mean demand in normal periods
			Sigma : standard deviation in all periods
			label : demands label
		"""
		Demand.__init__(self, label)
		self.peak_0 = peak_period_0
		self.peak = peak_duration
		self.transition = transition_periods
		self.Step = 1
		self.Mu = demand_base
		self.multiplier_peak = multiplier_peak
		self.multiplier_transition = multiplier_transition
		self.Sigma = Sigma
		self.Min = self.Mu
		self.Max = self.Mu * self.multiplier_peak 
		self.demand_avg = []


	def generate (self, size):
		"""
		Generates a list containing a demand for each period
		Also generates a list that will have the averages (Mu's) of each period.
		This list is necessary to the creation of good benchmark agents
		Args :
			size : size of the list that will be generated
		"""
		demands = []
		self.demand_avg = []

		for i in range (size):
			if i < self.peak_0 - self.transition or i >= self.peak_0 + self.peak + self.transition:
				demand_avg = self.Mu
			elif i < self.peak_0:
				demand_avg = self.Mu * self.multiplier_transition
			elif i < self.peak_0 + self.peak :
				demand_avg = self.Mu * self.multiplier_peak
			elif i < self.peak_0 + self.peak + self.transition:
				demand_avg = self.Mu * self.multiplier_transition

			demand = np.round(np.random.normal(demand_avg, self.Sigma))
			
			demands.append(demand)
			self.demand_avg.append(demand_avg)

		return demands

	@property
	def demand_list(self):
		return self.demand_avg
	
	def display(self):
		return "Seasonal demand with peak in the periods "+str(self.peak)+" by a factor of "+str(self.multiplier_peak)+" with a transition period of "+str(self.transition)

	def get_possible_demands (self):
		return np.arange(self.Min, self.Max, self.Step)

	def bench_agent(self,agent_pos, TS,periods):
		self.generate(periods)
		agent_temp = Agent.BS_Agent_Seas(agent_pos, self.Sigma, TS,self)
		return agent_temp

class Growing_Demand (Demand):
	def __init__(self, base_demand, growing_step_1, growing_step_2, Sigma, label = "growing"):
		"""
		Initialize growing demand
		Args : 
			base_demand : c coefficient
			growing_step_1 : b coefficient
			growing_step_2 : a coefficient
			demand = a t² + b t + c

			t = cycle element (in our example with cycles number and cycle durations we have months and weeks)
			Sigma : standard deviation used on the creation of gaussian random variables
			label :
		"""
		#courbes de la demande planifiés pour des périodes qui sont des semaines et taux de croissance mensuels
		Demand.__init__(self, label)
		self.base = base_demand
		self.growth2 = growing_step_2
		self.Step = 1
		self.growth1 = growing_step_1
		self.Sigma = Sigma
		self.max_nb_cycles = 12 #12 months
		self.cycle_duration = 4 #each month has 4 weeks

		self.Min = self.base
		self.Max = (self.base + (self.max_nb_cycles*self.max_nb_cycles)*self.growth2+ self.max_nb_cycles * self.growth1 )
		self.Mu = (self.Min + self.Max)/2
		self.demand_avg = []

	def generate (self, size):
		"""
		Generates a list containing a demand for each period
		Args :
			size : size of the list that will be generated
		"""
		demands = []
		self.demand_avg = []
		for i in range(size):
			demand_avg = (i//self.cycle_duration)*(i//self.cycle_duration)*self.growth2 + (i//self.cycle_duration	)*self.growth1 + self.base 
			demand = np.round(np.random.normal(demand_avg, self.Sigma))
			demands.append(demand)
			self.demand_avg.append(demand_avg)
		return demands

	@property
	def demand_list(self):
		return self.demand_avg

	def display(self):
		return "Growing demand with a rate per period of "+str(self.growth2)

	def get_possible_demands (self):
		return np.arange(self.Min, self.Max, self.Step)

	def bench_agent(self,agent_pos, TS,periods):
		self.generate(periods)
		agent_temp = Agent.BS_Agent_Seas(agent_pos, self.Sigma, TS,self)
		return agent_temp

class Sporadic_Demand (Demand):
	def __init__(self, base_demand, probability, multiplier, label = "Sporadic"):
		"""
		Initialize Sporadic demand
		Args : 
			base_demand : normal quantity ordered when there is an order
			probability : probabilty of each period to have a order in place
			multiplier : something to simulate bigger orders that can arrive
			label :
		"""
		Demand.__init__(self, label)
		self.bdemand = base_demand
		self.prob = probability
		self.multi = multiplier
		self.Min = 0
		self.Max = self.bdemand * self.multi
		self.Step = 1
		self.Mu = base_demand * self.prob * self.multi
		self.demand_avg = []

	def generate (self, size):
		"""
		Generates a list containing a demand for each period
		Args :
			size : size of the list that will be generated
		"""
		demands = []
		for i in range (size):
			if np.random.random() < self.prob:
				demand = self.bdemand * (np.random.random() * self.multi + 1)
			else :
				demand = 0
			demands.append(demand)
		return demands
	
	def display(self):
		return "Sporadic demand with a demand probability of "+str(self.prob)

	def get_possible_demands (self):
		return np.arange(0, self.bdemand*self.multi, self.Step)

	def bench_agent(self,agent_pos, TS,periods):
		self.generate(periods)
		agent_temp = Agent.BS_Agent_Gauss(agent_pos, (self.Mu*self.Mu)/4, TS,self.Mu)
		return agent_temp

class Growing_Seasonal_Demand (Demand):
	def __init__(self,linear_growth, whole_period_seas, sigma, label = "growing_seasonal"):
		"""
		Initialize Growing Seasonal demand
		Args : 
			linear_growth : growth by cycle element
			whole_period_seas : list with cycles seasonalities coefficients
			sigma : standard deviation
			label :
		"""
		#courbes de la demande planifiés pour des périodes qui sont des semaines et taux de croissance mensuels
		#linear growth = int, whole_period_seas = liste de mus, sigma = ecart type moyen
		Demand.__init__(self, label)
		self.max_nb_cycles = len(whole_period_seas)
		self.growth = linear_growth
		self.Sigma = sigma
		self.cycle_mu = []
		for cycle in range(self.max_nb_cycles):
			self.cycle_mu.append(whole_period_seas[cycle]*(1+self.growth*cycle))
		self.Min = min(self.cycle_mu)
		self.Max = max(self.cycle_mu)
		self.demand_avg = []
		self.Step = 1


	def generate (self, size):
		"""
		Generates a list containing a demand for each period
		Args :
			size : size of the list that will be generated
		"""
		demands = []
		self.demand_avg = []
		cycle_duration = np.round(size/self.max_nb_cycles)+1
		for i in range(size):
			demand_avg = self.cycle_mu[int(i//cycle_duration)]
			demand = np.round(np.random.normal(demand_avg, self.Sigma)) # Corrected division par 12 au lieu de 4, à valider !
			demands.append(demand)
			self.demand_avg.append(demand_avg)
		return demands

	@property
	def demand_list(self):
		return self.demand_avg

	def display(self):
		return "Growing demand with monthly seasonal coefficients"

	def get_possible_demands (self):
		return np.arange(self.Min, self.Max, self.Step)

	def bench_agent(self,agent_pos, TS,periods):
		self.generate(periods)
		agent_temp = Agent.BS_Agent_Seas(agent_pos, self.Sigma, TS,self)
		return agent_temp

class Mixed_Saisonnalities_Demand (Demand):
	def __init__(self,base_demande, year_seas, monthly_seas, sigma, label = "mixed_seasonal"):
		"""
		Initialize Mixed Saisonnalities demand
		Args : 
			base_demande : base demand
			year_seas : coefficients for each month in a yearly point of view
			monthly_seas : coefficients for each week in monthly point of view
			sigma : standard deviation
			label :
		"""
		#courbes de la demande planifiés pour des périodes qui sont des semaines et taux de croissance mensuels
		#12 mois dans year_seas et 4 semaines dans monthly seas
		#donc j'utilise le numero du periode et la division par 4 comme inputs
		#le resultat entier du nb periode/4 (utilisant la formule i//4) donne l'indice du mois
		#pareil i%4 donne la periodicité intérior à un mois
		Demand.__init__(self, label)
		self.base = base_demande
		self.Sigma = sigma
		self.yearly = year_seas
		self.monthly = monthly_seas
		self.Min = self.base * min (self.yearly) * min (self.monthly)
		self.Max = self.base * max (self.yearly) * max (self.monthly)
		self.max_nb_cycles = len(year_seas)
		self.cycle_duration = len(monthly_seas)
		self.demand_avg = []
		self.Step = 1

	def generate (self, size):
		"""
		Generates a list containing a demand for each period
		Args :
			size : size of the list that will be generated
		"""
		demands = []
		self.demand_avg = []
		simulation = min(size,self.cycle_duration*self.max_nb_cycles)
		for i in range(simulation):
			demand_avg = self.base * self.yearly[i//self.cycle_duration] * self.monthly[i % self.cycle_duration]
			demand = np.round(np.random.normal(demand_avg, self.Sigma))
			demands.append(demand)
			self.demand_avg.append(demand_avg)
		return demands

	@property
	def demand_list(self):
		return self.demand_avg

	def display(self):
		return "Demand with monthly and yearly seasonality"

	def get_possible_demands (self):
		return np.arange(self.Min, self.Max, self.Step)

	def bench_agent(self,agent_pos, TS,periods):
		self.generate(periods)
		agent_temp = Agent.BS_Agent_Seas(agent_pos, self.Sigma, TS,self)
		return agent_temp

class Growth_Stable_Demand (Demand):
	def __init__(self,base_demande, growth, stability_period, sigma, label = "growing_stable"):
		"""
		Initialize Growing s demand
		Args : 
			base_demande : demande at time 0
			growth : steps of the growing demande
			stability_period : from this point onwards the demand remains stable
			sigma : standard deviation
			label :
		"""
		#taux de croissance par semaine
		Demand.__init__(self, label)
		self.base = base_demande
		self.Sigma = sigma
		self.stable = np.round(np.random.normal(stability_period, 4))
		self.growth = growth
		self.Min = self.base
		self.Max = self.base + self.stable * self.growth
		self.Mu = (self.Min + self.Max)/2
		self.demand_avg = []
		self.Step = 1

	def generate (self, size):
		demands = []
		self.demand_avg = []
		for i in range(size):
			if i < self.stable :
				demand_avg = self.base + i * self.growth
				demand = max(0,np.round(np.random.normal(demand_avg, self.Sigma)))
			else:
				demand_avg = self.base + self.stable * self.growth 
				demand = max(0,np.round(np.random.normal(demand_avg, self.Sigma)))
			demands.append(demand)
			self.demand_avg.append(demand_avg)
		return demands

	@property
	def demand_list(self):
		return self.demand_avg

	def display(self):
		return "Demand with monthly and yearly seasonality"

	def get_possible_demands (self):
		return np.arange(self.Min, self.Max, self.Step)

	def bench_agent(self,agent_pos, TS,periods):
		self.generate(periods)
		agent_temp = Agent.BS_Agent_Seas(agent_pos, self.Sigma, TS,self)
		return agent_temp