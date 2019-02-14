import numpy as np

class LeadTime:
	def __init__(self, label = ""):
		self.label = label
	
	def generate(self, size):
		pass
	
	def display(self):
		pass

class Uniform_LeadTime:
	def __init__(self, Min, Max, Step = 1, label = "Uniform"):
		LeadTime.__init__(self, label)
		self.Min = Min
		self.Max = Max
		self.Mean = (Min + Max) / 2 
		self.Step = Step
		self.Mean = (self.Max + self.Min)/2

	@property
	def Moyenne(self):return self.Mean

	def generate(self, size):
		possible_lead_times = np.arange(self.Min, self.Max + 1, self.Step)
		n = len(possible_lead_times)
		return possible_lead_times[np.random.randint(0,n, size)]

	def display(self):
		return "U("+ str(self.Min)+", "+ str(self.Max)+")"

class Constant_LeadTime:
	def __init__(self, value, label = "Constant"):
		LeadTime.__init__(self, label)
		self.Max = value
		self.Min = value
		self.Mean = value
		self.value = value
		self.Step = 0
		self.Mean = value
	@property
	def Moyenne(self):return self.Mean

	def generate(self, size):
		return np.ones(size) * self.Max

	def display(self):
		return "C("+ str(self.value)+")"

class Guaussian_LeadTime:
	def __init__(self, Mu, Sigma, max_value = 4, min_value = 2, label = "Gaussian"):
		LeadTime.__init__(self, label)
		self.Mu = Mu
		self.Sigma = Sigma
		self.Max = max_value
		self.Min = min_value
		self.Mean = Mu
		self.Step = 1
		self.Mean = Mu
		
	@property
	def Moyenne(self):return self.Mean

	def generate(self, size):
		lead_times = []
	
		for i in range(size):
			lead_time = np.round(np.random.normal(self.Mu, self.Sigma))
			while lead_time < self.Min or demand > self.Max:
				lead_time = np.round(np.random.normal(self.Mu, self.Sigma))

			lead_times.append(lead_time)

		return lead_times


	def display(self):
		return "G("+ str(self.Mu)+", "+ str(self.Sigma)+")"
	
	