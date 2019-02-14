import Demand as Demand
import LeadTime as ld
import numpy as np

DEFAULT_PARAMS = {
    'client_demand': Demand.Uniform_Demand(1,10,Step = 1),
    'lead_times':[ld.Uniform_LeadTime(2,3), ld.Uniform_LeadTime(2,3), 
              ld.Uniform_LeadTime(2,3), ld.Uniform_LeadTime(2,3)],
    'AI_possible_actions': np.arange(-10,11),
    'm' :1,
    'shortage_cost':4,
    'holding_cost':1,
    'initial_inventory':20,
    'number_periods':52,
    'use_backorders':1,
    'state_features':["IL" ,"d", "BO", "RS", "OO"],
    'AI_DN':[10,10]
}


