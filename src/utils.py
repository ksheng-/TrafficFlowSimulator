from collections import deque
from itertools import repeat
import numpy as np
import pandas as pd

class Agent():
    """This object chooses a road based on given probabilities."""
    def __init__(self, tag, tresholds, weights, trip, routes, change_percent, trpf_use_percent):
        self.tag = tag
        sorted_inds = np.argsort(np.array(tresholds))[::-1]
        self.tresholds = np.array(tresholds)[sorted_inds]
        self.weights = np.array(weights)[sorted_inds]
        self.change_percent = change_percent
        self.route_count = len(routes[trip])
        self.trip = trip
        self.routes = np.arange(self.route_count)
        self.last_choice = np.random.randint(self.route_count)
        self.route_travel_counts = np.zeros(self.route_count)
        self.historic_route_costs = np.zeros(self.route_count)
        
        # choose_route method is defined here
        if np.random.rand() < trpf_use_percent:
            self._uses_trpf = True
            self.choose_route = self.__choose_trpf_route
            self.trpf = np.ones(self.route_count)
        else:
            self._uses_trpf = False
            self.choose_route = self.__choose_rand_route

    def uses_trpf(self):
        return self._uses_trpf

    def __choose_rand_route(self):
        decide = np.random.rand()
        explore = np.random.rand()
        if decide <  self.change_percent:
            if explore < 0.05:
                route_choice = np.random.randint(self.route_count)
                self.route_travel_counts[route_choice] += 1
                self.last_choice = route_choice
                return self.trip, route_choice
            
            else:
                min_cost = np.amin(self.historic_route_costs)
                excess_route_costs = self.historic_route_costs - np.ones(self.route_count) * min_cost

                options = []
                for i, option in enumerate(np.absolute(excess_route_costs) < 0.1):
                    if option:
                        options.append(i)

                route_choice = np.random.choice(options)
                self.route_travel_counts[route_choice] += 1
                self.last_choice = route_choice
                return self.trip, route_choice

        else:
            route_choice = self.last_choice
            self.route_travel_counts[route_choice] += 1
            return self.trip, route_choice

    def __choose_trpf_route(self):
        decide = np.random.rand()

        if decide < self.change_percent:
            max_trpf = np.amax(self.trpf)

            options = []
            for i, option in enumerate(self.trpf == max_trpf):
                if option:
                    options.append(i)

            route_choice = np.random.choice(options)
            self.route_travel_counts[route_choice] += 1
            self.last_choice = route_choice
            return self.trip, route_choice

        else:
            route_choice = self.last_choice
            self.route_travel_counts[route_choice] += 1
            return self.trip, route_choice


    def recieve_travel_cost(self, route_costs):
        new_cost = route_costs[self.trip][self.last_choice]
        previous_cost = self.historic_route_costs[self.last_choice]
        count = self.route_travel_counts[self.last_choice]        
        self.historic_route_costs[self.last_choice] = (previous_cost * (count-1) + new_cost) / count

    def report_congestion(self, excess_car_counts):
        congestion_levels = np.where(self.tresholds <= excess_car_counts[self.trip][self.last_choice], self.tresholds, 0)

        if congestion_levels[-1]: # Least treshold is bigger than excess_car_count
            congestion = congestion_levels.argmax()
            return self.trip, self.last_choice, self.weights[congestion]

        else:
            return self.trip, self.last_choice, 0

    def recieve_trpf(self, trpfs):
        self.trpf = trpfs[self.trip]

class Road():
    """This class simulates road congestion."""
    def __init__(self, start_node, end_node, alpha, beta):
        self.start_node = start_node
        self.end_node = end_node
        self.alpha = alpha
        self.beta = beta
        self._traveller_count = 0

    def add_travellers(self, count):
        self._traveller_count = count

    def report_cost(self):
        return self.alpha + self._traveller_count * self.beta


class Trpf():
    """This object assignes roads points so that trafic converges to a syste optimum."""
    def __init__(self, routes, round_count, memory_length):
        self.routes = routes
        self.round_count = round_count
        self.memory_length = memory_length
        self.report_counts = {i[0] : np.ones((round_count, len(i[1]))) for i in routes.items()}
        self.reports = {i[0] : np.zeros((round_count, len(i[1]))) for i in routes.items()}
        self.current_round = -1
        #self.trpf_history = deque(np.ones((5, route_count)), 5)  #Commented out the moving average

    def start_new_round(self):
        self.current_round += 1

        if self.current_round == self.round_count:
            return False

        else:
            return True

    def recieve_report(self, trip, route, report):
        self.report_counts[trip][self.current_round, route] += 1
        self.reports[trip][self.current_round, route] += report

    def _calculate_route_trpf(self, trip, route):
        memory = np.max(self.current_round - self.memory_length, 0)
        route_report_sum = np.sum(self.reports[trip][memory:self.current_round+1,route])
        route_report_count = np.sum(self.report_counts[trip][memory:self.current_round+1,route])
        if route_report_count == 0:
            return 1
        else:
            route_trpf = 1 - route_report_sum / route_report_count
            return route_trpf
    
    def calculate_trpf(self):
        new_trpf = {i[0]: np.array(list(map(self._calculate_route_trpf,repeat(i[0]), i[1]))) \
            for i in self.routes.items()} 
        #self.trpf_history.append(new_trpf)                      #Commented out the moving average
        #trpf = np.average(self.trpf_history, axis=0)
        #if self.current_round > 200:
        #   return trpf
        #else:
        return new_trpf

def read_config(config_file, road_params_file):
    """This function parses the configuration files."""
    with open(config_file, 'r') as f:
        line = next(f)

        while line[:4] != 'w = ':
            line = next(f)
        
        tresholds = []
        weights = []
        congestion_params = line.split(' = ')[1].split(', ')
        for params in congestion_params:
            tresholds.append(int(params.split(':')[0]))
            weights.append(int(params.split(':')[1]))
        assert len(tresholds) == len(weights)
        
        while line[:8] != 'p_values':
            line = next(f)
        
        if line[11] == '[':
            p_values = [float(p) for p in line.split(' = ')[1][1:-2].split(',')]
        elif line[11] == '(':
            params = [float(param) for param in line.split(' = ')[1][1:-2].split(',')]
            p_values = np.linspace(*params)
        else:
            p_values = [float(line.split(' = ')[1])]
        
        while line[:11] != 't_values = ':
            line = next(f)
        
        if line[11] == '[':
            t_values = [int(t) for t in line.split(' = ')[1][1:-2].split(',')]
        else:
            t_values = [int(line.split(' = ')[1])]

        while line[:4] != 'trip':
            line = next(f)
            
        trip_names = []
        while line[:4] == 'trip':
            trip_name = line.split('trip_')[1].split(' = ')[1][:-1]
            trip_names.append(trip_name)
            line = next(f)
       
        while line[:10] != 'num_agents':
            line = next(f)
        
        agent_counts = []
        while line[:10] == 'num_agents':
            agent_count = int(line.split('num_agents_')[1].split(' = ')[1])
            agent_counts.append(agent_count)
            assert agent_count >= 0
            line = next(f)
       
        while line[:11] != 'g_values = ':
            line = next(f)
        
        if line[11] == '[':
            g_values = [float(g) for g in line.split(' = ')[1][1:-2].split(',')]
        elif line[11] == '(':
            params = [float(param) for param in line.split(' = ')[1][1:-2].split(',')]
            g_values = np.linspace(*params)
        else:
            g_values = [float(line.split(' = ')[1])]

        while line[:5] != 'route':
            line = next(f)
        
        routes = {i: [] for i in range(len(agent_counts))}
        while line[:5] == 'route':
            trip = int(line.split('_')[1])
            routes[trip].append(line.split(' = ')[1][:-1].split(','))
            line = next(f)

        while line[:9] != 'route_opt':
            line = next(f)
                                
        route_opts = {i: [] for i in range(len(agent_counts))}
        while line[:9] == 'route_opt':
            trip = int(line.split('_')[2])
            opt = float(line.split(' = ')[1])
            route_opts[trip].append(opt)
            line = next(f)
        assert len(routes) == len(route_opts)

        while line[:17] != 'num_iterations = ':
            line = next(f)
        round_count = int(line.split(' = ')[1])
        assert round_count > 0
        
    road_params = pd.read_csv(road_params_file, skiprows=0)

    config = {'tresholds': tresholds, 'weights' : weights, 'g_values': g_values, 't_values': t_values, \
        'agent_counts': agent_counts, 'p_values': p_values, 'routes': routes, 'route_opts': route_opts, \
        'round_count': round_count, 'road_params': road_params, 'trip_names': trip_names}

    return config

def get_route_choices(agents, routes):
    """This function gets choices from agents and returns choices in an array with shape (trip_count, agent_count)
       and traveller counts in a dictionary with an array shaped (trip_route_count,) for each trip."""
    choices = np.array(list(map(lambda agent: agent.choose_route(), agents)))
    
    route_traveller_counts = {i[0]: np.zeros(len(i[1])) for i in routes.items()}
    for trip, route in choices:
        route_traveller_counts[trip][route] += 1
    #route_traveller_counts = np.bincount(choices[0,:], minlength = route_count)[:, np.newaxis]
    
    return choices, route_traveller_counts

def get_route_costs(roads, choices, route_traveller_counts, route_to_road, road_to_route):
    """This function takes the choices and returns the route costs in a dictionary with an array 
       shaped (trip_route_count,) for each trip."""
    trip_road_traveller_counts = {i[0]: np.dot(j[1], i[1]) \
        for i, j in zip(route_traveller_counts.items(), route_to_road.items())}
    
    road_traveller_counts = 0
    for i, trip_road_traveller_count in trip_road_traveller_counts.items():
        road_traveller_counts += trip_road_traveller_count
    
    list(map(lambda count, road: road.add_travellers(count), road_traveller_counts, roads))
    road_costs = np.array(list(map(lambda road: road.report_cost(), roads)))
    route_costs = {i[0]: np.dot( i[1], road_costs)  for i in road_to_route.items()}
    return route_costs

def give_costs(agents, route_costs):    
    """This functions gives the route costs to the agents."""
    list(map(lambda agent, route_costs: agent.recieve_travel_cost(route_costs), agents, repeat(route_costs)))
    
def get_reports(trpf_agents, trpf, choices, excess_traveller_counts):
    """This function gets the reports from agents and gives them to trpf."""
    reports = list(map(lambda agent, excess: agent.report_congestion(excess),\
        trpf_agents, repeat(excess_traveller_counts)))
    list(map(lambda report: trpf.recieve_report(*report), reports))
    
def give_trpfs(trpf_agents, route_trpfs):
    """This function gives the trpf scores to the agents."""
    list(map(lambda agent, trpf: agent.recieve_trpf(trpf), trpf_agents, repeat(route_trpfs)))
