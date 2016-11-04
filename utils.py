from collections import deque
from itertools import repeat
import numpy as np
import pandas as pd

class Agent():
    # choose_route method is defined inside __init__
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
        
        if decide <  self.change_percent:
            if decide < 0.05:
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

    def report_congestion(self, excess_car_count):
        congestion_levels = np.where(self.tresholds <= excess_car_count, self.tresholds, 0)

        if congestion_levels[-1]: # Least treshold is bigger than excess_car_count
            congestion = congestion_levels.argmax()
            return self.weights[congestion]

        else:
            return 0

    def recieve_trpf(self, trpfs):
        self.trpf = trpfs[self.trip]

class Road():
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
        memory = self.current_round - self.memory_length
        route_report_sum = np.sum(self.reports[trip][memory:self.current_round+1,route])
        route_report_count = np.sum(self.report_counts[trip][memory:self.current_round+1,route])
        if route_report_count == 0:
            return 1
        else:
            route_trpf = 1 - route_report_sum / route_report_count
            return route_trpf
    
    def calculate_trpf(self):
        new_trpf = {np.array(list(map(self._calculate_route_trpf,repeat(i[0]), i[1]))) \
            for i in self.routes.items()} 
        #self.trpf_history.append(new_trpf)                      #Commented out the moving average
        #trpf = np.average(self.trpf_history, axis=0)
        #if self.current_round > 200:
        #   return trpf
        #else:
        return new_trpf

def read_config(config_file, road_params_file):
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
        
        while line[:4] != 'p = ':
            line = next(f)
        trpf_use_percent = float(line.split(' = ')[1])
        assert 0 <= trpf_use_percent <= 1
        
        while line[:4] != 'T = ':
            line = next(f)
        t = int(line.split(' = ')[1])
        assert t >= 0
        
        while line[:10] != 'num_agents':
            line = next(f)
        
        agent_counts = []
        while line[:10] == 'num_agents':
            agent_count = int(line.split('num_agents_')[1].split(' = ')[1])
            agent_counts.append(agent_count)
            assert agent_count >= 0
            line = next(f)
       
        while line[:4] != 'G = ':
            line = next(f)
        change_percent = float(line.split(' = ')[1])
        assert 0 <= change_percent <= 1

        while line[:5] != 'route':
            line = next(f)
        
        routes = {i: [] for i in range(len(agent_counts))}
        while line[:5] == 'route':
            trip = int(line.split('_')[1])
            routes[trip].append(line.split(' = ')[1][:-1].split(','))
            line = next(f)

        while line[:5] != 'route':
            line = next(f)
                                
        route_opts = {i: [] for i in range(len(agent_counts))}
        while line[:5] == 'route':
            trip = int(line.split('_')[1])
            opt = int(line.split(' = ')[1])
            route_opts[trip].append(opt)
            line = next(f)
        assert len(routes) == len(route_opts)

        while line[:17] != 'num_iterations = ':
            line = next(f)
        round_count = int(line.split(' = ')[1])
        assert round_count > 0
        
    road_params = pd.read_csv(road_params_file, skiprows=0)

    config = {'tresholds': tresholds, 'weights' : weights, 'trpf_use_percent': trpf_use_percent, \
            't': t, 'agent_counts': agent_counts, 'change_percent': change_percent, 'routes': routes, \
            'route_opts': route_opts, 'round_count': round_count, 'road_params': road_params}

    return config