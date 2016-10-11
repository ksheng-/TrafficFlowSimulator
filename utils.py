import numpy as np
import pandas as pd

class Agent():
    def __init__(self, tag, tresholds, weights, route_count, change_percent, trpf_use_percent):
        self.tag = tag
        sorted_inds = np.argsort(np.array(tresholds))[::-1]
        self.tresholds = np.array(tresholds)[sorted_inds]
        self.weights = np.array(weights)[sorted_inds]
        self.change_percent = change_percent
        self.route_count = route_count
        self.routes = np.arange(route_count)
        self.last_choice = np.random.randint(route_count)
        self.route_travel_counts = np.zeros(route_count)
        self.historic_route_costs = np.zeros(route_count)

        if np.random.rand() < trpf_use_percent:
            self._uses_trpf = True
            self.choose_route = self.__choose_trpf_route
            self.trpf = np.ones(route_count)
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
                return route_choice
            
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
                return route_choice

        else:
            route_choice = self.last_choice
            self.route_travel_counts[route_choice] += 1
            return route_choice

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
            return route_choice

        else:
            route_choice = self.last_choice
            self.route_travel_counts[route_choice] += 1
            return route_choice


    def recieve_travel_cost(self, new_cost):
        count = self.route_travel_counts[self.last_choice]
        previous_cost = self.historic_route_costs[self.last_choice]
        self.historic_route_costs[self.last_choice] = (previous_cost * (count-1) + new_cost) / count

    def report_congestion(self, excess_car_count):
        congestion_levels = np.where(self.tresholds <= excess_car_count, self.tresholds, 0)

        if congestion_levels[-1]: # Least treshold is bigger than excess_car_count
            congestion = congestion_levels.argmax()
            return self.weights[congestion]

        else:
            return 0

    def recieve_trpf(self, trpf):
        self.trpf = trpf

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


class Trpf(): # Check route ~ road
    def __init__(self, route_count, round_count, memory_length):
        self.route_count = route_count
        self.round_count = round_count
        self.memory_length = memory_length
        self.report_counts = np.zeros((round_count, route_count))
        self.reports = np.zeros((round_count, route_count))
        self.current_round = -1

    def start_new_round(self):
        self.current_round += 1

        if self.current_round == self.round_count:
            return False

        else:
            return True

    def recieve_report(self, route, report):
        self.report_counts[self.current_round, route] += 1
        self.reports[self.current_round, route] += report

    def _calculate_route_trpf(self, route):
        memory = self.current_round - self.memory_length
        route_report_sum = np.sum(self.reports[memory:self.current_round,route])
        route_report_count = np.sum(self.report_counts[memory:self.current_round,route])
        if route_report_count == 0:
            return 1
        else:
            route_trpf = 1 - route_report_sum / route_report_count
            return route_trpf
    
    def calculate_trpf(self):
        trpf = np.array(list(map(self._calculate_route_trpf,range(self.route_count)))) 
        return trpf

def read_config(config_file, road_params_file):
    with open(config_file, 'r') as f:
        line = next(f)

        while line[:4] != 'w = ':
            line = next(f)
        congestion_params = line.split(' = ')[1].split(', ')
        tresholds = []
        weights = []
        for params in congestion_params:
            tresholds.append(int(params.split(':')[0]))
            weights.append(int(params.split(':')[1]))

        while line[:4] != 'p = ':
            line = next(f)
        trpf_use_percent = float(line.split(' = ')[1])

        while line[:4] != 'T = ':
            line = next(f)
        t = int(line.split(' = ')[1])

        while line[:13] != 'num_agents = ':
            line = next(f)
        agent_count = int(line.split(' = ')[1])
       
        while line[:4] != 'G = ':
            line = next(f)
        change_percent = float(line.split(' = ')[1])

        while line[:5] != 'route':
            line = next(f)
        routes = []
        while line[:5] == 'route':
            routes.append(line.split(' = ')[1][:-1].split(','))
            line = next(f)

        while line[:5] != 'route':
            line = next(f)
        route_opts = []
        while line[:5] == 'route':
            route_opts.append(int(line.split(' = ')[1][:-1]))
            line = next(f)

        while line[:17] != 'num_iterations = ':
            line = next(f)
        round_count = int(line.split(' = ')[1])

    road_params = pd.read_csv(road_params_file, skiprows=0)

    config = {'tresholds': tresholds, 'weights' : weights, 'trpf_use_percent': trpf_use_percent, \
            't': t, 'agent_count': agent_count, 'change_percent': change_percent, 'routes': routes, \
            'route_opts': route_opts, 'round_count': round_count, 'road_params': road_params}

    assert len(tresholds) == len(weights)
    assert 0 <= trpf_use_percent <= 1
    assert t >= 0
    assert agent_count > 0
    assert 0 <= change_percent <= 1
    assert len(routes) == len(route_opts)
    assert round_count > 0

    return config
