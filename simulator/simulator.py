import os
import numpy as np
import pandas as pd
from collections import deque
from itertools import repeat, compress, product
from altair import Chart
from datetime import datetime
from dateutil import tz

class Agent():
    """This object chooses a road based on given probabilities."""
    def __init__(self, tag, thresholds, weights, trip, routes, change_percent, trpf_use_percent):
        self.tag = tag
        sorted_inds = np.argsort(np.array(thresholds))[::-1]
        self.thresholds = np.array(thresholds)[sorted_inds]
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
        congestion_levels = np.where(self.thresholds <= excess_car_counts[self.trip][self.last_choice], self.thresholds, 0)

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

def get_route_choices(agents, routes):
    """This function gets choices from agents and returns choices in an array with shape (trip_count, agent_count)
       and traveller counts in a dictionary with an array shaped (trip_route_count,) for each trip."""
    choices = np.array(list(map(lambda agent: agent.choose_route(), agents)))
    
    route_traveller_counts = {i[0]: np.zeros(len(i[1])) for i in routes.items()}
    for trip, route in choices:
        route_traveller_counts[trip][int(route)] += 1
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

def simulate(thresholds, weights, agent_counts, route_opts, routes, 
             round_count, road_params, change_percent, t, trpf_use_percent, 
             simulation_name, trip_names, outfile):

    trips = np.hstack([np.ones(agent_counts[i])*i for i in range(len(agent_counts))])

    roads = []
    road_inds = {}
    for i, params in enumerate(road_params.itertuples(index=False)):
        road = Road(*params)
        roads.append(road)
        road_name = params[0] + params[1] # Node 1 + Node 2
        road_inds[road_name] = i

    route_to_road = {i[0] : np.zeros((len(roads), len(i[1]))) for i in routes.items()}
    for trip, trip_routes in routes.items():
        for i, route in enumerate(trip_routes):
            for e in range(1, len(route)):
                try:
                    road_name = route[e-1] + route[e]
                    j = road_inds[road_name]
                except KeyError:
                    road_name = route[e] + route[e-1]
                    j = road_inds[road_name]

                route_to_road[trip][j, i] = 1

    road_to_route = {i[0]: i[1].T for i in route_to_road.items()}
    route_counts = {i[0]: len(i[1]) for i in road_to_route.items()}

    routes = {i[0]: np.arange(len(i[1])) for i in routes.items()}
    agents = []
    trpf_agents = []
    for tag, trip in zip(range(sum(agent_counts)), trips):
        agent = Agent(tag, thresholds, weights, trip, routes, change_percent, trpf_use_percent)
        agents.append(agent)
        if agent.uses_trpf():
            trpf_agents.append(agent)

    trpf = Trpf(routes, round_count, t)

    data = pd.DataFrame(columns = ['Trip', 'Route', 'Count', 'Cost', 'Trpf', 'Round'])
    data2 = pd.DataFrame(columns=['Trip', 'AverageCost', 'Round'])

    # print('G:{}, T:{}, P:{}'.format(change_percent, t, trpf_use_percent))

    while trpf.start_new_round():
        choices, route_traveller_counts = get_route_choices(agents, routes)

        excess_traveller_counts = {i[0]: i[1]-j[1] for i, j in zip(route_traveller_counts.items(), \
            route_opts.items())}

        route_costs = get_route_costs(roads, choices, route_traveller_counts, route_to_road, road_to_route)

        give_costs(agents, route_costs)

        get_reports(agents, trpf, choices, excess_traveller_counts)

        route_trpfs = trpf.calculate_trpf()
        give_trpfs(trpf_agents, route_trpfs)

        # Save the choices
        overall_cost = 0
        for trip, trip_routes in routes.items():
            total_trip_cost = 0
            for route in trip_routes:
                route_cost = route_costs[trip][route]
                route_trpf = route_trpfs[trip][route]
                route_traveller_count = route_traveller_counts[trip][route]

                total_trip_cost += route_cost * route_traveller_count

                data.loc[data.shape[0]] = [trip, route, route_traveller_count, \
                                                        route_cost, route_trpf, trpf.current_round]
            overall_cost += total_trip_cost
            average_trip_cost = total_trip_cost / agent_counts[trip]
            data2.loc[data2.shape[0]] = [trip, average_trip_cost, trpf.current_round]

        average_cost = overall_cost / sum(agent_counts)
        data2.loc[data2.shape[0]] = ['Overall', average_cost, trpf.current_round]

    # Drop the first 100 rounds
    first_hundred_rounds = 100 * len(trip_names)
    data_by_trips = data2[first_hundred_rounds:].drop(['Round'], axis=1).groupby('Trip')
    means = data_by_trips.mean()
    means.columns = ['CostMean']

    stds = data_by_trips.std()
    stds.columns = ['CostStd']

    stats = pd.concat([means, stds], axis=1)
    stats = stats.values.flatten()[np.newaxis,:]

    trips = trip_names
    aggrs = ['Average', 'Std']
    columns = pd.MultiIndex.from_product([trips, aggrs])
    weightstr = '|'.join(['{}:{}'.format(t, w) for t, w  in zip(thresholds, weights)])
    index = pd.MultiIndex.from_tuples([(weightstr, change_percent, t, trpf_use_percent)], names=['Thresholds', 'G', 'T', 'P'])

    report = pd.DataFrame(stats, index=index, columns=columns)

    if outfile:
        # Add the report to the excel
        try:
            xl = pd.read_excel(outfile, header=[0,1], index_col=[0,1,2,3], sheetname=simulation_name)
            xl.loc[weightstr, change_percent, t, trpf_use_percent] = report.values.flatten()
            writer = pd.ExcelWriter(outfile, engine='xlsxwriter')
            xl.to_excel(writer, sheet_name=simulation_name)
            writer.save()
        except:
            writer = pd.ExcelWriter(outfile, engine='xlsxwriter')
            report.to_excel(writer, sheet_name=simulation_name)
            writer.save()

    return data, data2, report

def run(properties, filename=None, show=False, save=True):
    if save:
        if filename:
            filename.split('.xlsx')[0] + '.xlsx'
        else:
            filename = (datetime.now(tz.tzutc()).strftime('%Y%m%dT%H%M%Sz_')
                        + properties['name'] + '.xlsx')
        outfile = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
                               '..', 'outputs', filename)
    else:
        outfile = None
    threshold_list = properties['thresholds']
    weight_list = properties['weights']
    agent_counts = properties['agents']
    routes = properties['routes']
    route_opts = properties['optimums']
    round_count = properties['steps']
    road_params = properties['graph']
    trip_names = properties['trips']
    simulation_name = filename.split('.xlsx')[0]
    values = list(product(properties['trpf']['g'],
                          properties['trpf']['t'],
                          properties['trpf']['p']))
   
    first = True
    for thresholds, weights in zip(threshold_list, weight_list): 
        for params in values:
            data, data2, report = simulate(thresholds, weights, agent_counts, 
                    route_opts, routes, round_count, road_params, *params, 
                    simulation_name, trip_names, outfile)
            if show:
                # The most disgusting code ever
                if first:
                    print('{:*>27} {}'.format(' ', ' '.join(['{:-^21}'.format(trip.upper()) for trip in trip_names])))
                    print('{:>14} {:>3} {:>3} {:>3} {}'.format(
                        'weights', 'G', 'T', 'P', 
                        ' '.join(['{:>10} {:>10}'.format('mean', 'std') 
                            for trip in trip_names])))
                index = '{:>14} {:>3} {:>3} {:>3}'.format(*report.index.values[0])
                results = ' '.join(['{:>10} {:>10}'.format(
                    report[trip, 'Average'].to_string(index=False, header=False), 
                    report[trip, 'Std'].to_string(index=False, header=False),
                    ) for trip in trip_names])
                print('{} {}'.format(index, results))
                first = False

if __name__ == '__main__':
    import netlist as nl
    properties = nl.load('example')
    run(properties, 'example', show=True, save=True)
