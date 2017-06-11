import ast, shlex, os, pprint, numbers
import pandas as pd

def load(netlist):
    """Parse the netlist file, return simulation parameters."""
    nlpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
                          '..', 'netlists', netlist.split('.ntl')[0] + '.ntl')
    
    properties = {
        'name': netlist.split('.ntl')[0], 
        'graph': pd.DataFrame(),
        'routes': {},
        'thresholds': [],
        'weights': [],
        'trpf': {
            'g': [], 
            't': [], 
            'p': []
        },
        'trips': [],
        'agents': [],
        'optimums': {},
        'steps': 200
    }   
    
    with open(nlpath, 'r') as f: 
        # One initial pass to count number of trips
        ntrips = 0
        for line in f:
            if line[0] == '#' or len(line.strip()) == 0:
               continue
            if shlex.split(line.lower().strip())[0] == 'trip':
                ntrips += 1
        properties['routes'] = {trip: [] for trip in range(ntrips)}
        properties['optimums'] = {trip: [] for trip in range(ntrips)}
        properties['trips'] = [None for trip in range(ntrips)]
        properties['trips'].append('Overall')
        properties['agents'] = [None for trip in range(ntrips)]
    
    with open(nlpath, 'r') as f:
        for line in f:
            if line[0] == '#' or len(line.strip()) == 0:
                continue

            args = shlex.split(line.lower().strip())
            command = args.pop(0)

            kwargs = {arg.split('=')[0]: arg.split('=')[1]
                      for arg in args if '=' in arg}
            posargs = [arg for arg in args if '=' not in arg]

            # TODO: don't think this needs to be specified
            if command == 'type' and args[0] == 'multiod':
                pass
            else:
                pass
            
            if command == 'edge':
                if len(args) >= 2:
                    row = {'node 1': args[0], 'node 2': args[1]}
                    if 'fft' in kwargs:
                        row.update({'fft': float(kwargs['fft'])})
                    else:
                        row.update({'fft': 0})
                    if 'ddt' in kwargs:
                        row.update({'ddt': float(kwargs['ddt'])})
                    else:
                        row.update({'ddt': 0})
                properties['graph'] = pd.concat(
                    [properties['graph'], pd.DataFrame(row, index=[0], 
                        columns=['node 1', 'node 2', 'fft', 'ddt'])], 
                    ignore_index=True)
             
            if command == 'trip':
                properties['routes'][int(kwargs['index'])] \
                    = [[] for i in range(int(kwargs['routes']))] 
                properties['optimums'][int(kwargs['index'])] \
                    = [None for i in range(int(kwargs['routes']))]
                properties['trips'][int(kwargs['index'])] \
                    = kwargs['name'].upper()
                properties['agents'][int(kwargs['index'])] \
                    = int(kwargs['agents'])

            if command == 'route':
                trip = int(kwargs['trip'])
                route = int(kwargs['route'])
                properties['routes'][trip][route] = args[0].split(',')
                properties['optimums'][trip][route] = int(kwargs['so'])

            if command == 'weights':
                properties['thresholds'].append([0 for bp in args])
                properties['weights'].append([0 for bp in args])
                for i, breakpoint in enumerate(args):
                    cars, weight = breakpoint.split(':')
                    properties['thresholds'][-1][i] = int(cars)
                    properties['weights'][-1][i] = int(weight)

            if command == 'trpf':
                if len(args) >= 3:
                    gvals = ast.literal_eval(kwargs['g'])
                    if isinstance(gvals, list):
                        properties['trpf']['g'] = gvals
                    elif isinstance(gvals, numbers.Number):
                        properties['trpf']['g'] = [gvals]
                    
                    tvals = ast.literal_eval(kwargs['t'])
                    if isinstance(tvals, list):
                        properties['trpf']['t'] = tvals
                    elif isinstance(tvals, numbers.Number):
                        properties['trpf']['t'] = [tvals]
                        
                    pvals = ast.literal_eval(kwargs['p'])
                    if isinstance(pvals, list):
                        properties['trpf']['p'] = pvals
                    elif isinstance(pvals, numbers.Number):
                        properties['trpf']['p'] = [pvals]
                
                if 'steps' in kwargs:
                    properties['steps'] = int(kwargs['steps'])

    return properties

def show(netlist):
    """ Print text of netlist file """
    nlpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
                          '..', 'netlists', netlist.split('.ntl')[0] + '.ntl')
        
    with open(nlpath, 'r') as f: 
        for line in f:
            print(line)


