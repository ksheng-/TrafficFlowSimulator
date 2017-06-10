import ast
import shlex
import pandas as pd
import os

def loadsim(netlist):
    """Parse the netlist file, return simulation parameters."""
    nlpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
                          '..', 'netlists', netlist.split('.cfg')[0] + '.cfg')
   
    print(nlpath)
    properties = {}
    with open(nlpath, 'r') as f:
        
        graph = pd.DataFrame()
        
        for line in f:
            if line[0] == '#' or len(line.strip()) == 0:
                continue

            print(line.lower().strip())
            args = shlex.split(line.lower().strip())
            command = args.pop(0)

            kwargs = {arg.split('=')[0]: arg.split('=')[1]
                      for arg in args if '=' in arg}
            posargs = [arg for arg in args if '=' not in arg]
            print(posargs)
            print(kwargs)

            if command == 'type' and args[0] == 'multiod':
                pass
            else:
                pass
            
            if command == 'edge':
                if len(args) >= 2:
                    row = {'node 1': args[0], 'node 2': args[1]}
                    if 'fft' in kwargs:
                        row.update({'fft': kwargs['fft']})
                    if 'ddt' in kwargs:
                        row.update({'ddt': kwargs['ddt']})
                graph = pd.concat([graph, pd.DataFrame(row, index=[0], 
                    columns=['node 1', 'node 2', 'fft', 'ddt'])], 
                    ignore_index=True)
            
            if command == 'route':
                if len(args) >= 3:
                    args[0].split(',')

        print(graph)
                
                
    return properties


loadsim('example')
