import cvxpy as cvx
import numpy as np

def optimize(properties):
    print(properties['graph'])
    df = properties['graph'].set_index(['node 1', 'node 2'])
    
    print(df.index.values)
   
    constraints = []
    drivers = [[cvx.Variable() for j in properties['routes'][i]] for i in properties['routes']]
    for i, trip in enumerate(properties['routes']):
        constraints.append(sum(drivers[i]) = properties['agents'][i])
    
    
    for i, trip in enumerate(properties['routes']):
        for j, route in enumerate(properties['routes'][trip]):
            print(route)
                           
                
    print(sum([df.loc[route[n], route[n+1]]['fft']
         + 100 * df.loc[route[n], route[n+1]]['ddt']
            for i, trip in enumerate(properties['routes'])
                for j, route in enumerate(properties['routes'][trip]) 
                    for n in range(len(route) - 1)])/3)
    obj = cvx.Minimize(
        sum([df.loc[route[n], route[n+1]]['fft']
             + drivers[i][j] * df.loc[route[n], route[n+1]]['ddt']
                for i, trip in enumerate(properties['routes'])
                    for j, route in enumerate(properties['routes'][trip]) 
                        for n in range(len(route) - 1)])
    )

    prob = cvx.Problem(obj, constraints)
    prob.solve(solver='SCS', verbose=True)
    print(prob.status)
    print(prob.value)
    for i, trip in enumerate(drivers):
        for j, route in enumerate(trip):
            print(drivers[i][j].value)

if __name__ == '__main__':
    import netlist as nl
    properties = nl.load('pigou_single')
    
    optimize(properties)

