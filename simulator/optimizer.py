import cvxpy as cvx
import numpy as np

if __name__ == '__main__':
    import netlist as nl
    properties = nl.load('example')
    
    df = properties['graph'].set_index(['node 1', 'node 2'])
    print(df)
   
    constraints = []
    drivers = [[cvx.Variable() for j in properties['routes'][i]] for i in properties['routes']]
    for i, trip in enumerate(properties['routes']):
        constraints.append(sum(drivers[i]) == properties['agents'][i])
    
    
    for i, trip in enumerate(properties['routes']):
        for j, route in enumerate(properties['routes'][trip]):
            print(route)
                           
                
    print(sum([df.loc[route[n], route[n+1]]['fft'].iloc[0]
         + 100 * df.loc[route[n], route[n+1]]['ddt'].iloc[0] 
            for i, trip in enumerate(properties['routes'])
                for j, route in enumerate(properties['routes'][trip]) 
                    for n in range(len(route) - 1)]))
    obj = cvx.Minimize(
        sum([df.loc[route[n], route[n+1]]['fft'].iloc[0]
             + drivers[i][j] * df.loc[route[n], route[n+1]]['ddt'].iloc[0] 
                for i, trip in enumerate(properties['routes'])
                    for j, route in enumerate(properties['routes'][trip]) 
                        for n in range(len(route) - 1)])
    )

    prob = cvx.Problem(obj, constraints)
    prob.solve(solver='SCS', verbose=False)
    print(prob.status)
    print(prob.value)
    for i, trip in enumerate(drivers):
        for j, route in enumerate(trip):
            print(drivers[i][j].value)
