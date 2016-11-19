# TrafficFlowSimulator

This is the codebase for our research project on convention formation in traffic networks.

## How to use

To use the simulator, you need the following tools:
* Jupyter Notebook 4.2.0
* Python 3.5.2
* Numpy 1.11.2
* Pandas 0.19.1
* XlsxWriter 0.9.3
* Altair 1.1.0.dev0

To install all of the required tools, follow these steps:

1. Download Anaconda for Python 3.5 from here: https://www.continuum.io/downloads
2. Type the following in the command line:
```
bash Anaconda3-4.2.0-Linux-x86_64.sh 
conda install -c anaconda xlsxwriter=0.9.3
conda install altair --channel conda-forge
```

To open jupyter notebook type:
```
jupyter-notebook
```
This will open a home page for jupyter. Open Simulator.ipynb and you are good to go.

## How to specify simulations
If you want to run the simulations inside the MultiODNetworks, you can just change the simulation_name to your desired simulation. Only thing you need to be careful about in the traffic.properties file is the routes and route opts should be named as route_[trip]_[route]

## Hardcoded parts
1. You need to change the simulation_folder for your own computer
2. Inside the simulate function, trips is assigned the list of OD pairs and overall which needs to be changed for each simulation.
3. G, T and P values are assigned inside the Initialize the simulation cell
4. If you want to plot the simulation for a set of parameters, you will need to enter them in the first cell of visualizations.
