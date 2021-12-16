# -*- coding: utf-8 -*-


# imports
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
import random

# functions
def make_network(slopes, intercepts, size):
    if len(slopes) != size or len(intercepts) != size:
        print('Error')
        return
    if size % 4 != 0:
        print('Error')
        return
    
    charint = 65
    G = nx.DiGraph()
    network = []
    edge_labels = {}
    
    for i in range(int(size / 4)):
        a = chr(charint)
        b = chr(charint+1)
        c = chr(charint+2)
        d = chr(charint+3)
        
        G.add_nodes_from([a,b,c,d])
        G.add_edges_from([(a,b), (a,c), (b,d), (c,d)])
        
        network.append([chr(charint), chr(charint+1), 'Directed', slopes[4*i], intercepts[4*i]])
        network.append([chr(charint), chr(charint+2), 'Directed', slopes[4*i+1], intercepts[4*i+1]])
        network.append([chr(charint+1), chr(charint+3), 'Directed', slopes[4*i+2], intercepts[4*i+2]])
        network.append([chr(charint+2), chr(charint+3), 'Directed', slopes[4*i+3], intercepts[4*i+3]])
    
        edge_labels[(chr(charint),chr(charint+1))] = '{}x+{}'.format(slopes[4*i], intercepts[4*i])
        edge_labels[(chr(charint),chr(charint+2))] = '{}x+{}'.format(slopes[4*i+1], intercepts[4*i+1])
        edge_labels[(chr(charint+1),chr(charint+3))] = '{}x+{}'.format(slopes[4*i+2], intercepts[4*i+2])
        edge_labels[(chr(charint+2),chr(charint+3))] = '{}x+{}'.format(slopes[4*i+3], intercepts[4*i+3])
        
        charint += 3

    return (G, network, edge_labels)

def draw_network(G, edge_labels):
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color = 'green')
    nx.draw_networkx_edge_labels(G, pos, edge_labels, label_pos=0.5)
    
def all_paths(network, begin, end):
    allpaths = []
    path = []
    path.append(begin)
    paths_helper(network, begin, end, path, allpaths)
    return allpaths
    
def paths_helper(network, begin, end, path, allpaths):
    if begin == end:
        allpaths.append(copy.deepcopy(path))
    else:
        net = network[network['Source'] == begin]
        next_nodes = net['Target']
        for node in next_nodes:
            path.append(node)
            paths_helper(network, node, end, path, allpaths)
            path.pop()

def get_path_cost(network, path, players_per_edge):
    cost = 0
    for i in range(len(path)-1):
        net = network[network['Source'] == path[i]]
        net = net[net['Target'] == path[i+1]]
        slope = int(net['slope'])
        intercept = int(net['intercept'])
        players = players_per_edge[(path[i],path[i+1])]
        cost = cost + slope*players + intercept
    return cost

def player_choose_path(path, players_per_edge, step):
    for i in range(len(path)-1):
        players_per_edge[(path[i], path[i+1])] += step
    return players_per_edge

def populate_network(network, num_players, allpaths, begin, end):
    players_per_edge = {}
    for i in range(network.shape[0]):
        edge = [network['Source'][i], network['Target'][i]]
        players_per_edge[tuple(edge)] = 0

    length = len(allpaths)
    for i in range(length):
        # Player will initially choose the lowest cost path
        step = 0
        if i < num_players % length:
            step = int(num_players / length) + 1
        else:
            step = int(num_players / length)
        players_per_edge = player_choose_path(allpaths[i], players_per_edge, step)
    
    return players_per_edge

def get_nash_eq(players_per_edge, allpaths):    
    # After initial player choices, each type of player will have a chance to
    # determine their best response
    cond = True
    while(cond):
        cond_int = 0
        for removed_path in allpaths:
            if path_exists(removed_path, players_per_edge):
                players_per_edge = remove_player(removed_path, players_per_edge)
                least_cost_path = removed_path
                lowest_cost = get_path_cost(network, least_cost_path, players_per_edge)
                
                for path in allpaths:
                    cost = get_path_cost(network, path, players_per_edge)
                    if cost < lowest_cost:
                        least_cost_path = path
                        lowest_cost = cost
                
                # Add player back into network per best response
                players_per_edge = player_choose_path(least_cost_path, players_per_edge, 1)
                
                # Keep track of if path changed for condition
                if least_cost_path == removed_path:
                    cond_int += 1
        
        # Terminate loop if system is in equilibrium
        if cond_int == len(allpaths):
            cond = False
    
    return players_per_edge

def path_exists(path, players_per_edge):
    for i in range(len(path)-1):
        if players_per_edge[(path[i], path[i+1])] == 0:
            return False
    return True
    
def remove_player(path, players_per_edge):
    for i in range(len(path)-1):
        players_per_edge[(path[i], path[i+1])] -= 1
    return players_per_edge

def add_edge(G, network, edge_labels, allpaths, players_per_edge, source, target, slope, intercept):
    G.add_edges_from([(source, target)])
    network.loc[len(network.index)] = [source, target, 'Directed', slope, intercept]
    edge_labels[(source, target)] = '{}x+{}'.format(slope, intercept)
    begin = allpaths[0][0]
    end = allpaths[0][len(allpaths[0])-1]
    allpaths = all_paths(network, begin, end)
    players_per_edge[(source, target)] = 0
    return(G, network, edge_labels, allpaths, players_per_edge)

def total_network_cost(network, players_per_edge):
    slopes = network['slope']
    intercepts = network['intercept']
    vals = list(players_per_edge.values())
    total_cost = 0
    for i in range(len(vals)):
        total_cost += slopes[i]*vals[i] + intercepts[i]
    return total_cost
    
def get_example_plot(size):
    slopes = []
    intercepts = []
    for i in range(size):
        slopes.append('m$_{}$'.format(i))
        intercepts.append('b$_{}$'.format(i))
    (G,network,edge_labels) = make_network(slopes,intercepts,size)
    draw_network(G, edge_labels)
    plt.savefig('../plots/exampleplotn2')
    plt.show()
    
def get_random_network(min_slope, max_slope, min_int, max_int):
    size = 4
    
    # Get discrete uniform random values for slopes and intercepts
    slopes = []
    intercepts = []
    for i in range(size):
        slopes.append(min_slope + (max_slope-min_slope)*random.random())
        intercepts.append(min_int + (max_int-min_int)*random.random())
    
    # Construct network
    (G,network,edge_labels) = make_network(slopes, intercepts, size)
    return (G,network,edge_labels)

def run_simulations(min_slope, max_slope, min_int, max_int, sims, num_players):
    before_edge_cost = []
    after_zero_edge_cost = []
    after_rand_edge_cost = []
    begin = 'A'
    end = 'D'
    for i in range(sims):
        (G,network,edge_labels) = get_random_network(min_slope, max_slope, min_int, max_int)
        network = pd.DataFrame(network)
        network.columns = ['Source', 'Target', 'Type', 'slope', 'intercept']
        allpaths = all_paths(network, begin, end)
        players_per_edge = populate_network(network, num_players, allpaths, begin, end)
        players_per_edge = get_nash_eq(players_per_edge, allpaths)
        before_edge_cost.append(total_network_cost(network, players_per_edge))
        
        # Add edge with zero weight
        (G,network,edge_labels,allpaths,players_per_edge) = add_edge(G, network, edge_labels, allpaths, players_per_edge, 'B', 'C', 0, 0)
        players_per_edge = get_nash_eq(players_per_edge, allpaths)
        after_zero_edge_cost.append(total_network_cost(network, players_per_edge))
        
        # Add random weights to new edge
        new_slope = min_slope + (max_slope-min_slope)*random.random()
        new_int = min_int + (max_int-min_int)*random.random()
        num_edges = network.shape[0]
        network.drop([num_edges-1], axis=0, inplace=True)
        network.loc[len(network.index)] = ['B', 'C', 'Directed', new_slope, new_int]
        
        # Redo equilibrium and total cost calculations
        players_per_edge = get_nash_eq(players_per_edge, allpaths)
        after_rand_edge_cost.append(total_network_cost(network, players_per_edge))
    
    return([before_edge_cost, after_zero_edge_cost, after_rand_edge_cost])


    
def large_sim_analysis():
    # Analysis on 1000 simulations with 100 players, for less noise
    sim_results = run_simulations(0,10,0,100,1000,100)
    
    # Plots plots plots
    plt.hist(sim_results[0], bins = 30)
    plt.xlabel('Total Cost')
    plt.ylabel('Count')
    plt.title('Total Cost Histogram for Random Weights')
    plt.savefig('../plots/simhist')
    plt.show()
    
    
    plt.scatter(sim_results[0],sim_results[1])
    plt.scatter(sim_results[0],sim_results[2])
    plt.plot(sim_results[0], sim_results[0], color = 'red')
    plt.legend(['Zero weight', 'Random weight'])
    plt.xlabel('Total Cost Original Network')
    plt.ylabel('Total Cost with Added Edge')
    plt.title('Orginal Network vs Added Edge Network')
    plt.savefig('../plots/compareadded_edge')
    
    
    # Get ratio results and plot CDFs
    ratio_zero = []
    ratio_rand = []
    cdf = []
    for i in range(len(sim_results[0])):
        ratio_zero.append(sim_results[0][i] / sim_results[1][i])
        ratio_rand.append(sim_results[0][i] / sim_results[2][i])
        cdf.append(i / len(sim_results[0]))
    
    
    
    plt.plot(sorted(ratio_zero), cdf)
    plt.plot(sorted(ratio_rand), cdf)
    plt.hlines(0.5, .72, 1.1, linestyles = 'dotted', color = 'red')
    plt.vlines(1,0,1, linestyles = 'dotted', color = 'red')
    plt.vlines(.92,0,1, linestyles = 'dotted', color = 'red')
    plt.xlabel('Cost Original Network / Cost Added Edge')
    plt.ylabel('Pr(X > x)')
    plt.legend(['Zero weight', 'Random weight'])
    plt.title('CDF of Original Network Ratio / Added Edge')
    plt.savefig('../plots/ratiocdfs')
    plt.show()
    
    
    plt.hist(ratio_zero)
    
    
def vary_players_analysis():
    
    # Lets vary num_players and observe resulting average ratios
    vary_players = []
    num_players = [10, 20, 60, 100, 200, 500, 1000]
    for players in num_players:
        results = run_simulations(0,10,0,100,100, players)
        r_zero = []
        r_rand = []
        for i in range(len(results[0])):
            r_zero.append(results[0][i] / results[1][i])
            r_rand.append(results[0][i] / results[2][i])
        vary_players.append([np.mean(r_zero), np.mean(r_rand)])
    
    # To pandas
    vary_players = pd.DataFrame(vary_players)
    vary_players.columns = ['Zero', 'Rand']
    
    # Plots plots plots
    plt.plot(num_players, vary_players['Zero'])
    plt.plot(num_players, vary_players['Rand'])
    plt.legend(['Zero weight', 'Random weight'])
    plt.xlabel('Number of Players')
    plt.ylabel('Average Cost Original Network / Cost Added Edge Ratio')
    plt.title('Ratio With Changing Number of Players')
    plt.savefig('../plots/varyplayers')
    plt.show()
    
def testing_code():
    slopes = [1,2,3,4]
    intercepts = [0,1,2,3]
    (G,network,edge_labels) = make_network(slopes,intercepts,4)
    network = pd.DataFrame(network)
    network.columns = ['Source', 'Target', 'Type', 'slope', 'intercept']
    #G = nx.from_pandas_edgelist(net, source='Source', target='Target', edge_attr='weight')
    draw_network(G, edge_labels)
    plt.show()
    
    allpaths = all_paths(network, 'A', 'D')
    players_per_edge = populate_network(network, 100, allpaths, 'A', 'D')
    players_per_edge = get_nash_eq(players_per_edge, allpaths)
    
    print('Total Cost Before Edge: ', total_network_cost(network, players_per_edge))
    
    # Add and edge from B to C with weight 0
    (G,network,edge_labels,allpaths,players_per_edge) = add_edge(G, network, edge_labels, allpaths, players_per_edge, 'B', 'C', 0, 0)
    players_per_edge = get_nash_eq(players_per_edge, allpaths)
    
    print('Total Cost After Edge: ', total_network_cost(network, players_per_edge))
    
    #get_example_plot(8)

def __main__():
    large_sim_analysis()
    vary_players_analysis()

