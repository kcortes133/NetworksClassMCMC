import igraph as ig
import os, time, random, math, statistics

import matplotlib.pyplot as plt
import numpy as np
import ConfigModel_MCMC as MCMC
import networkx as nx


file = 'Medici network/medici_network.gml'
graph = ig.Graph.Read_GML(file)
harmonic_Cent = graph.harmonic_centrality()


names =[]
namesFile = 'Medici network/medici_network.txt'
with open(namesFile,'r') as f:
    lines = f.readlines()
    for l in lines:
        line = l.strip().split(' ')
        names.append(line[1].strip(','))

hlist = tuple(zip(names, harmonic_Cent))
with open('Medici_Network_Harmonic_Centrality.txt', 'w') as of:
    for n, h in hlist:
        of.write(n + ',' + str(h) + '\n')


randHCs = []
graphNX = graph.to_networkx()
mcmc = MCMC.MCMC(graphNX)
# do 100 times
s = time.time()
ccs = []
eNum = graph.ecount()
for i in range(1000):
    randG = mcmc.get_graph(sampling_gap=eNum* 20, return_type='igraph')
    cc = randG.harmonic_centrality()
    ccs.append(cc)

nodeHCs = np.array(ccs)
avgNodeHCs = np.mean(nodeHCs, axis=0)
hRandlist = tuple(zip(names, avgNodeHCs))
with open('Medici_Network_Rand_Harmonic_Centrality.txt', 'w') as of:
    for n, h in hRandlist:
        of.write(n + '\t' + str(h) + '\n')

randHCs.append(np.mean(ccs))

expExpanded = np.outer(harmonic_Cent, np.ones(1000)).transpose()

plt.boxplot(expExpanded-nodeHCs, showfliers=False)
plt.title('Harmonic Centrality Difference for MCMC Simple Graph Space')
plt.xlabel('Vertex Label')
plt.ylabel('Difference in Harmonic Centrality (Difference - Observed)')
plt.show()



degree_seq = [d for n, d in graphNX.degree()]

configHCs = []
for i in range(1000):
    gConfigModel = nx.configuration_model(degree_seq)
    gConfigModel = nx.Graph(gConfigModel)
    gConfigModel.remove_edges_from(nx.selfloop_edges(gConfigModel))
    gConfigIG = ig.Graph.from_networkx(gConfigModel)
    configHCs.append(gConfigIG.harmonic_centrality())

expExpanded = np.outer(harmonic_Cent, np.ones(1000)).transpose()

plt.boxplot(expExpanded-configHCs, showfliers=False)
plt.title('Harmonic Centrality Difference for Stub-Labeled Loopy Multigraph Space')
plt.xlabel('Vertex Label')
plt.ylabel('Difference in Harmonic Centrality (Difference - Observed)')
plt.show()