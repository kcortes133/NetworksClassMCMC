import igraph as ig
import os, time, random, math, statistics

import matplotlib.pyplot as plt
import numpy as np
import ConfigModel_MCMC as MCMC
import networkx as nx



calc = True
if calc:
    def getRand25():
        random.seed(13)
        allFiles = []
        path = 'facebook100txt'
        for file in os.listdir('facebook100txt/'):
            if 'attr' not in file and file.endswith('.txt') and 'facebook' not in file:
                allFiles.append(path+'/'+file)
        subset = random.sample(allFiles, 25)

        empCCoeffs = []
        randCCoeffs = []
        allCCoeffs = []
        erCCoeffs = []
        nodeNums = []

        for file in subset:
            print(file)
            #graphT = ig.Graph.Read_Ncol(file, directed=False)
            #n = graphT.vcount()
            #nodeNums.append(n)
            #clusteringCoeff = graphT.transitivity_undirected()
            #empCCoeffs.append(clusteringCoeff)

            #p = ig.mean(graphT.degree())/(graphT.vcount()-1)
            #del graphT
            # ER random graph
            #erG = ig.Graph.Erdos_Renyi(n=n, p=p)
            #erCCoeffs.append(erG)
            #del erG

            graphNX = nx.read_edgelist(file)
            mcmc = MCMC.MCMC(graphNX)
            # do 100 times
            s = time.time()
            ccs = []

            for i in range(20):
                randG = mcmc.get_graph(sampling_gap=graphNX.number_of_edges()*20, return_type='igraph')
                cc = randG.transitivity_undirected()
                ccs.append(cc)
            allCCoeffs.append(ccs)
            randCCoeffs.append(np.mean(ccs))
        allCCoeffs = np.array(allCCoeffs)
        np.savez_compressed('Rand_Clustering_Coefficients3.npz',a=allCCoeffs)

        return
    getRand25()

erCalc = False
if erCalc:
    def getRand25():
        random.seed(13)
        allFiles = []
        path = 'facebook100txt'
        for file in os.listdir('facebook100txt/'):
            if 'attr' not in file and file.endswith('.txt') and 'facebook' not in file:
                allFiles.append(path+'/'+file)
        subset = random.sample(allFiles, 25)
        erCCoeffs = []
        erGDPLs = []

        c = 0
        for file in subset:
            print(c)
            graphT = ig.Graph.Read_Ncol(file, directed=False)
            n = graphT.vcount()
            p = ig.mean(graphT.degree())/(graphT.vcount()-1)
            # ER random graph
            erG = ig.Graph.Erdos_Renyi(n=n, p=p)
            erCoeff = erG.transitivity_undirected()
            erCCoeffs.append(erCoeff)
            #erGDPLs.append(calcMDPL(erG))

            c+=1
        with open('ER_Connectivity_Coefficients.txt', 'w')as f:
            for l in erCCoeffs:
                f.write(str(l) + '\n')

        #with open('ER_GDPLs.txt', 'w')as f:
        #    for l in erGDPLs:
        #        f.write(str(l) + '\n')
        return
    getRand25()


eCCalc = False
if eCCalc:
    def getRand25():
        random.seed(13)
        allFiles = []
        path = 'facebook100txt'
        for file in os.listdir('facebook100txt/'):
            if 'attr' not in file and file.endswith('.txt') and 'facebook' not in file:
                allFiles.append(path+'/'+file)
        subset = random.sample(allFiles, 25)

        empCCoeffs = []
        nodeNums = []
        gdpls = []
        c = 0
        for file in subset:
            print(c)
            graphT = ig.Graph.Read_Ncol(file, directed=False)
            n = graphT.vcount()
            nodeNums.append(n)
            clusteringCoeff = graphT.transitivity_undirected()
            empCCoeffs.append(clusteringCoeff)
            #gdpls.append(calcMDPL(graphT))
            c+=1

        with open('Empirical_Clustering_Coefficients.txt', 'w') as f:
            for l in empCCoeffs:
                f.write(str(l)+'\n' )

        with open('Number_of_Nodes.txt', 'w') as f:
            for l in nodeNums:
                f.write(str(l)+'\n' )

        with open('Empirical_GDPLs.txt', 'w') as f:
            for l in gdpls:
                f.write(str(l)+'\n' )

        return
    getRand25()


def loadNodeNums():
    nodeNums = []
    with open('Number_of_Nodes.txt', 'r') as f:
        lines = f.readlines()
        for l in lines:
            nodeNums.append(float(l.strip()))
    return nodeNums


def loadERCCs():
    erCC = []
    with open('ER_Connectivity_Coefficients.txt', 'r') as f:
        lines = f.readlines()
        for l in lines:
            erCC.append(float(l.strip()))
    return erCC

def loadEmpiricalCCs():
    empirCCs = []
    with open('Empirical_Clustering_Coefficients.txt', 'r') as f:
        lines = f.readlines()
        for l in lines:
            empirCCs.append(float(l.strip()))
    return empirCCs


def loadRandCCs():
    cc = np.load('Rand_Clustering_Coefficients.npz', allow_pickle=True)['a']
    cc1 = np.load('Rand_Clustering_Coefficients1.npz', allow_pickle=True)['a']
    cc2 = np.load('Rand_Clustering_Coefficients2.npz', allow_pickle=True)['a']
    cc3 = np.load('Rand_Clustering_Coefficients3.npz', allow_pickle=True)['a']
    allCCs = np.concatenate((cc,cc1), axis=1)
    allCCs = np.concatenate((allCCs, cc2), axis=1)
    allCCs = np.concatenate((allCCs, cc3), axis=1)
    avgCCs = np.mean(allCCs, axis=1)
    return avgCCs


nodeNums = loadNodeNums()
erCCs = loadERCCs()
empCCs = loadEmpiricalCCs()
randCCs = loadRandCCs()

er = plt.scatter(nodeNums, erCCs, marker='o')
em = plt.scatter(nodeNums, empCCs, marker='v')
ra = plt.scatter(nodeNums, randCCs, marker='*')
plt.title('Clustering Coefficient vs Number of Nodes for Empirical and Null Models')
plt.xlabel('Number of Nodes')
plt.ylabel('Clustering Coefficient')
plt.xscale('log')
plt.yscale('log')
plt.legend((er, em, ra), ('Edge Graph Model', 'Empirical Graph', 'Degree Graph Model'))
plt.show()


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

plt.plot(list(range(1,17)), expExpanded)
plt.plot(list(range(1,17)), nodeHCs)
plt.scatter(expExpanded)
plt.scatter(nodeHCs)
plt.xlabel('Vertex Label')
plt.ylabel('Harmonic Centrality Scores for MCMC Simple Graph Space')
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

plt.plot(list(range(1,17)), expExpanded)
plt.plot(list(range(1,17)), configHCs)
plt.scatter(expExpanded)
plt.scatter(configHCs)
plt.xlabel('Vertex Label')
plt.ylabel('Harmonic Centrality Scores for Stub-Labeled Loopy Multigraph Space')
plt.show()
