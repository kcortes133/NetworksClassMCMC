import igraph as ig
import os, time, random, math, statistics

import matplotlib.pyplot as plt
import numpy as np
import ConfigModel_MCMC as MCMC
import networkx as nx


# calculate mean geodisic path length
def calcMDPL(graph):
    dists = graph.distances()
    dists = [i for ilist in dists for i in ilist]
    filteredDists= [v for v in dists if not math.isnan(v) and not math.isinf(v)]
    return np.mean(filteredDists)


test = False
if test:

    graphF = 'facebook100txt/American75.txt'
    graphTestIG = ig.Graph.Read_Ncol(graphF, directed=False)
    # calculate empirical clustering coefficient
    graphTestIG.transitivity_undirected()
    # calculate p for making null model
    p = ig.mean(graphTestIG.degree())/(graphTestIG.ecount()-1)
    # calculate mean degree of nodes
    l = ig.mean(graphTestIG.degree())

    # calculate mean geodisic path length
    dists = graphTestIG.distances
    dists = [i for ilist in dists for i in ilist]    # number of nodes
    filteredDists= [v for v in dists if not math.isnan(v) and not math.isinf(v)]

    graphTestNX = nx.read_edgelist(graphF)
    mcmc = MCMC.MCMC(graphTestNX)
    # do 100 times
    s = time.time()
    ccs = []
    for i in range(10):
        randG = mcmc.get_graph(sampling_gap=435324*20, return_type='igraph')
        cc = randG.transitivity_undirected()
        ccs.append(cc)
    print(time.time()-s)


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

        randCCoeffs = []
        allCCoeffs = []

        for file in subset:
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

erCalc = True
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
            c+=1
        with open('ER_Connectivity_Coefficients.txt', 'w')as f:
            for l in erCCoeffs:
                f.write(str(l) + '\n')

        return
    getRand25()


eCCalc = True
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

