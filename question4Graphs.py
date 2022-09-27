import matplotlib.pyplot as plt
import numpy as np


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
