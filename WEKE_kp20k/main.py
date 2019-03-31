import networkx as nx
from Train import *


def readFile(path):
    G = nx.DiGraph()
    with open(path, mode='r', encoding='utf-8') as f:
        text = f.read()
    edges = []
    for line in text.split('\n'):
        if len(line.strip()) == 0:
            continue
        elif len(line.split(',')) != 3:
            continue
        else:
            edge = tuple(line.split(','))
            if 'a' in edge:
                continue
            else:
                edges.append(edge)
    G.add_weighted_edges_from(edges)
    return G


def combinationVec(sourceVec, distinationVec, alpha):
    result = {}
    for key in sourceVec:
        result[key] = alpha * sourceVec[key] + \
            (1 - alpha) * distinationVec[key]
    return result


def we(total_iter, dim, dataset):
    # '''
    # total_iter:number of sample edge
    # '''
    input_wordsG_path = 'kp20k/wordsG.data'
    input_topicG_path = 'kp20k/topicG.data'
    path = 'kp20k/embedding/ke.emb'

    print("*****Read Data*****")
    wG = readFile(input_wordsG_path)
    print(dataset + "'s wordsG's number of edges is ", len(wG.edges()))
    wtG = readFile(input_topicG_path)
    print(dataset + "'s topicG's number of edges is ", len(wtG.edges()))

    print("*****train*****")
    trainAll = Train(wG, wtG, dim)
    trainAll.initial()
    trainAll.train(total_iter, path)
