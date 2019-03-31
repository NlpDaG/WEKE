import networkx as nx
from train import *
from concatenate import *


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
            edges.append(edge)
    G.add_weighted_edges_from(edges)
    return G


def combinationVec(sourceVec, distinationVec, alpha):
    result = {}
    for key in sourceVec:
        result[key] = alpha * sourceVec[key] + \
            (1 - alpha) * distinationVec[key]
    return result


def main(total_iter, dim, dataset, topicN):
    '''
    total_iter:number of sample edge
    dim: the dimension of word embeddings
    dataset: KDD or WWW
    topicN: the number of topics for constructing word-topic graph
    '''
    input_wordsG_path = 'data_preparation/result_graph/' + \
        dataset + '/wordsG_tf.data'
    input_topicG_path = 'data_preparation/result_graph/' + \
        dataset + '/topicG' + str(topicN) + '.data'

    path1 = 'result/' + dataset + '/c.emb'
    path2 = 'result/' + dataset + '/t.emb'
    path3 = 'result/' + dataset + '/h.emb'
    path4 = 'result/' + dataset + '/c+t.emb'

    print("*****Read Data*****")
    wG = readFile(input_wordsG_path)
    print(dataset + "'s wordsG's number of edges is ", len(wG.edges()))
    wtG = readFile(input_topicG_path)
    print(dataset + "'s topicG's number of edges is ", len(wtG.edges()))

    print("*****train*****")
    trainAll = Train(wG, wtG, dim)
    trainAll.initial()
    trainAll.train(total_iter, path1, path2)
    trainAll.initial()
    trainAll.jointtrain(total_iter, path3)
    concatenate(path1, path2, path4)


if __name__ == '__main__':
    datasets = ['WWW','KDD']
    for dataset in datasets:
        if dataset == 'KDD':
            main(total_iter=int(300000), dim=100,
                 dataset=dataset, topicN=50)
        if dataset == 'WWW':
            main(total_iter=int(1000000), dim=100,
                 dataset=dataset, topicN=50)
