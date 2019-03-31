# our
import os
import csv
import math
import networkx as nx
import itertools
from text_process import read_file, filter_text, is_word, normalized_token
from nltk import word_tokenize


def dict2list(dict):
    output = []
    for key in dict:
        if isinstance(key, str):
            tmp = [key]
        else:
            tmp = list(key)
        tmp.append(dict[key])
        output.append(tmp)
    return output


def build_graph(edge_weight):
    graph = nx.Graph()
    graph.add_weighted_edges_from(edge_weight)
    return graph


def get_edge_freq(text_stemmed, window=2):
    edges = []
    edge_freq = {}
    tokens = text_stemmed.split()
    for i in range(0, len(tokens) - window + 1):
        edges += list(itertools.combinations(tokens[i:i + window], 2))
    for i in range(len(edges)):
        for edge in edges:
            if edges[i][0] == edge[1] and edges[i][1] == edge[0]:
                edges[i] = edge
    for edge in edges:
        # * 2 / (tokens.count(edge[0]) + tokens.count(edge[1]))
        edge_freq[tuple(sorted(edge))] = edges.count(edge)
    return edge_freq


def euc_distance(vec1, vec2):
    """欧式距离"""
    tmp = map(lambda x: abs(x[0] - x[1]), zip(vec1, vec2))
    distance = math.sqrt(sum(map(lambda x: x * x, tmp)))
    # distance==0时如何处理？
    if distance == 0:
        distance = 0.1
    return distance


def wash_doc(text):
    """
    Return stemmed text.
    :param text: text without tags
    """
    words_stem = [normalized_token(w).strip('.?,:')
                  for w in word_tokenize(text) if is_word(w)]
    return ' '.join(words_stem)


def read_tfidf(name):

    tfidf_path = 'kp20k/tfidf/' + name
    tfidf_raw = read_file(tfidf_path).split('\n')
    if tfidf_raw[-1] == '':
        tfidf_raw = tfidf_raw[:-1]
    tfidf = {}
    for line in tfidf_raw:
        key, value = line.split()
        tfidf[key] = float(value)
    return tfidf


def cal_w(edge_freq, text_candidates, text, vecs):

    def force(freq1, freq2, distance):
        return freq1 * freq2 / distance

    def dice(freq1, freq2, edge_count):
        if freq1 == 0 and freq2 == 0:
            freq1 = 0.1
        return 2 * edge_count / (freq1 + freq2)

    words = [word for word in text_candidates.split() if word != '']
    all_words = [word for word in wash_doc(text).split() if word != '']
    tf_word = {}
    for word in words:
        tf_word[word] = all_words.count(word)
    for key, value in edge_freq.items():
        default_vec = [1] * 100
        vec1 = vecs.get(key[0], default_vec)
        vec2 = vecs.get(key[1], default_vec)
        distance = euc_distance(vec1, vec2)
        new_w = force(tf_word[key[0]], tf_word[key[1]], distance) * \
            dice(tf_word[key[0]], tf_word[key[1]], value)
        edge_freq[key] = new_w
    return edge_freq


def kewe(name, vecs):
    doc_path = 'kp20k/abstracts/' + name
    text = read_file(doc_path)
    text_candidates = filter_text(text, with_tag=False)
    edge_freq = get_edge_freq(text_candidates, window=2)
    edge_freq = cal_w(edge_freq, text_candidates, text, vecs)
    edges = dict2list(edge_freq)
    graph = build_graph(edges)
    tfidf = read_tfidf(name)
    nodes = set(graph.nodes())
    tfidf_nodes = set(tfidf.keys())
    add_nodes = nodes - tfidf_nodes

    for node in add_nodes:
        tfidf[node] = 1e-7
    pr = nx.pagerank(graph, alpha=0.85, personalization=tfidf)
    return pr, graph
