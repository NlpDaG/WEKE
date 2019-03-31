# coding:utf-8

import csv
import math
import os
import numpy as np

from ke_preprocess import read_file, filter_text, normalized_token
from ke_postprocess import rm_tags
from ke_main import evaluate_extraction


def read_vec(path, standard=True):
    """
    read vec: word, 1, 3, 4, ....
    return word:[1,...] dict
    """
    vec_dict = {}
    with open(path, encoding='utf-8') as file:
        # 标准csv使用','隔开，有的文件使用空格，所以要改变reader中的delimiter参数
        if standard:
            table = csv.reader(file)
        else:
            table = csv.reader(file, delimiter=' ')
        for row in table:
            try:
                vec_dict[row[0]] = list(float(i) for i in row[1:])
            except:
                continue
    return vec_dict


def read_edges(path):
    """
    read csv edge features
    return a (node1, node2):[features] dict
    """
    edges = {}
    with open(path, encoding='utf-8') as file:
        table = csv.reader(file)
        for row in table:
            edges[(row[0], row[1])] = [float(i) for i in row[2:]]
    return edges


def text2_stem_dict(text_notag):
    """
    convert text to a stem:word dict
    """
    stem_dict = {}
    for word in text_notag.split():
        stem_dict[normalized_token(word)] = word
    return stem_dict


def edgefeatures2file(path, edge_features):
    output = []
    for edge in edge_features:
        output.append(list(edge) + edge_features[edge])
    # print(output)
    with open(path, mode='w', encoding='utf-8', newline='') as file:
        table = csv.writer(file)
        table.writerows(output)


def cosine_sim(vec1, vec2):
    """余弦相似度"""
    def magnitude(vec):
        return math.sqrt(np.dot(vec, vec))
    cosine = np.dot(vec1, vec2) / (magnitude(vec1) * magnitude(vec2) + 1e-10)
    return cosine


def euc_distance(vec1, vec2):
    """欧式距离"""
    tmp = map(lambda x: abs(x[0] - x[1]), zip(vec1, vec2))
    distance = math.sqrt(sum(map(lambda x: x * x, tmp)))
    # distance==0时如何处理？
    if distance == 0:
        distance = 0.1
    return distance


def add_word_attr(filtered_text, edge_features, node_features, vec_dict,
                  part=None, **kwargs):
    """
    edge feature
    word attraction rank
    filterted_text为空格连接的单词序列，edge_features和vecs为dict
    特征计算后append到edge_features中

    params: filtered_text, filtered normalized string
            edge_features, a edge:feature dict
            vec_dict,
    """
    # 词向量的格式不统一，要想办法处理
    def force(freq1, freq2, distance):
        return freq1 * freq2 / (distance * distance)

    def dice(freq1, freq2, edge_count):
        return 2 * edge_count / (freq1 + freq2)

    splited = filtered_text.split()

    freq_sum = len(splited)

    for edge in edge_features:
        freq1 = splited.count(edge[0])
        freq2 = splited.count(edge[1])

        # 读不到的词向量设为全1
        default_vec = [1] * len(list(vec_dict.values())[0])
        vec1 = vec_dict.get(edge[0], default_vec)
        vec2 = vec_dict.get(edge[1], default_vec)
        distance = euc_distance(vec1, vec2)

        edge_count = edge_features[edge][0]

        force_score = force(freq1, freq2, distance)
        dice_score = dice(freq1, freq2, edge_count)
        srs_score = force_score * distance
        ctr_score = sum(edge_features[edge][0:3])

        word_attr = srs_score * dice_score * ctr_score

        temp = edge_features[edge][0:3]
        temp.append(word_attr)
        edge_features[edge] = temp

    return edge_features


def main(dataset, part, sub_vec_type, damping):

    print(sub_vec_type)
    phi = '1'
    if 'node' in part:
        phi = 'tfidf'

    dataset_dir = os.path.join('data/', dataset)
    edgefeature_dir = os.path.join(dataset_dir, 'edge_features')
    nodefeature_dir = os.path.join(dataset_dir, 'node_features')
    filenames = read_file(os.path.join(
        dataset_dir, 'abstract_list')).split(',')

    root_path = '../WordEmbedding/result/' + dataset
    vec_dict = read_vec(os.path.join(root_path, sub_vec_type))

    for filename in filenames:
        # print(filename)
        text = read_file(os.path.join(dataset_dir, 'abstracts', filename))
        filtered_text = filter_text(text)
        edge_features = read_edges(os.path.join(edgefeature_dir, filename))
        node_features = read_vec(os.path.join(nodefeature_dir, filename))

        edge_features_new = add_word_attr(filtered_text, edge_features, node_features, vec_dict,
                                          part=part)
        edgefeatures2file(os.path.join(
            edgefeature_dir, filename), edge_features_new)

    method_name = '_'.join(["WeKe", sub_vec_type])
    evaluate_extraction(dataset, method_name, omega='-1',
                        phi=phi, damping=damping, alter_node=None)
