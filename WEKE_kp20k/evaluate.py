import os
import time
import csv
import gensim
import json
from text_process import filter_text, get_phrases, normalized_token, read_file
from rank import kewe
from train import we


def read_vec(path, standard=True):
    """
    read vec: word, 1, 3, 4, ....
    return word:[1,...] dict
    """
    vec_dict = {}
    with open(path, encoding='utf-8') as file:
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


def evaluate_pagerank(method_name='textrank', dataset='kp20k'):
    # read config
    names = read_file('kp20k/filelist').split(',')
    abstract_dir = 'kp20k/abstracts/'
    gold_dir = 'kp20k/golds/'
    extracted = 'kp20k/extracted/'
    topn = 6
    with_tag = False

    ngrams = 2
    weight2 = 0.5
    weight3 = 0.2

    count = 0
    gold_count = 0
    extract_count = 0
    mrr = 0
    prcs_micro = 0
    recall_micro = 0
    vecs = read_vec('kp20k/embedding/ke.emb')
    for name in names:

        pr, graph = kewe(name, vecs)

        # logger.debug(str(pr)) #Python3.6后字典有序，此处未做处理
        doc_path = os.path.join(abstract_dir, name)
        keyphrases = get_phrases(
            pr, graph, doc_path, ng=ngrams, pl2=weight2, pl3=weight3, with_tag=with_tag)
        top_phrases = []
        for phrase in keyphrases:
            if phrase[0] not in str(top_phrases):
                top_phrases.append(phrase[0])
            if len(top_phrases) == topn:
                break
        # detailedresult_dir = os.path.join(extracted, method_name)
        # if not os.path.exists(detailedresult_dir):
        #     os.makedirs(detailedresult_dir)
        # with open(os.path.join(detailedresult_dir, name), encoding='utf-8', mode='w') as file:
        #     file.write('\n'.join(top_phrases))

        standard = read_file(os.path.join(gold_dir, name)).split('\n')
        if standard[-1] == '':
            standard = standard[:-1]
        standard = list(' '.join(list(normalized_token(w)
                                      for w in g.split())) for g in standard if g != '')
        count_micro = 0
        position = []
        for phrase in top_phrases:
            if phrase in standard:
                count += 1
                count_micro += 1
                position.append(top_phrases.index(phrase))
        if position != []:
            mrr += 1 / (position[0] + 1)
        gold_count += len(standard)
        extract_count += len(top_phrases)
        prcs_micro += count_micro / len(top_phrases)
        recall_micro += count_micro / len(standard)
        # break
    prcs = count / extract_count
    recall = count / gold_count
    f1 = 2 * prcs * recall / (prcs + recall)
    mrr /= len(names)
    prcs_micro /= len(names)
    recall_micro /= len(names)
    f1_micro = 2 * prcs_micro * recall_micro / (prcs_micro + recall_micro)
    result_print = (method_name, count, prcs, recall, f1, mrr)
    print(str(result_print))

    eval_result = method_name + '@' + str(topn) + ',' + dataset + ',' + str(prcs) + ',' \
        + str(recall) + ',' + str(f1) + ',' + str(mrr) + ',\n'
    with open('result/kp20k.result', mode='a', encoding='utf-8') as file:
        file.write(eval_result)


if __name__ == '__main__':
    # 1.learning word embedding
    we(total_iter=3000000, dim=100, dataset='kp20k')
    # 2. keyphrase eatraction
    evaluate_pagerank(method_name='kewe')
