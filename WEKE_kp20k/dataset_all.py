from text_process import normalized_token, is_word, read_file
from nltk import word_tokenize
from math import log


def wash_doc(text):
    """
    Return stemmed text.
    :param text: text without tags
    """
    words_stem = [normalized_token(w).strip('')
                  for w in word_tokenize(text) if is_word(w)]
    return ' '.join(words_stem)


def writeAfile(path, text):
    with open(path, mode='a', encoding='utf-8') as f:
        f.write(text)
        f.write('\n')


def write_dict(path, dic):
    with open(path, 'w', encoding='utf-8') as f:
        for key, value in dic.items():
            f.write(key + ' ' + str(value) + '\n')


filelist = read_file('./kp20k/filelist').split(',')
abstracts_path = './kp20k/abstracts/'
out_path = './kp20k/kp20k.data'

# get all docs to a file
for file in filelist:
    new_file = wash_doc(read_file(abstracts_path + file))
    writeAfile(out_path, new_file)

# tfidf
tfidf_path = './kp20k/tfidf/'
docs = read_file(out_path).split('\n')
for i in range(len(filelist)):
    print(i)
    name = filelist[i]
    words = [word for word in docs[i].split() if word != '']
    tfidf = {}
    for w in set(words):
        df = 0
        for d in docs:
            if w in d:
                df += 1
        idf = log(len(filelist) / df)  # log底数可调整
        tf = words.count(w)
        tfidf[w] = tf * idf
    write_dict(tfidf_path + str(name), tfidf)
