# -*- coding: utf-8 -*-

import os
import shutil
import json


def load():
    with open('kp20k/kp20k_validation.json', 'r', encoding='utf-8') as json_file:
        text = json_file.readlines()
        print(len(text))
        file_num = 0
        for line in text:
            data = json.loads(line)
            abstract = data['title'] + data['abstract']
            golds = [key.strip().replace('\n', ' ')
                     for key in data['keyword'].split(';') if key != '']
            write_golds('kp20k/golds/' + str(file_num), golds)
            write_file('kp20k/abstracts/' + str(file_num), abstract)
            file_num += 1
        print(file_num)
        filenames = ','.join(map(str, range(file_num)))
        write_file('kp20k/filelist', filenames)
        print('ok')


def file_name(file_dir, suffix):
    L = []
    names = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == suffix:
                L.append(os.path.join(root, file))
                names.append(os.path.splitext(file)[0])
    return L, names


def read_file(path):
    with open(path, mode='r', encoding='utf-8') as f:
        text = f.read()
    return text


def write_golds(path, clist):
    with open(path, mode='w', encoding='utf-8') as f:
        for line in clist:
            f.write(line + '\n')


def write_file(path, strr):
    with open(path, mode='w', encoding='utf-8') as f:
        f.write(strr)


load()
