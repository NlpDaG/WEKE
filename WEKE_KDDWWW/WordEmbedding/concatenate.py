import csv


def readFile(file):
    dict1 = {}
    with open(file, encoding='utf-8', mode='r')as f:
        txt = f.readlines()
    for line in txt:
        items = line.split(',')
        k = items[0]
        vec = [float(item) for item in items[1:]]
        dict1[str(k)] = vec
    return dict1


def concatenate(file1, file2, path):
    dict1 = readFile(file1)
    dict2 = readFile(file2)
    for key, vec in dict1.items():
        dict1[key] = dict1[key] + dict2[key]
    writeTofile(path, dict1)


def add(file1, file2, path, alpha):
    dict1 = readFile(file1)
    dict2 = readFile(file2)
    for key, vec in dict1.items():
        vec1 = [item * alpha for item in dict1[key]]
        vec2 = [item * (1 - alpha) for item in dict2[key]]
        vec = [e1 + e2 for (e1, e2) in zip(vec1, vec2)]
        dict1[key] = vec
    writeTofile(path, dict1)


def writeTofile(path, dict1):
    with open(path, mode='w', encoding='utf-8')as f:
        csvWriter = csv.writer(f)
        for key, value in dict1.items():
            row = []
            row.append(key)
            for item in value:
                row.append(float(item))
            csvWriter.writerow(row)
        f.close()


if __name__ == '__main__':
    file1 = '../result/KDD/w.emb'
    file2 = '../result/KDD/t.emb'
    path = '../result/KDD/w+t.emb'
    concatenate(file1, file2, path)
