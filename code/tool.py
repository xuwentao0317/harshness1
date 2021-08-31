import re
from collections import Counter

import xlrd
import gensim
import datetime
import numpy as np
from xlutils.copy import copy
from gensim import corpora
from gensim.models import TfidfModel
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import accuracy_score
from stanfordcorenlp import StanfordCoreNLP


# music数据集
class Dataset(object):
    def __init__(self, data_path):
        self.user = []
        self.pro = []
        self.label = []
        self.comm = []
        with open(data_path, 'r', encoding='UTF-8') as f:
            for line in f:
                line = line.strip().split('\t')
                self.user.append(line[0])
                self.pro.append(line[1])
                self.label.append(self.change_label(int(float(line[2]))))
                self.comm.append(line[3].lower())

    def change_label(self, label):
        if label < 4:  # 负向1-3---0
            label = 0
        elif label < 5:  # 中性4---1
            label = 1
        else:  # 正向5---2
            label = 2
        return label

class Dataset_sen(object):
    def __init__(self, data_path):
        self.label = []
        self.comm = []
        ExcelFile = xlrd.open_workbook(data_path)
        sheet = ExcelFile.sheet_by_index(0)
        rows = sheet.nrows
        flag = 0
        for row in range(rows):
            # print(row)
            if (row + 2) % 4 == 0:
                sentences = sheet.row_values(row)
                for sen in sentences:
                    if sen == 0.0:
                        flag = 1
                        break
                    if sen == '':
                        break
                    self.comm.append([sen])
            if (row + 1) % 4 == 0:
                if flag == 1:
                    flag = 0
                    continue
                labels = sheet.row_values(row)
                for label_ in labels:
                    if label_ == '':
                        break
                    if int(label_) not in [0, 1, 2]:
                        print("error---", row, int(label_))
                    self.label.append(int(label_))
        print(Counter(self.label))



# # movie
# class Dataset(object):
#     def __init__(self, data_path):
#         self.user = []
#         self.pro = []
#         self.label = []
#         self.comm = []
#         with open(data_path, 'r', encoding='UTF-8') as f:
#             for line in f:
#                 line = line.strip().split('\t\t')
#                 self.user.append(line[0])
#                 self.pro.append(line[1])
#                 self.label.append(self.change_label(int(line[2])))
#                 self.comm.append(line[3].lower())
#
#     def change_label(self, label):
#         if label < 5:  # 负向1-4---0
#             label = 0
#         elif label < 7:  # 中性5-6---1
#             label = 1
#         else:  # 正向7-10---2
#             label = 2
#         return label
# # movie
# class Dataset_sen(object):
#     def __init__(self, data_path):
#         self.label = []
#         self.comm = []
#         ExcelFile = xlrd.open_workbook(data_path)
#         sheet = ExcelFile.sheet_by_index(0)
#         rows = sheet.nrows
#         for row in range(rows):
#             # print(row)
#             if row % 2 == 0:
#                 self.comm.append(sheet.row_values(row))
#             else:
#                 labels = sheet.row_values(row)
#                 if int(labels[0]) not in [0, 1, 2]:
#                     print("error---", row, int(labels[0]))
#                 self.label.append(int(labels[0]))
#         self.comm_size = len(self.comm)



class Write2File:
    def __init__(self):
        pass

    @staticmethod
    def append(filepath, content):
        if filepath is not None:
            with open(filepath, "a", encoding="utf-8") as f:
                for item in content:
                    f.write(str(item) + ' ')
                f.write('\n')
            f.close()

    @staticmethod
    def write(filepath, content):
        if filepath is not None:
            with open(filepath, "w", encoding="utf-8") as f:
                for item in content:
                    f.write(str(item) + ' ')
                f.write('\n')
            f.close()

# 特征嵌入
def word_embedding(feature_embedding_path, words):
    word_embedding = []
    flag = 0
    for word in words:
        with open(feature_embedding_path, 'r', encoding='UTF-8') as f:
            for line in f:
                line = line.strip().split()
                if line[0] == word:
                    flag = 1
                    embedding = [float(s) for s in line[1:]]
                    word_embedding.append(embedding)
                    break
        if flag == 1:
            flag = 0
        else:
            # pass
            print(word + " don't have word embedding")
    return np.array(word_embedding)

# 特征词嵌入
def feature_embedding(embedding_file_path, feature_path, k):
    feature_set = []
    delete_set = []
    feature_embedding = []
    if k == 0:
        with open(feature_path, 'r', encoding='UTF-8') as lines:
            for line in lines:
                feature_set.append(line.split('\'')[1])
    else:
        with open(feature_path, 'r', encoding='UTF-8') as lines:
            for i, line in enumerate(lines):
                if i == k:
                    feature_set = line.split()
                    break
    dict_ = dict()
    embedding = []
    index = 0
    with open(embedding_file_path, 'r', encoding='UTF-8') as f:
        for line in f:
            line = line.strip().split()
            if len(line) == 2: continue
            embedding_ = [float(s) for s in line[1:]]
            embedding.append(embedding_)
            dict_[line[0]] = index
            index += 1
    for feature in feature_set:
        if feature in dict_:
            index = dict_[feature]
            feature_embedding.append(embedding[index])
        else:
            print("error: " + feature + " don't have word embedding")
            delete_set.append(feature)
    for feature in delete_set:
        feature_set.remove(feature)
    return np.asarray(feature_embedding), feature_set

# 写入excel
def write_excel_data(excel_path, row, col, value_data):
    # 打开文件，并且保留原格式
    rbook = xlrd.open_workbook(excel_path, formatting_info=True)
    # 使用xlutils的copy方法使用打开的excel文档创建一个副本
    wbook = copy(rbook)
    # 使用get_sheet方法获取副本要操作的sheet
    w_sheet = wbook.get_sheet(0)
    # 写入数据参数包括行号、列号、和值（其中参数不止这些）
    w_sheet.write(row, col, value_data)
    # 保存
    wbook.save(excel_path)

# 写入excel
def write2excel(excel_path, datas):
    rbook = xlrd.open_workbook(excel_path)
    table = rbook.sheets()[0]
    rows = table.nrows
    for i, data in enumerate(datas):
        write_excel_data(excel_path, rows, i, data)

def tf_idf(sents):
    tfidf_Model = TfidfModel.load("../model/tf_idf.model")
    dictionary = corpora.Dictionary.load('../model/sampleDict.dict')
    vector = []
    for sen in sents:
        sen_doc = dictionary.doc2bow(sen[0].split())
        sen_tfidf = tfidf_Model[sen_doc]
        vector.append(sen_tfidf)
    rows = []
    cols = []
    datas = []
    length = len(dictionary.token2id)
    from scipy.sparse import csr_matrix
    for i, doc in enumerate(vector):
        for (row, col) in doc:
            rows.append(i)
            cols.append(row)
            datas.append(col)
    print(rows, cols, datas)
    print(max(rows) + 1, max(cols) + 1, length)
    # vector = csr_matrix((datas, (rows, cols)), shape=(max(rows) + 1, max(cols) + 1))
    vector = csr_matrix((datas, (rows, cols)), shape=(max(rows) + 1, length))
    print(vector)
    return vector

def verify_classification(y_pred, y_true):
    accuracy = accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred, average=None)
    precision = " ".join(str(precision).split())
    recall = metrics.recall_score(y_true, y_pred, average=None)
    recall = " ".join(str(recall).split())
    f1_score = metrics.f1_score(y_true, y_pred, average=None)
    f1_score = " ".join(str(f1_score).split())
    return accuracy, precision, recall, f1_score

def doc2vec(datas):
    vector = []
    # doc2vec = joblib.load('../model/doc2vec2019-01-08-14-54-28.model')  # label为012
    # doc2vec = joblib.load('../model/doc2vec2019-01-08-17-33-23.model')   #label为01
    doc2vec = gensim.models.Doc2Vec.load('../model/doc2vec' +
             datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '.model')  # label为sen序号
    print('doc2vec start: ' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    for data in datas:
        vector.append(doc2vec.infer_vector(data))
    s = StandardScaler()
    s.fit(vector)
    vector = s.transform(vector)
    print('doc2vec end: ' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    # filepath = '../data/vector/doc2vec-sen-train-' + \
    #            datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '.csv'
    # np.savetxt(filepath, vector)
    return vector

# 语料库预处理
def Corpus_pre():
    comment_file = '../data/test.ss'
    corpus_file = '../data/corpus-test.txt'
    commentset = Dataset(comment_file)
    sentences = commentset.comm
    p1 = re.compile(r'-\{.*?(zh-hans|zh-cn):([^;]*?)(;.*?)?\}-')
    p2 = re.compile(r'[(][: @ . , ？！\s][)]')
    p3 = re.compile(r'[「『]')
    p4 = re.compile(r'[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）0-9 , : ; \-\ \[\ \]\ ]')
    for i, sentence in enumerate(sentences):
        sentence = p1.sub(r' ', sentence)
        sentence = p2.sub(r' ', sentence)
        sentence = p3.sub(r' ', sentence)
        sentence = p4.sub(r' ', sentence)
        sentence = sentence.split()
        sentence = [sen for sen in sentence if sen != "<sssss>"]
        sentence = [sen for sen in sentence if sen != "``"]
        Write2File.append(corpus_file, sentence)
        print("评论处理进度： " + str(i) + " " + str(i / len(commentset.comm)))





if __name__ == "__main__":

    content = ["111", "222"]


    Corpus_pre()