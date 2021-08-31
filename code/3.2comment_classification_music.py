import re
import random
import gensim
import xlrd
import datetime
import numpy as np
from autocorrect import spell
from nltk.corpus import names as corpus_names
from gensim.models import word2vec
from nltk.corpus import stopwords
from gensim import corpora, models
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from gensim.models import TfidfModel
import joblib
from stanfordcorenlp import StanfordCoreNLP
from tool import doc2vec, word_embedding, Dataset, write2excel, Write2File
from scipy.sparse import csr_matrix

# 属性词嵌入
def class_embedding(feature_embedding_path, feature_class_path, k):

    class_vec = []
    class_ = []

    with open(feature_class_path, 'r', encoding='UTF-8') as lines:
        for i, line in enumerate(lines):
            if (i == 0) and (k == 0):
                continue
            line = line.strip().split(' ')
            class_.append(line)
            word_embedding_ = word_embedding(feature_embedding_path, line)
            if len(word_embedding_) != len(line):
                print("error: class_embedding is wrong")
            else:
                class_vec.append(np.array(word_embedding_))
    return class_vec, class_

# 确定属性语句评论类别
def sent_class(attribute, key_score):
    threshold = 0.001
    word_embedding_ = []
    for word in attribute:
        if word in word2vec_model.wv:
            embed = word2vec_model.wv[word]
            word_embedding_.append(embed)
    dis_ = [0 for i in range(len(class_vec))]
    for word in word_embedding_:
        for i in range(len(class_vec)):
            dis_[i] += sum(np.linalg.norm(class_vec[i] - word, axis=1)) / len(class_vec[i])
    for word in attribute:
        for i, class_w in enumerate(class_):
            if word in class_w:
                id = class_w.index(word)

                dis_[i] -= key_score * (len(class_w) - id) / len(class_w)
                # dis_[i] -= key_score

                break
    if min(dis_) < threshold:
        c_class = len(class_vec)
    else:
        c_class = dis_.index(min(dis_))
    return c_class


# 将评论进行分割
def comm_split(num, pro, comment, key_score, label, tf_score, threshold):
    feature_class = ["歌手", "乐器", "旋律", "风格", "其他"]
    attribute = []
    classification = [[] for i in range(len(class_vec) + 1)]
    comment = comment.strip().split('<sssss>')  # 将评论拆分成句子列表
    sentence_num = []
    debug = 0
    write_flag = 0
    for sentence in comment:
        if len(sentence) < 1:
            print("continue")
            continue
        # 处理歌名（单词两个及以上）
        sentence_or = sentence
        for song_name in song_names:
            if song_name in sentence:
                sentence = sentence.replace(song_name, '')
        # print("sentence_new: ", sentence)
        if debug == 1: print("sentence_new: ", sentence)
        # 分词，去停用词，词性标注
        word = nlp.word_tokenize(sentence)  # 先对句子进行分词 ['Excellent', 'food', '.']
        word_filtered = [w for w in word if (w not in stopwords.words('english'))]
        sentence_f = " ".join(word_filtered)
        word_tagged = nlp.pos_tag(sentence_f)  # 再对这个分好的句子进行词性标注 [('Excellent', 'JJ'), ('food', 'NN'), ('.', '.')]
        if debug == 1: print('word_tagged: ', word_tagged)
        # 根据词性，抽取关键词
        flag = 0
        for item in word_tagged:
            if item[0] == '"':  # "..."中间的词多为歌名，不计入
                if flag == 0:
                    flag = 1
                    continue
                else:
                    flag = 0
            if flag == 1 and item[0] in feature_word:  # "..."关键词计入(有关键词通常为话语)
                attribute.append(item[0])
            if flag == 0 and (item[1] in ['NN', 'NNS']):  # , 'JJ', 'JJR', 'JJS'
                if (item[0] not in ['album', 'songs', 'song', 'music', 'cd', 'albums', 'tracks', 'track', 'hit',
                                    'hits', 'singles', 'single']):
                    attribute.append(item[0])
        if debug == 1: print("step1: attribute", attribute)
        if len(attribute) != 0:
            # 根据命名实体标注，去掉除人名以外的实体
            word_ner = nlp.ner(" ".join(attribute))
            for word, tag in word_ner:
                if tag != 'O' and word not in feature_word:
                    if tag != 'PERSON':
                        if word in attribute:
                            attribute.remove(word)
                        else:
                            print("error", word, word_ner)
                    else:
                        attribute.append('artist')
            if debug == 1: print("step2: attribute", attribute)
            # 去掉一些无用词（tf-idf过低或者tf过高）
            attribute_or = attribute.copy()
            for att in attribute:
                if dictionary.doc2bow([att]):
                    id = dictionary.doc2bow([att])[0][0]
                else:
                    continue
                sen_doc = dictionary.doc2bow(sentence.split())
                corpus_tfidf = tfidf_Model[sen_doc]
                tf_idf = [tf_idf for word, tf_idf in corpus_tfidf if word == id]
                if att not in feature_word and wn.morphy(att) not in feature_word and tf_idf:
                    if tf_idf[0] < tf_score or dictionary.dfs[id] > threshold:
                        attribute_or.remove(att)
            if debug == 1:
                print("step3: attribute", attribute_or)
                if len(attribute_or) != len(attribute):
                    print(attribute, " != ", attribute_or)
            # 进行句子属性标注
            if len(attribute_or) == 0:
                s_class = len(class_)
            else:
                s_class = sent_class(attribute_or, key_score)
        else:
            s_class = len(class_)
        if debug == 1: print('sentence class is: ' + str(s_class) + " " + feature_class[s_class])
        classification[s_class].append(sentence_or)
        sentence_num.append(s_class)
        attribute.clear()
    # 写入文件
    write = []
    id_sen = []
    for i, cl in enumerate(classification[:-1]):
        if cl != []:
            if len(cl) != 0:
                cl = [" ".join(cl)]
                write.append(cl)
                id_sen.append(i)
    if write_flag == 1:
        excel_path = "../data/classification/svm-classification.xls"
        print("writing--------------------------")
        write2excel(excel_path, [num, pro, label])
        if len(id_sen) == 0:
            write2excel(excel_path, [0])
            write2excel(excel_path, [0])
        write2excel(excel_path, id_sen)
        write2excel(excel_path, write)
        write2excel(excel_path, [0])
    return write

# 初始化，加载模型
def initialization():
    global tfidf_Model, dictionary, svm, nlp, word2vec_model
    global feature_word, song_names, class_vec, class_
    # feature_embedding
    feature_embedding_path = '../data/attribute/word-embedding-music.txt'
    feature_class_path = '../data/cluster/Mean_Shift/mean_shift_feature-music.txt'
    class_vec, class_ = class_embedding(feature_embedding_path, feature_class_path, 0)
    # model
    tfidf_Model = TfidfModel.load("../model/tf_idf_music.model")

    dictionary = corpora.Dictionary.load('../model/sampleDict_music.dict')
    nlp = StanfordCoreNLP(r'stanford-corenlp-full-2018-10-05')
    word2vec_model = word2vec.Word2Vec.load('../model/word2vec_music.model')
    # key_word
    lines = open('../data/name/song_name.csv', 'r', encoding='utf-8')
    song_names = []
    for line in lines:
        song_names.append(line.strip())
    feature_word = []
    for class_word in class_:
        for word in class_word:
            if word not in feature_word:
                feature_word.append(word)
                word_mor = wn.morphy(word)
                if word_mor not in feature_word and word_mor != None:
                    feature_word.append(word_mor)

# 获取数据集
def split_dataset():
    train_file = '../data/train-music.csv'
    trainset = Dataset(train_file)
    pros = trainset.pro
    comment = trainset.comm
    labels = trainset.label
    print('start: ' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    key_score = 2.4
    tf_score = 0.2
    threshold = 500
    sentences = []
    for num in range(len(labels)):  #len(labels)
        print(num, "-" * 20)
        comm_split(num, pros[num], comment[num], key_score, labels[num], tf_score, threshold)
    print('end: ' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

# 获取属性语句向量情感
def comm_sentiment(classification, u_num=0, p_num=0):
    print(" " * 20, "获取属性语句的用户情感")
    write = [u_num]
    k = 4
    for i, comment in enumerate(classification):
        print("属性语句：", comment[0].strip())
        write.append(p_num * k + i)
        if len(comment) != 0:
            if len(comment) != 1:
                comment = [" ".join(comment)]
            sen_doc = dictionary.doc2bow(comment[0].split())
            tfidf = tfidf_Model[sen_doc]
            length = len(dictionary.token2id)
            rows = []
            cols = []
            datas = []
            for row, col in tfidf:
                rows.append(0)
                cols.append(row)
                datas.append(col)
            vector = csr_matrix((datas, (rows, cols)), shape=(1, length))
            result_pro = svm.predict_proba(vector)
            write.append(result_pro[0])
            print("消极：{} 中立：{} 积极：{}".format(round(result_pro[0][0], 4), round(result_pro[0][1], 4), round(result_pro[0][2], 4)))
        else:
            write.append([0.0, 0.0, 0.0])
        print()
        Write2File.append("../data/EM/em_data" +
             datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".csv", write)
        write.clear()
        write.append(u_num)

if __name__ == "__main__":
    # 初始化，加载模型
    initialization()
    # 分割数据集
    split_dataset()
    # 判断属性语句中用户情感
    # comm_sentiment(att_sent_vector)
