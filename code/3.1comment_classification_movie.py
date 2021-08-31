import datetime

import joblib
from scipy.sparse import csr_matrix
import numpy as np
from gensim.models import word2vec
from gensim import corpora
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from gensim.models import TfidfModel
from stanfordcorenlp import StanfordCoreNLP
from tool import word_embedding, Dataset, Write2File
import warnings


warnings.filterwarnings("ignore")

# 属性词嵌入
def class_embedding(feature_embedding_path, feature_class_path, k):
    global class_vec, class_
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
                dis_[i] -= key_score
                break
    if min(dis_) < threshold:
        c_class = len(class_vec)
    else:
        c_class = dis_.index(min(dis_))
    return c_class

# 将评论进行分割
def comm_split(comment, u_num=0, p_num=0, key_score=0.6, tf_score=0.1, threshold=12000):
    print("--" * 50)
    print(" " * 20, "将评论分割为属性语句向量")
    print("--" * 50)
    feature_class = ["情节", "演员", "配乐", "场景", "其他"]
    attribute = []
    classification = [[] for i in range(len(class_vec) + 1)]
    comment = comment.strip().split('<sssss>')  # 将评论拆分成句子列表
    sentence_num = []
    for sentence in comment:
        sentence = sentence.strip()
        print(sentence)
        # 将名字进行替换 电影名--空 演员名，导演名--演员
        for movie_name in movie_names:
            if movie_name in sentence:
                sentence = sentence.replace(movie_name, '')
        print("sentence ", sentence)
        for people_name in people_names:
            if people_name in sentence:
                if '-lrb-' in sentence and '-rrb-' in sentence:
                    sentence = sentence.replace(people_name, '')
                else:
                    sentence = sentence.replace(people_name, 'actor')
        print("sentence ", sentence)
        word = nlp.word_tokenize(sentence)  # 先对句子进行分词 ['Excellent', 'food', '.']
        word_filtered = [w for w in word if (w not in stopwords.words('english'))]
        sentence_f = " ".join(word_filtered)
        word_tagged = nlp.pos_tag(sentence_f)  # 再对这个分好的句子进行词性标注 [('Excellent', 'JJ'), ('food', 'NN'), ('.', '.')]
        for item in word_tagged:
            if (item[1] in ['NN', 'NNS']):
                if (item[0] not in ['film', 'movie', 'films', 'movies']):
                    attribute.append(item[0])
        print("attribute ", attribute)
        # 去掉一些无用词（tf-idf过低或者tf过高）
        # 将名字进行替换 人物名--角色
        for role_name in role_names:
            if role_name in attribute:
                attribute.remove(role_name)
                attribute.append('actor')
        print("attribute ", attribute)
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
        print("attribute_or ", attribute_or)
        if len(attribute_or) == 0:
            s_class = len(class_)
        else:
            s_class = sent_class(attribute_or, key_score)

        print(len(classification), s_class)
        print(sentence)

        print('属性语句所评论的商品属性: ' + feature_class[s_class])


        print(len(classification), s_class)
        print(sentence)


        classification[s_class].append(sentence)
        sentence_num.append(s_class)
        attribute.clear()
    print(" " * 20, "分割后的属性语句向量")
    print("属性语句向量", classification[:-1])
    for idx, cf in enumerate(classification[:-1]):
        print(feature_class[idx] + ":", cf) #cf[0]
    return classification[:-1]

# 初始化，加载模型
def initialization():
    global tfidf_Model, dictionary, svm, nlp, word2vec_model
    global feature_word, movie_names, people_names, role_names
    feature_embedding_path = '../data/attribute/word-embedding.txt'
    feature_class_path = '../data/cluster/Mean_Shift/mean_shift_feature.txt'
    tfidf_Model = TfidfModel.load('../model/tf_idf.model')
    dictionary = corpora.Dictionary.load('../model/sampleDict.dict')
    nlp = StanfordCoreNLP(r'stanford-corenlp-full-2018-10-05')
    word2vec_model = word2vec.Word2Vec.load('../model/word2vec.model')

    svm = joblib.load('../model/svm.model')
    class_vec, class_ = class_embedding(feature_embedding_path, feature_class_path, 0)

    lines = open('../data/name/movie_name.csv', 'r', encoding='utf-8')
    movie_names = []
    for line in lines:
        movie_names.append(line.strip().lower())
    lines = open('../data/name/people_name.csv', 'r', encoding='utf-8')
    people_names = []
    for line in lines:
        people_names.append(line.strip().lower())
    lines = open('../data/name/role_name.csv', 'r', encoding='utf-8')
    role_names = []
    for line in lines:
        role_names.append(line.strip().lower())
    feature_word = []
    for class_word in class_:
        for word in class_word:
            if word not in feature_word:
                feature_word.append(word)
                word_mor = wn.morphy(word)
                if word_mor not in feature_word and word_mor != None:
                    feature_word.append(word_mor)

# 分割单个评论
def split_comment():
    comm = "the plot is very attractive, <sssss> but the music is very annoying. "
    print("评论：", comm)
    return comm_split(comm)

# 获取数据集
def split_dataset():
    train_file = '../data/train.ss'
    trainset = Dataset(train_file)
    users = trainset.user
    users_dic = list(set(users))
    users_dic.sort(key=users.index)
    pros = trainset.pro
    pros_dic = list(set(pros))
    pros_dic.sort(key=pros.index)
    comment = trainset.comm
    print('sum : ', len(comment))
    print('start: ' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    threshold = 12000
    key_score = 0.6
    tf_score = 0.1
    for i, com in enumerate(comment):
        print(i, "-" * 20)
        u_num = users_dic.index(users[i])
        p_num = pros_dic.index(pros[i])
        classification = comm_split(com, u_num, p_num, key_score, tf_score, threshold)
        comm_sentiment(classification, u_num, p_num)
    nlp.close()
    print('end: ' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

# 获取属性语句向量情感
def comm_sentiment(classification, u_num=0, p_num=0):
    print(" " * 20, "获取属性语句的用户情感")
    write = [u_num]
    k = 4

    print("*******")
    print(len(classification))

    for i, comment in enumerate(classification):
        # print("属性语句：", comment[0].strip())
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
        Write2File.append("../data/EM/em_data" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".csv", write)
        write.clear()
        write.append(u_num)

if __name__ == "__main__":
    # 初始化，加载模型
    initialization()
    # 分割数据集
    split_dataset()
    #分割单个评论，将评论分割为属性语句向量
    # att_sent_vector = split_comment()  # 不必运行

