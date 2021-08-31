from gensim import corpora, models
from collections import defaultdict
from gensim.models import TfidfModel
import datetime

def process_corpus(num):
    train_path = '../data/corpus/corpus-train.txt'
    train_comm = []
    dic_comm = []
    train_file = open(train_path, 'r', encoding='UTF-8')
    for line in train_file:
        dic_comm.append(line.split())
        if num == 0:  # 删除词频低于10的
            train_comm.append(line.split())
    frequency = defaultdict(int)
    for text in dic_comm:
        for token in text:
            frequency[token] += 1
    if num == 0:
        dic_comm = [[token for token in text if frequency[token] > 10]
                    for text in dic_comm]
        dictionary = corpora.Dictionary(dic_comm)
    else:
        dictionary = corpora.Dictionary(dic_comm)
        dictionary.save('../model/sampleDict' +
             datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '.dict')
        train_comm = dic_comm
    corpus = [dictionary.doc2bow(text) for text in train_comm]
    corpora.MmCorpus.serialize('../model/corpuse-movie' +
             datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '.mm', corpus)
    return corpus

def train_tf_idf(num):
    corpus = process_corpus(num)
    tfidf_Model = models.TfidfModel(corpus)
    tfidf_Model.save("../model/tf_idf" +
             datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".model")

def tf_idf():
    tfidf_Model = TfidfModel.load("../model/tf_idf.model")
    # 对整个语料库应用转换
    corpus = corpora.MmCorpus('../model/corpuse-movie.mm')
    corpus_tfidf = tfidf_Model[corpus]
    rows = []
    cols = []
    datas = []
    from scipy.sparse import csr_matrix
    for i, doc in enumerate(corpus_tfidf):
        for (row, col) in doc:
            rows.append(i)
            cols.append(row)
            datas.append(col)
    vector = csr_matrix((datas, (rows, cols)), shape=(max(rows) + 1, max(cols) + 1))
    return vector

if __name__ == "__main__":
    process_corpus(1)
    # train_tf_idf(1)
    # tf_idf()
