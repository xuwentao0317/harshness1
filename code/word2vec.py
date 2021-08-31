import logging  # 引入日志配置
from gensim.models import word2vec
import datetime

def train_word2vec():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # 引入数据集
    sentences = word2vec.Text8Corpus('../data/corpus/corpus-train.txt')

    # 构建模型  最低词频5 skip-Gram 词向量维度200 迭代次数20 并行次数为5
    model = word2vec.Word2Vec(sentences, min_count=5, sg=1, workers=5)
    model.save('../data/model/word2vec' +
             datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '.model')

def retrain(model_path):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = word2vec.Word2Vec.load(model_path)
    sentences = word2vec.Text8Corpus('../results/data_corpus-dev.txt')
    model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
    model.save('../results/model_word2vec' +
             datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '.model')

if __name__ == "__main__":

    train_word2vec()

    # model_path = '../result/model_word2vec.model'    #no
    # retrain(model_path)
