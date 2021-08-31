import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import joblib
from sklearn.model_selection import cross_val_score
from tool import write2excel, Dataset_sen, tf_idf

class SVMClassifier:
    def __init__(self, C):
        self.clf = SVC(C=C, probability=True, gamma='auto')

    def train(self, train_data, train_labels, c):
        svm = self.clf.fit(train_data, np.array(train_labels))
        joblib.dump(svm, '../model/svm/svm' +
             datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '.model')

    def predict(self, test_data):
        prediction = self.clf.predict(test_data)
        return prediction

def parameter_sen(train_file, train_num):
    trainset = Dataset_sen(train_file)
    train_vec = tf_idf(trainset.comm[:train_num])
    train_label = trainset.label[:train_num]
    excel_path = "../model/svm_pa.xls"
    c_ran = [c for c in range(2, 10)]
    scores = []
    for c in c_ran:
        print('start: ' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        print("-" * 10 + str(c) + " is testing" + "-" * 10)
        svm = SVC(C=c, probability=True, gamma="auto", class_weight='balanced')
        score = cross_val_score(svm, train_vec, train_label, cv=5, scoring='accuracy')
        scores.append(score.mean())
        write2excel(excel_path,
                    [c, train_file, train_num, 'tf', score.mean()])
    plt.plot(c_ran, scores)
    plt.xlabel('Value of C for SVM')
    plt.ylabel('Cross_Validation Accuracy')
    plt.show()
    print('end: ' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

def train_svm_sen(train_file, train_num, C=2):
    trainset = Dataset_sen(train_file)
    train_vec = tf_idf(trainset.comm[:train_num])
    train_label = trainset.label[:train_num]
    print("train svm start: " + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    print("train_num: " + str(train_num))
    svm = SVMClassifier(C)
    svm.train(train_vec, train_label, C)
    print("train svm end: " + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

if __name__ == "__main__":

    # movie
    train_file = '../model/movie-train-svm.xls'
    train_num = 2000
    # 参数优化
    parameter_sen(train_file, train_num)
    # 训练模型
    train_svm_sen(train_file, train_num, C=15)
