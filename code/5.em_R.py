import os
import pandas as pd
import logging
import random
import datetime
import numpy as np
import scipy as sp
import scipy.stats
import scipy.optimize
from collections import Counter
from tool import verify_classification
from tool import Dataset as ds

np.set_printoptions(suppress=True)  # 不使用科学计数
THRESHOLD = 1e-5

verbose = True
debug = True
logger = None

# warnings.filterwarnings('error')
def init_logger():
    global logger
    LOG_FILE = r'../data/EM/log'
    logger = logging.getLogger("SAM''")
    logger.setLevel(logging.DEBUG)
    log_fmt = '%(asctime)s/%(name)s[%(levelname)s]: %(message)s'
    logging.basicConfig(format=log_fmt)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def logsigmoid(x):
    return - np.log(1 + np.exp(-x))

class Dataset(object):
    def __init__(self, labels=None,
                 numLabels=-1, numUsers=-1, numProducts=-1, numClasses=-1,
                 priorAlpha=None, priorBeta=None, priorZ=None,
                 alpha=None, beta=None, probZ=None,
                 products=None, users=None):
        self.labels = labels
        self.numLabels = numLabels
        self.numUsers = numUsers
        self.numProducts = numProducts
        self.numClasses = numClasses
        self.priorAlpha = priorAlpha
        self.priorBeta = priorBeta
        self.priorZ = priorZ
        self.alpha = alpha
        self.beta = beta
        self.probZ = probZ
        self.products = products
        self.users = users

def load_data(label_path, priorZ_path):
    data = Dataset()
    k = 4
    with open(label_path) as f:
        # Read parameters
        header = f.readline().split()
        data.numLabels = int(header[0])
        data.numUsers = int(header[1])
        data.numProducts = int(header[2]) * k
        data.numClasses = int(header[3])
        data.products = [[] for i in range(data.numProducts)]
        data.users = [[] for i in range(data.numUsers)]
        if verbose:
            logger.info('Reading {} labels of {} users over {} products '.format(data.numLabels, data.numUsers,
                                                                                 data.numProducts))
        # Read Labels
        data.labels = [[[] for i in range(data.numProducts)] for i in range(data.numUsers)]
        for lines in f:  # 读入用户、产品、评论等级概率
            line = lines.split()
            user = int(line[0])
            product = int(line[1])
            line[2] = line[2].split('[')[1]
            line[2] = line[2].split(',')[0]
            line[3] = line[3].split(',')[0]
            line[4] = line[4].strip().split(']')[0]
            label = list(map(float, line[2:5]))
            if np.sum(label) == 0:
                continue
            data.labels[user][product] = label
            data.products[product].append(user)
            data.users[user].append(product)
    # Initialize Probs
    data.priorZ = [[] for i in range(data.numProducts)]
    data.priorAlpha = np.ones(data.numUsers) / 1000
    data.priorBeta = np.ones(data.numProducts) / 1000
    data.probZ = np.zeros((data.numProducts, data.numClasses))
    data.alpha = np.ones(data.numUsers) / 1000
    data.beta = np.ones(data.numProducts) / 1000
    with open(priorZ_path) as f1:
        for pro, line in enumerate(f1):
            temp = list(map(float, line.split()))
            for i, item in enumerate(temp):
                if item < 0.001:
                    temp[i] = 0.001
            for i in range(k):
                data.priorZ[pro * k + i] = temp
    return data

def EM(data):
    EStep(data)
    epoch = 0
    print("epoch:", epoch)
    epoch = epoch + 1
    lastQ = computeQ(data)
    MStep(data)
    Q = computeQ(data)
    counter = 1
    while abs((Q - lastQ) / lastQ) > THRESHOLD:
        if verbose: logger.info('EM: iter={} ----------------'.format(counter))
        lastQ = Q
        EStep(data)
        MStep(data)
        Q = computeQ(data)
        counter += 1
    if verbose:
        logger.info('EM ending ------------------')

# 单个属性  P(lij|lj,ai,bj) = sum(P(lij=k) * P(lij=k|lj,ai,bj))
# sum(P(lij|lj,ai,bj))= P1*P2*P3*P4
def plij_lj_ai_bj(user, pro, true_label, data):
    sum = 0
    labels = data.labels[user][pro]
    for label_, label_pro in enumerate(labels):  # 对属性评级的概率
        if label_ == true_label:
            maxtrix_R = label_pro
        else:
            maxtrix_R = (1 - label_pro) / 2
        p_label = 1 / (1 + np.exp(-data.alpha[user] * data.beta[pro])) * maxtrix_R
        assert 1 > p_label >= 0, 'data.alpha[user]={} data.beta[pro]={} maxtrix_R={} p_label={}'.format \
            (data.alpha[user], data.beta[pro], maxtrix_R, p_label)
        # if debug:
        #     logger.debug('data.alpha[user]={} maxtrix_R={} p_label={}'.format
        #                  (data.alpha[user], maxtrix_R, p_label))
        if true_label == label_:
            sum += label_pro * p_label
        else:
            sum += label_pro * (1 - p_label) / (data.numClasses - 1)
        # if debug:
        #     logger.debug('label_pro={} p_label={} sum={}'.format(label_pro, p_label, sum))
    assert 1 > sum > 0, 'Invalid Value plij_lj_ai_bj = {}'.format(sum)
    return sum

def EStep(data):
    u"""Evaluate the posterior probability of true labels given observed labels and parameters
    """
    # P(true|pre,a,b) = P(z)*P(pre|true,a,b)
    # P(pre|true,a,b) = exp((alpha+1-beta)*R(pre,true))
    if verbose: logger.info('EStep--------------------------------')
    for pro in range(data.numProducts):
        for true_label in range(data.numClasses):
            sum = 1.0
            for user in data.products[pro]:
                sum *= plij_lj_ai_bj(user, pro, true_label, data)
            data.probZ[pro][true_label] = sum * data.priorZ[pro][true_label]
            assert 1 >= data.probZ[pro][true_label] >= 0, 'Invalid Value pZk = {}'.format(data.probZ[pro][true_label])

    # 归一化
    s = data.probZ.sum(axis=1)
    data.probZ = (data.probZ.T / s).T
    assert not np.any(np.isnan(data.probZ)), 'Invalid Value [EStep]'
    return data

def computeQ(data):
    u"""Calculate the expectation of the joint likelihood
    """
    # PklnP(lj=k)
    Q1 = (data.probZ * np.log(data.priorZ)).sum()

    # PklnP(lij|zj,ai,bj)
    Q2 = 0
    for user in range(data.numUsers):
        for product in data.users[user]:
            for true_label in range(data.numClasses):
                labels = data.labels[user][product]
                for label_, label_pro in enumerate(labels):
                    if label_ == true_label:
                        maxtrix_R = label_pro
                    else:
                        maxtrix_R = (1 - label_pro) / 2
                    sigma = 1 / (1 + np.exp(-data.alpha[user] * data.beta[product])) * maxtrix_R
                    if label_ == true_label:
                        Q2 += data.probZ[product][true_label] * np.log(sigma) * label_pro
                    else:
                        Q2 += data.probZ[product][true_label] * np.log(1 - sigma) * label_pro

    # Add Gaussian (standard normal) prior for alpha
    Q_alpha = np.log(sp.stats.norm.pdf(data.alpha - data.priorAlpha)).sum()

    # Add Gaussian (standard normal) prior for beta
    Q_beta = np.log(sp.stats.norm.pdf(data.beta - data.priorBeta)).sum()

    Q = - Q1 + Q2 + Q_alpha + Q_beta
    if debug:
        logger.debug('a[0]={} b[0]={}'.format(data.alpha[0], data.beta[0]))
        logger.debug('Q={} Q1={} Q2={} Q_alpha={} Q_beta={}'.format(Q, -Q1, Q2, Q_alpha, Q_beta))
    if np.isnan(Q):
        return -np.inf
    return Q

def MStep(data):
    # if verbose: logger.info('MStep--------------------------------')
    initial_params = np.r_[data.alpha.copy(), data.beta.copy()]
    params = sp.optimize.minimize(fun=f, x0=initial_params, args=(data,), method='SLSQP',  # SLSQP BFGS CG
                                  jac=df, tol=0.01,
                                  options={'maxiter': 20, 'disp': verbose})  # 'ftol': 0.01
    # if debug:
    #     logger.info('MStep end-----------------------------')
    #     logger.debug(params)
    data.alpha = params.x[:data.numUsers].copy()
    data.beta = params.x[data.numUsers:].copy()

def f(x, *args):
    u"""Return the value of the objective function
    """
    data = args[0]
    d = Dataset(labels=data.labels, numLabels=data.numLabels, numUsers=data.numUsers,
                numProducts=data.numProducts, numClasses=data.numClasses,
                priorAlpha=data.priorAlpha, priorBeta=data.priorBeta, priorZ=data.priorZ, probZ=data.probZ,
                products=data.products, users=data.users)
    d.alpha = x[:data.numUsers].copy()
    d.beta = x[data.numUsers:].copy()
    return - computeQ(d)

def df(x, *args):
    u"""Return gradient vector
    """
    data = args[0]
    d = Dataset(labels=data.labels, numLabels=data.numLabels, numUsers=data.numUsers,
                numProducts=data.numProducts, numClasses=data.numClasses,
                priorAlpha=data.priorAlpha, priorBeta=data.priorBeta, priorZ=data.priorZ, probZ=data.probZ,
                products=data.products, users=data.users)
    d.alpha = x[:data.numUsers].copy()
    d.beta = x[data.numUsers:].copy()
    dQdAlpha, dQdBeta = gradientQ(d)
    # Flip the sign since we want to minimize
    return np.r_[-dQdAlpha, -dQdBeta]

def gradientQ(data):
    # prior prob.
    # dQdAlpha = - (data.alpha - data.priorAlpha)
    dQ = np.zeros((data.numUsers, data.numProducts))
    for user in range(data.numUsers):
        for product in data.users[user]:
            for true_label in range(data.numClasses):
                gradient = 0
                labels = data.labels[user][product]
                for label_, label_pro in enumerate(labels):  # 分别计算A,B,C类概率和
                    if label_ == true_label:
                        maxtrix_R = label_pro
                    else:
                        maxtrix_R = (1 - label_pro) / 2
                    sigma = 1 / (1 + np.exp(-data.alpha[user])) * maxtrix_R
                    beta = 1 / (1 + np.exp(-data.alpha[user]))
                    if true_label == label_:
                        gradient += label_pro * data.probZ[product][true_label] * (1 - beta)
                    else:
                        gradient -= label_pro * data.probZ[product][true_label] * sigma / (1 - sigma) * (1 - beta)
            dQ[user][product] = gradient
    dQdAlpha = dQ.sum(axis=1)
    dQdBeta = dQ.sum(axis=0)
    # if debug:
    #     logger.debug('dQdAlpha[0]={} dQdBeta[0]={}'.format(dQdAlpha[0], dQdBeta[0]))
    return dQdAlpha, dQdBeta

def max_label(s0, s1, s2, p):
    max_len = [s0, s1, s2]
    max_k = max_len.index(max(max_len))
    if max_len[1] == max_len[2] and max_len[0] == max_len[2]:
        if p != -1:
            print(p, " 0=1=2")
        max_k = random.randint(0, 2)
        return max_k
    if max_k == 0:
        if max_len[0] == max_len[1]:
            if p != -1:
                print(p, " 0=1")
            ran = random.randint(0, 1)
            if ran == 0:
                max_k = 0
            else:
                max_k = 1
        elif max_len[0] == max_len[2]:
            if p != -1:
                print(p, " 0=2")
            ran = random.randint(0, 1)
            if ran == 0:
                max_k = 0
            else:
                max_k = 2
    elif max_k == 1:
        if max_len[1] == max_len[2]:
            if p != -1:
                print(p, " 1=2")
            ran = random.randint(0, 1)
            if ran == 0:
                max_k = 1
            else:
                max_k = 2
    return max_k

def kappa_kendall(prd_labels, true_labels):
    # kappa
    testData = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for pro in range(100):
        testData[prd_labels[pro]][true_labels[pro]] += 1
    dataMat = np.mat(testData)
    P0 = 0.0
    for i in range(3):
        P0 += dataMat[i, i] * 1.0
    xsum = np.sum(dataMat, axis=1)
    ysum = np.sum(dataMat, axis=0)
    Pe = float(ysum * xsum) / 100 ** 2
    P0 = float(P0 / 100.0)
    kappa = float((P0 - Pe) / (1 - Pe))
    print(Counter(prd_labels))
    # kendall
    df = pd.DataFrame({'prd_labels': prd_labels, 'true_labels': true_labels})
    # print(df)
    print('kendall:')
    print(df.corr('kendall'))
    print('kappa: ', kappa)

# def compute_svm():
#     probZ = [[] for i in range(1635 * 4)]
#     user_dis = []
#     user_path = "../data/EM/sp_user_distribution_nor.txt"
#     svm_path = "../data/EM/svm.csv"
#     true_label_path = "../data/EM/true_label_movie.txt"
#     with open(user_path) as f:
#         for lines in f:
#             line = lines.split()
#             line = list(map(float, line[1:]))
#             user_dis.append(line)
#     with open(svm_path) as f:
#         header = f.readline().split()
#         numUsers = int(header[1])
#         numProducts = int(header[2]) * 4
#         products = [[] for i in range(numProducts)]
#         users = [[] for i in range(numUsers)]
#         # Read Labels
#         svm_labels = [[[] for i in range(numProducts)] for i in range(numUsers)]
#         for lines in f:  # 读入用户、产品、评论等级概率
#             line = lines.split()
#             user = int(line[0])
#             product = int(line[1])
#             line[2] = line[2].split('[')[1]
#             line[2] = line[2].split(',')[0]
#             line[3] = line[3].split(',')[0]
#             line[4] = line[4].strip().split(']')[0]
#             label = list(map(float, line[2:5]))
#             if np.sum(label) == 0:
#                 continue
#             svm_labels[user][product] = label
#             products[product].append(user)
#             users[user].append(product)
#     with open(true_label_path) as f:
#         true_ij = []
#         true_labels = []
#         for lines in f:  # 读入用户、产品、评论等级概率
#             line = lines.split()
#             true_ij.append((int(line[0]), int(line[1])))
#             true_labels.append(int(line[2]))
#     accuracy = 0
#     while accuracy != 0.7:
#         label_sen = []
#         label_sen_all = []
#         true_labels_sum = 0
#         for product in range(len(probZ)):
#             max_svm = [0, 0, 0]
#             for user in products[product]:
#                 user_label = np.multiply(np.array(user_dis[user]), np.array(svm_labels[user][product]))
#                 user_label = user_label.tolist()
#                 # print(user_dis[user], svm_labels[user][product], user_label)
#                 svm_label = user_label.index(max(user_label))
#                 # svm_label = svm_labels[user][product].index(max(svm_labels[user][product]))
#                 max_svm[svm_label] += 1
#             max_s = max_label(max_svm[0], max_svm[1], max_svm[2], -1)
#             if len(products[product]) == 0:  # no comment
#                 # print("no comment: ", product)
#                 max_s = 1
#             label_sen_all.append(max_s)
#             if (int(product / 4), product % 4) in true_ij:
#                 label_sen.append(max_s)
#                 true_labels_sum += 1
#         print('label_sen', Counter(label_sen))
#         print('-' * 20)
#         accuracy, precision, recall, f1_score = verify_classification(label_sen, true_labels)
#         print('sen: ', accuracy, precision, recall, f1_score)
#         kappa_kendall(label_sen, true_labels)

def label_change(label, th1, th2):
    if label <= th1:
        return 0
    elif label <= th2:
        return 1
    else:
        return 2
    # labels_new = []
    # th1 = 0.3
    # the2 = 0.5
    # for label in labels:
    #     if label <=  th1:
    #         labels_new.append(0)
    #     elif label <= the2:
    #         labels_new.append(1)
    #     else:
    #         labels_new.append(2)
    # return labels_new

def output(data):
    alpha = np.c_[np.arange(1, data.numUsers + 1), data.alpha]
    k = 4
    np.savetxt('../results/EM_alpha-music-' + str(k) +
             datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '.txt', alpha, fmt=['%d', '%.5f'], delimiter=',',
               header='id,alpha')
    beta = np.c_[np.arange(1, data.numProducts + 1), np.exp(data.beta)]
    np.savetxt('../results/EM_beta-music-' + str(k) +
             datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") +'.txt', beta, fmt=['%d', '%.5f'], delimiter=',', header='id,beta')
    label = np.c_[np.arange(1, data.numProducts + 1), data.probZ]
    np.savetxt('../results/EM_label-music-' + str(k) +
             datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") +'.txt', label, fmt=['%d', '%.5f', '%.5f', '%.5f'], delimiter=',',
               header='id,z')



def analysis_model(obj):
    print(" " * 20, "分析模型结果")

    if obj == "music":
        probZ = [[] for i in range(58 * 4)]
    elif obj == "movie":
        probZ = [[] for i in range(1635 * 4)]

    pem_path = "../results/"
    true_label_path = "../data/EM/true_label_music.txt"
    with open(pem_path) as f:
        for pro, line in enumerate(f):
            if pro == 0:
                continue
            temp = list(map(float, line.split(',')[1:]))
            probZ[pro - 1] = temp
    with open(true_label_path) as f:
        true_ij = []
        true_labels = []
        for lines in f:  # 读入用户、产品、评论等级概率
            line = lines.split()
            true_ij.append((int(line[0]), int(line[1])))
            true_labels.append(int(line[2]))
    label_pz = []
    label_pz_all = []
    for product in range(len(probZ)):
        max_probZ = probZ[product].index(max(probZ[product]))
        label_pz_all.append(max_probZ)
        if (int(product / 4), product % 4) in true_ij:
            label_pz.append(max_probZ)
    print('SAM‘’的分布：好评：{}% 中评：{}% 差评：{}%'.format(Counter(label_pz)[2], Counter(label_pz)[1], Counter(label_pz)[0]))
    print('专家评估的分布：好评：{}% 中评：{}% 差评：{}%'.format(Counter(true_labels)[2], Counter(true_labels)[1], Counter(true_labels)[0]))
    accuracy, precision, recall, f1_score = verify_classification(label_pz, true_labels)
    print('准确率:{} 召回率:{} F值:{}'.format(accuracy, recall, f1_score))

def main(label_path, priorZ_path):

    global debug, verbose
    debug = True
    verbose = True

    data = load_data(label_path, priorZ_path)

    EM(data)

    output(data)

if __name__ == '__main__':

    init_logger()

    label_path = '../data/EM/data_music.csv'  # 文本分析结果
    priorZ_path = '../data/EM/priorZ_eq_music.txt'  # 商品评估好中差评为离散均匀分布0.33:0.33:0.33


    # 进行模型训练
    # main(label_path, priorZ_path)


    analysis_model("music")
