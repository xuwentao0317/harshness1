import sys
import datetime
from nltk.corpus import stopwords
from stanfordcorenlp import StanfordCoreNLP
from tool import Dataset
import nltk
nltk.download('stopwords')

#  进度条
def View_Bar(flag, sum):
    rate = float(flag) / sum
    rate_num = rate * 100
    if flag % 15.0 == 0:
        print('\r%.2f%%: ' % (rate_num))
        sys.stdout.flush()

#  词性标注
def Tag_Word(path):  # path 是所有用户的评论文件路径
    nlp = StanfordCoreNLP(r'stanford-corenlp-full-2018-10-05')
    # 分词、赋词性
    f = open('../data/attribute/word_tagged_sentences-' +
             datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '.txt', 'w', encoding='utf-8')  # 保存一下词性标注后的结果
    flag = 0  # 进度条
    trainset = Dataset(path)
    for text in trainset.comm:
        flag += 1
        sentences = text.split('<sssss>')  # 将文本拆分成句子列表
        # 先对每个句子进行分词，在对这个句子进行词性标注
        for sentence in sentences:
            word = nlp.word_tokenize(sentence)  # 先对句子进行分词 ['Excellent', 'food', '.']
            word_filtered = [w for w in word if (w not in stopwords.words('english'))]
            sentence_f = " ".join(word_filtered)
            word_tagged = nlp.pos_tag(sentence_f)  # 再对这个分好的句子进行词性标注 [('Excellent', 'JJ'), ('food', 'NN'), ('.', '.')]
            for item in word_tagged:  # 将标注好的词写入文件中
                f.write(item[0] + '/' + item[1] + ' ')  # 'Excellent/JJ food/NN ./. '
            f.write('\n')
        print('分词进度: ' + str(flag))
    nlp.close()
    return 0

#  选择属性
def Featuer_Word(path, window):  # path 是词性标注后的评论句子
    lines = open(path, 'r', encoding='UTF-8').readlines()
    len_lines = float(len(lines))
    tagged_sentences = []  # 保存所有标注好的句子
    feature_list = []  # 挖到的feature

    # 设置一个滑窗，寻找距离这个滑窗最近的一个NN、NNS
    def Slip_Window_Func(tagged_sentence, i, window):
        len_sentence = len(tagged_sentence)
        feature = ''
        k = 1

        while k <= window:  # 同时向目标词两边找 NN\NNS
            if i - k >= 0:
                if len(tagged_sentence[i - k]) == 1:
                    break
                if tagged_sentence[i - k][1] in ['NN', 'NNS']:
                    feature = tagged_sentence[i - k][0]
            if i + k < len_sentence:
                if len(tagged_sentence[i + k]) == 1:
                    break
                if tagged_sentence[i + k][1] in ['NN', 'NNS']:
                    feature = tagged_sentence[i + k][0]
            if feature == '':
                k += 1
                continue
            else:
                break
        return feature

    # 数据预处理
    flag = 0  # 进度条
    print('-' * 20 + '数据预处理进度: ' + '-' * 20)
    for line in lines:  # 预处理一下字符串 'Excellent/JJ food/NN ./. \n'
        sentence = line[:-3].split(' ')  # ['Excellent/JJ', 'food/NN', './.']
        tagged_sentence = []  # 标注好的一个句子 [('Excellent','JJ'), ('food','NN'), ('.','.')]
        for item in sentence:
            tagged_sentence.append(item.split('/'))
        tagged_sentences.append(tagged_sentence)
        flag += 1
        View_Bar(flag, len_lines)

    # 使用滑窗window确定 feature
    flag = 0  # 进度条
    print('-' * 20 + 'feature挖掘进度: ' + '-' * 20)
    for tagged_sentence in tagged_sentences:
        for i, tagged_word in enumerate(tagged_sentence):  # ('Excellent','JJ')
            if len(tagged_sentence) == 1:
                continue
            if len(tagged_word) == 1:
                continue
            if tagged_word[1] == ('JJ' or 'JJR' or 'JJS'):  # 如果遇到形容词、比较级、最高级的话
                feature = Slip_Window_Func(tagged_sentence, i, 5)  # 设置一个滑窗，寻找距离这个滑窗最近的一个NN、NNS
                if feature != '' and feature_list != []:  # 如果挖到了feature的话
                    if feature != feature_list[-1]:  # 这一步是防止挖到有滑窗交集的feature
                        feature_list.append(feature)
                elif feature != '' and feature_list == []:
                    feature_list.append(feature)
                else:
                    continue
        flag += 1
        View_Bar(flag, len_lines)

    # 将feature词汇保存一下
    f = open('../data/attribute/feature-' +
             datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '.txt', 'w', encoding='UTF-8')
    for item in feature_list:
        f.write(str(item) + '\n')
    print('feature词汇保存完毕')

#  筛选属性
def Feature_Data_Cleaning(path):  # path是装有feature词汇的文件路径
    lines = open(path, 'r', encoding='UTF-8').readlines()
    feature_dict = {}  # 保存feature的字典

    # 把原始文件放到字典中
    for feature in lines:
        print(feature)
        feature = feature[:-1]
        if feature not in feature_dict:  # 如果字典里没有这个feature
            feature_dict[feature] = 1  # 赋一下key-value对
        else:  # 如果有这个feature
            feature_dict[feature] += 1

    # 对字典排序
    feature_dict = sorted(feature_dict.items(), key=lambda asd: asd[1], reverse=True)  # 对value进行降序排序

    sum = 0
    # 将feature出现次数超过阀值的保存
    threshold = 600
    f = open('../data/attribute/feature_dict_select-' + str(threshold) + "-" +
             datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '.txt', 'w', encoding='UTF-8')
    for item in feature_dict:
        if (int(item[1]) >= threshold):
            sum += 1
            print(sum)
            f.write(str(item) + '\n')

    print('原始feature数目: ' + str(len(lines)))
    print('放到dict中的数目：' + str(len(feature_dict)))
    print('选择的feature数目：' + str(sum))
    return 0

if __name__ == "__main__":

    #   词性标注
    Tag_Word('../data/train.ss')

    #   选择属性
    # Featuer_Word('../data/attribute/word_tagged_sentences.txt', window=5)


    #   筛选属性
    # Feature_Data_Cleaning('../results/attribute_feature.txt')

