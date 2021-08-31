import numpy as np
from operator import itemgetter
from collections import defaultdict
from sklearn.metrics import silhouette_score
from sklearn.cluster import MeanShift, estimate_bandwidth
from tool import feature_embedding, Write2File

def get_seeds(self, data):
    # 获取可以作为起始质心的点（seed）
    seed_list = []
    seeds_fre = defaultdict(int)
    for sample in data:
        seed = tuple(np.round(sample / self.bin_size))  # 将数据粗粒化，以防止非常近的样本点都作为起始质心
        seeds_fre[seed] += 1
    for seed, fre in seeds_fre.items():
        if fre >= 3:
            seed_list.append(np.array(seed))
    if not seed_list:
        raise ValueError('the bin size and min_fre are not proper')
    if len(seed_list) == data.shape[0]:
        return data
    return np.array(seed_list) * self.bin_size

# 电影数据集聚类
# def Mean_Shift_cluster(feature_embedding, feature_set, quantile=0.22, k=7, n_samples=6250):
#     bandwidth = estimate_bandwidth(feature_embedding, quantile=quantile, n_samples=n_samples)
#     if bandwidth <= 0:
#         return
#     ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
#     ms.fit(feature_embedding)  # 训练模型
#     label_pred = ms.labels_  # 所有点的的labels
#     cluster_centers = ms.cluster_centers_  # 聚类得到的中心点
#     n_clusters = len(set(label_pred)) - (1 if -1 in label_pred else 0)
#     clusters_sum = []
#     print(n_clusters)
#     if n_clusters > 1:
#         score = silhouette_score(feature_embedding, label_pred)
#         print('分簇的数目: ' + str(n_clusters) + " 轮廓系数: %0.6f" % score)
#
#         filepath = '../data/cluster/Mean_Shift/mean_shift_feature.txt'
#
#         for i in range(n_clusters):
#             clusters_sum.append(len(feature_embedding[label_pred == i]))
#             print('number of data in Cluster %s is : %s' % (i, len(feature_embedding[label_pred == i])))
#         Write2File.append(filepath, clusters_sum)
#         for i in range(n_clusters):
#             feature_sort = []
#             feature_group = np.array(feature_set)[label_pred == i]
#             for word in enumerate(feature_group):
#                 dis = np.linalg.norm(cluster_centers[i] - feature_embedding[feature_set.index(word[1])])
#                 feature_sort.append((word[1], dis))
#             feature_sort = sorted(feature_sort, key=itemgetter(1), reverse=False)
#             feature_sort = [fea[0] for fea in feature_sort]
#             Write2File.append(filepath, feature_sort)


# 音乐数据集聚类
def Mean_Shift_cluster(feature_embedding, feature_set, quantile=0.12, k=7, n_samples=6250):
    bandwidth = estimate_bandwidth(feature_embedding, quantile=quantile, n_samples=n_samples)
    if bandwidth <= 0:
        return
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(feature_embedding)  # 训练模型
    label_pred = ms.labels_  # 所有点的的labels
    cluster_centers = ms.cluster_centers_  # 聚类得到的中心点
    n_clusters = len(set(label_pred)) - (1 if -1 in label_pred else 0)
    clusters_sum = []
    print(n_clusters)
    if n_clusters > 1:
        score = silhouette_score(feature_embedding, label_pred)
        print('分簇的数目: ' + str(n_clusters) + " 轮廓系数: %0.6f" % score)


        filepath = '../data/cluster/Mean_Shift/mean_shift_feature-music.txt'

        for i in range(n_clusters):
            clusters_sum.append(len(feature_embedding[label_pred == i]))
            print('number of data in Cluster %s is : %s' % (i, len(feature_embedding[label_pred == i])))
        Write2File.append(filepath, clusters_sum)
        for i in range(n_clusters):
            feature_sort = []
            feature_group = np.array(feature_set)[label_pred == i]
            for word in enumerate(feature_group):
                dis = np.linalg.norm(cluster_centers[i] - feature_embedding[feature_set.index(word[1])])
                feature_sort.append((word[1], dis))
            feature_sort = sorted(feature_sort, key=itemgetter(1), reverse=False)
            feature_sort = [fea[0] for fea in feature_sort]
            Write2File.append(filepath, feature_sort)



if __name__ == "__main__":

    #电影数据集
    # embedding_file_path = "../data/attribute/word-embedding-tsne.txt"
    # feature_path = "../data/attribute/feature_select.txt"

    # 音乐数据集
    embedding_file_path = "../data/attribute/feature-embedding-tsne.txt"
    feature_path = "../data/attribute/feature_select_music.txt"

    feature_embedding_, feature_set = feature_embedding(embedding_file_path, feature_path, 0)

    Mean_Shift_cluster(feature_embedding_, feature_set)
