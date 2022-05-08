import csv
import numpy as np


def calc_distance():
    global distance, data, center
    for i1 in range(data.shape[0]):
        for j1 in range(center.shape[0]):
            distance[i1, j1] = np.linalg.norm(data[i1, ...] - center[j1, ...], ord=2)


step = 0
max_itr = 50
K = 5  # 聚类中心
d = 4  # 数据维度
N = 150  # 数据大小
with open('data.csv', 'r') as f:
    reader = csv.reader(f)
    data = np.zeros((N, d), dtype=float)
    data_id = np.zeros((N, 1), dtype=int)
    i = -1
    for row in reader:
        if i == -1:
            i = i + 1
            continue
        data[i, ...] = row[1:d + 1]
        if row[5] == 'Iris-setosa':
            data_id[i] = 1
        elif row[5] == 'Iris-versicolor':
            data_id[i] = 2
        elif row[5] == 'Iris-virginica':
            data_id[i] = 3
        i = i + 1

accuracy_matrix = np.zeros((K, K), dtype=float)
accuracy = np.zeros((1, K), dtype=float)
accuracy_record = np.zeros((max_itr, K), dtype=float)
distance = np.zeros((N, K), dtype=float)
train_id = np.zeros((N, 1), dtype=int)
center = np.zeros((K, d), dtype=float)
initial = np.random.randint(0, high=150, size=None, dtype='int')
center[0, ...] = data[initial, ...]
center_probability = np.zeros((N, 1), dtype=float)
for i in range(K - 1):
    calc_distance()
    for k in range(N):
        center_probability[k] = np.min(distance[k, 0:i + 1])
    idx = np.where(center_probability == np.max(center_probability))[0][0]
    center[i + 1, ...] = data[idx, ...]

while step < max_itr:
    step += 1
    calc_distance()
    for i in range(N):  # 更新聚类结果
        train_id[i] = np.where(distance[i] == np.min(distance[i]))[0][0] + 1
    center = np.zeros((K, d), dtype=float)
    for i in range(N):  # 更新聚类中心
        center[train_id[i] - 1] += data[i]
    for i in range(K):
        classified_data_i_idx = np.where(train_id == i + 1)[0]  # 被分类至第i类的数据的下标
        real_id = data_id[classified_data_i_idx]  # 被分类至第i类的数据的实际类别
        center[i] /= classified_data_i_idx.shape[0]  # 更新聚类中心的最后一步
        for j in range(K):  # 计算准确度
            accuracy_matrix[i, j] = np.where(real_id == j + 1)[0].shape[0]
    for i in range(K):
        accuracy_matrix[..., i] /= np.where(data_id == i + 1)[0].shape[0]
        accuracy[0, i] = np.max(accuracy_matrix[..., i])
    accuracy_record[step-1] = accuracy
# print(accuracy_record)
# print(center)
# print(np.where(train_id == 1)[0])
# print(np.where(train_id == 2)[0])
# print(np.where(train_id == 3)[0])
final_dis = np.zeros((N, 1), dtype=float)
for i in range(N):
    final_dis[i] = np.min(distance[i, ...])
print(np.sum(final_dis))
