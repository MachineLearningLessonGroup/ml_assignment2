#-*-coding:utf-8-*-
'''
Generate by Python3.5

'''
import numpy as np
import time
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.datasets.base import Bunch
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import data_utils

'''

原数据格式:
| 类别:
unacc, acc, good, vgood

| 特征:
buying:   vhigh, high, med, low.
maint:    vhigh, high, med, low.
doors:    2, 3, 4, 5more.
persons:  2, 4, more.
lug_boot: small, med, big.
safety:   low, med, high.

'''
#数据链接:
url="http://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"

#进行数值化处理的dict:
str2int={
    'vhigh':4,
    'high':3,
    'med':2,
    'low':1,
    '2':2,
    '3':3,
    '4':4,
    '5more':5,
    'more':5,
    'small':3,
    'big':1,
    'unacc':0,
    'acc':1,
    'good':2,
    'vgood':3,
}

X, y = data_utils.dispose_data(url, str2int)

#将数据集切分为训练集和测试集:
train_data, test_data, train_target, test_target = train_test_split(X, y, test_size=0.3, random_state=0)

# hyh 生成一个默认的数据结构,方便使用
dataArray = np.empty((len(train_target), 6))
for i in range(len(train_target)):
    dataArray[i] = np.asarray(train_data[i], dtype=np.float)
targetArray = np.asarray(train_target, dtype=np.int)
target_names = np.asarray(['unacc', 'acc', 'good', 'vgood'])
fdescr = "Train data for the car"
carSet = Bunch(data=dataArray, target=targetArray,
               target_names=target_names,
               DESCR=fdescr,
               feature_names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])

# 设定各种参数
max_depth = 1
numRound = 33
labels = 4

# 维持一个m * k的权重分布, 共有m个样本,k种潜在label,初始化
weightArrays = [np.ndarray] * (1 + numRound)
weightArrays[0] = np.empty((len(carSet.data), labels))
for i in range(len(carSet.data)):
    for j in range(labels):
        weightArrays[0][i][j] = 1.0 / (labels * len(carSet.data))

# 结果数组
targetArray = np.empty((len(carSet.data), labels))
for i in range(len(carSet.data)):
    k = carSet.target[i]
    for j in range(labels):
        targetArray[i][j] = -1
    targetArray[i][k] = 1

clf_trees = [[DecisionTreeClassifier] * labels] * numRound
preds = [np.ndarray] * numRound
preds_probs = [np.ndarray] * numRound
Z = [float]*numRound

print("决策树开始训练!")
tree_train_start=time.time()

for nr in range(numRound):
    # 每一轮训练对每一个标签训练
    preds[nr] = np.empty((len(carSet.data), labels))
    preds_probs[nr] = np.empty((len(carSet.data), labels))
    for l in range(labels):
        clf_trees[nr][l] = DecisionTreeClassifier(
            criterion='entropy',
            splitter='best',
            max_depth=max_depth,
            min_samples_split=1,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features=None,
            random_state=42,
            max_leaf_nodes=None,
            class_weight=None,
            presort=False)

        # 训练该轮并记录数据
        clf_trees[nr][l] = clf_trees[nr][l].fit(carSet.data, targetArray[:, l], sample_weight=weightArrays[nr][:, l])
        preds[nr][:, l] = clf_trees[nr][l].predict(carSet.data)
        pbs = clf_trees[nr][l].predict_proba(carSet.data)
        #p = clf_trees[nr][l].predict(carSet.data)  # 预测结果
        #for xx in range(100):
        #    print ps[xx], p[xx]
        for index in range(len(carSet.data)):
            if preds[nr][index, l] == -1:
                preds_probs[nr][index, l] = pbs[index][0]
            else:
                preds_probs[nr][index, l] = pbs[index][1]

    # 调整数据, 准备下一轮循环
    weightArrays[nr+1] = np.empty((len(carSet.data), labels))
    Z[nr] = 0.0
    for i in range(len(carSet.data)):
        for j in range(labels):
            Z[nr] += weightArrays[nr][i][j] * (np.e ** (-1 * preds[nr][i][l] * preds_probs[nr][i][l]))

    #print Z[nr]
    print preds_probs[nr]
    for i in range(len(carSet.data)):
        for l in range(labels):
            res = -1
            if targetArray[i][l] == preds[nr][i][l]:
                res = 1
            weightArrays[nr + 1][i][j] = weightArrays[nr][i][j] * (np.e ** (-1 * res * preds_probs[nr][i][l])) / Z[nr]

print("决策树训练结束!\n")

print("进行预测\n")
test_pred_prob = np.ndarray((len(test_data), labels))
for i in range(numRound):
    for l in range(labels):
        test_pred_prob[:, l] += clf_trees[nr][l].predict_proba(test_data)[:, 1]
print test_pred_prob

test_pre = np.argmax(test_pred_prob, axis=1)
print test_pre

right = 0
for index in range(len(test_data)):
    if test_pre[index] == test_target[index]:
        right += 1
print right, len(test_data)
