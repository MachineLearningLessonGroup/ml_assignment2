#-*-coding:utf-8-*-
'''
Generate by Python3.5

'''
import time

import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.datasets.base import Bunch
from sklearn.tree import DecisionTreeClassifier
import data_utils
import test_utils
import math

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
train_data, test_data, train_target, test_target = train_test_split(X, y, test_size=0.3, random_state=11)

# 生成一个默认的数据结构,方便使用
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
maxdepth = 4
numRound = 100
labels = 4

# 维持一个m * k的权重分布, 共有m个样本,k种潜在label,初始化
weightArrays = np.ones((len(carSet.data), labels)) * (1.0 / (len(carSet.data) * labels))

# 结果数组
targetArray = np.ones((len(carSet.data), labels)) * -1
for i in range(len(carSet.data)):
    targetArray[i][carSet.target[i]] = 1

print("决策树开始训练!")
tree_train_start = time.time()
trs = []
alphas = []
for nr in range(numRound):
    tr = []
    # 每一轮训练对每一个标签训练
    preds = np.zeros((len(carSet.data), labels))
    preds_probs = np.zeros((len(carSet.data), labels))
    for l in range(labels):
        clf_tree = DecisionTreeClassifier(
            criterion='entropy',
            splitter='best',
            max_depth=maxdepth,
            min_samples_split=1,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features=None,
            random_state=42,
            max_leaf_nodes=None,
            class_weight=None,
            presort=False)

        # 训练该轮并记录数据
        clf_tree = clf_tree.fit(carSet.data, targetArray[:, l], sample_weight=weightArrays[:, l])
        tr.append(clf_tree)
        preds[:, l] = clf_tree.predict(carSet.data)
        pbs = clf_tree.predict_proba(carSet.data)
        for index in range(len(carSet.data)):
            if preds[index, l] == -1:
                preds_probs[index][l] = pbs[index][0]
            else:
                preds_probs[index][l] = pbs[index][1]

    r = 0.0
    # 调整数据, 准备下一轮循环 确定r和alpha权重

    for l in range(labels):
        for i in range(len(carSet.data)):
            sign = -1
            if targetArray[i][l] == preds[i][l]:
                sign = 1
            r += weightArrays[i][l] * sign * preds_probs[i][l]
            weightArrays[i][l] = weightArrays[i][l] * np.exp(-1 * sign * preds_probs[i][l])


    # print r
    rr = (1.0 + r)/ (1.0 - r)
    # print rr
    alpha = math.log(rr)
    Z = np.array(weightArrays).sum()
    weightArrays /= Z
    alphas.append(alpha)
    trs.append(tr)

print("决策树训练结束!\n")


print("进行预测\n")
# 测试数据与期望目标
test_utils.test(test_data, test_target, trs, alphas, numRound, labels)
