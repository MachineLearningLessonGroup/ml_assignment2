# -*-coding:utf-8-*-
'''
Generate by Python3.5
This solution performs much better than adaboost_MR.py
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
# 数据链接:
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"

# 进行数值化处理的dict:
str2int = {
    'vhigh': 4,
    'high': 3,
    'med': 2,
    'low': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5more': 5,
    'more': 5,
    'small': 3,
    'big': 1,
    'unacc': 0,
    'acc': 1,
    'good': 2,
    'vgood': 3,
}

X, y = data_utils.dispose_data(url, str2int)

# 将数据集切分为训练集和测试集:
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
maxdepth = 5
numRound = 200
labels = 4

D=np.zeros([len(carSet.data),4,4])

for i in range(len(carSet.data)):
    l1 = int(carSet.target[i])
    for l0 in range(4):
        if l0!=l1:
            D[i, l0, l1] = 1.0 / (3 * 1 * len(carSet.data) / 4)

'''
for i in range(len(carSet.data)):
			l1 = int(carSet.target[i])
			l0s = [0,1,2,3]
			l0s.remove(l1)
			D[i,l0s,l1] = 1.0/(3*1*len(carSet.data)/4)
'''

#weightArrays=np.zeros(len(carSet.data))
print("决策树开始训练!")
tree_train_start = time.time()
trs = []
alphas = []
for nr in range(numRound):
    preds = np.zeros((len(carSet.data)))
    weightArrays = np.zeros(len(carSet.data))#weightArrays should not use the D in previous round
    for i in range(len(carSet.data)):
        for l0 in range(4):
            l1=int(carSet.target[i])
            weightArrays[i]=weightArrays[i]+D[i,l0,l1]

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
    clf_tree = clf_tree.fit(carSet.data, carSet.target,sample_weight=weightArrays)
    trs.append(clf_tree)
    preds = clf_tree.predict(carSet.data)
    pbs = clf_tree.predict_proba(carSet.data)
    # 调整数据, 准备下一轮循环 确定r和alpha权重
    r = 0.0
    '''
    for i in range(len(carSet.data)):
        l1=carSet.target[i]
        for l0 in range(labels):
            if l0!=l1:
                r+=D[i,l0,l1]*(pbs[i][l1]-pbs[i][l0])
    '''
    #r = 1 - clf_tree.score(carSet.data, carSet.target, sample_weight=weightArrays)
    alpha=1# the pepar just give the discrete form of a , r ,Z ,so i just use the alpha=1 and it  performs better than use the form of dicrete form
    alphas.append(alpha)
    for i in range(len(carSet.data)):
        for l0 in range(labels):
            for l1 in range(labels):
               D[i,l0,l1] = D[i,l0,l1] * math.exp(0.5 * alpha  * (pbs[i][l0]-pbs[i][l1]))
    Z=0

    for i in range(len(carSet.data)):
        for l0 in range(3):
            Z=Z+D[i,l0,carSet.target[i]]

    D /= Z
print("决策树训练结束!\n")
print("进行预测\n")
# 测试数据与期望目标
test_utils.test(test_data, test_target, trs, alphas, numRound, labels)

