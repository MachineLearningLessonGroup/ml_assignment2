#-*-coding:utf-8-*-
# 这是adaboost提供的方法
'''
Generate by Python3.5

'''
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.datasets.base import Bunch
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

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


ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=1000, learning_rate=1)
ada_clf.fit(carSet.data, carSet.target)
print ada_clf.score(test_data, test_target)