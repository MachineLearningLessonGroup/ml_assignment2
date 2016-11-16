#-*-coding:utf-8-*-
# 此文件用于检查分析数据分布情况

import data_utils
import numpy as np

url="http://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"

raw_data=data_utils.download(url)
data_set=np.loadtxt(raw_data, delimiter=",", dtype=bytes).astype(str)
attribute=data_set[:,:6]
labels=data_set[:,6]

classes={
    'unacc':0,
    'acc':0,
    'good':0,
    'vgood':0,
}

for label in labels:
    classes[label]+=1

sum=0
for label in classes:
    sum+=classes[label]

print("\n数据类别分布情况如下:\n")
print("类别".ljust(9)+"数目".ljust(8)+"百分比%".ljust(9))
print('-'*30)
print("unacc".ljust(10)+"{:<10d}".format(classes['unacc'])+"{:<10.3f}".format(classes['unacc']*1.0/sum*100))
print("acc".ljust(10)+"{:<10d}".format(classes['acc'])+"{:<10.3f}".format(classes['acc']*1.0/sum*100))
print("good".ljust(10)+"{:<10d}".format(classes['good'])+"{:<10.3f}".format(classes['good']*1.0/sum*100))
print("vgood".ljust(10)+"{:<10d}".format(classes['vgood'])+"{:<10.3f}".format(classes['vgood']*1.0/sum*100))
print("总计".ljust(8)+"{:<10d}".format(sum)+"{:<10.3f}".format(1))