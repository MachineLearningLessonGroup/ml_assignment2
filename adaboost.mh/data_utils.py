#-*-coding:utf-8-*-
import numpy as np
import os
import urllib
from typing import Dict

#下载数据:
def download(url):
    """如果文件尚未下载,则从url地址下载并返回文件名"""
    filename=url.split('/')[-1]
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url,filename)
    if os.path.exists(filename):
        print("文件{0}已准备完毕!".format(filename))
    else:
        raise Exception('文件'+filename+'尚未下载,请检查网络!')
    return filename

def dispose_data(url, dictionary):
    '''将数据数值化处理,切分为特征X与类别y'''
    raw_data=download(url)
    data_set=np.loadtxt(raw_data,delimiter=",",dtype=bytes).astype(str)
    X_=data_set[:,:6]
    y_=data_set[:,6]
    X=[]
    for x in X_:
        X.append(list(map(lambda c:dictionary[c],x)))
    X=np.array(X)
    y=list(map(lambda c:dictionary[c], y_))
    return X,y

