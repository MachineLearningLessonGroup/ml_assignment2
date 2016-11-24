# -*- coding: utf-8 -*-
import numpy as np


# 测试数据与期望目标
def test(data, target, clf_trees, alphas, numRound, labels):
    test_pred_prob = np.zeros((len(data), labels))
    for nr in range(numRound):
        test_pred_prob += clf_trees[nr].predict_proba(data) * alphas[nr]
    test_pred_prob=test_pred_prob.tolist()
    test_pre=[]
    for i in range(len(test_pred_prob)):
        maxidx=test_pred_prob[i].index(max(test_pred_prob[i]))
        test_pre.append(maxidx)

    right = 0
    for index in range(len(data)):
        if test_pre[index] == target[index]:
            right += 1

    print (right, len(data))
    per = (right * 1.0 / len(data))
    print ("正确数目：", right, len(data))
    print ("正确率： ", per)

