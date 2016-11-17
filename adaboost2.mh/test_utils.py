# -*- coding: utf-8 -*-
import numpy as np


# 测试数据与期望目标

def test(data, target, clf_trees, numRound, labels):
    test_pred_prob = np.zeros((len(data), labels))

    for nr in range(numRound):
        for l in range(labels):
            test_pred_prob[:, l] += clf_trees[nr][l].predict_proba(data)[:, 1]

    test_pre = np.argmax(test_pred_prob, axis=1)

    right = 0
    for index in range(len(data)):
        if test_pre[index] == target[index]:
            right += 1

    print right, len(data)
    per = (right * 1.0 / len(data))
    print "正确数目：", right, len(data)
    print "正确率： ", per

