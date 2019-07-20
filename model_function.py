import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import xgboost as xgb
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import random
import sklearn
import os


def upper_sample(data, label):
    np.random.seed(1)
    ind1 = []
    ind0 = []
    for i in range(len(label)):
        if (label[i] == 0):
            ind0.append(i)
        if (label[i] == 1):
            ind1.append(i)
    ind1 = pd.DataFrame(ind1)
    ind0 = pd.DataFrame(ind0)

    if (len(ind1) > len(ind0)):
        pic_0 = ind0.sample(n=len(ind1), replace=True)
        pic_0 = pic_0.values
        pic_0 = pic_0[:, 0]
        ind1 = ind1.values
        ind1 = ind1[:, 0]
        data_0 = data[pic_0]
        label_0 = label[pic_0]
        data_1 = data[ind1]
        label_1 = label[ind1]
        return data_0, data_1, label_0, label_1
    if (len(ind1) < len(ind0)):
        pic_1 = ind1.sample(n=len(ind0), replace=True)
        pic_1 = pic_1.values
        pic_1 = pic_1[:, 0]
        ind0 = ind0.values
        ind0 = ind0[:, 0]
        data_1 = data[pic_1]
        label_1 = label[pic_1]
        data_0 = data[ind0]
        label_0 = label[ind0]
        return data_0, data_1, label_0, label_1


def loocvLGB(data, label):
    from sklearn.metrics import accuracy_score
    res = []
    from sklearn.model_selection import LeaveOneOut
    loo = LeaveOneOut()
    i = 0
    for train, test in loo.split(data):
        x_train = data[train]
        x_test = data[test]
        y_train = label[train]
        y_test = label[test]
        lgb_train = lgb.Dataset(x_train, y_train)
        lgb_val = lgb.Dataset(x_test, y_test, reference=lgb_train)
        params = {
            'task': 'train',
            'boosting_type': 'gbdt',  # 设置提升类型
            'objective': 'binary',  # 目标函数
            'metric': {'l2'},  # 评估函数
            'num_leaves': 15,  # 叶子节点数
            'learning_rate': 0.1,  # 学习速率
            'feature_fraction': 0.9,  # 建树的特征选择比例
            'bagging_fraction': 0.9,  # 建树的样本采样比例
            'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
            'verbose': 1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
        }
        gbm = lgb.train(params, lgb_train, num_boost_round=300, valid_sets=lgb_train,
                        early_stopping_rounds=5000)  # 训练数据需要参数列表和数据集
        pred = gbm.predict(x_test)
        # for i in range(len(pred)):
        # if(pred[i]>=0.5):
        # pred[i]=1
        # else:
        # pred[i]=0

        res.append(pred)

    return res


def loocvSVC(data, label):
    from sklearn.metrics import accuracy_score
    res = []
    from sklearn.model_selection import LeaveOneOut
    loo = LeaveOneOut()
    i = 0
    for train, test in loo.split(data):
        x_train = data[train]
        x_test = data[test]
        y_train = label[train]
        y_test = label[test]
        clf = svm.SVC(C=0.9, kernel='rbf', gamma=20, decision_function_shape='ovr', probability=True)
        clf.fit(x_train, y_train)

        pred = clf.predict_proba(x_test)
        # for i in range(len(pred)):
        # if(pred[i]>=0.5):
        # pred[i]=1
        # else:
        # pred[i]=0

        res.append(pred[0, 1])

    return res


def loocvKnn(data, label):
    from sklearn.metrics import accuracy_score
    res = []
    from sklearn.model_selection import LeaveOneOut
    loo = LeaveOneOut()
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier()
    i = 0
    for train, test in loo.split(data):
        x_train = data[train]
        x_test = data[test]
        y_train = label[train]
        y_test = label[test]
        knn.fit(x_train, y_train)

        pred = knn.predict_proba(x_test)
        # for i in range(len(pred)):
        # if(pred[i]>=0.5):
        # pred[i]=1
        # else:
        # pred[i]=0

        res.append(pred[0, 1])

    return res


def loocvRF(data, label):
    from sklearn.metrics import accuracy_score
    res = []
    from sklearn.model_selection import LeaveOneOut
    loo = LeaveOneOut()
    i = 0
    for train, test in loo.split(data):
        x_train = data[train]
        x_test = data[test]
        y_train = label[train]
        y_test = label[test]
        a, b, c, d = upper_sample(x_train, y_train)
        x_train = np.vstack((a, b))
        y_train = np.append(c, d)
        rnd_clf = RandomForestClassifier(n_estimators=15, max_leaf_nodes=16, n_jobs=1)
        rnd_clf.fit(x_train, y_train)

        pred = rnd_clf.predict_proba(x_test)

        # pred=clf.predict_proba(x_test)
        # for i in range(len(pred)):
        # if(pred[i]>=0.5):
        # pred[i]=1
        # else:
        # pred[i]=0

        res.append(pred[0, 1])

    return res


def loocvGBDT(data, label):
    from sklearn.metrics import accuracy_score
    res = []
    from sklearn.model_selection import LeaveOneOut
    loo = LeaveOneOut()
    i = 0
    for train, test in loo.split(data):
        x_train = data[train]
        x_test = data[test]
        y_train = label[train]
        y_test = label[test]
        gbr = GradientBoostingClassifier(n_estimators=3000, max_depth=2, min_samples_split=2, learning_rate=0.1)
        gbr.fit(x_train, y_train)

        pred = gbr.predict_proba(x_test)

        # pred=clf.predict_proba(x_test)
        # for i in range(len(pred)):
        # if(pred[i]>=0.5):
        # pred[i]=1
        # else:
        # pred[i]=0

        print(test[0], 'finished')
        res.append(pred[0, 1])

    return res


def loocvLR(data, label):
    from sklearn.metrics import accuracy_score
    res = []
    from sklearn.model_selection import LeaveOneOut
    loo = LeaveOneOut()
    i = 0
    for train, test in loo.split(data):
        x_train = data[train]
        x_test = data[test]
        y_train = label[train]
        y_test = label[test]
        log_model = LogisticRegression()
        # 训练逻辑回归模型
        log_model.fit(x_train, y_train)
        # y_predict_rf = gbr.predict(x_test)

        pred = log_model.predict_proba(x_test)

        # pred=clf.predict_proba(x_test)
        # for i in range(len(pred)):
        # if(pred[i]>=0.5):
        # pred[i]=1
        # else:
        # pred[i]=0

        res.append(pred[0, 1])

    return res


def upperLoocvLGB(data, label):
    from sklearn.metrics import accuracy_score
    res = []
    from sklearn.model_selection import LeaveOneOut
    loo = LeaveOneOut()
    i = 0
    for train, test in loo.split(data):
        x_train = data[train]
        x_test = data[test]
        y_train = label[train]
        y_test = label[test]
        a, b, c, d = upper_sample(x_train, y_train)
        x_train = np.vstack((a, b))
        y_train = np.append(c, d)
        lgb_train = lgb.Dataset(x_train, y_train)
        lgb_val = lgb.Dataset(x_test, y_test, reference=lgb_train)
        params = {
            'task': 'train',
            'boosting_type': 'gbdt',  # 设置提升类型
            'objective': 'binary',  # 目标函数
            'metric': {'binary_logloss'},  # 评估函数
            'num_leaves': 20,  # 叶子节点数
            'learning_rate': 0.1,  # 学习速率
            'feature_fraction': 0.9,  # 建树的特征选择比例
            'bagging_fraction': 0.9,  # 建树的样本采样比例
            'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
            'verbose': 1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
        }
        gbm = lgb.train(params, lgb_train, num_boost_round=500, valid_sets=lgb_val,
                        early_stopping_rounds=5000)  # 训练数据需要参数列表和数据集
        pred = gbm.predict(x_test)
        # for i in range(len(pred)):
        # if(pred[i]>=0.5):
        # pred[i]=1
        # else:
        # pred[i]=0

        res.append(pred)

    return res


def normUpperLoocvLGB(data, label):
    random.seed(1)
    from sklearn.metrics import accuracy_score
    res = []
    from sklearn.model_selection import LeaveOneOut
    loo = LeaveOneOut()
    i = 0
    for train, test in loo.split(data):
        x_train = data[train]
        x_test = data[test]
        y_train = label[train]
        y_test = label[test]
        a, b, c, d = upper_sample(x_train, y_train)
        x_train = np.vstack((a, b))
        y_train = np.append(c, d)
        for i in range(len(x_train[0, :])):
            col_max = max(x_train[:, i])
            col_min = min(x_train[:, i])
            x_train[:, i] = (2 * (x_train[:, i] - col_min) / (col_max - col_min)) - 1
            x_test[:, i] = (2 * (x_test[:, i] - col_min) / (col_max - col_min)) - 1
        lgb_train = lgb.Dataset(x_train, y_train)
        lgb_val = lgb.Dataset(x_test, y_test, reference=lgb_train)
        params = {
            'task': 'train',
            'boosting_type': 'gbdt',  # 设置提升类型
            'objective': 'binary',  # 目标函数
            'metric': {'binary_logloss'},  # 评估函数
            'num_leaves': 20,  # 叶子节点数
            'learning_rate': 0.1,  # 学习速率
            'feature_fraction': 0.9,  # 建树的特征选择比例
            'bagging_fraction': 0.9,  # 建树的样本采样比例
            'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
            'verbose': 1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
        }

        gbm = lgb.train(params, lgb_train, num_boost_round=100, valid_sets=lgb_val,
                        early_stopping_rounds=40)  # 训练数据需要参数列表和数据集
        pred = gbm.predict(x_test)
        # for i in range(len(pred)):
        # if(pred[i]>=0.5):
        # pred[i]=1
        # else:
        # pred[i]=0

        res.append(pred)

    return res


def randNormUpperLoocvLGB(data, label):
    from sklearn.metrics import accuracy_score
    res = []
    from sklearn.model_selection import LeaveOneOut
    loo = LeaveOneOut()
    i = 0
    for train, test in loo.split(data):
        x_train = data[train]
        x_test = data[test]
        y_train = label[train]
        y_test = label[test]
        x_train_train, x_train_test, y_train_train, y_train_test = train_test_split(x_train, y_train, test_size=0.4)
        a, b, c, d = upper_sample(x_train_train, y_train_train)
        x_train_train = np.vstack((a, b))
        y_train_train = np.append(c, d)
        for i in range(len(x_train[0, :])):
            col_max = max(x_train_train[:, i])
            col_min = min(x_train_train[:, i])
            x_train_train[:, i] = (2 * (x_train_train[:, i] - col_min) / (col_max - col_min)) - 1
            x_train_test[:, i] = (2 * (x_train_test[:, i] - col_min) / (col_max - col_min)) - 1
            x_test[:, i] = (2 * (x_test[:, i] - col_min) / (col_max - col_min)) - 1
        lgb_train = lgb.Dataset(x_train_train, y_train_train)
        lgb_val = lgb.Dataset(x_train_test, y_train_test, reference=lgb_train)
        params = {
            'task': 'train',
            'boosting_type': 'gbdt',  # 设置提升类型
            'objective': 'binary',  # 目标函数
            'metric': {'binary_logloss'},  # 评估函数
            'num_leaves': 20,  # 叶子节点数
            'learning_rate': 0.1,  # 学习速率
            'feature_fraction': 0.9,  # 建树的特征选择比例
            'bagging_fraction': 0.9,  # 建树的样本采样比例
            'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
            'verbose': 1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
        }
        gbm = lgb.train(params, lgb_train, num_boost_round=500, valid_sets=lgb_val,
                        early_stopping_rounds=5000)  # 训练数据需要参数列表和数据集
        pred = gbm.predict(x_test)
        # for i in range(len(pred)):
        # if(pred[i]>=0.5):
        # pred[i]=1
        # else:
        # pred[i]=0

        res.append(pred)

    return res


def cvNormUpperLoocvLGB2(data, label, k):
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import KFold
    from sklearn.model_selection import LeaveOneOut
    loo = LeaveOneOut()
    i = 0
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',  # 设置提升类型
        'objective': 'binary',  # 目标函数
        'metric': {'binary_logloss'},  # 评估函数
        'num_leaves': 20,  # 叶子节点数
        'learning_rate': 0.1,  # 学习速率
        'feature_fraction': 0.9,  # 建树的特征选择比例
        'bagging_fraction': 0.9,  # 建树的样本采样比例
        'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
        'verbose': 1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
    }
    res = []
    for train, test in loo.split(data):
        x_test = data[test, :]
        y_test = label[test]
        x_train = data[train, :]
        y_train = label[train]
        best_train = []
        best_label = []
        KF = KFold(n_splits=10)  # 建立4折交叉验证方法  查一下KFold函数的参数
        best_score = 10
        best_test = []
        best_label_test = []
        for j in range(0, k):
            # print("TRAIN:",train_index,"TEST:",test_index)
            x_train_train, x_train_test, y_train_train, y_train_test = train_test_split(x_train, y_train,
                                                                                        test_size=0.01)

            a, b, c, d = upper_sample(x_train_train, y_train_train)
            x_train_train = np.vstack((a, b))
            y_train_train = np.append(c, d)
            x_train_train1 = x_train_train
            x_train_test1 = x_train_test
            # for i in range(len(x_train_train[0,:])):
            # col_max=max(x_train_train[:,i])
            # col_min=min(x_train_train[:,i])
            # x_train_train1[:,i]=(2*(x_train_train[:,i]-col_min)/(col_max-col_min))-1
            # x_train_test1[:,i]=(2*(x_train_test[:,i]-col_min)/(col_max-col_min))-1
            lgb_train = lgb.Dataset(x_train_train1, y_train_train)
            lgb_val = lgb.Dataset(x_train_test1, y_train_test, reference=lgb_train)

            gbm = lgb.train(params, lgb_train, num_boost_round=500, valid_sets=lgb_val, early_stopping_rounds=100)
            # 训练数据需要参数列表和数据集
            score = gbm.best_score['valid_0']['binary_logloss']
            if score < best_score:
                best_score = score
                best_train = x_train_train
                best_label = y_train_train
                best_test = x_train_test
                best_label_test = y_train_test
        # for i in range(len(best_train[0,:])):
        # col_max=max(best_train[:,i])
        # col_min=min(best_train[:,i])
        # best_train[:,i]=(2*(best_train[:,i]-col_min)/(col_max-col_min))-1
        # best_test[:,i]=(2*(best_test[:,i]-col_min)/(col_max-col_min))-1
        # x_test[:,i]=(2*(x_test[:,i]-col_min)/(col_max-col_min))-1
        lgb_train = lgb.Dataset(best_train, best_label)
        lgb_val = lgb.Dataset(best_test, best_label_test)
        gbm = lgb.train(params, lgb_train, num_boost_round=500, valid_sets=lgb_val, early_stopping_rounds=5000)
        tmp = gbm.predict(x_test)
        print(test, 'fold prop:', tmp, '##################################\n')
        res.append(tmp)

    return res


def cvNormUpperSplLGB2(data, label):
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import KFold
    from sklearn.model_selection import LeaveOneOut
    loo = LeaveOneOut()
    i = 0
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',  # 设置提升类型
        'objective': 'binary',  # 目标函数
        'metric': {'binary_logloss'},  # 评估函数
        'num_leaves': 20,  # 叶子节点数
        'learning_rate': 0.1,  # 学习速率
        'feature_fraction': 0.9,  # 建树的特征选择比例
        'bagging_fraction': 0.9,  # 建树的样本采样比例
        'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
        'verbose': 1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
    }
    res = []
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.05)

    best_train = []
    best_label = []
    KF = KFold(n_splits=100)  # 建立4折交叉验证方法  查一下KFold函数的参数
    best_score = 10
    best_test = []
    best_label_test = []
    for train_index, test_index in KF.split(x_train):
        # print("TRAIN:",train_index,"TEST:",test_index)
        x_train_train, x_train_test = x_train[train_index], x_train[test_index]
        y_train_train, y_train_test = y_train[train_index], y_train[test_index]
        a, b, c, d = upper_sample(x_train_train, y_train_train)
        x_train_train = np.vstack((a, b))
        y_train_train = np.append(c, d)
        x_train_train1 = x_train_train
        x_train_test1 = x_train_test
        # for i in range(len(x_train_train[0,:])):
        # col_max=max(x_train_train[:,i])
        # col_min=min(x_train_train[:,i])
        # x_train_train1[:,i]=(2*(x_train_train[:,i]-col_min)/(col_max-col_min))-1
        # x_train_test1[:,i]=(2*(x_train_test[:,i]-col_min)/(col_max-col_min))-1
        lgb_train = lgb.Dataset(x_train_train1, y_train_train)
        lgb_val = lgb.Dataset(x_train_test1, y_train_test, reference=lgb_train)

        gbm = lgb.train(params, lgb_train, num_boost_round=500, valid_sets=lgb_val, early_stopping_rounds=5000)
        # 训练数据需要参数列表和数据集
        score = gbm.best_score['valid_0']['binary_logloss']
        if score < best_score:
            best_score = score
            best_train = x_train_train
            best_label = y_train_train
            best_test = x_train_test
            best_label_test = y_train_test
        # for i in range(len(best_train[0,:])):
        # col_max=max(best_train[:,i])
        # col_min=min(best_train[:,i])
        # best_train[:,i]=(2*(best_train[:,i]-col_min)/(col_max-col_min))-1
        # best_test[:,i]=(2*(best_test[:,i]-col_min)/(col_max-col_min))-1
        # x_test[:,i]=(2*(x_test[:,i]-col_min)/(col_max-col_min))-1
    lgb_train = lgb.Dataset(best_train, best_label)
    lgb_val = lgb.Dataset(best_test, best_label_test)
    gbm = lgb.train(params, lgb_train, num_boost_round=500, valid_sets=lgb_val, early_stopping_rounds=5000)
    tmp = gbm.predict(x_test)
    # print(test,'fold prop:',tmp,'##################################\n')
    res.append(tmp)

    return (res, y_test)


def batchFS(data_dir, label_dir, start, end):
    label = pd.read_csv(open(label_dir))
    label = label.values
    label = label[:, 1]
    res = []
    cnt = 0
    for i in range(start, end + 1):
        data_path = data_dir + str(i) + '.csv'
        data = pd.read_csv(open(data_path))
        data = data.values
        data = data[:, 1:len(data[0, :])]
        print(data.shape)
        prop = normUpperLoocvLGB(data, label)
        res.append(prop)
        print(i, 'times finished\n')
    res = np.array(res)
    return res


def batchFS2(data_dir, label_dir, start, end):
    label = pd.read_csv(open(label_dir))
    label = label.values
    label = label[:, 0]
    res = []
    cnt = 0
    i = start
    while i <= end:
        data_path = data_dir + str(i) + '.csv'
        data = pd.read_csv(open(data_path))
        data = data.values
        data = data[:, 1:len(data[0, :])]
        print(data.shape)
        prop = normUpperLoocvLGB(data, label)
        res.append(prop)
        print(i, 'times finished\n')
        i = i + 0.05
    res = np.array(res)
    return res


def svmCVTR(train_dir, test_dir, size, start, end):
    res = []
    genes = start
    while genes <= end:
        tmp = []
        for i in range(1, size + 1):
            train_data_path = train_dir + str(genes) + 'features_' + str(i) + 'fold.csv'
            train_label_path = train_dir + str(genes) + 'features_' + str(i) + 'fold_label.csv'
            test_data_path = test_dir + str(genes) + 'features_' + str(i) + 'fold.csv'
            test_label_path = test_dir + str(genes) + 'features_' + str(i) + 'fold_label.csv'
            train_data = pd.read_csv(open(train_data_path))
            train_label = pd.read_csv(open(train_label_path))
            test_data = pd.read_csv(open(test_data_path))
            test_label = pd.read_csv(open(test_label_path))
            train_data = train_data.values
            train_data = train_data[:, 1:len(train_data[0, :])]
            train_label = train_label.values
            train_label = train_label[:, 1]
            test_data = test_data.values
            test_data = test_data[:, 1:len(test_data[0, :])]
            test_label = test_label.values
            test_label = test_label[:, 1]
            for i in range(len(train_data[0, :])):
                col_max = max(train_data[:, i])
                col_min = min(train_data[:, i])
                train_data[:, i] = (2 * (train_data[:, i] - col_min) / (col_max - col_min)) - 1
                test_data[:, i] = (2 * (test_data[:, i] - col_min) / (col_max - col_min)) - 1
            lgb_train = lgb.Dataset(train_data, train_label)
            lgb_val = lgb.Dataset(test_data, test_label, reference=lgb_train)
            params = {
                'task': 'train',
                'boosting_type': 'gbdt',  # 设置提升类型
                'objective': 'binary',  # 目标函数
                'metric': {'binary_logloss'},  # 评估函数
                'num_leaves': 20,  # 叶子节点数
                'learning_rate': 0.1,  # 学习速率
                'feature_fraction': 0.9,  # 建树的特征选择比例
                'bagging_fraction': 0.9,  # 建树的样本采样比例
                'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
                'verbose': 1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
            }
            gbm = lgb.train(params, lgb_train, num_boost_round=500, valid_sets=lgb_val,
                            early_stopping_rounds=5000)  # 训练数据需要参数列表和数据集
            pred = gbm.predict(test_data)
            tmp.append(pred)
        res.append(tmp)
        genes = genes + 20
    res = np.array(res)
    return res


def readTestLabel(test_dir, size):
    genes = 20
    res = []
    while genes <= 1000:
        tmp = []
        for i in range(1, size + 1):
            test_label_path = test_dir + str(genes) + 'features_' + str(i) + 'fold_label.csv'
            test_label = pd.read_csv(open(test_label_path))
            test_label = test_label.values
            test_label = test_label[:, 1]
            tmp.append(test_label)
        res.append(tmp)
        genes = genes + 20
    res = np.array(res)
    return res


def cptRes(data, label):
    res = []
    prop = data
    for i in range(len(prop[:, 0])):
        tmp = []
        fpr, tpr, threshold = roc_curve(label, prop[i, :])  ###计算真正率和假正率
        roc_auc = auc(fpr, tpr)  ###计算auc的值
        for j in range(len(prop[0, :])):
            if (prop[i, j] > 0.5):
                prop[i, j] = 1
            else:
                prop[i, j] = 0

        num_0 = sum(label == 0)
        num_1 = sum(label == 1)
        pred_0 = 0
        pred_1 = 0
        for j in range(len(prop[i, :])):
            if prop[i, j] == 0:
                if label[j] == 0:
                    pred_0 = pred_0 + 1
            if prop[i, j] == 1:
                if label[j] == 1:
                    pred_1 = pred_1 + 1
        sen = pred_1 / num_1
        spe = pred_0 / num_0
        print('thresh', i, 'pred 0:', pred_0, 'pred 1:', pred_1, '\n')
        acc = (pred_0 + pred_1) / (num_0 + num_1)
        tmp.append(roc_auc)
        tmp.append(sen), tmp.append(spe), tmp.append(acc)
        res.append(tmp)
    res = np.array(res)
    return res


def drawCurves(label, prop):
    for i in range(len(prop[:, 0])):
        print(i, '\n')
        fpr, tpr, threshold = roc_curve(label, prop[i, :])  ###计算真正率和假正率
        roc_auc = auc(fpr, tpr)  ###计算auc的值
        plt.figure()
        lw = 2
        plt.figure(figsize=(10, 10))
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
        # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()


def getDim(dir, start, end):
    res = []
    for i in range(start, end + 1):
        data_path = dir + str(i) + '.csv'
        data = pd.read_csv(open(data_path))
        data = data.values
        data = data[:, 1:len(data[0, :])]
        print(len(data[0, :]))
    # res=np.array(res)
    # return res


def cptRes2(data, label):
    res = []
    for i in range(len(data)):

        tmp2 = []
        fpr, tpr, threshold = roc_curve(label[i], data[i])  ###计算真正率和假正率
        roc_auc = auc(fpr, tpr)  ###计算auc的值
        for k in range(len(data[i])):
            if data[i][k][0] > 0.5:
                data[i][k][0] = 1
            else:
                data[i][k][0] = 0
        num_0 = sum(label[i] == 0)
        num_1 = sum(label[i] == 1)
        pred_0 = 0
        pred_1 = 0
        for k in range(len(data[i])):
            if (data[i][k] == 0):
                if label[i][k] == 0:
                    pred_0 = pred_0 + 1
            if (data[i][k] == 1):
                if label[i][k] == 1:
                    pred_1 = pred_1 + 1
        sen = pred_1 / num_1
        spe = pred_0 / num_0
        acc = ((pred_1 + pred_0) / (num_0 + num_1))
        print(sen, spe, '\n')
        tmp2.append(roc_auc)
        tmp2.append(sen)
        tmp2.append(spe)
        tmp2.append(acc)
        res.append(tmp2)
    # res.append(tmp1)
    res = np.array(res)
    return res
    final = []
    for i in range(len(data[:, 0])):
        tmp = []
        mean_auc = np.mean(res[i][:, 0])
        mean_sen = np.mean(res[i][:, 1])
        mean_spe = np.mean(res[i][:, 2])
        mean_acc = np.mean(res[i][:, 3])
        tmp.append(mean_auc)
        tmp.append(mean_sen), tmp.append(mean_spe), tmp.append(mean_acc)
        final.append(tmp)
    final = np.array(final)
    return final

