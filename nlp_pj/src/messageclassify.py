#-*- coding: utf8 -*-

import loaddata
import numpy as np
import sklearn
import jieba
import jieba.posseg as pseg
import sklearn.feature_extraction.text


messages = loaddata.load_message()
content = np.array([m[0] for m in messages])
target = np.array([m[1] for m in messages])

def check_content():
    for cls in '1234':
        print('cls ', cls)
        _ = content[target==cls]
        for i in range(10):
            print(_[i])


### 处理繁体字
def process_fantizi(content)
    fantizi = loaddata.load_fantizi()
    content_after_fantizi = []
    processed = set()
    for i in content:
        new_words=''
        for k in i:
            if k in fantizi:
                new_words += fantizi[k]
                processed.add((k, fantizi[k]))
            else:
                new_words += k
        content_after_fantizi.append(new_words)
    return content_after_fantizi, processed


### 处理拆分字
def process_chaifenzi(content):
    chaifenzi = loaddata.load_chaifenzi()
    content_after_chaifenzi = []
    found_chaifenzi = set()
    for line in content:
        result = line
        for k,v in chaifenzi.items():
            if k in line:
                found_chaifenzi.add((k,v))
                result = result.replace(k,v)
        content_after_chaifenzi.append(result)
    return content_after_chaifenzi, found_chaifenzi


### 将连续的数字转变为长度的维度
def process_cont_numbers(content):
    digits_features = np.zeros((len(content),16))
    import re
    for i,line in enumerate(content):
        for digits in re.findall(r'\d+', line):
            length = len(digits)
            if 0 < length <= 15:
                digits_features[i, length-1] += 1
            elif length > 15:
                digits_features[i, 15] += 1
    return process_cont_numbers

### 将电子邮箱转换为特征
def process_cont_email(content):
    ema_features = 0
    import re
    for i,line in enumerate(content):
        for digits in re.findall(r'([a-zA-Z0-9_\.\-])+\@(([a-zA-Z0-9\-])+\.)+([a-zA-Z0-9]{2,4})+', line):
            ema_features += 1
    return ema_features


### 正常分词非TFID
class MessageCountVectorizer(sklearn.feature_extraction.text.CountVectorizer):
    def build_analyzer(self):
        def analyzer(doc):
            words = pseg.cut(doc)
            new_doc = ''.join(w.word for w in words if w.flag != 'x')
            words = jieba.cut(new_doc)
            return words
        return analyzer

vec_count = MessageCountVectorizer(min_df=2,max_df=0.8)
data_count = vec_count.fit_transform(content)
vec_count.get_feature_names()
print(data_count.shape)

#直接采用TFID生成对应的
class TfidfVectorizer(sklearn.feature_extraction.text.TfidfVectorizer):
    def build_analyzer(self):
        #analyzer = super(TfidfVectorizer, self).build_analyzer()
        def analyzer(doc):
            words = pseg.cut(doc)
            new_doc = ''.join(w.word for w in words if w.flag != 'x')
            words = jieba.cut(new_doc)
            return words
        return analyzer

vec_tfidf = TfidfVectorizer(min_df=2,max_df=0.8)
data_tfidf = vec_tfidf.fit_transform(content)
print(data.shape)

## 繁体字、拆分字、奇异短语（谐音字）

## 初步采用SVM实现
def test_SVM(data, target):
    from sklearn.grid_search import GridSearchCV
    from sklearn.svm import SVC
    clf = SVC()

    C = [0.1, 0.5, 0.75, 1, 2, 3, 4, 5, 10, 30, 50]
    kernel = ['linear','rbf'] # 'poly',
    param_grid = [{'C': C, 'kernel':kernel}]
    grid_search = GridSearchCV(clf, param_grid=param_grid, cv=10)
    grid_search.fit(data, target)

    return grid_search
grid_search = test_SVM(data_tfidf, target)
print(grid_search.grid_scores_)
grid_search.best_score_, grid_search.best_estimator_, grid_search.best_params_

##### 查看混淆矩阵
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(data, target, test_size=0.2,
                                                random_state=0)
clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
   decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
   max_iter=-1, probability=False, random_state=None, shrinking=True,
   tol=0.001, verbose=False)
clf.fit(Xtrain, ytrain)
clf.score(Xtest, ytest)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(ytest, clf.predict(Xtest)))

##### 求分类平均值
pre = clf.predict(Xtest)
for i in '123456':
    print('score for class', i, ':', np.logical_and(pre==i, ytest==i).sum()/ (ytest==i).sum())

from sklearn.cross_validation import train_test_split
def get_classes_accury(data, target, test_times = 10, test_size=0.1):
    target_list = list(set(target))
    target_list.sort()
    scores = np.zeros((test_times,len(target_list)))
    for t in range(test_times):
        clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,   decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,   tol=0.001, verbose=False)
        Xtrain, Xtest, ytrain, ytest = train_test_split(data, target, test_size=test_size,
                                                    random_state=t)
        clf.fit(Xtrain, ytrain)
        print(t, clf.score(Xtest, ytest))
        pre = clf.predict(Xtest)
        for i,c in enumerate(target_list):
            s = np.logical_and(pre==c, ytest==c).sum()/ (ytest==c).sum()
            scores[t, i] = s

    ##### 生成表格
    print('|'+'class'+'|'+'|'.join([str(i) for i  in target_list]) +'|')
    print('|'+'-'+'|')
    for i,score in enumerate(scores):
        print( '|'+str(i)+'|'+ '|'.join(['{:.4f}'.format(_) for _ in score])+ '|' )
    print( '|'+'max'+ '|'+ '|'.join(['{:.4f}'.format(_) for _ in scores.max(axis=0)])+ '|' )
    print( '|'+'min'+ '|'+ '|'.join(['{:.4f}'.format(_) for _ in scores.min(axis=0)])+ '|' )
    print( '|'+'mean'+'|'+  '|'.join(['{:.4f}'.format(_) for _ in scores.mean(axis=0)])+ '|' )

    return scores
scores = get_classes_accury(data, target)


##### 单类去一验证
def cross_validation_for_class_x(data, target):
    all_sample_n = len(target)
    all_x_sample, right_sample = 0, 0
    train = np.ones(all_sample_n, dtype=bool)
    errs_list = []
    for i in range(all_sample_n):
        if target[i] == 0:
            continue
        all_x_sample += 1
        train[i] = False
        clf = SVC(C=2, cache_size=200, class_weight=None, coef0=0.0,
   decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
   max_iter=-1, probability=False, random_state=None, shrinking=True,
   tol=0.001, verbose=False)
        clf.fit(data[train], target[train])
        pre = clf.predict(data[i])
        if int(pre[0]) != 0:
            right_sample += 1
        else:
            errs_list.append(i)
        train[i] = True
        print('times:', i, ' answers:', pre, ' all_times:', all_x_sample, ' right', right_sample)
    return all_x_sample, right_sample, errs_list


##### 生成表格
for i,score in enumerate(scores):
    print( '|'+str(i)+'|'+ '|'.join(['{:.4f}'.format(_) for _ in score])+ '|' )
    print( '|'+'max'+ '|'+ '|'.join(['{:.4f}'.format(_) for _ in scores.max(axis=0)])+ '|' )
    print( '|'+'min'+ '|'+ '|'.join(['{:.4f}'.format(_) for _ in scores.min(axis=0)])+ '|' )
    print( '|'+'mean'+'|'+  '|'.join(['{:.4f}'.format(_) for _ in scores.mean(axis=0)])+ '|' )
