import nltk

def gender_features(word):
    return {'first_letter': word[-2:]}

from nltk.corpus import names
import random

def process():
    namelist = ([(name, 'male') for name in names.words('male.txt')] + [(name, 'female') for name in names.words('female.txt')])
    random.shuffle(namelist)
    train_names = namelist[800:]
    test_names = namelist[:400]
    devtest_names = namelist[400:800]
    train_set = [(gender_features(n), g) for (n,g) in train_names]
    devtest_set = [(gender_features(n), g) for (n,g) in devtest_names]
    test_set = [(gender_features(n), g) for (n,g) in test_names]
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    print(nltk.classify.accuracy(classifier, devtest_set))
    print(nltk.classify.accuracy(classifier, test_set))
