from csv import DictReader, DictWriter

import numpy as np
from numpy import array
from pylab import *
from scipy.stats import norm

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn import cross_validation as cv
from sklearn.metrics import confusion_matrix as cm
from sklearn import metrics
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import TransformerMixin
from scipy import sparse

class Featurizer:
    def __init__(self):
        self.vectorizer = CountVectorizer()

    def train_feature(self, examples):
        return self.vectorizer.fit_transform(examples)

    def test_feature(self, examples):
        return self.vectorizer.transform(examples)


class CapsCount(TransformerMixin):

    def __init__(self,dozscore=0):
        self.dozscore = dozscore


    def transform(self, X, dozscore=0, **transform_params):
        out = []
        for line in X:
            tmp = sum(1 for x in line if x.isupper())
            if self.dozscore == 1:
                out.append((tmp-self.catmean)/self.catstd)
            else:
                out.append(tmp)

        # out = 1.0*copy(self.ncaps)
        # for i in range(len(X)):
        #     ind = self.y[i]
        #     out[i] = (X[i]-self.catmean[ind]) / self.catstd[ind]

        return sparse.coo_matrix(out).transpose() 

    def fit(self, X, y=[], **fit_params):
        ncaps = []
        labels = []
        for line in X:
            ncaps.append(sum(1 for x in line if x.isupper()))

        ncaps = array(ncaps)
#        capsbycat = [ncaps[y==i] for i in unique(y)]
        self.catmean = mean(ncaps)#[mean(x) for x in capsbycat]
        self.catstd = std(ncaps)#[std(x) for x in capsbycat]
#        self.y = y
 #       self.ncaps = ncaps

        return self

    def get_feature_names(self):
        return ['zscoreNCaps']  

class SentCount(TransformerMixin):

    def __init__(self,dozscore=0):
        self.dozscore = dozscore    

    def transform(self, X, dozscore=0, **transform_params):
        out = []
        for line in X:
            tmp = sum(1 for x in line if x is '.')
            if self.dozscore == 1:
                out.append((tmp-self.catmean)/self.catstd)
            else:
                out.append(tmp)

#         out = 1.0*copy(self.nsent)
#         for i in range(len(X)):
# #            ind = self.y[i]
#             out[i] = (X[i]-self.catmean)/self.catstd
# #            out[i] = (self.nsent[i]-self.catmean[ind])/self.catstd[ind]

        return sparse.coo_matrix(out).transpose()     
        

    def fit(self, X, y=[], **fit_params):
        nsent = []
        for line in X:
            nsent.append(sum(1 for x in line if x is '.'))

        nsent = array(nsent)
        self.nsent = nsent

        #sentbycat = [nsent[y==i] for i in unique(y)]        
        self.catmean = mean(nsent)#[mean(x) for x in sentbycat]
        self.catstd = std(nsent)#[std(x) for x in sentbycat]
        #self.y = y

        return self


    def get_feature_names(self):
        return ['zscoreNSent']

class AlphaCount(TransformerMixin):

    def __init__(self,dozscore=0):
        self.dozscore = dozscore

    def transform(self, X, dozscore=0, **transform_params):
        out = []
        for line in X:
            tmp = len(filter(str.isalpha,line))
            if self.dozscore == 1:
                out.append((tmp-self.catmean)/self.catstd)
            else:
                out.append(tmp)

#         out = 1.0*copy(self.nalpha)
#         for i in range(len(X)):
# #            ind = self.y[i]
# #            out[i] = (self.nalpha[i]-self.catmean[ind])/self.catstd[ind]
#             out[i] = (X[i]-self.catmean)/self.catstd

        return sparse.coo_matrix(out).transpose()     
        

    def fit(self, X, y=[], **fit_params):
        nalpha = []
        for line in X:
            nalpha.append(len(filter(str.isalpha,line)))

        nalpha = array(nalpha)
        self.catmean = mean(nalpha)#[mean(x) for x in alphabycat]
        self.catstd = std(nalpha)#[std(x) for x in alphabycat]

        return self


    def get_feature_names(self):
        return ['zscoreNAlpha']


class NonAlphaCount(TransformerMixin):

    def __init__(self,dozscore=0):
        self.dozscore = dozscore

    def transform(self, X, **transform_params):
        out = []
        for line in X:
            tmp = len(filter(str.isalpha,line))
            if self.dozscore == 1:
                out.append((tmp-self.catmean)/self.catstd)
            else:
                out.append(tmp)

        return sparse.coo_matrix(out).transpose()     
        

    def fit(self, X, y=[], **fit_params):
        nnalpha = []
        for line in X:
            nnalpha.append(len(filter(str.isalpha,line)))

        nnalpha = array(nnalpha)
        self.catmean = mean(nnalpha)#[mean(x) for x in nnalphabycat]
        self.catstd = std(nnalpha)#[std(x) for x in nnalphabycat]

        return self


    def get_feature_names(self):
        return ['zscoreNNonAlpha']



def show_top10(classifier, categories, feats):
    feature_names = np.asarray(feats.get_feature_names())
    for i, category in enumerate(categories):
        top10 = np.argsort(classifier.coef_[i])[-10:]
        print("%s: %s" % (category, " ".join(feature_names[top10])))

if __name__ == "__main__":

    # Cast to list to keep it all in memory
    train = list(DictReader(open("train.csv", 'r')))
    test = list(DictReader(open("test.csv", 'r')))

    feat = Featurizer()

    labels = []
    ncaps = []
    nsent = []
    for line in train:
        ncaps.append(sum(1 for x in line['text'] if x.isupper()))
        nsent.append(line['text'].count('\t'))
        if not line['cat'] in labels:
            labels.append(line['cat'])

    xtext = [x['text'] for x in train]
    xtext_test = [x['text'] for x in test]

    y_train = array(list(labels.index(x['cat']) for x in train))


    countvec = CountVectorizer(analyzer='word',stop_words='english',ngram_range=(1,2))
    countvecchar = CountVectorizer(analyzer='char',stop_words='english',ngram_range=(2,3))
    #countvec = CountVectorizer()
    tf_idf = TfidfTransformer()
    ncaps = CapsCount(dozscore=1)
    nsent = SentCount(dozscore=1)
    nalpha = AlphaCount(dozscore=1)
    nnalpha = NonAlphaCount(dozscore=1)

    combfeat = FeatureUnion([('countvec',countvec),
                             #('countvecchar',countvecchar),
                             #('tfidf', Pipeline([
                             #   ('countvec',countvec),
                             #   ('tf_idf',tf_idf)
                             #   ])),
                           ('ncaps',ncaps),
                           ('nsent',nsent),
                           ('nalpha',nalpha),
                           ('nnalpha',nnalpha),
                            ])
    x_train = combfeat.fit_transform(xtext,y_train)
    x_test = combfeat.transform(xtext_test)


    lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)

    #split for 10 fold cross_vallidation
    kf = cv.KFold(len(train), 10, shuffle=True)
    acc = []
    lrs = []
    predicted = zeros(len(train))
    errors = zeros(len(train))
    for train_indices, test_indices in kf:
        lr.fit(x_train[train_indices], y_train[train_indices])
        acc.append(lr.score(x_train[test_indices],y_train[test_indices]))
        lrs.append(copy(lr))
        predicted[test_indices] = lr.predict(x_train[test_indices])
        errors[test_indices] = y_train[test_indices] != lr.predict(x_train[test_indices])

    print(metrics.classification_report(y_train, predicted, target_names=labels))

    # for ind in find(errors):
    #     print 'ID: %d, labeled %s, true label %s\n' % 
                                                    
    mycm = cm(y_train,predicted)    
    out = hist(y_train,range(15))
    normcm = array([i/out[0][j] for j, i in enumerate(mycm)])

    dprime = norm.ppf(diag(mycm)/out[0]) - norm.ppf((sum(mycm,1)-diag(mycm))/(sum(out[0])-out[0]))

    show_top10(lr, labels, combfeat)

    run full trainning set
    lr.fit(x_train,y_train)
    predictions = lr.predict(x_test)

    o = DictWriter(open("predictions.csv", 'w'), ["id", "cat"])
    o.writeheader()
    for ii, pp in zip([x['id'] for x in test], predictions):
        d = {'id': ii, 'cat': labels[pp]}
        o.writerow(d)
