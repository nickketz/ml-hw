from numpy import array, zeros, dot, transpose, logical_and, logical_or, concatenate, reshape, random
import argparse
from sklearn import svm
import matplotlib.pyplot as plt
from mpltools import style





class Numbers:
    """
    Class to store MNIST data
    """

    def __init__(self, location):
        import cPickle, gzip

        # Load the dataset
        f = gzip.open(location, 'rb')
        train_set, valid_set, test_set = cPickle.load(f)

        self.train_x, self.train_y = train_set
        self.test_x, self.test_y = valid_set
        f.close()

    def select_classes(self,pos,neg,limit=0):
        """
        label classes as either +1(pos) or 0(neg)
        """
        poslog_train = zeros(len(self.train_y))
        neglog_train = zeros(len(self.train_y))
        poslog_test = zeros(len(self.test_y))
        neglog_test = zeros(len(self.test_y))
        #get logical arrays for positive and negative classes
        for i,j in zip(pos,neg):
          poslog_train = logical_or(poslog_train, self.train_y==i)
          neglog_train = logical_or(neglog_train, self.train_y==j)
          poslog_test = logical_or(poslog_test, self.test_y==i)
          neglog_test = logical_or(neglog_test, self.test_y==j)
        #get indicies from logicals
        posind_train = [i for i,x in enumerate(poslog_train) if x==True]
        negind_train = [i for i,x in enumerate(neglog_train) if x==True]
        posind_test = [i for i,x in enumerate(poslog_test) if x==True]
        negind_test = [i for i,x in enumerate(neglog_test) if x==True]
        #create selected samples
        limpos = len(posind_train)
        limneg = len(negind_train)
        if limit:
            if logical_and( limit<len(posind_train), limit<len(negind_train) ):
              limpos = limit
              limneg = limit

        self.selind_train = concatenate((posind_train[:limpos],negind_train[:limneg]))
        self.selind_test = concatenate((posind_test,negind_test))

        self.seltrain_y = concatenate((zeros(limpos)+1,zeros(limneg)))
        self.seltrain_x = self.train_x[self.selind_train]
        self.seltest_y = concatenate((zeros(len(posind_test))+1,zeros(len(negind_test))))
        self.seltest_x = self.test_x[self.selind_test]

def show_samples(xin, label=[], nrows=5, ncols=5, ind=[], fignum=[]):
    """
    show plot of sample
    """

    if not ind:
      ind = random.randint(len(xin),size=nrows*ncols)

    if fignum:
        plt.figure(fignum, figsize=(4, 3))
    else:
        plt.figure()



    plt.ion()
    for cnt,i in enumerate(ind):
          samp = reshape(xin[i],(len(xin[i])**.5,len(xin[i])**.5))
          plt.subplot(nrows,ncols,cnt+1)
          if label.any():
            plt.title(label[i])
          plt.axis('off')
          plt.imshow(samp)
          plt.draw()

    plt.show()


if __name__ == '__main__':
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--limit", help="limit on samples per class",
                           type=int, default=100, required=False)

    args = argparser.parse_args()

    data = Numbers("../data/mnist.pkl.gz")
    data.select_classes([3],[8],limit=args.limit)
    datalimit = len(data.seltrain_y)/2
    #show_samples(data.seltest_x, data.seltest_y)
    print("Done loading data\n\nUsing %d samples per class\n\n"  % (datalimit))

    fignum = 1
    clfs = []
    trainacc = {}
    testerrs = {}
    testacc = {}
    testacclist = []
    trainacclist = []
    labels = []
    for kk, dd, gg in [('linear', 0, 0),
                         ('poly', 3, 10),
                         ('sigmoid', 3, 10),
                         ('rbf', 3, 10)]:

      for cc in range(-5,6,2):
        label = "%s-C=10^%i" % (kk,cc)
        
        labels.append("%s-C=10^%i" % (kk,cc))
        cc = 10**cc
        # Fit the model
        clf = svm.SVC(kernel=kk, degree=dd, C=cc, gamma=gg)
        clf.fit(data.seltrain_x, data.seltrain_y)
        clfs.append(clf)
        testacc[label] = clf.score(data.seltest_x,data.seltest_y)
        testacclist.append(clf.score(data.seltest_x,data.seltest_y))
        trainacc[label] = clf.score(data.seltrain_x,data.seltrain_y)
        trainacclist.append(clf.score(data.seltrain_x,data.seltrain_y))
        print "%s, train: %f, test: %f" % (label, trainacc[label],testacc[label])
      print "\n"

    myclf = clfs[2]
    show_samples(myclf.support_vectors_,data.seltrain_y[myclf.support_],0)

    print "Evaluation Complte"
    style.use('ggplot')

    tmp = array(testacclist)
    plt.figure()
    plt.plot(reshape(tmp,(4,6)).transpose(),'-s',linewidth=3)
    plt.xticks(range(0,6),['10^-5','10^-3','10^-1','10^1','10^3','10^5'])
    plt.legend(['linear','poly','sigmoid','rbf'],loc='center right')
    plt.xlabel('C')
    plt.ylabel('Test Accuracy')
    plt.ax = plt.gca()
    plt.ax.xaxis.label.set_fontsize(20)
    plt.ax.yaxis.label.set_fontsize(20)
    plt.ylim((.45,1.05))

    tmp = array(trainacclist)
    plt.figure()
    plt.plot(reshape(tmp,(4,6)).transpose(),'--s',linewidth=3)
    plt.xticks(range(0,6),['10^-5','10^-3','10^-1','10^1','10^3','10^5'])
    plt.legend(['linear','poly','sigmoid','rbf'],loc='center right')
    plt.xlabel('C')
    plt.ylabel('Train Accuracy')
    plt.ax = plt.gca()
    plt.ax.xaxis.label.set_fontsize(20)
    plt.ax.yaxis.label.set_fontsize(20)
    plt.ylim((.45,1.05))

    plt.show()


        #find errors



        #show_samples(clf.support_vectors_, data.seltrain_y[clf.support_],fignum=fignum)
      








