from logreg import *
from pylab import *
from mpltools import style
train, test, vocab = read_dataset("../data/hockey_baseball/positive","../data/hockey_baseball/negative","../data/hockey_baseball/vocab")
print("Read in %i train and %i test" % (len(train), len(test)))

mu = 0.001
passes = 3
ftacc = []
fhacc = []
for istep in [.5]:#logspace(-6,0,3):

    # Initialize model
    lr = LogReg(len(vocab), mu, lambda x: istep)

    # Iterations
    update_number = 0
    updcnt = []
    tacc = []
    hacc = []
    for pp in xrange(passes):
        for ii in train:
            update_number += 1
            lr.sg_update(ii, update_number)

            if update_number % 100 == 1:       
                updcnt.append(update_number)         
                train_lp, train_acc = lr.progress(train)
                ho_lp, ho_acc = lr.progress(test)
                tacc.append(train_acc)
                hacc.append(ho_acc)
                #print("Update %i\tTP %f\tHP %f\tTA %f\tHA %f" %
                #      (update_number, train_lp, ho_lp, train_acc, ho_acc))

    print("Update %i\tTP %f\tHP %f\tTA %f\tHA %f" %
            (update_number, train_lp, ho_lp, train_acc, ho_acc))

    sortedwords = [vocab[i[0]] for i in sorted(enumerate(lr.beta),key=lambda x:x[1])]
    swlen = len(sortedwords)
    swmid = int(round(swlen/2))
    zippedwords = zip(sortedwords[1:10],sortedwords[swlen-10:swlen],sortedwords[swmid-5:swmid+4])
    print("top 10 negative, positive, and useless words:\n")
    for i in zippedwords:
        print '{0:<17} {1:<17} {2:17}'.format(*i)


    ftacc.append(tacc[:])
    fhacc.append((hacc[:]))

style.use('ggplot')
h = figure()
hold
plot(updcnt,transpose(ftacc),linewidth=2)
h = plot(updcnt,transpose(fhacc),'--',linewidth=2)
mylogs = logspace(-6,0,3)
names = []
for i in mylogs:
    names.append("train %f" % i)
for i in mylogs:
    names.append("test %f" % i)
legend(names,loc='lower right')
xlabel('Trainning examples',fontsize=20)
ylabel('Accuracy',fontsize=20)