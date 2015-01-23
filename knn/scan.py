from collections import Counter, defaultdict
from knn import Knearest, Numbers

data = Numbers("../data/mnist.pkl.gz")

acc = defaultdict(dict)
lstart = 100
dindex = 0
for i in range(1,10):
    for j in range(1,10):
        dindex += 1
        print("k=%i, limit=%i\n" % (i, j*lstart))
        myknn = Knearest(data.train_x[:j*lstart],data.train_y[:j*lstart:],i)
        cm = myknn.confusion_matrix(data.test_x,data.test_y)
        acc[i][j] = myknn.acccuracy(cm)
        
        

print("done")
