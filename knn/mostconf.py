#input is confusion matrix cm
conf = zeros((10,10))
for i in cm.keys():
    d = cm[i]
    mykeys = d.keys()
    mykeys.remove(i)
    mymax = max(mykeys, key = lambda k: d[k])
    print('label %i confused with %i %i times' % (i, mymax, max([d[x] for x in mykeys])))
    for j in mykeys:
        conf[i][j] += d[j]

