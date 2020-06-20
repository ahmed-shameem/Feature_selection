import numpy as np
import random
from copy import deepcopy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import math,time,sys
from matplotlib import pyplot
import pandas as pd
from datetime import datetime
from functools import partial
import seaborn as sns 
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

MaxIter = 70
pop_size = 8
omega = 0.99


def initialise(partCount, dim, trainX, testX, trainy, testy):    
    population=np.zeros((partCount,dim))
    minn = 1
    maxx = math.floor(0.5*dim)
    
    if maxx<minn:
        maxx = minn + 1
        #not(c[i].all())
    
    for i in range(partCount):
        random.seed(i**3 + 10 + time.time() ) 
        no = random.randint(minn,maxx)
        if no == 0:
            no = 1
        random.seed(time.time()+ 100)
        pos = random.sample(range(0,dim-1),no)
        for j in pos:
            population[i][j]=1
            
    return population

def fitness(agent, trainX, testX, trainy, testy):
    # print(agent)
    cols=np.flatnonzero(agent)
    # print(cols)
    val=1
    if np.shape(cols)[0]==0:
        return val    
    clf=KNeighborsClassifier(n_neighbors=5)
    #clf=MLPClassifier(alpha=0.001, hidden_layer_sizes=(1000,500,100),max_iter=2000,random_state=4)
    train_data=trainX[:,cols]
    test_data=testX[:,cols]
    clf.fit(train_data,trainy)
    val=1-clf.score(test_data,testy)

    #in case of multi objective  []
    set_cnt=sum(agent)
    set_cnt=set_cnt/np.shape(agent)[0]
    val=omega*val+(1-omega)*set_cnt
    return val

def test_accuracy(agent, trainX, testX, trainy, testy):
    cols=np.flatnonzero(agent)
    val=1
    if np.shape(cols)[0]==0:
        return val    
    # clf = RandomForestClassifier(n_estimators=300)
    #clf=MLPClassifier(alpha=0.001, hidden_layer_sizes=(1000,500,100),max_iter=2000,random_state=4)
    clf=KNeighborsClassifier(n_neighbors=5)
    # clf=MLPClassifier( alpha=0.01, max_iterno=1000) #hidden_layer_sizes=(1000,500,100)
    #cross=4
    #test_size=(1/cross)
    #X_train, X_test, y_train, y_test = train_test_split(trainX, trainy,  stratify=trainy,test_size=test_size)
    train_data=trainX[:,cols]
    test_data=testX[:,cols]
    clf.fit(train_data,trainy)
    val=clf.score(test_data,testy)
    return val

def onecnt(agent):
    return sum(agent)

def sigmoid(gamma):
    if gamma < 0:
        return 1 - 1/(1 + math.exp(gamma))
    else:
        return 1/(1 + math.exp(-gamma))
    
def WOA(dataset):
    df = pd.read_csv(dataset)
    a, b = np.shape(df)
    data = df.values[:,0:b-1]
    label = df.values[:,b-1]
    dimension = data.shape[1]
    
    cross = 5
    test_size = (1/cross)
    trainX, testX, trainy, testy = train_test_split(data, label,stratify=label ,test_size=test_size,random_state=(7+17*int(time.time()%1000)))
    clf=KNeighborsClassifier(n_neighbors=5)
    clf.fit(trainX,trainy)
    val=clf.score(testX,testy)
    whole_accuracy = val
    print("Total Acc: ",val)
    
    pop = initialise(pop_size, dimension, trainX, testX, trainy, testy)
    fit = []
    for i in range(pop_size):
        fit.append(fitness(pop[i], trainX, testX, trainy, testy))
    ind = np.argsort(fit)
    gbest = pop[ind[0]].copy()
    gbest_fit = fit[ind[0]].copy()
    
    for n in range(MaxIter):
        a = 2 - 2 * n / (MaxIter - 1)            # linearly decreased from 2 to 0

        for j in range(pop_size):

            r = np.random.rand()
            A = 2 * a * r - a
            C = 2 * r
            l = np.random.uniform(-1, 1)
            p = np.random.rand()
            b = 1

            if (p < 0.5) :
                if np.abs(A) < 1:
                    D = np.abs(C * gbest - pop[j] )
                    pop[j] = gbest - A * D
                else :
                    x_rand = pop[np.random.randint(pop_size)] 
                    D = np.abs(C * x_rand - pop[j])
                    pop[j] = (x_rand - A * D)
            else:
                D1 = np.abs(gbest - pop[j])
                pop[j] = D1 * np.exp(b * l) * np.cos(2 * np.pi * l) + gbest
                
                
        for i in range(pop_size):
            for j in range(dimension):
                if (sigmoid(pop[i][j]) > random.random()):
                    pop[i][j] = 1
                else:
                    pop[i][j] = 0
                    
    ind = np.argsort(fit)
    bestpop = pop[ind[0]].copy()
    bestfit = fit[ind[0]].copy()
    
    testAcc = test_accuracy(bestpop, trainX, testX, trainy, testy)
    featCnt = onecnt(bestpop)
    #print("best agent: ", bestpop)
    print("Test Accuracy: ", testAcc)
    print("#Features: ", featCnt)
            
    return testAcc, featCnt, bestpop


datasetlist = ["BreastCancer.csv", "BreastEW.csv", "CongressEW.csv", "Exactly.csv", "Exactly2.csv", "HeartEW.csv", "Ionosphere.csv", "Lymphography.csv", "M-of-n.csv", "PenglungEW.csv", "Sonar.csv", "SpectEW.csv", "Tic-tac-toe.csv", "Vote.csv", "Wine.csv", "Zoo.csv","KrVsKpEW.csv", "WaveformEW.csv" ]

for datasetname  in datasetlist:    
    print(datasetname)
    accuArr = []
    featArr = []
    agenArr = []
    #start_time = datetime.now()
    for i in range(15):
        # print(i)
        testAcc, featCnt, gbest = WOA(datasetname)
        # print(testAcc)
        accuArr.append(testAcc)
        featArr.append(featCnt)
        agenArr.append(gbest)
    #time_required = datetime.now() - start_time
    maxx = max(accuArr)
    k = np.argsort(accuArr)
    bagent = agenArr[k[-1]]
    currFeat= 20000
    for i in range(np.shape(accuArr)[0]):
        if accuArr[i]==maxx and featArr[i] < currFeat:
            currFeat = featArr[i]
            bagent = agenArr[i]
    datasetname = datasetname.split('.')[0]
    print(datasetname)
    print(maxx,currFeat)
    print(bagent)

