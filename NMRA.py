
# coding: utf-8

# In[6]:


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
# from sklearn.naive_bayes import 
# from sklearn.ensemble import RanfomForestClassifier
from sklearn.ensemble import RandomForestClassifier

MaxIter = 30
pop_size = 20
omega = 0.99
bp = 0.5
bsize = 4
wsize = 16

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

def allfit(population, trainX, testX, trainy, testy):
    x=np.shape(population)[0]
    acc=np.zeros(x)
    for i in range(x):
        acc[i]=fitness(population[i],trainX, testX, trainy, testy)     
        #print(acc[i])
    return acc

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
    
def NMRA(dataset):
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
    fit = allfit(pop, trainX, testX, trainy, testy)
    
    ind = np.argsort(fit)
    index_b, index_w = [], []

    for i in range(bsize):
        index_b.append(ind[i])
        
    for i in range(wsize):
        index_w.append(ind[i])
    
    bestpop = pop[ind[0]]
    bestfit = fit[ind[0]]
    
    for i in range(MaxIter):
        for j in range(len(index_w)):
            l = []
            for k in range(dimension):
                random.seed(time.time()+k*2)
                l.append(random.random())
                
            pos = random.sample(range(0,len(index_w)-1),2)
            s = np.add(pop[index_w[j]], np.multiply(l, np.subtract(pop[index_w[pos[0]]],pop[index_w[pos[1]]])))
            
            if(fitness(s,trainX, testX, trainy, testy) < fit[index_w[j]]):
                fit[index_w[j]] = fitness(s,trainX, testX, trainy, testy)
                pop[index_w[j]] = s.copy()
                
                
        for j in range(len(index_b)):
            random.seed(time.time())
            if(random.random() < bp): 
                l, nl = [], []
                for k in range(dimension):
                    random.seed(time.time()+k*2)
                    l.append(random.random())

                for k in l:
                    nl.append(1-k)

                s = np.add(np.multiply(nl, pop[index_b[j]]), np.multiply(l, (np.subtract(bestpop, pop[index_b[j]]))))

                if(fitness(s,trainX, testX, trainy, testy) < fit[index_b[j]]):
                    fit[index_b[j]] = fitness(s,trainX, testX, trainy, testy)
                    pop[index_b[j]] = s.copy()  
                    
        for j in range(pop_size):
            for k in range(dimension):
                random.seed(time.time())
                if (sigmoid(pop[j][k]) > 0.5):
                    pop[j][k] = 1
                    
                else:
                    pop[j][k] = 0
                
                
        fit = allfit(pop, trainX, testX, trainy, testy)
        ind = np.argsort(fit)
        index_b, index_w = [], []

        for i in range(bsize):
            index_b.append(ind[i])

        for i in range(wsize):
            index_w.append(ind[i])

        bestpop = pop[ind[0]]
        bestfit = fit[ind[0]]
    
    fit = allfit(pop, trainX, testX, trainy, testy)
    ind = np.argsort(fit)
    bestpop = pop[ind[0]]
    bestfit = fit[ind[0]]
    
    testAcc = test_accuracy(bestpop, trainX, testX, trainy, testy)
    featCnt = onecnt(bestpop)
    #print("best agent: ", bestpop)
    print("Test Accuracy: ", testAcc)
    print("#Features: ", featCnt)
            
    return testAcc, featCnt, bestpop



datasetlist = ["BreastCancer.csv", "Tic-tac-toe.csv", "Wine.csv", "HeartEW.csv", "Exactly.csv", "Exactly2.csv", "M-of-n.csv", "Zoo.csv", "Vote.csv", "CongressEW.csv", "Lymphography.csv", "SpectEW.csv", "BreastEW.csv", "Ionosphere.csv", "KrVsKpEW.csv", "WaveformEW.csv", "Sonar.csv", "PenglungEW.csv"]

for datasetname  in datasetlist:    
    print(datasetname)
    accuArr = []
    featArr = []
    agenArr = []
    #start_time = datetime.now()
    for i in range(15):
        # print(i)
        testAcc, featCnt, gbest = NMRA(datasetname)
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
#             currAgent = agentBest.copy()
    datasetname = datasetname.split('.')[0]
#     print(datasetname)
    print(maxx,currFeat)

