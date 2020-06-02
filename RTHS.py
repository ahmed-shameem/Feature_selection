import numpy as np
import random
from copy import deepcopy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import math,time,sys
from matplotlib import pyplot
import pandas as pd
from datetime import datetime


pop_size = 5
Pm = 0.2
omega = 0.9

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

def combFact(dim):
    f = []
    for i in range(dim):
        f.append(np.random.choice([-1,0,1]))
        
    return f

def SMO(x):
    for i in range(len(x)):
        random.seed(i**3 + 10 + time.time() ) 
        rnd = random.random()
        if (rnd <= Pm):
            x[i] = 1 - x[i]
        
    return x

def HS(pop, fit, dimension, trainX, testX, trainy, testy):    
    
    hybrid = np.array([])
    counter = 0

    for j in range(dimension):
        random.seed(j**3 + 10 + time.time())
        ra = random.randint(0, pop_size-1)
        hybrid = np.append(hybrid, pop[ra][j])

    worst = pop[0]
    for j in range(pop_size):
        if(fit[j] > fitness(worst, trainX, testX, trainy, testy)):
            worst = deepcopy(pop[j])
            counter = j

    if(fitness(worst, trainX, testX, trainy, testy) > fitness(hybrid, trainX, testX, trainy, testy)):
        fit[counter] = deepcopy(fitness(hybrid, trainX, testX, trainy, testy))
        pop[counter] = deepcopy(hybrid)

    return pop, fit

def ringOpt(dataset):
    df = pd.read_csv(dataset)
    a, b = np.shape(df)
    data = df.values[:,0:b-1]
    label = df.values[:,b-1]
    dimension = data.shape[1]
    MIT = 30
    
    cross = 5
    test_size = (1/cross)
    trainX, testX, trainy, testy = train_test_split(data, label,stratify=label ,test_size=test_size,random_state=(7+17*int(time.time()%1000)))
    clf=KNeighborsClassifier(n_neighbors=5)
    clf.fit(trainX,trainy)
    val=clf.score(testX,testy)
    whole_accuracy = val
    print("Total Acc: ",val)
    
    pop = initialise(pop_size, dimension, trainX, testX, trainy, testy)
#     print("pop = ", pop)
    fit = []
    for i in range(pop_size):
        fit.append(fitness(pop[i], trainX, testX, trainy, testy))
     
            
    for i in range(MIT):
        for j in range(pop_size):
            pop, fit = HS(pop, fit, dimension, trainX, testX, trainy, testy)
        for j in range(pop_size):
            random.seed(i**3 + 10 + time.time() )
            one = random.randint(0,pop_size-1)
            two = random.randint(0,pop_size-1)
            three = random.randint(0,pop_size-1)
            four = random.randint(0, pop_size-1)

            One, Two, Three, Four = pop[one], pop[two], pop[three], pop[four]

                           
            y,z = np.array([]), np.array([])
        
            random.seed(i**4 + 40 + time.time()*500)
            r = random.random()
            if (r <= 0.5):
                y = np.append(y, np.add(One, np.multiply(Four, np.add(Two, Three)))%2)
            else:
                y = np.append(y, np.add(One, np.add(Two, Three))%2)

            
            z = np.append(z,SMO(y))

            if(fitness(z, trainX, testX, trainy, testy) < fitness(pop[j], trainX, testX, trainy, testy)): 
                pop[j] = deepcopy(z)
                fit[j] = deepcopy(fitness(z, trainX, testX, trainy, testy))
                
     
    gbest = pop[0]
    for i in range(len(pop)):
        if (fitness(pop[i], trainX, testX, trainy, testy) < fitness(gbest, trainX, testX, trainy, testy)):
            gbest = deepcopy(pop[i])
        
    testAcc = test_accuracy(gbest, trainX, testX, trainy, testy)
    #print(gbest)
    featCnt = onecnt(gbest)
    print("Test Accuracy: ", testAcc)
    print("#Features: ", featCnt)
    
    return gbest, testAcc, featCnt
                

datasetlist = ["BreastCancer.csv", "BreastEW.csv", "CongressEW.csv", "Exactly.csv", "Exactly2.csv", "HeartEW.csv", "Ionosphere.csv", "Lymphography.csv", "M-of-n.csv", "PenglungEW.csv", "Sonar.csv", "SpectEW.csv", "Tic-tac-toe.csv", "Vote.csv", "Wine.csv", "Zoo.csv","KrVsKpEW.csv", "WaveformEW.csv" ]

for datasetname  in datasetlist:    
    print(datasetname)
    accuArr = []
    featArr = []
    currAgent = []
    start_time = datetime.now()
    maxx = -1
    currFeat= 20000
    currAgent = []
    for i in range(20):
        agentBest, testAcc, featCnt = ringOpt(datasetname)
    if testAcc>maxx:
        maxx = testAcc
        currFeat = featCnt
        currAgent = agentBest.copy()

	#time_required = datetime.now() - start_time

    datasetname = datasetname.split('.')[0]
    print(datasetname)
    print(maxx,currFeat)
    with open("result_RTHS_KNN.csv","a") as f:
        print(datasetname,"%.2f"%(100*maxx),currFeat,sep=',',file=f,end=',')
        for x in currAgent:
            print(int(x),end=' ',file=f)
            print('',file=f)
