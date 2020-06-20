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

MaxIter = 100
pop_size = 20
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
    
def RSGW(dataset):
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
        
    
    for n in range(MaxIter):
        fit = []
        for i in range(pop_size):
            fit.append(fitness(pop[i], trainX, testX, trainy, testy))
        ind = np.argsort(fit)
        alpha = pop[ind[0]]
        alpha_fit = fit[ind[0]]
        beta = pop[ind[1]]
        beta_fit = pop[ind[1]]
        delta = pop[ind[2]]
        delta_fit = fit[ind[2]]
        
        a=2-n*((2)/MaxIter); # a decreases linearly fron 2 to 0
        
        # Update the Position of search agents including omegas
        for i in range(pop_size):
            for j in range (dimension):     
                           
                r1=random.random() # r1 is a random number in [0,1]
                r2=random.random() # r2 is a random number in [0,1]
                
                A1=2*a*r1-a; # Equation (3.3)
                C1=2*r2; # Equation (3.4)
                
                D_alpha=abs(C1*alpha[j]-pop[i,j]); # Equation (3.5)-part 1
                X1=alpha[j]-A1*D_alpha; # Equation (3.6)-part 1
                           
                r1=random.random()
                r2=random.random()
                
                A2=2*a*r1-a; # Equation (3.3)
                C2=2*r2; # Equation (3.4)
                
                D_beta=abs(C2*beta[j]-pop[i,j]); # Equation (3.5)-part 2
                X2=beta[j]-A2*D_beta; # Equation (3.6)-part 2       
                
                r1=random.random()
                r2=random.random() 
                
                A3=2*a*r1-a; # Equation (3.3)
                C3=2*r2; # Equation (3.4)
                
                D_delta=abs(C3*delta[j]-pop[i,j]); # Equation (3.5)-part 3
                X3=delta[j]-A3*D_delta; # Equation (3.5)-part 3             
                
                pop[i,j]=(X1+X2+X3)/3  # Equation (3.7)
                
        if(random.random() > 0.5):           
            for j in range(pop_size):

                r = np.random.rand()
                A = 2 * a * r - a
                C = 2 * r
                l = np.random.uniform(-1, 1)
                p = np.random.rand()
                b = 1

                if (p < 0.5) :
                    if np.abs(A) < 1:
                        D = np.abs(C * alpha - pop[j] )
                        pop[j] = alpha - A * D
                    else :
                        x_rand = pop[np.random.randint(pop_size)] 
                        D = np.abs(C * x_rand - pop[j])
                        pop[j] = (x_rand - A * D)
                else:
                    D1 = np.abs(alpha - pop[j])
                    pop[j] = D1 * np.exp(b * l) * np.cos(2 * np.pi * l) + alpha
                    
        for i in range(pop_size):
            for j in range(dimension):
                if (sigmoid(pop[i][j]) > random.random()):
                    pop[i][j] = 1
                else:
                    pop[i][j] = 0

                
        
    ind = np.argsort(fit)
    bestpop = pop[ind[0]]
    bestfit = fit[ind[0]]
    
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
        testAcc, featCnt, gbest = RSGW(datasetname)
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
    with open("RSGW_result.csv","a") as f:
        print(datasetname,"%.2f"%(100*maxx),currFeat,sep=',',file=f,end=',')
        for x in bagent:
            print(int(x),end=' ',file=f)
        print('',file=f)

