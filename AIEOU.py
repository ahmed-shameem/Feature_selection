import numpy as np
import pandas as pd
import random
import math,time,sys
from matplotlib import pyplot
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from copy import deepcopy

def Ufunc(gamma, alpha, beta):
    return alpha * abs(pow(gamma, beta))


def fitness(particle,trainX,trainy,testX,testy):
    cols=np.flatnonzero(particle)
    val=1
    if np.shape(cols)[0]==0:
        return val
    
    clf=KNeighborsClassifier(n_neighbors=5)
    
    train_data=trainX[:,cols]
    test_data=testX[:,cols]
    clf.fit(train_data,trainy)
    val=1-clf.score(test_data,testy)

    set_cnt=sum(particle)
    set_cnt=set_cnt/np.shape(particle)[0]
    val=omega*val+(1-omega)*set_cnt
    return val


def allfit(population,trainX,trainy,testX,testy):
    x=np.shape(population)[0]
    acc=np.zeros(x)
    for i in range(x):
        acc[i]=fitness(population[i],trainX,trainy,testX,testy)     
    return acc

def initialize(partCount,dim):
    population=np.zeros((partCount,dim))
    minn = 1
    maxx = math.floor(0.8*dim)
    if maxx<minn:
        maxx = minn + 1

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

def avg_concentration(eqPool,poolSize,dimension):
    # simple average
    (r,) = np.shape(eqPool[0])
    avg = np.zeros(np.shape(eqPool[0]))
    for i in range(poolSize):
        x = np.array(eqPool[i])
        avg = avg + x

    avg = avg/poolSize

    for i in range(dimension):
        if avg[i]>=0.5:
            avg[i] = 1
        else:
            avg[i] = 0
    return avg



def signFunc(x): #signum function? or just sign ?
    if x<0:
        return -1
    return 1

def toBinary(currAgent, al, beta):
    Xnew = np.zeros(np.shape(currAgent))
    for i in range(np.shape(currAgent)[0]):
        random.seed(time.time()+i)
        temp = Ufunc(currAgent[i], al, beta)
        if temp > 0.5: # sfunction
            Xnew[i] = float(1)
        else:
            Xnew[i] = float(0)
        
    return Xnew


def updateLA(prevDec,beta,pvec):
    a= 0.01
    b= 0.01
    r=3
    if beta==0: 
        for j in range(3): 
            if j-1 == prevDec:
                pvec[j]=pvec[j]+a*(1-pvec[j])
            else:
                pvec[j]=(1-a)*pvec[j]
    elif beta==1: 
        for j in range(3): 
            if j-1 == prevDec:
                pvec[j]=(1-b)*pvec[j]
            else:
                pvec[j]= b/(r-1)+ (1-b)*pvec[j]
    return pvec

def randomwalk(agent):
    percent = 30
    percent /= 100
    neighbor = agent.copy()
    size = len(agent)
    upper = int(percent*size)
    if upper <= 1:
        upper = size
    x = random.randint(1,upper)
    pos = random.sample(range(0,size - 1),x)
    for i in pos:
        neighbor[i] = 1 - neighbor[i]
    return neighbor


def adaptiveBeta(agent, agentFit, trainX,trainy,testX,testy):
    bmin = 0.1 #parameter: (can be made 0.01)
    bmax = 1
    maxIter = 10 # parameter: (can be increased )
    
    for curr in range(maxIter):
        neighbor = agent.copy()
        size = len(neighbor)
        neighbor = randomwalk(neighbor)

        beta = bmin + (curr / maxIter)*(bmax - bmin)
        for i in range(size):
            random.seed( time.time() + i )
            if random.random() <= beta:
                neighbor[i] = agent[i]
        neighFit = fitness(neighbor,trainX,trainy,testX,testy)
        if neighFit <= agentFit:
            agent = neighbor.copy()
            agentFit = neighFit
            


    return (agent,agentFit)


def iEO(dataset, randomstate, al, beta):
    #========================================================================================
    df=pd.read_csv(dataset)
    (a,b)=np.shape(df)
    data = df.values[:,0:b-1]
    label = df.values[:,b-1]
    dimension = np.shape(data)[1] #solution dimension

    #========================================================================================
    cross = 5
    test_size = (1/cross)
    trainX, testX, trainy, testy = train_test_split(data, label,stratify=label ,test_size=test_size,random_state=randomstate)

    clf=KNeighborsClassifier(n_neighbors=5)
    clf.fit(trainX,trainy)
    val=clf.score(testX,testy)

    x_axis = []
    y_axis = []
    population = initialize(popSize,dimension)
    LAA1 = np.zeros((popSize,3))
    LAA2 = np.zeros((popSize,3))
    LAGP = np.zeros((popSize,3))
    A1=np.zeros(popSize)
    A2=np.zeros(popSize)
    GP=np.zeros(popSize)
    for i in range(popSize):
        LAA1[i][0] = (1/3)
        LAA1[i][1] = (1/3)
        LAA1[i][2] = (1/3)

        LAA2[i][0] = (1/3)
        LAA2[i][1] = (1/3)
        LAA2[i][2] = (1/3)

        LAGP[i][0] = (1/3)
        LAGP[i][1] = (1/3)
        LAGP[i][2] = (1/3)


        A1[i]=(Amax+Amin)/2
        A2[i]=(Amax+Amin)/2
        GP[i]=(GPmax+GPmin)/2

    eqPool = np.zeros((poolSize+1,dimension))
    eqfit = np.zeros(poolSize+1)
    for i in range(poolSize+1):
        eqfit[i] = 100

    start_time = datetime.now()
    accList = allfit(population,trainX,trainy,testX,testy)
    for curriter in range(maxIter):
        popnew = np.zeros((popSize,dimension))
        for i in range(popSize):
            for j in range(poolSize):
                if accList[i] <= eqfit[j]:
                    eqfit[j] = deepcopy(accList[i])
                    eqPool[j] = population[i].copy()
                    break

        Cave = avg_concentration(eqPool,poolSize,dimension)
        Cave = toBinary(Cave, al, beta)
        eqPool[poolSize] = Cave.copy()
        eqfit[poolSize] = fitness(Cave,trainX,trainy,testX,testy)     
        
        for p in range(len(eqPool)):
            eqPool[p], eqfit[p] = adaptiveBeta(eqPool[p], eqfit[p], trainX,trainy,testX,testy)
            
        fitListnew=[]
        for i in range(popSize):
            #choose THE BEST candidate from the equillibrium pool
            bfit = eqfit[0]
            bcan = eqPool[0]
            for e in range(1,len(eqPool)):
                if eqfit[e] < bfit:
                    bfit = eqfit[e]
                    bcan = eqPool[e]
            
            Ceq = bcan
            
            lambdaVec = np.zeros(np.shape(Ceq))
            rVec = np.zeros(np.shape(Ceq))
            for j in range(dimension):
                random.seed(time.time() + 1.1)
                lambdaVec[j] = random.random()
                random.seed(time.time() + 10.01)
                rVec[j] = random.random()

            random.seed(time.time()+17)
            decisionGP = np.random.choice([-1,0,1],1,p=LAGP[i])[0]
            GP[i] = GP[i] + decisionGP*deltaGP
            if GP[i]>GPmax:
                GP[i]=GPmax
            if GP[i]<GPmin:
                GP[i]=GPmin


            random.seed(time.time()+17)
            decisionA1 = np.random.choice([-1,0,1],1,p=LAA1[i])[0]
            A1[i] = A1[i] + decisionA1*deltaA1
            if A1[i]>Amax:
                A1[i]=Amax
            if A1[i]<Amin:
                A1[i]=Amin

            random.seed(time.time()+19)
            decisionA2 = np.random.choice([-1,0,1],1,p=LAA2[i])[0]
            A2[i] = A2[i] + decisionA2*deltaA2
            if A2[i]>Amax:
                A2[i]=Amax
            if A2[i]<Amin:
                A2[i]=Amin

            t = (1 - (curriter/maxIter))**(A2[i]*curriter/maxIter)
            FVec = np.zeros(np.shape(Ceq))
            for j in range(dimension):
                x = -1*lambdaVec[j]*t 
                x = math.exp(x) - 1
                x = A1[i] * signFunc(rVec[j] - 0.5) * x

            random.seed(time.time() + 200)
            r1 = random.random()
            random.seed(time.time() + 20)
            r2 = random.random()
            if r2 < GP[i]:
                GCP = 0
            else:
                GCP = 0.5 * r1
            G0 = np.zeros(np.shape(Ceq))
            G = np.zeros(np.shape(Ceq))
            for j in range(dimension):
                G0[j] = GCP * (Ceq[j] - lambdaVec[j]*population[i][j])
                G[j] = G0[j]*FVec[j]
            temp=[]
            for j in range(dimension):
                temp.append(Ceq[j] + (population[i][j] - Ceq[j])*FVec[j] + G[j]*(1 - FVec[j])/lambdaVec[j])
            temp=np.array(temp)
            popnew[i]=toBinary(temp, al, beta)
            fitNew = fitness(popnew[i],trainX,trainy,testX,testy)    

            fitListnew.append(fitNew)
            beta=1 
            if fitNew<=accList[i]:
                beta = 0
            LAA1[i]= deepcopy(updateLA(decisionA1,beta,LAA1[i]))
            LAA2[i]= deepcopy(updateLA(decisionA2,beta,LAA2[i]))
            LAGP[i]= deepcopy(updateLA(decisionGP,beta,LAGP[i]))
            

        population = popnew.copy()
        accList = deepcopy(fitListnew)
        bestfit=[]
        for pop in population:
            bestfit.append(fitness(pop,trainX,trainy,testX,testy))


    output = eqPool[0].copy()
    cols = np.flatnonzero(output)
    X_test = testX[:,cols]
    X_train = trainX[:,cols]

    clf=KNeighborsClassifier(n_neighbors=5)
    
    clf.fit(X_train,trainy)
    val=clf.score(X_test, testy )
    print(val,output.sum())
    return output,val

############################################################################################################
poolSize = 4
popSize=20
maxIter=30
omega = 0.99

Amax=5
Amin=0.1
A2max=5
A2min=0.1
GPmax = 1
GPmin = 0
deltaA2=0.5
deltaA1=0.5
deltaGP=0.05
#can be tuned: t, GP,

# alpha = [0.5,0.67,0.83,1,1.17,1.33,1.5,1.67,1.83,2]
alpha = 2
beta = 4
# beta = [1.5,1.78,2.06,2.33,2.61,2.89,3.17,3.44,3.72,4]

datasetList = ["BreastCancer.csv", "BreastEW.csv", "CongressEW.csv", "Exactly.csv", "Exactly2.csv", "HeartEW.csv", "Ionosphere.csv", "KrVsKpEW.csv","Lymphography.csv", "M-of-n.csv", "PenglungEW.csv", "Sonar.csv", "SpectEW.csv", "Tic-tac-toe.csv", "Vote.csv", "WaveformEW.csv", "Wine.csv", "Zoo.csv"]
randomstateList=[15,5,15,26,12,7,10,8,37,19,35,2,49,26,1,25,47,12]


for datasetinx in range(len(datasetList)):
#     for r in range(len(alpha)):
    dataset=datasetList[datasetinx]
    randomstate=randomstateList[datasetinx]
    maxRun = 20
    print(dataset)


    best_accuracy = -1
    best_no_features = -1
    accuracyList = []
    featureList = []

    for runNo in range(20):
# 		print(runNo)
        #===============================================================================================================
        # start_time = time.time()
        agent,val=iEO(dataset, randomstate, alpha, beta)
        accuracyList.append(val)
        featureList.append(agent.sum())
        if val>best_accuracy:
            best_accuracy = val
            best_no_features = agent.sum()
        
        if ( val == best_accuracy ) and ( agent.sum()  < best_no_features ):
            best_accuracy = val
            best_no_features = agent.sum()

    print("Test Accuracy: ", best_accuracy)
    print("#Features: ", best_no_features)
    