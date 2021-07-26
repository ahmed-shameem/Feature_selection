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
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_curve

MaxIter = 30
p_size = [20]
omega =  0.99


def initialise(partCount, dim, trainX, testX, trainy, testy):
    population=np.zeros((partCount,dim))
    minn = 1
    maxx = math.floor(0.5*dim)

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

def fitness(agent, trainX, testX, trainy, testy):

    cols=np.flatnonzero(agent)

    val=1
    if np.shape(cols)[0]==0:
        return val
    clf=KNeighborsClassifier(n_neighbors=5)

    train_data=trainX[:,cols]
    test_data=testX[:,cols]
    clf.fit(train_data,trainy)
    val=1-clf.score(test_data,testy)


    set_cnt=sum(agent)
    set_cnt=set_cnt/np.shape(agent)[0]
    val=omega*val+(1-omega)*set_cnt
    return val

def allfit(pop, trainX, testX, trainy, testy):
    fit = np.zeros((len(pop), 1))
    for p in range(len(pop)):
        fit[p] = fitness(pop[p], trainX, testX, trainy, testy)
    return fit

def test_accuracy(agent, trainX, testX, trainy, testy):
    cols=np.flatnonzero(agent)
    val=1
    if np.shape(cols)[0]==0:
        return val

    clf=KNeighborsClassifier(n_neighbors=5)

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

def bestAgent(fit):
    ind = np.argsort(fit, axis=0)
    return ind[0]


def randomwalk(agent):
    percent = 30
    percent /= 100
    neighbor = agent.copy()
    size = np.shape(agent)[0]
    upper = int(percent*size)
    if upper <= 1:
        upper = size
    x = random.randint(1,upper)
    pos = random.sample(range(0,size - 1),x)
    for i in pos:
        neighbor[i] = 1 - neighbor[i]
    return neighbor

## Simulated Anealing
def SA(agent, fitAgent, trainX, testX, trainy, testy):
    # initialise temprature
    T = 4*np.shape(agent)[0]; T0 = 2*np.shape(agent)[0];

    S = agent.copy();
    bestSolution = agent.copy();
    bestFitness = fitAgent;

    while T > T0:
        neighbor = randomwalk(S)
        neighborFitness = fitness(neighbor, trainX, testX, trainy, testy)

        if neighborFitness < bestFitness:
            S = neighbor.copy()
            bestSolution = neighbor.copy()
            bestFitness = neighborFitness

        elif neighborFitness == bestFitness:
            if np.sum(neighbor) == np.sum(bestSolution):
                S = neighbor.copy()
                bestSolution = neighbor.copy()
                bestFitness = neighborFitness

        else:
            theta = neighborFitness - bestFitness
            if np.random.rand() < math.exp(-1*(theta/T)):
                S = neighbor.copy()

        T *= 0.925

    return bestSolution, bestFitness




def GNDO(dataset, pop_size):
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


    pop = initialise(pop_size, dimension, trainX, testX, trainy, testy)

    best_agent, best_fit = pop[0], fitness(pop[0], trainX, testX, trainy, testy)

    for Iter in range(MaxIter):

        fit = allfit(pop, trainX, testX, trainy, testy)
        best_i = bestAgent(fit[0]) # index of the best agent in population
        mo = np.mean(pop, axis=0)

        for i in range(pop_size):

            a = np.random.randint(pop_size)
            b = np.random.randint(pop_size)
            c = np.random.randint(pop_size)

            while a==i or a==b or a==c or b==c or b==i or c==i:

                a = np.random.randint(pop_size)
                b = np.random.randint(pop_size)
                c = np.random.randint(pop_size)

            if fit[a][0] < fit[i][0]:
                v1 = np.subtract(pop[a], pop[i])
            else:
                v1 = np.subtract(pop[i], pop[a])



            if fit[b][0] < fit[c][0]:
                v2 = np.subtract(pop[b], pop[c])
            else:
                v2 = np.subtract(pop[c], pop[b])



            random.seed(time.time()*10%7)
            if random.random() > 0.5:
                u = (1/3) * np.add(pop[i], np.add(pop[best_i], mo))


                delta = np.sqrt((1/3) * (np.add(np.square(np.subtract(pop[i], u)), np.add(np.square(np.subtract(pop[best_i], u)), np.square(np.subtract(mo, u))))))


                random.seed(time.time()*10%10)
                vc1 = np.random.rand(1, dimension)
                vc2 = np.random.rand(1, dimension)

                vc1 = vc1.flatten()
                vc2 = vc2.flatten()


                Z1 = np.sqrt(-1*np.log(vc2)) * np.cos(2*np.pi*vc1)
                Z2 = np.sqrt(-1*np.log(vc2)) * np.cos(2*np.pi*vc1 + np.pi)

                random.seed(time.time()*10%9)
                a = np.random.uniform()
                random.seed(time.time()*10%2)
                b = np.random.uniform()

                if a <= b:
                    eta = np.add(u, np.multiply(delta, Z1))
                else:
                    eta = np.add(u, np.multiply(delta, Z2))

                newsol = eta.copy()

            else:

                random.seed(time.time()*10%70)
                beta = np.random.uniform()

                v = np.add(pop[i], np.add(beta * np.random.uniform() * v1, (1-beta) * np.random.uniform() * v2))

                newsol = v.copy()


            for k in range(dimension):
                if(sigmoid(newsol[k]) < random.random()):
                    newsol[k] = 0
                else:
                    newsol[k] = 1

            newfit = fitness(newsol, trainX, testX, trainy, testy)

            if newfit < fit[i][0]:
                pop[i] = newsol.copy()
                fit[i][0] = newfit
                if(newfit < fit[best_i][0]):
                    fit[best_i][0] = newfit
                    pop[best_i] = newsol

        #########################################################
        # LOCAL SEARCH
        ########################################################
        for agentNum in range(pop_size):
            pop[agentNum], fit[agentNum][0] = SA(pop[agentNum], fit[agentNum][0], trainX, testX, trainy, testy)


        least_fit = min(fit)
        least_fit = float(least_fit)
        ### ADDED LATER ###
        if best_fit > least_fit:
            best_fit = least_fit


    testAcc = test_accuracy(pop[best_i], trainX, testX, trainy, testy)
    featCnt = onecnt(pop[best_i])

    return testAcc, featCnt, pop[best_i]


if __name__ == "__main__":
    datasetlist = ["BreastCancer.csv", "BreastEW.csv", "CongressEW.csv", "Exactly.csv", "Exactly2.csv", "HeartEW.csv", "Ionosphere.csv", "Lymphography.csv", "M-of-n.csv", "PenglungEW.csv", "Sonar.csv", "SpectEW.csv", "Tic-tac-toe.csv", "Vote.csv", "Wine.csv", "Zoo.csv","KrVsKpEW.csv", "WaveformEW.csv" ]

    for pop_size in p_size:
        print(pop_size)
        for datasetname  in datasetlist:
            print(datasetname)
            accuArr = []
            featArr = []
            agenArr = []

            for i in range(3):

                testAcc, featCnt, gbest = GNDO(datasetname, pop_size)

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
            print('Best: ', maxx,currFeat)
            with open("GNDO+SA_RF.csv", "a") as f:
                print(datasetname, maxx, currFeat, sep=',', file=f)
            
