# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 14:02:24 2017

@author: Admin
"""
import numpy as np
from scipy import linalg
import copy
from scipy.optimize import minimize
#import matrixops
import inspyred
from random import Random
from time import time
from inspyred import ec
import math as m
from pyKriging import samplingplan
from numpy.matlib import rand,zeros,ones,empty,eye
import scipy
#from matrixops.matrixops import matrixops
class matrixops():
    
    def __init__(self):
        self.LnDetPsi = None
        self.Psi = np.zeros((self.n,self.n), dtype=np.float)
        self.psi = np.zeros((self.n,1))
        self.one = np.ones(self.n)
        self.mu = None
        self.U = None
        self.SigmaSqr = None
        self.Lambda = 1
        self.updateData()

    def updateData(self):
        self.distance = np.zeros((self.n,self.n, self.k))
        for i in range(self.n):
            for j in range(i+1,self.n):
                self.distance[i,j]= np.abs((self.X[i]-self.X[j]))

    def updatePsi(self):
        self.Psi = np.zeros((self.n,self.n), dtype=np.float)
        self.one = np.ones(self.n)
        self.psi = np.zeros((self.n,1))
        newPsi = np.exp(-np.sum(self.theta*np.power(self.distance,self.pl), axis=2))
        self.Psi = np.triu(newPsi,1)
        self.Psi = self.Psi + self.Psi.T + np.mat(eye(self.n))+np.multiply(np.mat(eye(self.n)),np.spacing(1))
        self.U = np.linalg.cholesky(self.Psi)
        self.U = self.U.T

    def regupdatePsi(self):
        self.Psi = np.zeros((self.n,self.n), dtype=np.float)
        self.one = np.ones(self.n)
        self.psi = np.zeros((self.n,1))
        newPsi = np.exp(-np.sum(self.theta*np.power(self.distance,self.pl), axis=2))
        self.Psi = np.triu(newPsi,1)
        self.Psi = self.Psi + self.Psi.T + eye(self.n) + eye(self.n) * (self.Lambda)
        self.U = np.linalg.cholesky(self.Psi)
        self.U = np.matrix(self.U.T)


    def neglikelihood(self):
        self.LnDetPsi=2*np.sum(np.log(np.abs(np.diag(self.U))))

        a = np.linalg.solve(self.U.T, self.one.T)
        b = np.linalg.solve(self.U, a)
        c = self.one.T.dot(b)
        d = np.linalg.solve(self.U.T, self.y)
        e = np.linalg.solve(self.U, d)
        self.mu=(self.one.T.dot(e))/c

        self.SigmaSqr = ((self.y-self.one.dot(self.mu)).T.dot(np.linalg.solve(self.U,np.linalg.solve(self.U.T,(self.y-self.one.dot(self.mu))))))/self.n
        self.NegLnLike=-1.*(-(self.n/2.)*np.log(self.SigmaSqr) - 0.5*self.LnDetPsi)

    def regneglikelihood(self):
        self.LnDetPsi=2*np.sum(np.log(np.abs(np.diag(self.U))))

        mu=(self.one.T.dot(np.linalg.solve(self.U, np.linalg.solve(self.U.T, self.y))))/self.one.T.dot(np.linalg.solve(self.U, np.linalg.solve(self.U.T, self.one)))
        self.mu=mu

        self.SigmaSqr = ((self.y-self.one.dot(self.mu)).T.dot(np.linalg.solve(self.U,np.linalg.solve(self.U.T,(self.y-self.one.dot(self.mu))))))/self.n

        self.NegLnLike=-1.*(-(self.n/2.)*np.log(self.SigmaSqr) - 0.5*self.LnDetPsi)

    def predict_normalized(self,x):
        for i in range(self.n):
            self.psi[i]=np.exp(-np.sum(self.theta*np.power((np.abs(self.X[i]-x)),self.pl)))##相关性矩阵
        z = self.y-self.one.dot(self.mu)
        a = np.linalg.solve(self.U.T, z)
        b=np.linalg.solve(self.U, a)
        c=self.psi.T.dot(b)

        f=self.mu + c
        return f[0]

    def predicterr_normalized(self,x):
        for i in range(self.n):
            try:
                self.psi[i]=np.exp(-np.sum(self.theta*np.power((np.abs(self.X[i]-x)),self.pl)))
            except Exception as e:
                print(Exception,e)
        try:
            SSqr=self.SigmaSqr*(1-self.psi.T.dot(np.linalg.solve(self.U, np.linalg.solve(self.U.T,self.psi))))
        except Exception as e:
            print(self.U.shape)
            print(self.SigmaSqr.shape)
            print(self.psi.shape)
            print(Exception,e)
            pass

        SSqr = np.abs(SSqr[0])
        return np.power(SSqr,0.5)[0]

    def regression_predicterr_normalized(self,x):
        for i in range(self.n):
            try:
                self.psi[i]=np.exp(-np.sum(self.theta*np.power((np.abs(self.X[i]-x)),self.pl)))
            except Exception as e:
                print(Exception,e)
        try:
            SSqr=self.SigmaSqr*(1+self.Lambda-self.psi.T.dot(np.linalg.solve(self.U, np.linalg.solve(self.U.T,self.psi))))
        except Exception as e:
            print(Exception,e)
            pass

        SSqr = np.abs(SSqr[0])
        return np.power(SSqr,0.5)[0]

class Prs(matrixops):
    def __init__(self,x,y,**kwargs):
        self.normRange = []
        self.ynormRange = []
        self.X = copy.deepcopy(x)
        self.y = copy.deepcopy(y)
        self.n,self.k = self.X.shape#增加平方项
        self.yn = self.y.shape[0]
        if not self.n ==self.yn:
            raise ValueError("All arrays must be equal length.")
        self.xsqt = self.X**2
        self.normalizeData()
        self.theta = np.ones(self.k)
        self.pl = np.ones(self.k) * 2.
        self.sigma = 0        
        # = self.sp = samplingplan.samplingplan(self.k)
        self.updateData()        
        self.updateModel()
        self.thetamin = 1e-5
        self.thetamax = 100
        self.pmin = 1
        self.pmax = 2        
        matrixops.__init__(self)
        
    def predict_var(self, X):##模型的预测误差
        '''
        The function returns the model's predicted 'error' at this point in the model.
        :param X: new design variable to evaluate, in physical world units
        :return: Returns the posterior variance (model error prediction)
        '''
        X = copy.deepcopy(X)
        X = self.normX(X)
        # print X, self.predict_normalized(X), self.inversenormy(self.predict_normalized(X))
        return self.predicterr_normalized(X)
        
    def normX(self,X):
        X = copy.deepcopy(X)
       
        for i in range(self.k):
            X[:,i] = (X[:,i] - self.normRange[i][0]) / float(self.normRange[i][1] - self.normRange[i][0])
        return X
    
    def inversenormX(self,X): 
        X = copy.deepcopy(X)
        for i in range(self.k):
            X[:,i] = (X[:,i] * float(self.normRange[i][1] - self.normRange[i][0] )) + self.normRange[i][0]
        return X
        
    def normy(self,y):
        return (y - self.ynormRange[0]) / (self.ynormRange[1] - self.ynormRange[0])
    
    def inversenormy(self, y):
        '''
        :param y: A normalized array of model units in the range of [0,1]
        :return: An array of observed values in real-world units
        '''
        return (y * (self.ynormRange[1] - self.ynormRange[0])) + self.ynormRange[0]
        
    def normalizeData(self):
        '''
        This function is called when the initial data in the model is set.
        We find the max and min of each dimension and norm that axis to a range of [0,1]
        '''
        for i in range(self.k):
            #print min(self.X[:,i]),max(self.X[:,i])
            self.normRange.append([min(self.X[:,i]), max(self.X[:,i])])
        #print self.normRange
              
        #self.X = self.normX(self.X)
        #print self.X        
        self.ynormRange.append(min(self.y))
        self.ynormRange.append(max(self.y))
        
        #for i in range(self.n):
            #self.y[i] = self.normy(self.y[i])        
        #print self.X,'self.x',self.y,'y'
        
    def train(self):
        self.updateData()
        newrow = []
        for j in range(self.k):
            for k in range(j+1,self.k):
                newrow.append(self.X[:,j] * self.X[:,k])
        newrow = np.array(newrow).T
        #print newrow 
        self.xi = np.c_[self.X, self.xsqt,newrow]
        self.prsones = np.ones((self.n,1))##加单位阵
        self.xi =  np.c_[self.prsones,self.xi]

        self.x_i = np.transpose(self.xi)
        self.y_fix = np.dot(self.x_i, self.y)##x'y
        self.x_fix = np.dot(self.x_i,self.xi)
        self.nodes = linalg.solve(self.x_fix, self.y_fix)       
        self.updateModel()
        self.neglikelihood()
        
    def predict(self,x):
        self.xp = copy.deepcopy(x)
        if isinstance(self.xp, list):
           self.xp = np.asarray(self.xp)
            
        Xprs = self.xp**2
        #print Xprs,'xprs'
        '''
        for i in range(self.k):
            for j in range(i+1,self.k):
                newrow = np.asarray([0 for i in range(self.pn)])
                for m in range(self.pn):
                    print self.xp[:,i]
                    print self.xp[:,i][m]
                    newrow[m] = self.xp[:,i][m] * self.xp[:,j][m]
                self.xp = np.c_[self.xp,newrow]
        #newcol = np.array(newcol).T
        #print newrow 
        '''
        for i in range(self.k):
            for j in range(i+1,self.k):
                newrow = np.asarray([0 for v in range(self.k)])
                #print newrow.shape,'newrow'
                #for mv in range(self.k):
                newrow = self.xp[i] * self.xp[j]
                #print newrow
                self.xp = np.append(self.xp, [newrow], axis = 0)  
        #print self.xp
        
        xpi = np.append(self.xp, Xprs, axis = 0)        
        #prsones = np.ones((self.pn,1))##加单位阵
        Xi =  np.r_[1,xpi]        
        #self.Yprs = np.zeros(((self.pn,1)))
        #for i in range(self.pn):
        self.Yprs = np.dot(Xi,self.nodes)
        #self.Yprs = self.inversenormy(self.Yprs)
        return self.Yprs
        
    def Rsquared(self):
        #y = self.inversenormy(self.y)
        y=self.y
        ym = np.mean(y)
        sst = np.sum((self.Yprs-ym)**2)
        ssr = np.sum((y-ym)**2)
        return ssr/sst
        
    def addPoint(self, newX, newy):
        '''
        This add points to the model.
        :param newX: A new design vector point
        :param newy: The new observed value at the point of X
        :param norm: A boolean value. For adding real-world values, this should be True. If doing something in model units, this should be False
        '''
        #if norm:
            #newX = self.normX(newX)
            #newy = self.normy(newy)

        self.X = np.append(self.X, [newX], axis=0)
        self.y = np.append(self.y, newy)
        self.n = self.X.shape[0]
        self.updateData()
        while True:
            try:
                self.updateModel()
            except:
                self.train()
            else:
                break
    
    def updateModel(self):
        '''
        The function rebuilds the Psi matrix to reflect new data or a change in hyperparamters
        
        self.updatePsi()
        self.U = np.asarray(self.U)
        print self.U.shape
        '''
        try:
            self.updatePsi()
        except Exception as err:
            #pass
            # print Exception, err
            raise Exception("bad params")
                  
    def generate_population(self, random, args):
        '''
        Generates an initial population for any global optimization that occurs in pyKriging
        :param random: A random seed
        :param args: Args from the optimizer, like population size
        :return chromosome: The new generation for our global optimizer to use
        '''
        size = args.get('num_inputs', None)
        bounder = args["_ec"].bounder
        chromosome = []
        for lo, hi in zip(bounder.lower_bound, bounder.upper_bound):
            chromosome.append(random.uniform(lo, hi))
        return chromosome
    
    def expimp(self, x):
        '''
        Returns the expected improvement at the design vector X in the model
        :param x: A real world coordinates design vector
        :return EI: The expected improvement value at the point x in the model
        '''
        S = self.predicterr_normalized(x)
        y_min = np.min(self.y)
        if S <= 0.:
            EI = 0.
        elif S > 0.:
            EI_one = ((y_min - self.predict_normalized(x)) * (0.5 + 0.5*m.erf((
                      1./np.sqrt(2.))*((y_min - self.predict_normalized(x)) /
                                       S))))
            EI_two = ((S * (1. / np.sqrt(2. * np.pi))) * (np.exp(-(1./2.) *
                      ((y_min - self.predict_normalized(x))**2. / S**2.))))
            EI = EI_one + EI_two
        return EI
        
    def infill_objective_mse(self,candidates, args):##设计变量的均方差
        '''
        This acts
        :param candidates: An array of candidate design vectors from the infill global optimizer
        :param args: args from the optimizer
        :return fitness: An array of evaluated MSE values for the candidate population
        '''
        fitness = []
        for entry in candidates:
            fitness.append(-1 * self.predicterr_normalized(entry))
        return fitness

    def infill_objective_ei(self,candidates, args):#期望的改善值
        '''
        The infill objective for a series of candidates from infill global search
        :param candidates: An array of candidate design vectors from the infill global optimizer
        :param args: args from the optimizer
        :return fitness: An array of evaluated Expected Improvement values for the candidate population
        '''
        fitness = []
        for entry in candidates:
            fitness.append(-1 * self.expimp(entry))
        return fitness
    
    def infill(self, points, method='error', addPoint=True):
        '''
        The function identifies where new points are needed in the model.
        :param points: The number of points to add to the model. Multiple points are added via imputation.
        :param method: Two choices: EI (for expected improvement) or Error (for general error reduction)
        :return: An array of coordinates identified by the infill
        '''
        # We'll be making non-permanent modifications to self.X and self.y here, so lets make a copy just in case
        initX = np.copy(self.X)
        inity = np.copy(self.y)##得到初始x，y

        # This array will hold the new values we add
        returnValues = np.zeros([points, self.k], dtype=float)##建立一个加点零阵
        for i in range(points):
            rand = Random()#随机产生一个数
            rand.seed(int(time()))
            ea = inspyred.swarm.PSO(Random())
            ea.terminator = self.no_improvement_termination
            ea.topology = inspyred.swarm.topologies.ring_topology
            if method=='ei':
                evaluator = self.infill_objective_ei
            else:
                evaluator = self.infill_objective_mse

            final_pop = ea.evolve(generator=self.generate_population,
                                  evaluator=evaluator,
                                  pop_size=155,
                                  maximize=False,
                                  bounder=ec.Bounder([0] * self.k, [1] * self.k),
                                  max_evaluations=20000,
                                  neighborhood_size=30,
                                  num_inputs=self.k)
            final_pop.sort(reverse=True)##从大大小排列
            #print final_pop
        
            newpoint = final_pop[0].candidate
            returnValues[i][:] = newpoint
        #print returnValues
        #if addPoint:##函数里的参数为真时 则执行下列代码
            #self.addPoint(returnValues, self.predict(returnValues))##把加的点加到初始x中

        self.Xc = np.copy(initX)
        self.y = np.copy(inity)
        self.n = len(self.Xc)
        self.updateData()
        while True:
            try:
                self.updateModel()
            except:
                self.train()
            else:
                break
        return returnValues
        
    def no_improvement_termination(self, population, num_generations, num_evaluations, args):
        '''
        Return True if the best fitness does not change for a number of generations of if the max number
        of evaluations is exceeded.

        .. Arguments:
           population -- the population of Individuals
           num_generations -- the number of elapsed generations
           num_evaluations -- the number of candidate solution evaluations
           args -- a dictionary of keyword arguments

        Optional keyword arguments in args:

        - *max_generations* -- the number of generations allowed for no change in fitness (default 10)

        '''
        max_generations = args.setdefault('max_generations', 10)
        previous_best = args.setdefault('previous_best', None)
        max_evaluations = args.setdefault('max_evaluations', 30000)
        current_best = np.around(max(population).fitness, decimals=4)
        if previous_best is None or previous_best != current_best:   
            args['previous_best'] = current_best
            args['generation_count'] = 0
            return False or (num_evaluations >= max_evaluations)
        else:
            if args['generation_count'] >= max_generations:
                return True
            else:
                args['generation_count'] += 1
                return False or (num_evaluations >= max_evaluations)
