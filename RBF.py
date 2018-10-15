# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 16:16:40 2017

@author: Admin
"""

"""rbf - Radial basis functions for interpolation/smoothing scattered Nd data.

Written by John Travers <jtravs@gmail.com>, February 2007
Based closely on Matlab code by Alex Chirokov
Additional, large, improvements by Robert Hetland
Some additional alterations by Travis Oliphant

Permission to use, modify, and distribute this software is given under the
terms of the SciPy (BSD style) license.  See LICENSE.txt that came with
this distribution for specifics.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.

Copyright (c) 2006-2007, Robert Hetland <hetland@tamu.edu>
Copyright (c) 2007, John Travers <jtravs@gmail.com>

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above
       copyright notice, this list of conditions and the following
       disclaimer in the documentation and/or other materials provided
       with the distribution.

    * Neither the name of Robert Hetland nor the names of any
       contributors may be used to endorse or promote products derived
       from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
#from __future__ import division, print_function, absolute_import

import sys

from numpy import (sqrt, log, asarray, newaxis, all, dot, exp, eye,
                   float_)
from scipy import linalg
from scipy.lib.six import callable, get_method_function, \
     get_function_code
import copy
import numpy as np
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

__all__ = ['Rbf']   # __all__用于将文件内的对象导出，

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
        return np.power(SSqr,0.5)[0]                    # 可以在其他文件中使用 from <module> import *的形式


class Rbf(matrixops):
    """
    Rbf(*args)

    A class for radial basis function approximation/interpolation of
    n-dimensional scattered data.

    Parameters
    ----------
    *args : arrays
        x, y, z, ..., d, where x, y, z, ... are the coordinates of the nodes
        and d is the array of values at the nodes
    function : str or callable, optional
        The radial basis function, based on the radius, r, given by the norm
        (default is Euclidean distance); the default is 'multiquadric'::

            'multiquadric': sqrt((r/self.epsilon)**2 + 1)
            'inverse': 1.0/sqrt((r/self.epsilon)**2 + 1)
            'gaussian': exp(-(r/self.epsilon)**2)
            'linear': r
            'cubic': r**3
            'quintic': r**5
            'thin_plate': r**2 * log(r)

        If callable, then it must take 2 arguments (self, r).  The epsilon
        parameter will be available as self.epsilon.  Other keyword
        arguments passed in will be available as well.

    epsilon : float, optional
        Adjustable constant for gaussian or multiquadrics functions
        - defaults to approximate average distance between nodes (which is
        a good start).
    smooth : float, optional
        Values greater than zero increase the smoothness of the
        approximation.  0 is for interpolation (default), the function will
        always go through the nodal points in this case.
    norm : callable, optional
        A function that returns the 'distance' between two points, with
        inputs as arrays of positions (x, y, z, ...), and an output as an
        array of distance.  E.g, the default::

            def euclidean_norm(x1, x2):
                return sqrt( ((x1 - x2)**2).sum(axis=0) )

        which is called with x1=x1[ndims,newaxis,:] and
        x2=x2[ndims,:,newaxis] such that the result is a matrix of the
        distances from each point in x1 to each point in x2.

    Examples
    --------
    >>> rbfi = Rbf(x, y, z, d)  # radial basis function interpolator instance
    >>> di = rbfi(xi, yi, zi)   # interpolated values

    """

    def _euclidean_norm(self, x1, x2):  # 求两点间的欧几里得距离，输入的两个列表为两点的坐标
        return sqrt(((x1 - x2)**2).sum(axis=0))

    def _h_multiquadric(self, r):   #高次曲面函数
        return sqrt((1.0/self.epsilon*r)**2 + 1)

    def _h_inverse_multiquadric(self, r):   #反高次曲面函数
        return 1.0/sqrt((1.0/self.epsilon*r)**2 + 1)

    def _h_gaussian(self, r):   #高斯函数
        return exp(-(1.0/self.epsilon*r)**2)

    def _h_linear(self, r):
        return r

    def _h_cubic(self, r):
        return r**3

    def _h_quintic(self, r):
        return r**5

    def _h_thin_plate(self, r): #薄板样条函数
        result = r**2 * log(r)
        result[r == 0] = 0  # the spline is zero at zero
        return result

    # Setup self._function and do smoke test on initial r
    def _init_function(self, r):    # 选择径向基函数使用的方程，最终返回径向基函数用r计算后的距离
        if isinstance(self.function, str):  # 选择已提供的径向基函数形式，若self.function是一个字符串
            self.function = self.function.lower()   #字符串调整为小写
            _mapped = {'inverse': 'inverse_multiquadric',
                       'inverse multiquadric': 'inverse_multiquadric',
                       'thin-plate': 'thin_plate'}  #函数字典
            if self.function in _mapped:    #若self.function在函数字典内，变为函数名
                self.function = _mapped[self.function]

            func_name = "_h_" + self.function   #调整函数真名
            if hasattr(self, func_name):    #若类中有该函数
                self._function = getattr(self, func_name)   #调用该函数
            else:   #若类中无函数真名
                functionlist = [x[3:] for x in dir(self) if x.startswith('_h_')]    #指向所有函数名
                raise ValueError("function must be a callable or one of " + #抛出赋值异常，函数必须为一可调用对象
                                     ", ".join(functionlist))
            self._function = getattr(self, "_h_"+self.function)
        elif callable(self.function):   # 若方程为一个可调用的对象：反正这段看不懂我特么就不看了吧算了我特么还是看看吧
            allow_one = False   #
            if hasattr(self.function, 'func_code') or \
                   hasattr(self.function, '__code__'):  #若方程对象有func_code属性或__code__属性
                val = self.function # val+函数对象
                allow_one = True    #allow_one设为1
            elif hasattr(self.function, "im_func"):
                val = get_method_function(self.function)
            elif hasattr(self.function, "__call__"):
                val = get_method_function(self.function.__call__)
            else:
                raise ValueError("Cannot determine number of arguments to function")    #抛出赋值异常，不能确定有函数的赋值变量数

            argcount = get_function_code(val).co_argcount   #获取变量数
            if allow_one and argcount == 1:
                self._function = self.function
            elif argcount == 2:
                if sys.version_info[0] >= 3:
                    self._function = self.function.__get__(self, Rbf)
                else:
                    import new
                    self._function = new.instancemethod(self.function, self,
                                                        Rbf)
            else:
                raise ValueError("Function argument must take 1 or 2 arguments.")

        a0 = self._function(r)
        if a0.shape != r.shape:
            raise ValueError("Callable must take array and return array of the same shape")
        return a0

    def __init__(self,X,y, **kwargs):    #初始化设置，初始化后获得权重矩阵
        self.normRange = []
        self.ynormRange = []
        self.X = copy.deepcopy(X)  # 将变量实验取值都展平为一维向量，xi中有多组行向量，每组行向量代表一个实验变量取值
        self.kwargs = kwargs
        self.xi =self.X.T               
        self.n = self.xi.shape[-1]  #记录一维向量的长度（即试验点的数量）行数
        self.k = self.X.shape[1]
        self.y = copy.deepcopy(y)    #将响应向量展平为一维
        self.yn = self.y.shape[0]
        self.di = self.y
        if not self.n ==self.yn:
            raise ValueError("All arrays must be equal length.")
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
        #if not all([x.size == self.di.size for x in self.xi]):  # 检测输入变量组数和实验结果组数是否相同
            #raise ValueError("All arrays must be equal length.")
#        print kwargs
#        self.norm = kwargs.pop('norm', self._euclidean_norm)    # 可自定义距离形式，若无输入的距离定义则默认采用欧氏距离
#        print self.norm
#        r = self._call_norm(self.xi, self.xi)   # r为两点间距离
#        #print r,'r'
#        self.epsilon = kwargs.pop('epsilon', None)  #设定epsilon，收敛系数
#        if self.epsilon is None:
#            self.epsilon = r.mean()
#        self.smooth = kwargs.pop('smooth', 0.0) #设定收敛平整性
#        self.function = kwargs.pop('function', 'multiquadric')  #默认函数格式为高次曲面函数
        # attach anything left in kwargs to self
        #  for use by any user-callable function or
        #  to save on the object returned.
#        for item, value in kwargs.items():  #用输入的列表中的属性和变量赋值
#            setattr(self, item, value)
        #print self._init_function(r),'funcr'
        #print self.smooth
        #print self.di    
#        self.A = self._init_function(r) - eye(self.n)*self.smooth
#        self.nodes = linalg.solve(self.A, self.di)  #计算权重，需要调用权重时可要求打印instance.nodes
    
    def train(self):
        self.updateData()
        #print self.kwargs
        self.norm = self.kwargs.pop('norm', self._euclidean_norm)    # 可自定义距离形式，若无输入的距离定义则默认采用欧氏距离
        #print self.norm
        r = self._call_norm(self.xi, self.xi)   # r为两点间距离
        #print r,'r'
        self.epsilon = self.kwargs.pop('epsilon', None)  #设定epsilon，收敛系数
        if self.epsilon is None:
            self.epsilon = r.mean()
        self.smooth = self.kwargs.pop('smooth', 0.0) #设定收敛平整性
        self.function = self.kwargs.pop('function', 'multiquadric')  #默认函数格式为高次曲面函数
        # attach anything left in kwargs to self
        #  for use by any user-callable function or
        #  to save on the object returned.
        for item, value in self.kwargs.items():  #用输入的列表中的属性和变量赋值
            setattr(self, item, value)
        #print self._init_function(r),'funcr'
        #print self.smooth
        #print self.di    
        self.A = self._init_function(r) - eye(self.n)*self.smooth
        self.nodes = linalg.solve(self.A, self.di)  #计算权重，需要调用权重时可要求打印instance.nodes
        self.updateModel()
        self.neglikelihood()
        
    def predict(self,X):
        '''
        get the predict value 
        '''
        #args = [asarray(x) for x in args]
        #if not all([x.shape == y.shape for x in args for y in args]):
            #raise ValueError("Array lengths must be equal")
        #shp = args[0].shape
        self.xaa = copy.deepcopy(X)
        self.xa = self.xaa.T
        #shp = self.xa.shape
        #print shp
        #self.xa = asarray([a.flatten() for a in args], dtype=float_)
        r = self._call_norm(self.xa, self.xi)
        #print dot(self._function(r), self.nodes).shape
        return dot(self._function(r), self.nodes)#.reshape(shp) 
        
    def _call_norm(self, x1, x2):   #计算两点间距离
        if len(x1.shape) == 1:  #判断x1和x2是否都只有一个方向，若只有一个方向则在该方向前插入另一方向。
            x1 = x1[newaxis, :]
        if len(x2.shape) == 1:
            x2 = x2[newaxis, :]
        x1 = x1[..., :, newaxis]
        x2 = x2[..., newaxis, :]
        return self.norm(x1, x2)
    """"
    def __call__(self, *args):  #设置该实例为可调用，设置调用格式
        args = [asarray(x) for x in args]
        if not all([x.shape == y.shape for x in args for y in args]):
            raise ValueError("Array lengths must be equal")
        shp = args[0].shape
        self.xa = asarray([a.flatten() for a in args], dtype=float_)
        r = self._call_norm(self.xa, self.xi)
        return dot(self._function(r), self.nodes).reshape(shp)
      
    def _weight_vec(self):  #返回初始化后的权重向量
        return self.nodes
    """
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
    
    def infill(self, points, method='error'):
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
        self.X = np.copy(initX)
        self.y = np.copy(inity)
        self.n = len(self.X)
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