# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 11:47:52 2018

@author: Rohit Tawde
"""
import numpy as np
import sys


maze_input = sys.argv[1]
_output = open(sys.argv[2],'w')
actn = sys.argv[3]
        
class Env():
    
    def __init__(self,filename):
        self.t1 = open(filename)
        self.t1=self.t1.readlines()
        for i in range(0,len(self.t1)):
            self.t1[i]=self.t1[i].strip('\n')
            
        self.op=[]
        for i in range(0,len(self.t1)):
            for j in range(0,len(list(self.t1[0]))):
                self.op.append(self.t1[i][j])
        self.t1 = np.reshape(self.op,(len(self.t1),len(self.t1[0])))
        self.a = [len(self.t1)-1,0]
        self.is_terminal=0
        
    def step(self,action):
        #West
        if (action == 0):
            if (self.a[1] == 0):
                self.a[1] = self.a[1]
            else:
                if(self.t1[self.a[0],self.a[1]-1]=='*'):
                    self.a[1] = self.a[1]
                else:
                    self.a[1] = self.a[1]-1
        #North
        if (action == 1):
            if (self.a[0] == 0):
                self.a[0] = self.a[0]
            else:
                if(self.t1[self.a[0]-1,self.a[1]]=='*'):
                    self.a[0] = self.a[0]
                else:
                    self.a[0] = self.a[0]-1
        #East
        if (action == 2):
            if (self.a[1] == len(self.t1[0])-1):    
                self.a[1] = self.a[1]
            else:
                if(self.t1[self.a[0],self.a[1]+1]=='*'):
                    self.a[1] = self.a[1]
                else:   
                    self.a[1] = self.a[1]+1
        #South    
        if (action == 3):
            if (self.a[0] == len(self.t1)-1):
                self.a[0] == self.a[0]
            else:
                if(self.t1[self.a[0]+1,self.a[1]]=='*'):
                    self.a[0] == self.a[0]
                else:
                    self.a[0] = self.a[0]+1
                
        if (self.t1[self.a[0]][self.a[1]]=='G'):
            self.is_terminal = 1
        else:
            self.is_terminal = 0
            
    def reset(self):
        self.a = [len(self.t1)-1,0]


myEnv=Env(maze_input)
#print(myEnv.t1)
#print(myEnv.a,myEnv.is_terminal)

myEnv.step(1)
#print(myEnv.a,myEnv.is_terminal)

myEnv.step(2)
#print(myEnv.a,myEnv.is_terminal)

myEnv.step(3)
#print(myEnv.a,myEnv.is_terminal)

myEnv.reset()
#print(myEnv.a)

ac=open(actn)
ac=ac.readlines()
ac=ac[0].strip("\n")
ac=ac.split()


#action_seq = [0,1,3]

for i in range(0,len(ac)):
    myEnv.step(int(ac[i]))
    print"{} {} -1 {}\n".format(myEnv.a[0],myEnv.a[1],myEnv.is_terminal)
    _output.write("{} {} -1 {}\n".format(myEnv.a[0],myEnv.a[1],myEnv.is_terminal))
