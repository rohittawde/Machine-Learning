# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 20:13:48 2018

@author: Rohit Tawde
"""
import numpy as np
import sys
import random


maze_input = sys.argv[1]
value_file = open(sys.argv[2],'w')
q_value_file = open(sys.argv[3],'w')
policy_file = open(sys.argv[4],'w')
num_episodes = int(sys.argv[5])
max_epi_length = int(sys.argv[6])
learning_rate = float(sys.argv[7])
discount_factor = float(sys.argv[8])
eps = float(sys.argv[9])
        
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

myEnv.reset()
#print(myEnv.a)

Q = np.zeros(shape=[len(myEnv.t1),len(myEnv.t1[0]),4])

alpha = learning_rate
gamma = discount_factor
epsilon = eps
max_episodes = num_episodes
max_episode_length = max_epi_length
episode_length = 0
episode = 0

#start state
ci = myEnv.a[0]
cj = myEnv.a[1]
while (episode<max_episodes):
        
    decision = random.uniform(0,1)
    if(decision>epsilon):
        action_taken = np.argmax(Q[ci,cj,:])
    else:
        action_taken = random.choice([0,1,2,3])
    ci = myEnv.a[0]
    cj = myEnv.a[1]
    #Taking a step in the direction
    myEnv.step(action_taken)
    episode_length = episode_length + 1
    ni = myEnv.a[0]
    nj = myEnv.a[1]
    #print(action_taken)
    #print(ni,nj)
    Q[ci,cj,action_taken] = (1-alpha)*Q[ci,cj,action_taken]+alpha*(-1+gamma*(max(Q[ni,nj,:])))
    if (episode_length == max_episode_length):
        myEnv.reset()
        episode = episode + 1
        episode_length = 0
        ci = myEnv.a[0]
        cj = myEnv.a[1]
        
    
    if (myEnv.is_terminal == 0):
        
        ci = ni
        cj = nj
        
        
    if (myEnv.is_terminal == 1):
        #Q[ci,cj,action_taken] = (1-alpha)*Q[ci,cj,action_taken]+alpha*(-1)
        myEnv.reset()
        ci = myEnv.a[0]
        cj = myEnv.a[1]
        episode = episode + 1
        episode_length = 0
    #print(Q[ci,cj,:])

#Printing the q-value file
for i in range(0,len(myEnv.t1)):
    for j in range(0,len(myEnv.t1[0])):
        for k in range(0,4):    
            if (myEnv.t1[i,j] !='*'):
                print'{} {} {} {}'.format(i,j,k,Q[i,j,k])
                q_value_file.write('{} {} {} {}\n'.format(i,j,k,float(Q[i,j,k])))

#Printing the policy file                
for i in range(0,len(myEnv.t1)):
    for j in range(0,len(myEnv.t1[0])):
        if (myEnv.t1[i,j] != '*'):
            print'{} {} {}'.format(i,j,np.argmax(Q[i,j,:]))
            policy_file.write('{} {} {}\n'.format(i,j,float(np.argmax(Q[i,j,:]))))

#Printing the value file                
for i in range(0,len(myEnv.t1)):
    for j in range(0,len(myEnv.t1[0])):
        if (myEnv.t1[i,j] != '*'):
            print'{} {} {}'.format(i,j,np.max(Q[i,j,:]))
            value_file.write('{} {} {}\n'.format(i,j,float(np.max(Q[i,j,:]))))

q_value_file.close()
policy_file.close()
value_file.close()