# -*- coding: utf-8 -*-
"""
Created on Sat Apr 07 22:58:16 2018

@author: Rohit Tawde
"""

#Hidden Markov Models
import numpy as np
import sys

test_input = sys.argv[1]
index_to_word =  sys.argv[2]
index_to_tag = sys.argv[3]
hmm_prior = sys.argv[4]
hmm_emit = sys.argv[5]
hmm_trans = sys.argv[6]
predict = open(sys.argv[7],'w')

f1 = open(index_to_tag,"r")
t1 = f1.readlines()

mydict1={}
for i in range (0,len(t1)):
    t1[i]=t1[i].strip('\n')
    mydict1[t1[i]]=i

#Index to words
f2 = open(index_to_word,"r")
t2 = f2.readlines()

mydict2={}
for i in range (0,len(t2)):
    t2[i]=t2[i].strip('\n')
    mydict2[t2[i]]=i

K=len(mydict1)
M=len(mydict2)


#Reading the HMM parameters
t5 = np.genfromtxt(hmm_prior , delimiter=' ', dtype= None, unpack = True)
t6 = np.genfromtxt(hmm_trans , delimiter=' ', dtype= None, unpack = True)
t7 = np.genfromtxt(hmm_emit , delimiter=' ', dtype= None, unpack = True)

C1=np.zeros(shape=len(t5))
for i in range(0,len(t5)):
    C1[i] = t5[i]

t6=t6.T
A1=np.zeros(shape=(len(t6),len(t6[0])))   
for i in range(0,len(t6)):
    for j in range(0,len(t6[0])):
        A1[i][j]=t6[i][j]

t7=t7.T
B1=np.zeros(shape=(len(t7),len(t7[0])))
for i in range(0,len(t7)):
    for j in range(0,len(t7[0])):
        B1[i][j]=t7[i][j]
        
A1=np.matrix(A1)
B1=np.matrix(B1)
C1=np.matrix(C1)
C1=C1.T
        


t9 = np.genfromtxt(test_input, delimiter='\n', dtype= None, unpack = True)

for rohit in range(0,len(t9)):
    

    
    t4 = t9[rohit]
        
    x=[]
    for i in range(0,1):    
        eg=t4
        sg=eg.split()
        for j in range(0,len(sg)):    
            kg=sg[j].split("_")
            x.append(kg[0])
            
    alpha1=np.multiply(B1[:,mydict2[x[0]]],C1)
    
    alphao=alpha1
    dilly=[alphao]
    for i in range(1,len(x)):    
        prod=np.transpose(A1)*alphao
        #print(prod)
        #print("\n")
        alphan=np.multiply(B1[:,mydict2[x[i]]],prod)
        alphao=alphan
        #print(alphao)
        dilly.append(alphao)
        
    
    betaT=np.ones(len(B1))
    betan=np.matrix((betaT))
    betan=np.transpose(betan)
    swag=[betan]
    
    
    for i in range(len(x)-1,0,-1):    
        product=np.multiply(B1[:,mydict2[x[i]]],(betan))
        #print(product)
        betao=A1*product
        #print(betao)
        #print("\n")
        betan=betao
        swag.append(betan)   
 
    pred_mat=[]
    
    #Making Predictions 
    for i in range(0,len(dilly)):
        prediction=np.multiply(dilly[i],swag[-1-i])
        #print(dilly[i])
        #print(swag[-i-1])
        pred_mat.append(prediction)
        
    for i in range(0,len(pred_mat)):
        okay=(np.argmax(pred_mat[i]))
        #print "{}_{}".format(x[i],mydict1.keys()[mydict1.values().index(okay)]),
        predict.write("{}_{} ".format(x[i],mydict1.keys()[mydict1.values().index(okay)]),)
        
    if (rohit != len(t9)-1):    
        #print("\n")
        predict.write("\n")
        
predict.close()