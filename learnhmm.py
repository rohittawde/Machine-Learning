# -*- coding: utf-8 -*-
"""
Created on Sat Apr 07 22:58:16 2018

@author: Rohit Tawde
"""
#Hidden Markov Models
import sys
import numpy as np

tag_input=sys.argv[3]
word_input=sys.argv[2]
train_input=sys.argv[1]
prior=open(sys.argv[4],"w")
emit=open(sys.argv[5],'w')
trans=open(sys.argv[6],'w')

#Index to tags
f1 = open(tag_input,"r")
t1 = f1.readlines()

mydict1={}
for i in range (0,len(t1)):
    t1[i]=t1[i].strip('\n')
    mydict1[t1[i]]=i

#Index to words
f2 = open(word_input,"r")
t2 = f2.readlines()

mydict2={}
for i in range (0,len(t2)):
    t2[i]=t2[i].strip('\n')
    mydict2[t2[i]]=i

K=len(mydict1)
M=len(mydict2)

#Initializing the HMM indices
C=np.ones(shape=[K,1])
A=np.ones(shape=[K,K])
B=np.ones(shape=[K,M])

f3 = open(train_input)
t3 = f3.readlines()

for i in range(0,len(t3)):
    t3[i]=t3[i].strip('\n')
    
#Finding the prior vector C
for i in range(0,len(t3)):    
    eg=t3[i]
    sg=eg.split()
    kg=sg[0].split("_")    
    C[mydict1[kg[1]],0]=C[mydict1[kg[1]],0]+1

#Finding the emission matrix B   
for i in range(0,len(t3)):
    eg=t3[i]
    sg=eg.split()
    for j in range(0,len(sg)):
        kg=sg[j].split("_")
        B[mydict1[kg[1]],mydict2[kg[0]]]=B[mydict1[kg[1]],mydict2[kg[0]]]+1
        
#Finding the transmission matrix A
for i in range(0,len(t3)):
    eg=t3[i]
    sg=eg.split()    
    for j in range(0,len(sg)-1):
        kg=sg[j].split("_")
        kg1=sg[j+1].split("_")
        A[mydict1[kg[1]],mydict1[kg1[1]]]=A[mydict1[kg[1]],mydict1[kg1[1]]]+1
        
#Normalizing the probabilities
A1=np.matrix(A)
for i in range(0,len(A)):
    for j in range(0,len(np.transpose(A))):
        A1[i,j]=A[i,j]/np.sum(A[i,:])
        
B1=np.matrix(B)
for i in range(0,len(B)):
    for j in range(0,len(np.transpose(B))):
        B1[i,j]=B[i,j]/np.sum(B[i,:])
        
C1=np.matrix(C)
for i in range(0,len(C)):
    C1[i]=C[i]/np.sum(C)
    
np.savetxt(prior,C1,fmt='%.16e')
np.savetxt(trans,A1,fmt='%.16e')
np.savetxt(emit,B1,fmt='%.16e')

prior.close()
trans.close()
emit.close()
    
#C1=np.array(C1)
#A1=np.array(A1)
#B1=np.array(B1)
#
print("Priors")
print(C1)
#for i in range(0,len(C1)):
#    prior.write("{0:1.19f}\n".format(C1[i][0]))
#            
##prior.write("{}".format(C1))
#print("\n")
#
print("Transition")
print(A1)
##trans.write(A1)
#for i in range(0,len(A1)):
#    for j in range(0,len(np.transpose(A1))):
#        trans.write("{}\t".format(A1[i][j]))
#    trans.write("\n".format())
#
#print("\n")
#
print("Emission")
print(B1)
#for i in range(0,len(B1)):
#    for j in range(0,len(np.transpose(B1))):
#        emit.write("{}".format(B1[i][j]))
#        
#print("\n")