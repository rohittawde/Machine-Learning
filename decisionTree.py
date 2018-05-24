# -*- coding: utf-8 -*-
"""
Created on Thu Feb 01 22:35:07 2018

@author: Rohit Tawde
"""
# Importing modules

import csv
import math
import numpy as np
import pprint
import sys
globe=[]
mapping=[]
######################################################################
#Defining the system inputs
intrain = sys.argv[1]
intest = sys.argv[2]
depth = int(sys.argv[3])
trainout = sys.argv[4]
testout = sys.argv[5]
metrics = sys.argv[6]
##########################################################################
# Reading the CSV file into a list

with open(intrain) as csvfile:
    readCSV = csv.reader(csvfile,delimiter=',')
    my_list = list(readCSV)
#print(my_list)
#########################################################################
#Defining for label
label = [0,0]
label[1] = my_list[1][-1]
count = 2
while(count<len(my_list)):
    if my_list[count][-1] != my_list[1][-1]:
        label[0] = my_list[count][-1]
        break
    count = count + 1
###########################################################################
# Converting data into binary

def information(col):
    n = 1
    j = 0.
    i = 0.
    pl=[]
    ll=[]
    while(n<=len(my_list)-1):
        if my_list[n][col] == my_list[1][col]:
            i=i+1
            pl.append(1)
            ll.append(my_list[n][col])
        else:
            j=j+1
            pl.append(0)
            ll.append(my_list[n][col])
        n=n+1
    return pl

############################################################################
# Converting the initial data

a = len(my_list)
b = len(my_list[0])
k = 0
att = np.zeros(shape = [b,a-1])
while(k < b):
    att[k,:]=information(k)
    k = k + 1

#############################################################################
# Defining the entropy function

def nentropy(ml1):
    a = len(np.transpose(ml1))
    b = len(ml1[0])

    i = 0.
    j = 0.
    n = 0

    while(n<=a-1):
        if ml1[-1][0] == ml1[-1][n]:
            i = i + 1
        else:
            j = j + 1
        n = n + 1
    if (i == 0 or j == 0):
        H = 0
    else:
        H = -(i/(i+j))*math.log(i/(i+j))/math.log(2)-(j/(i+j))*math.log(j/(i+j))/math.log(2)
    return H
###############################################################################
# Function to calculate mutual information

def ngain(l1,l2,l3):
    n = 0
    i = 0.
    j = 0.
    while(n<=len(l1)-1):
        if (l1[n] == 1 and l2[n] == 1):
            i = i + 1
        elif (l1[n] == 1 and l2[n] == 0):
            j = j + 1
        else:
            k = 0
        n = n + 1

    if (i == 0 or j == 0):
        cont1 = 0
    else:
        cont1 = -1.*(i+j)/len(l1)*(i/(i+j)*math.log(i/(i+j))/math.log(2)+(j/(i+j))*math.log(j/(i+j))/math.log(2))


    n = 0
    i = 0.
    j = 0.
    while(n<=(len(l1)-1)):
        if (l1[n] == 0 and l2[n] == 0):
            i = i + 1
        elif (l1[n] == 0 and l2[n] == 1):
            j = j + 1
        else:
            k = 0
        n = n + 1

    if (i == 0 or j == 0):
        cont2 = 0
    else:
        cont2 = -1.*(i+j)/len(l1)*(i/(i+j)*math.log(i/(i+j))/math.log(2)+(j/(i+j))*math.log(j/(i+j))/math.log(2))

    info_gain = nentropy(l3)-cont1-cont2
    return (info_gain)
##############################################################################
# Split the unclassified tree

def split1(index_max,mat):
    index1 = []
    index2 = []

    count = 0
    while (count<len(np.transpose(mat))):
        if (mat[index_max,count] == 1):
            index1.append(count)
        else:
            index2.append(count)
        count = count + 1

    tr1 = np.zeros(shape = [len(mat),len(index1)])
    tr2 = np.zeros(shape = [len(mat),len(index2)])

    for c1 in range(0,len(mat)):
        for c2 in range(0,len(index1)):
            tr1[c1,c2] = mat[c1,index1[c2]]

    for c1 in range(0,len(mat)):
        for c2 in range(0,len(index2)):
            tr2[c1,c2] = mat[c1,index2[c2]]
    return tr1,tr2

##############################################################################
# Calculating maximum information gain

def max_info(mat):
    count = 0
    info_gain = np.zeros(shape = [len(mat)-1])
    while(count<len(mat)-1):
        info_gain[count] = ngain(mat[count,:],mat[-1,:],mat)
        count = count + 1
    #print(info_gain)
    return [np.argmax(info_gain),max(info_gain)]
##############################################################################
#Majority vote

def majority_vote(mat):
    n = 0
    i = 0
    j = 0
    l1 = []
    l2 = []
    while (n<len(np.transpose(mat))):
        if (mat[-1,0] == mat[-1,n]):
            i = i + 1
            l1.append(mat[-1,n])
        else:
            j = j + 1
            l2.append(mat[-1,n])
        n = n + 1
    #print(l1)
    #print(l2)
    if (i>=j):
        decision = l1[0]
        #print(label[int(decision)])
        return(label[int(decision)])
        #print("\n")
    elif(j>i):
        decision = l2[0]
        #print(label[int(decision)])
        return(label[int(decision)])
        #print("\n")
    else:
        return (label[decision])

#############################################################################
#Counting
def counting(mat):
    c1 = 0
    c2 = 0
    for i in range (0,len(np.transpose(mat))):
        if (mat[-1,i] == 0):
            c1 = c1 + 1
        else:
            c2 = c2 + 1
    counter = [c1,c2]
    print(counter[0],label[0],counter[1],label[1])
    return counter
#############################################################################
#Counting
def counting_mum(mat):
    c1 = 0
    c2 = 0
    for i in range (0,len(np.transpose(mat))):
        if (mat[-1,i] == 0):
            c1 = c1 + 1
        else:
            c2 = c2 + 1
    counter = [c1,c2]
    #print(counter[0],label[0],counter[1],label[1])
    return [c1,c2]
##########################################################################
#counting attributes
def count_att(mat):
    c1 = 0
    c2 = 0
    attc = []
    for j in range(0,len(mat)-1):
        for i in range(0,len(np.transpose(mat))):
            if (mat[j,i] == 0):
                c1 = c1 + 1
            else:
                c2 = c2 + 1
        if (c1>=c2):
            attc.append(0)
        else:
            attc.append(1)
    return attc
#############################################################################
#Recursing the function

def recurseprint(mat,maxdepth):

    while(maxdepth>0):
        #counting(mat)
        c = counting(mat)
        print("\n")
        #print(mat)
        #print("\n")
        #for i in range(0,maxdepth):
         #   print("|")
        k = max_info(mat)[0]
        print(my_list[0][k])
        [y,n] = split1(k,mat)
        #counting(mat)
        #if (c[0] == 0 or c[1] == 0):
            #decision = majority_vote(mat)
            #counting(mat)
            #print(my_list[0][k])
         #   return
        if (len(np.transpose(n))>0 ):
            pprint.pprint(my_list[1][k])
            nbranch = recurseprint(y,maxdepth-1)

        if (len(np.transpose(y))>0 ):
            pprint.pprint(my_list[1][k])
            ybranch = recurseprint(n,maxdepth-1)

        return

recurseprint(att,depth)
#########################################################################
#recurseprint(att,2)
def recurse(mat,maxdepth):
    global globe
    while(maxdepth>0):
        c = counting_mum(mat)
        #print(mat)
        #print(c)
        #decision = majority_vote(mat)
        #print(np.shape(mat))
        k = max_info(mat)[0]
        #print(k)
        #print("\n")
        decision = majority_vote(mat)
        kick=[k,c,maxdepth,decision,count_att(mat)]
        globe.append(kick)
        [y,n] = split1(k,mat)
        counting_mum(mat)
        if (c[0] == 0 or c[1] == 0):
            decision = majority_vote(mat)
            #print("Leaf \n")
            return[c]
        if (len(np.transpose(y))>0):
            ybranch = recurse(y,maxdepth-1)
        if (len(np.transpose(n))>0):
            nbranch = recurse(n,maxdepth-1)
        return[c]

recurse(att,len(att))
##########################################################################

with open(intest) as csvfile:
    readCSV = csv.reader(csvfile,delimiter=',')
    my_list1 = list(readCSV)
##########################################################################
# Converting data into binary

def information_two(col):
    n = 1
    j = 0.
    i = 0.
    pl1=[]
    ll=[]
    while(n<=len(my_list1)-1):
        if my_list[1][col] == my_list1[n][col]:
            i=i+1
            pl1.append(1)
            ll.append(my_list1[n][col])
        else:
            j=j+1
            pl1.append(0)
            ll.append(my_list1[n][col])
        n=n+1
    return pl1

############################################################################
# Creating the test data
a = len(my_list1)
b = len(my_list1[0])
k = 0
att1 = np.zeros(shape = [b,a-1])
while(k < b):
    att1[k,:]=information_two(k)
    k = k + 1
atest = np.zeros(shape = [b-1,a-1])

for i in range (0,b-1):
    for j in range(0,a-1):
        atest[i,j] = att1[i,j]
#########################################################################
# Evaluating the test data
val = []
valstr=''
f = []
f1 = open(trainout,'w')
f2 = open(testout,'w')
for j in range(0,np.transpose(len(np.transpose(atest)))):
    k=atest[0][j]
    l=atest[1][j]
    f.append([k,l])
    for i in range(0,len((globe))):
        
        if f[-1] == globe[i][4]:
            val.append(globe[i][3])
            valstr=valstr+str(globe[i][3])+'\n'
            f2.write("{}\n".format(val[i]))
            break
        else:
            val.append(globe[0][3])
            valstr=valstr+str(globe[0][3])+'\n'
            f2.write("{}\n".format(val[i]))
            break
#f1.write("valstr")

m = []
for j in range(0,len(my_list)-1):
    m.append([att[0,j],att[1,j]])
    for i in range(0,len((globe))):
        if m[-1] == globe[i][4]:

            val.append(globe[i][3])
            valstr=valstr+str(globe[i][3])+'\n'
            f1.write("{}\n".format(val[i]))
            break
        else:
            val.append(globe[0][3])
            valstr=valstr+str(globe[0][3])+'\n'
            f1.write("{}\n".format(val[i]))
            break
#f2.write(valstr)
cal = 0
for i in range(1,len(my_list1)-1):
    if my_list1[i][-1] == val[i]:
        cal = cal + 1
f3 = open(metrics,'w')
#print(cal)
error_test = float(cal)/float(len(np.transpose(att1)))

calt = 0
for i in range(1,len(my_list)-1):
    if my_list[i][-1] == val[i]:
        calt =calt + 1

#print(calt)
error_train = float(calt)/float(len(np.transpose(att)))

f3.write("error(train): {}\n".format(error_train))
f3.write("error(test): {}".format(error_test))


#predict(0,1)
