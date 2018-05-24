# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 14:51:41 2018

@author: Rohit Tawde
"""
import sys
import csv
import math

in_argument = sys.argv[1]
out_argument = sys.argv[2]

with open(in_argument) as csvfile:
    readCSV = csv.reader(csvfile,delimiter=',')
    my_list = list(readCSV)
    
#print(my_list[1][2])
a = len(my_list)
#print(a)

i = 0.;
j = 0.;
n = 1;

while(n<=a-1):
    if my_list[1][-1] == my_list[n][-1]:
        i = i + 1
    else:
        j = j + 1
    n = n + 1
    #print(n)
n=1
#print(i)
#print(j)
x = i / (i + j)
# Finding the Entropy
#import math
#x = (i/(i+j))
#print(x)
H=-(i/(i+j))*math.log(i/(i+j))/math.log(2)-(j/(i+j))*math.log(j/(i+j))/math.log(2)
#print(H)


if(i>=j):
    #print(my_list[1][-1])
    error = j/(i+j)
else:
    error = i/(i+j)
f1 = open(out_argument,'w')
f1.write("entropy: {}\n".format(H))
f1.write("error: {}".format(error)) 
  