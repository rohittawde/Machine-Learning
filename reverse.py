# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 19:46:46 2018

@author: Rohit Tawde
"""
import sys
example=sys.argv[1]
output=sys.argv[2]
f = open(example)
c = f.read()
forward = c.split('\n')
n = len(forward)
#print(len(forward))
i = n-1
f1 = open(output,'w')
while i!=-1:
    if i != len(forward)-1:
        #print(i)
        f1.write(forward[i])
        f1.write('\n')
        print(forward[i].strip())
    i = i-1
f1.close()
 