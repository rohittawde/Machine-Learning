# -*- coding: utf-8 -*-
"""
Created on Thu May 03 19:47:14 2018

@author: Rohit Tawde
"""
import numpy as np

D1 = [5.5,5.1,6.4,5.6,6.7]
D2 = [3.1,4.8,3.6,4.7,3.7]

m0 = [5.3,3.5]
m1 = [5.1,4.2]
m2 = [6.0,3.9]

#Interation 1
Eucledian1 = np.zeros(shape=[5,1])
Eucledian2 = np.zeros(shape=[5,1])
Eucledian3 = np.zeros(shape=[5,1])

for i in range(0,5):
    Eucledian1[i] = (m0[0] - D1[i])**2 + (m0[1] - D2[i])**2
    Eucledian2[i] = (m1[0] - D1[i])**2 + (m1[1] - D2[i])**2
    Eucledian3[i] = (m2[0] - D1[i])**2 + (m2[1] - D2[i])**2
    
print(Eucledian1)
print("\n")
print(Eucledian2)
print("\n")
print(Eucledian3)

