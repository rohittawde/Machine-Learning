# -*- coding: utf-8 -*-
"""
Created on Fri May 04 16:33:36 2018

@author: Rohit Tawde
"""
import numpy as np
D = [0.1666,0.25,0.1666,0.1666,0.25]
D_sum = np.sum(D)

err1 = (D[0]+D[1])/D_sum
print(err1)
err2 = (D[4]+D[3])/D_sum
print(err2)
err3 = (D[3]+D[0])/D_sum
print(err3)
err4 = (D[4]+D[1])/D_sum
print(err4)
#err5 = 