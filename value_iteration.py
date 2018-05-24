# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 10:20:46 2018

@author: Rohit Tawde
"""

import numpy as np
import sys

_input = open(sys.argv[1])
valueoutput=open(sys.argv[2],'w')
qoutput=open(sys.argv[3],'w')
policyoutput=open(sys.argv[4],'w')

t1=_input.readlines()
for i in range (0,len(t1)):
    t1[i]=t1[i].strip('\n')
  
op=[]
for i in range(0,len(t1)):
    for j in range(0,len(list(t1[0]))):
        op.append(t1[i][j])
        
s=np.reshape(op,(len(t1),len(t1[0])))


a=[0,1,2,3]

V=np.zeros(shape=[len(t1),len(t1[0])])
Vmax=np.zeros(shape=[len(t1),len(t1[0])])
Q=np.zeros(shape=[s.size,len(a)])
R=-1
gamma=float(sys.argv[6])
Q0=Q1=Q2=Q3=0

epoch=int(sys.argv[5])

for n in range(0,epoch):
        
    for i in range (0,len(t1)):
        for j in range (0,len(list(t1[0]))):
            if (s[i][j]=='*'):
                continue
            elif (s[i][j]=='G'):
                Q0=Q1=Q2=Q3=0
                if (n==epoch-1):
                    q=[Q0,Q1,Q2,Q3]
                    #print'{0} {1} 0 {2}\n{0} {1} 1 {3}\n{0} {1} 2 {4}\n{0} {1} 3 {5}\n'.format(i,j,Q0,Q1,Q2,Q3)
                    #qoutput.write('{0} {1} 0 {2}\n{0} {1} 1 {3}\n{0} {1} 2 {4}\n{0} {1} 3 {5}\n'.format(i,j,Q0,Q1,Q2,Q3))
                    #print('\n')
                    #print'{} {} {}'.format(i,j,np.argmax(q))
                    policyoutput.write('{} {} {}\n'.format(i,j,float(np.argmax(q))))
                    #print('\n')
                    #print'{} {} {}'.format(i,j,max(q))
                    valueoutput.write('{} {} {}\n'.format(i,j,float(max(q))))
                    #print('\n')
                    
            else:
                for k in range (len(a)):
                    if k==0:
                        if j==0:
                            Q0=R+gamma*V[i][j]
                        else:
                            if (s[i][j-1]=='*'):
                                Q0=R+gamma*V[i][j]
                            else:
                                Q0=R+gamma*V[i][j-1]
                    elif k==1:
                        if i==0:
                            Q1=R+gamma*V[i][j]
                        else:
                            if (s[i-1][j]=='*'):
                                Q1=R+gamma*V[i][j]
                            else:
                                Q1=R+gamma*V[i-1][j]
                    elif k==2:
                        if j==len(t1[0])-1:
                            Q2=R+gamma*V[i][j]
                        else:
                            if (s[i][j+1]=='*'):
                                Q2=R+gamma*V[i][j]
                            else:
                                Q2=R+gamma*V[i][j+1]
                    else:
                        if i==len(t1)-1:
                            Q3=R+gamma*V[i][j]
                        else:
                            if (s[i+1][j]=='*'):
                                Q3=R+gamma*V[i][j]
                            else:
                                Q3=R+gamma*V[i+1][j]
                Vmax[i][j]=max(Q0,Q1,Q2,Q3)
                if (n==epoch-1):  
                    q=[Q0,Q1,Q2,Q3]
                    #print('\n')
                    #print'{0} {1} 0 {2}\n{0} {1} 1 {3}\n{0} {1} 2 {4}\n{0} {1} 3 {5}\n'.format(i,j,Q0,Q1,Q2,Q3)
                    #qoutput.write('{0} {1} 0 {2}\n{0} {1} 1 {3}\n{0} {1} 2 {4}\n{0} {1} 3 {5}\n'.format(i,j,Q0,Q1,Q2,Q3))
                    #print('\n')
                    #print'{} {} {}'.format(i,j,np.argmax(q))
                    policyoutput.write('{} {} {}\n'.format(i,j,float(np.argmax(q))))
                    #print('\n')
                    #print'{} {} {}'.format(i,j,max(q))
                    valueoutput.write('{} {} {}\n'.format(i,j,float(max(q))))
                    #print('\n')
    V=Vmax
  
for i in range (0,len(t1)):
        for j in range (0,len(list(t1[0]))):
            if (s[i][j]=='*'):
                continue
            elif (s[i][j]=='G'):
                Q0=Q1=Q2=Q3=0
                
                q=[Q0,Q1,Q2,Q3]
                #print'{0} {1} 0 {2}\n{0} {1} 1 {3}\n{0} {1} 2 {4}\n{0} {1} 3 {5}\n'.format(i,j,Q0,Q1,Q2,Q3)
                qoutput.write('{0} {1} 0 {2}\n{0} {1} 1 {3}\n{0} {1} 2 {4}\n{0} {1} 3 {5}\n'.format(i,j,Q0,Q1,Q2,Q3))
                #print('\n')
                #print'{} {} {}'.format(i,j,np.argmax(q))
                #policyoutput.write('{} {} {}\n'.format(i,j,np.argmax(q)))
                #print('\n')
                #print'{} {} {}'.format(i,j,max(q))
                #valueoutput.write('{} {} {} \n'.format(i,j,max(q)))
                #print('\n')
                    
            else:
                for k in range (len(a)):
                    if k==0:
                        if j==0:
                            Q0=R+gamma*V[i][j]
                        else:
                            if (s[i][j-1]=='*'):
                                Q0=R+gamma*V[i][j]
                            else:
                                Q0=R+gamma*V[i][j-1]
                    elif k==1:
                        if i==0:
                            Q1=R+gamma*V[i][j]
                        else:
                            if (s[i-1][j]=='*'):
                                Q1=R+gamma*V[i][j]
                            else:
                                Q1=R+gamma*V[i-1][j]
                    elif k==2:
                        if j==len(t1[0])-1:
                            Q2=R+gamma*V[i][j]
                        else:
                            if (s[i][j+1]=='*'):
                                Q2=R+gamma*V[i][j]
                            else:
                                Q2=R+gamma*V[i][j+1]
                    else:
                        if i==len(t1)-1:
                            Q3=R+gamma*V[i][j]
                        else:
                            if (s[i+1][j]=='*'):
                                Q3=R+gamma*V[i][j]
                            else:
                                Q3=R+gamma*V[i+1][j]
                Vmax[i][j]=max(Q0,Q1,Q2,Q3)
                  
                q=[Q0,Q1,Q2,Q3]
                #print('\n')
                #print'{0} {1} 0 {2}\n{0} {1} 1 {3}\n{0} {1} 2 {4}\n{0} {1} 3 {5}\n'.format(i,j,Q0,Q1,Q2,Q3)
                qoutput.write('{0} {1} 0 {2}\n{0} {1} 1 {3}\n{0} {1} 2 {4}\n{0} {1} 3 {5}\n'.format(i,j,Q0,Q1,Q2,Q3))
                #print('\n')
                #print'{} {} {}'.format(i,j,np.argmax(q))
                #policyoutput.write('{} {} {} \n'.format(i,j,np.argmax(q)))
                #print('\n')
                #print'{} {} {}'.format(i,j,max(q))
                #valueoutput.write('{} {} {} \n'.format(i,j,max(q)))

policyoutput.close()
valueoutput.close()
qoutput.close()