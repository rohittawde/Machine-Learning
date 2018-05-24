# -*- coding: utf-8 -*-
"""
Created on Sun Mar 04 16:52:44 2018

@author: Rohit Tawde
"""
import sys
import numpy as np
import scipy as sp
train_input=sys.argv[1]
D_ = int(sys.argv[7])
init_flag=int(sys.argv[8])
l=float(sys.argv[9])
epoch=int(sys.argv[6])
val_input=sys.argv[2]
metrics=open(sys.argv[5],'w')
train_labels=open(sys.argv[3],'w')
val_labels=open(sys.argv[4],'w') 

#Reading dataset into a list 
data = sp.genfromtxt(train_input,delimiter=",",unpack=True,dtype=None)

#Separating the data into labels and features
#Number of training examples
N = len(np.transpose(data))
labels=data[0]
features=np.delete(data,0,axis=0)
features=np.transpose(features)
M=len(np.transpose(features))
bias=np.ones(shape=[N,1])
features=np.append(features,bias,axis=1)
classes=np.unique(labels)
K=len(classes)

#Reading dataset into a list 
vdata = sp.genfromtxt(val_input,delimiter=",",unpack=True,dtype=None)

#Separating the data into labels and features
#Number of training examples
vN = len(np.transpose(vdata))
vlabels=vdata[0]
vfeatures=np.delete(vdata,0,axis=0)
vfeatures=np.transpose(vfeatures)
vM=len(np.transpose(vfeatures))
vbias=np.ones(shape=[vN,1])
vfeatures=np.append(vfeatures,vbias,axis=1)

#initialization of alpha
if (init_flag==1):
    #Random initialization of weights
    alpha=np.random.rand(D_,M+1)*0.2-0.1
    for i in range(0,D_):
        alpha[i,-1]=0
else:
    #Zero initialization of weights
    alpha=np.zeros(shape=[D_,M+1])
    for i in range(0,D_):
        alpha[i,-1]=0
        
#initialization of beta
if (init_flag==1):
    #Random initialization of weights
    beta=np.random.rand(K,D_+1)*0.2-0.1
    for i in range(0,K):
        beta[i,-1]=0

else:
    #Zero initialization of weights
    beta=np.zeros(shape=[K,D_+1])
    for i in range(0,K):
        beta[i,-1]=0


for ep in range(0,epoch):
    J_sum=0
    vJ_sum=0
    for N_ in range(0,N):
        
        ok=np.matrix(features[N_,:])
        alpha=np.matrix(alpha)
        #Calculating a's
        a=np.zeros(shape=[D_,1])
        for i in range(0,D_):
            a[i]=((ok)*np.transpose(alpha[i,:]))
            
        #Calculating Z's
        z=np.zeros(shape=[D_,1])
        for i in range(0,D_):
            z[i]=1/(1+np.exp(-a[i]))
            
        #Adding in the bias term
        z=np.append(z,1)
        z=np.matrix(z)
        #Calculating b's
        b=np.zeros(shape=[K,1])
        beta=np.matrix(beta)
        for i in range(0,K):
            b[i]=((z)*np.transpose(beta[i,:]))
            
        #Calculating yhat's
        sum_=0
        for i in range(0,K):
            temp=np.exp(b[i])
            sum_=sum_+temp
            
        yhat=np.zeros(shape=[K,1])
        for i in range(0,K):
            yhat[i]=np.exp(b[i])/sum_
            
        #Converting the label into one hot vector
        yk=np.zeros(shape=[K,1])
        yk[labels[N_]]=1
        
        #Calculating the loss for training example
        J=-(np.matrix(np.transpose(yk))*np.matrix(np.log(yhat)))
        
        gyhat = np.divide(-yk,yhat)
        
#        diaghat=np.zeros(shape=[K,K])
#        for i in range(0,K):
#            diaghat[i,i]=yhat[i]
        diaghat=np.diagflat(yhat)
        yhat=np.matrix(yhat)
        #diaghat=np.diagflat(yhat)
        yhatdim=(np.matrix(yhat)*np.matrix(np.transpose(yhat)))
        
        gb=(np.matrix(np.transpose(gyhat))*np.matrix(diaghat-yhatdim))
        
        z=np.matrix(z)
        
        zn=np.delete(z,-1)
        
        gbeta=(np.transpose(gb)*z)  
        
        betan = np.delete(beta,-1,axis=1)
        
        gz=(np.transpose(betan)*np.transpose(gb))
        
        #z=np.delete(z,-1)
        #gz=np.delete(gz,-1)
        
        #gan=(zn*np.transpose(1-zn))
        #ga=gz*gan
        
        gan=np.multiply((zn),(1-zn))
        ga=np.multiply(gz,np.transpose(gan))
        
        x=np.matrix(features[N_])
        galpha=(ga*x)
        
        alpha=alpha-l*galpha
        beta=beta-l*gbeta
###############################################################################        
    for N_ in range(0,N):
        ok=np.matrix(features[N_,:])
        alpha=np.matrix(alpha)
        #Calculating a's
        a=np.zeros(shape=[D_,1])
        for i in range(0,D_):
            a[i]=((ok)*np.transpose(alpha[i,:]))
            
        #Calculating Z's
        z=np.zeros(shape=[D_,1])
        for i in range(0,D_):
            z[i]=1/(1+np.exp(-a[i]))
            
        #Adding in the bias term
        z=np.append(z,1)
        z=np.matrix(z)
        #Calculating b's
        b=np.zeros(shape=[K,1])
        beta=np.matrix(beta)
        for i in range(0,K):
            b[i]=((z)*np.transpose(beta[i,:]))
            
        #Calculating yhat's
        sum_=0
        for i in range(0,K):
            temp=np.exp(b[i])
            sum_=sum_+temp
            
        yhat=np.zeros(shape=[K,1])
        for i in range(0,K):
            yhat[i]=np.exp(b[i])/sum_
            
        #Converting the label into one hot vector
        yk=np.zeros(shape=[K,1])
        yk[labels[N_]]=1
        
        #Calculating the loss for training example
        J=-(np.matrix(np.transpose(yk))*np.matrix(np.log(yhat)))
        J_sum=J+J_sum
###############################################################################
    for vN_ in range(0,vN):
        vok=np.matrix(vfeatures[vN_,:])
        alpha=np.matrix(alpha)
        #Calculating a's
        a=np.zeros(shape=[D_,1])
        for i in range(0,D_):
            a[i]=((vok)*np.transpose(alpha[i,:]))
            
        #Calculating Z's
        z=np.zeros(shape=[D_,1])
        for i in range(0,D_):
            z[i]=1/(1+np.exp(-a[i]))
            
        #Adding in the bias term
        z=np.append(z,1)
        z=np.matrix(z)
        #Calculating b's
        b=np.zeros(shape=[K,1])
        beta=np.matrix(beta)
        for i in range(0,K):
            b[i]=((z)*np.transpose(beta[i,:]))
            
        #Calculating yhat's
        sum_=0
        for i in range(0,K):
            temp=np.exp(b[i])
            sum_=sum_+temp
            
        yhat=np.zeros(shape=[K,1])
        for i in range(0,K):
            yhat[i]=np.exp(b[i])/sum_
            
        #Converting the label into one hot vector
        yk=np.zeros(shape=[K,1])
        yk[vlabels[vN_]]=1
        
        #Calculating the loss for training example
        vJ=-(np.matrix(np.transpose(yk))*np.matrix(np.log(yhat)))
        vJ_sum=vJ_sum+vJ
        
    print('epoch={} crossentropy(train)= {}\n'.format(ep+1,float(J_sum/N)))
    metrics.write('epoch={} crossentropy(train)= {}\n'.format(ep+1,float(J_sum/N)))
    print('epoch={} crossentorpy(validation)= {}\n'.format(ep+1,float((vJ_sum)/vN)))
    metrics.write('epoch={} crossentorpy(validation)= {}\n'.format(ep+1,float((vJ_sum)/vN)))
    
# Making predictions
e=0.  
for N_ in range(0,N):        
    ok=np.matrix(features[N_,:])
    alpha=np.matrix(alpha)
    #Calculating a's
    a=np.zeros(shape=[D_,1])
    for i in range(0,D_):
        a[i]=((ok)*np.transpose(alpha[i,:]))
        
    #Calculating Z's
    z=np.zeros(shape=[D_,1])
    for i in range(0,D_):
        z[i]=1/(1+np.exp(-a[i]))
        
    #Adding in the bias term
    z=np.append(z,1)
    z=np.matrix(z)
    #Calculating b's
    b=np.zeros(shape=[K,1])
    beta=np.matrix(beta)
    for i in range(0,K):
        b[i]=((z)*np.transpose(beta[i,:]))
        
    #Calculating yhat's
    sum_=0
    for i in range(0,K):
        temp=np.exp(b[i])
        sum_=sum_+temp
        
    yhat=np.zeros(shape=[K,1])
    for i in range(0,K):
        yhat[i]=np.exp(b[i])/sum_
    train_labels.write(str(np.argmax(yhat)))
    train_labels.write('\n')
    #print(np.argmax(yhat))    
    if (np.argmax(yhat))!=labels[N_]:
        e=e+1

ve=0.  
for vN_ in range(0,vN):        
    vok=np.matrix(vfeatures[vN_,:])
    alpha=np.matrix(alpha)
    #Calculating a's
    a=np.zeros(shape=[D_,1])
    for i in range(0,D_):
        a[i]=((vok)*np.transpose(alpha[i,:]))
        
    #Calculating Z's
    z=np.zeros(shape=[D_,1])
    for i in range(0,D_):
        z[i]=1/(1+np.exp(-a[i]))
        
    #Adding in the bias term
    z=np.append(z,1)
    z=np.matrix(z)
    #Calculating b's
    b=np.zeros(shape=[K,1])
    beta=np.matrix(beta)
    for i in range(0,K):
        b[i]=((z)*np.transpose(beta[i,:]))
        
    #Calculating yhat's
    sum_=0
    for i in range(0,K):
        temp=np.exp(b[i])
        sum_=sum_+temp
        
    yhat=np.zeros(shape=[K,1])
    for i in range(0,K):
        yhat[i]=np.exp(b[i])/sum_
    val_labels.write(str(np.argmax(yhat)))
    val_labels.write('\n')
    #print(np.argmax(yhat))    
    if (np.argmax(yhat))!=vlabels[vN_]:
        ve=ve+1

print('error(train)= {}\n'.format(e/N))
metrics.write('error(train)= {}\n'.format(e/N)) 
print('error(validation)= {}'.format(ve/vN))
metrics.write('error(validation)= {}'.format(ve/vN))

metrics.close()
val_labels.close()
train_labels.close()