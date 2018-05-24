import sys
trainip = sys.argv[1]
valip = sys.argv[2]
testip = sys.argv[3]
epoch = int(sys.argv[7])
model = int(sys.argv[8])

trainop = open(sys.argv[4],'w')
testop = open(sys.argv[5],'w')
metricsop = open(sys.argv[6],'w')

if model == 2:
    
    #############################################################################
    #Importing modules
    import numpy as np
    import scipy as sp
    #Extracting data
    data = []
    with open(trainip,'r') as tsv:
        newdata = [line.strip().split('\t') for line in tsv]
    
    for i in range (len(newdata)):
        if newdata[i] ==['']:
            newdata[i] = ['EOS','Tawde']
        else:
            newdata[i] = newdata[i]
    
    data = np.transpose(newdata)
    
    ind_space = []
    
    features = data[0]
    labels = data[1]
    
    for i in range (len(features)):
        if(features[i]=='EOS'):
            ind_space.append(i)
        else:
            continue
        
    features=list(features)
    features.insert(0,'BOS')
    features.append('EOS')
    features=np.array(features)
    
    for i in range(len(ind_space)):
        features = np.insert(features,ind_space[i]+2+i,'BOS')
    
    features1 = features 
    unique_features = np.unique(features1)
    
    sparsity = []
    for i in range(1,len(features1)-1):
        sappend = []
        if features1[i] != 'EOS':
            if features1[i] !='BOS':
                for j in range(0,len(unique_features)):
                    if features1[i-1]==unique_features[j]:
                        f =(j)
                    elif features1[i]==unique_features[j]:
                        s=(j+len(unique_features))
                    elif features1[i+1]==unique_features[j]:
                        t=(j+2*len(unique_features))
                sappend = [f,s,t]
            if len(sappend) != 0:
                sparsity.append(sappend)
    labels1=[]
    for i in range(len(labels)):
        if labels[i]=='Tawde':
            continue
        else:
            labels1.append(labels[i])
    sig_label = np.unique(labels1)
    #############################################################################
    #Validation
    with open(valip,'r') as tsv:
        newdatax = [line.strip().split('\t') for line in tsv]
    
    for i in range (len(newdatax)):
        if newdatax[i] ==['']:
            newdatax[i] = ['EOS','Tawde']
        else:
            newdatax[i] = newdatax[i]
    
    datax = np.transpose(newdatax)
    
    ind_spacex = []
    
    featuresx = datax[0]
    labelsx = datax[1]
    
    for i in range (len(featuresx)):
        if(featuresx[i]=='EOS'):
            ind_spacex.append(i)
        else:
            continue
        
    featuresx=list(featuresx)
    featuresx.insert(0,'BOS')
    featuresx.append('EOS')
    featuresx=np.array(featuresx)
    
    for i in range(len(ind_spacex)):
        featuresx = np.insert(featuresx,ind_spacex[i]+2+i,'BOS')
    
    features1x = featuresx 
    unique_featuresx = np.unique(features1x)
    
    sparsityx = []
    for i in range(1,len(features1x)-1):
        sappendx = []
        if features1x[i] != 'EOS':
            if features1x[i] !='BOS':
                for j in range(0,len(unique_features)):
                    if features1x[i-1]==unique_features[j]:
                        fx=(j)
                    elif features1x[i]==unique_features[j]:
                        sx=(j+len(unique_features))
                    elif features1x[i+1]==unique_features[j]:
                        tx=(j+2*len(unique_features))
                sappendx = [fx,sx,tx]
            if len(sappendx) != 0:
                sparsityx.append(sappendx)
    labels1x=[]
    for i in range(len(labelsx)):
        if labelsx[i]=='Tawde':
            continue
        else:
            labels1x.append(labelsx[i])
    
    #############################################################################
    #Indicator function
    
    def I(i,k):
        if labels1[i] == sig_label[k]:
            return 1
        else:
            return 0
    #############################################################################
    #Indicator validation
            
    def Ix(i,k):
        if labels1x[i] == sig_label[k]:
            return 1
        else:
            return 0
    #############################################################################        
    K = len(sig_label)
    N = len(sparsity)
    M = len(unique_features)*3
    Nx= len(sparsityx)
    theta = np.zeros(shape=[K,M+1])
    grad = np.zeros(shape=[K,M+1])
    thetaC = np.zeros(shape=[K,M+1])
    
 
    res=0
    for ep in range(0,epoch):           
        for i in range(0,N):
            sum_ = 0             
            for j in range(0,K):
                sum_ = sum_ + np.exp(theta[j,sparsity[i][0]] + theta[j,sparsity[i][1]] + theta[j,sparsity[i][2]] + theta[j,-1])
            #print sum_   
            grad = np.zeros(shape=[K,M+1])
            for k in range(0,K):
                #grad = np.zeros(shape=[K,M+1])
                #num = (np.exp(theta[k,sparsity[i][0]] + theta[k,sparsity[i][1]] +theta[k,sparsity[i][2]] + theta[k,-1]))
                #print num
                ans=-(I(i,k) - (np.exp(theta[k,sparsity[i][0]] + theta[k,sparsity[i][1]] +theta[k,sparsity[i][2]] + theta[k,-1]))/sum_)
                grad[k,sparsity[i][0]] = ans
                grad[k,sparsity[i][1]] = ans
                grad[k,sparsity[i][2]] = ans
                grad[k,-1] = ans
            
            #print ans   
            for k in range(0,K):
                
                theta[k,sparsity[i][0]] = theta[k,sparsity[i][0]] - 0.5*grad[k,sparsity[i][0]]
                theta[k,sparsity[i][1]] = theta[k,sparsity[i][1]] - 0.5*grad[k,sparsity[i][1]]
                theta[k,sparsity[i][2]] = theta[k,sparsity[i][2]] - 0.5*grad[k,sparsity[i][2]]
                theta[k,-1] = theta[k,-1] - 0.5*grad[k,-1]
            
    
        #############################################################################
        # Calculatin NLL
        
        nl_ =  0
        
        for i in range(0,N):
            sum_ = 0             
            for j in range(0,K):
                sum_ = sum_ + np.exp(theta[j,sparsity[i][0]] + theta[j,sparsity[i][1]] + theta[j,sparsity[i][2]] + theta[j,-1])
            for k in range(0,K):
                nl_1 = I(i,k)*np.log((np.exp(theta[k,sparsity[i][0]] + theta[k,sparsity[i][1]] +theta[k,sparsity[i][2]] + theta[k,-1]))/sum_)
                nl_ = nl_ + nl_1
        metricsop.write('epoch={} likelihood(train): {}\n'.format(ep+1,-nl_/N))
        print('epoch={} likelihood(train): {}'.format(ep+1,-nl_/N))
        nlx_ =  0
        
        for i in range(0,Nx):
            sumx_ = 0             
            for j in range(0,K):
                sumx_ = sumx_ + np.exp(theta[j,sparsityx[i][0]] + theta[j,sparsityx[i][1]] + theta[j,sparsityx[i][2]] + theta[j,-1])
            for k in range(0,K):
                nl_1x = Ix(i,k)*np.log((np.exp(theta[k,sparsityx[i][0]] + theta[k,sparsityx[i][1]] +theta[k,sparsityx[i][2]] + theta[k,-1]))/sumx_)
                nlx_ = nlx_ + nl_1x
        metricsop.write('epoch={} likelihood(validation): {}\n'.format(ep+1,-nlx_/Nx))        
        print('epoch={} likelihood(validation): {}'.format(ep,-nlx_/Nx))
        
    
    #Importing modules
    import numpy as np
    import scipy as sp
    #Extracting data
    data = []
    with open(testip,'r') as tsv:
        newdata = [line.strip().split('\t') for line in tsv]
    
    for i in range (len(newdata)):
        if newdata[i] ==['']:
            newdata[i] = ['EOS','Tawde']
        else:
            newdata[i] = newdata[i]
    
    data = np.transpose(newdata)
    
    ind_space = []
    
    features = data[0]
    labels = data[1]
    
    for i in range (len(features)):
        if(features[i]=='EOS'):
            ind_space.append(i)
        else:
            continue
        
    features=list(features)
    features.insert(0,'BOS')
    features.append('EOS')
    features=np.array(features)
    
    for i in range(len(ind_space)):
        features = np.insert(features,ind_space[i]+2+i,'BOS')
    
    features1 = features 
    
    #unique_features = np.unique(features1)
    
    sparsity = []
    for i in range(1,len(features1)-1):
        sappend = []
        if features1[i] != 'EOS':
            if features1[i] !='BOS':
                for j in range(0,len(unique_features)):
                    if features1[i-1]==unique_features[j]:
                        f =(j)
                    elif features1[i]==unique_features[j]:
                        s=(j+len(unique_features))
                    elif features1[i+1]==unique_features[j]:
                        t=(j+2*len(unique_features))
                sappend = [f,s,t]
            if len(sappend) != 0:
                sparsity.append(sappend)
    labels1=[]
    for i in range(len(labels)):
        if labels[i]=='Tawde':
            continue
        else:
            labels1.append(labels[i])
    #sig_label = np.unique(labels1)
    #K = len(sig_label)
    N = len(sparsity)
    #M = len(unique_features)*3
    pred_mat = np.zeros(shape=[N,K])
    
    for i in range(0,N):    
        for k in range(0,K):
            a = theta[k,sparsity[i][0]] + theta[k,sparsity[i][1]] + theta[k,sparsity[i][2]] + theta[k,-1]
            pred_mat[i,k] = np.exp(a)
    testprint = []
    c = 0.        
    for i in range(len(pred_mat)):
        sol = np.argmax(pred_mat[i,:])
        #print(sig_label[sol])
        testprint.append(labels1[i])
        if sig_label[sol] != labels1[i]:
            c = c+1
    
    for i in range(len(ind_space)):
        testprint.insert(ind_space[i],'\n')
        
    for i in range(len(testprint)):
        if (testprint[i] == '\n'):
            testop.write('{}'.format(testprint[i]))
        else:
            testop.write('{}\n'.format(testprint[i]))
        
    p4 = (c/len(labels1))   
         
    ########################################################################## 
    #Importing modules
    import numpy as np
    import scipy as sp
    #Extracting data
    data = []
    with open(trainip,'r') as tsv:
        newdata = [line.strip().split('\t') for line in tsv]
    
    for i in range (len(newdata)):
        if newdata[i] ==['']:
            newdata[i] = ['EOS','Tawde']
        else:
            newdata[i] = newdata[i]
    
    data = np.transpose(newdata)
    
    ind_space = []
    
    features = data[0]
    labels = data[1]
    
    for i in range (len(features)):
        if(features[i]=='EOS'):
            ind_space.append(i)
        else:
            continue
        
    features=list(features)
    features.insert(0,'BOS')
    features.append('EOS')
    features=np.array(features)
    
    for i in range(len(ind_space)):
        features = np.insert(features,ind_space[i]+2+i,'BOS')
    
    features1 = features 
    
    #unique_features = np.unique(features1)
    
    sparsity = []
    for i in range(1,len(features1)-1):
        sappend = []
        if features1[i] != 'EOS':
            if features1[i] !='BOS':
                for j in range(0,len(unique_features)):
                    if features1[i-1]==unique_features[j]:
                        f =(j)
                    elif features1[i]==unique_features[j]:
                        s=(j+len(unique_features))
                    elif features1[i+1]==unique_features[j]:
                        t=(j+2*len(unique_features))
                sappend = [f,s,t]
            if len(sappend) != 0:
                sparsity.append(sappend)
    labels1=[]
    for i in range(len(labels)):
        if labels[i]=='Tawde':
            continue
        else:
            labels1.append(labels[i])
    #sig_label = np.unique(labels1)
    #K = len(sig_label)
    N = len(sparsity)
    #M = len(unique_features)*3
    pred_mat = np.zeros(shape=[N,K])
    
    for i in range(0,N):    
        for k in range(0,K):
            a = theta[k,sparsity[i][0]] + theta[k,sparsity[i][1]] + theta[k,sparsity[i][2]] + theta[k,-1]
            pred_mat[i,k] = np.exp(a)
    
    trainprint = []        
    for i in range(len(pred_mat)):
        sol = np.argmax(pred_mat[i,:])
        trainprint.append(sig_label[sol])
        #print(labels1[i])
        if sig_label[sol] != labels1[i]:
            c = c+1  
            
    for i in range(len(ind_space)):
        trainprint.insert(ind_space[i],'\n')
    
    p3 = (c/len(labels1))
    
    for i in range(len(trainprint)):
        if (trainprint[i] == '\n'):
            trainop.write('{}'.format(trainprint[i]))
        else:
            trainop.write('{}\n'.format(trainprint[i]))
    ############################################################################
    print('Train error: {}'.format(p3))
    print('Test error: {}'.format(p4))
    
    metricsop.write('error(train): {}\n'.format(p3))
    metricsop.write('error(test): {}'.format(p4))
    
else:
        
    """
    Created on Fri Feb 23 11:00:42 2018
    
    @author: Rohit Tawde
    """
    #############################################################################
    #Importing modules
    import numpy as np
    import scipy as sp
    #Extracting data
    data = sp.genfromtxt(trainip,delimiter="\t",unpack=True,dtype=None)
    position=[]
    #Sparse representation
    for i in range(0,len(np.transpose(data))):
        position.append(data[0,i])
    
    uniquepos = np.unique(position)
       
    sparse=np.ones(shape=[len(np.transpose(data)),len(uniquepos)+1])
    
    for i in range(0,len(position)):
        for j in range(0,len(uniquepos)):
            if position[i]==uniquepos[j]:
                sparse[i,j]=1
            else:
                sparse[i,j]=0
    sparse_rep = [uniquepos,sparse]
    #############################################################################
    datav = sp.genfromtxt(valip,delimiter="\t",unpack=True,dtype=None)
    positionv=[]
    #Sparse representation
    for i in range(0,len(np.transpose(datav))):
        positionv.append(datav[0,i])
    
    uniqueposv = np.unique(positionv)
       
    sparsev=np.ones(shape=[len(np.transpose(datav)),len(uniquepos)+1])
    
    for i in range(0,len(positionv)):
        for j in range(0,len(uniquepos)):
            if positionv[i]==uniquepos[j]:
                sparsev[i,j]=1
            else:
                sparsev[i,j]=0
    
    sig_label = np.unique(data[1,:])
    K = len(sig_label)
    N = len(np.transpose(data))
    M = len((sparse[0,:]))-1
    Nv=len(np.transpose(datav))
    #############################################################################
    #Indicator function
    
    def I(i,k):
        if data[1,i] == sig_label[k]:
            return 1
        else:
            return 0
    
    #Indicator for Val
            
    def Iv(i,k):
        if datav[1,i] == sig_label[k]:
            return 1
        else:
            return 0
    #############################################################################
    theta = np.zeros(shape=[K,M+1])
    grad = np.zeros(shape=[K,M+1])
    thetaC = np.zeros(shape=[K,M+1])
    index = []
    for i in range (0,N):
        for nzc in range(0,M+1):
            if sparse[i,nzc] == 1:
                index.append(nzc)
                break
            else:
                continue
    indexv = []
    for i in range (0,Nv):
        for nzcv in range(0,M+1):
            if sparsev[i,nzcv] == 1:
                indexv.append(nzcv)
                break
            else:
                continue
    
    
    for ep in range(0,epoch):
            
        for i in range(0,N):
            sum_ = 0 
                
            for j in range(0,K):
                sum_ = sum_ + np.exp(theta[j,index[i]] + theta[j,-1])
                
            for k in range(0,K):
                grad[k,:] = -(I(i,k) - (np.exp(theta[k,index[i]] + theta[k,-1]))/sum_)*sparse[i,:]
                
                
            theta = theta - 0.5*grad
            nl_ =  0
        
        for i in range(0,N):
            sum_ = 0             
            for j in range(0,K):
                sum_ = sum_ + np.exp(theta[j,index[i]] + theta[j,-1])
            for k in range(0,K):
                nl_1 = I(i,k)*np.log((np.exp(theta[k,index[i]] + theta[k,-1]))/sum_)
                nl_ = nl_ + nl_1
        print('epoch ={} likelihood(train): {}'.format(ep,-nl_/N))
        metricsop.write('epoch={} likelihood(train): {}\n'.format(ep+1,-nl_/N))
        nl_=0
        
        for i in range(0,Nv):
            sum_ = 0             
            for j in range(0,K):
                sum_ = sum_ + np.exp(theta[j,indexv[i]] + theta[j,-1])
            for k in range(0,K):
                nl_1 = Iv(i,k)*np.log((np.exp(theta[k,indexv[i]] + theta[k,-1]))/sum_)
                nl_ = nl_ + nl_1
        print('epoch={} likelihood(validation): {} '.format(ep,-nl_/Nv))
        metricsop.write('epoch={} likelihood(validation): {}\n'.format(ep+1,-nl_/Nv))
            
    ##############################################################################
    testdata = sp.genfromtxt(trainip,delimiter="\t",unpack=True,dtype=None)
    position1=[]
    position2=[]
    #Sparse representation
    for i in range(0,len(np.transpose(testdata))):
        position1.append(data[0,i])
        position2.append(data[1,i])
    
    uniquepos1 = np.unique(position1)
       
    sparse1=np.ones(shape=[len(np.transpose(testdata)),len(uniquepos)+1])
    
    for i in range(0,len(np.transpose(testdata))):
        for j in range(0,len(uniquepos)):
            if position1[i]==uniquepos[j]:
                sparse1[i,j]=1
            else:
                sparse1[i,j]=0
    sparse_rep1 = [uniquepos1,sparse1]
    N1 = len(sparse1)
    
    index1 = []
    for i in range (0,N1):
        for nzc in range(0,M+1):
            if sparse1[i,nzc] == 1:
                index1.append(nzc)
                break
            else:
                continue
    #print(index1)
    pred_mat = np.zeros(shape=[N1,K])
    
    for i in range(0,N1):   
        for k in range(0,K):        
            a = theta[k,index1[i]] + theta[k,-1]
            pred_mat[i,k] = np.exp(a)
    
    c1 = 0. 
    
    trainprintmod1 = []
    for i in range(len(pred_mat)):
        sol = np.argmax(pred_mat[i,:])
        trainprintmod1.append(sig_label[sol])
        #print(position2[i])
        #print("\n")
        if sig_label[sol] != position2[i] :
            c1 = c1 + 1
    
    
    print('error(train): {}'.format(c1/len(position2)))    
    metricsop.write('error(train): {}\n'.format(c1/len(position2)))
    #############################################################################
    testdata = sp.genfromtxt(testip,delimiter="\t",unpack=True,dtype=None)
    position1=[]
    position3=[]
    #Sparse representation
    for i in range(0,len(np.transpose(testdata))):
        position1.append(testdata[0,i])
        position3.append(testdata[1,i])
    
    uniquepos1 = np.unique(position1)
       
    sparse1=np.ones(shape=[len(np.transpose(testdata)),len(uniquepos)+1])
    
    for i in range(0,len(np.transpose(testdata))):
        for j in range(0,len(uniquepos)):
            if position1[i]==uniquepos[j]:
                sparse1[i,j]=1
            else:
                sparse1[i,j]=0
        
    sparse_rep1 = [uniquepos1,sparse1]
    N1 = len(sparse1)
    
    index1 = []
    for i in range (0,N1):
        for nzc in range(0,M+1):
            if sparse1[i,nzc] == 1:
                index1.append(nzc)
                break
            else:
                continue
    
    pred_mat = np.zeros(shape=[N1,K])
    for i in range(0,N1):  
      
    
        for k in range(0,K):
            a = theta[k,index1[i]] + theta[k,-1]
            pred_mat[i,k] = np.exp(a)
    
    c = 0. 
    testprintmod1 = []      
    for i in range(len(pred_mat)):
        sol = np.argmax(pred_mat[i,:])
        testprintmod1.append(sig_label[sol])
        #print(position3[i])
        #print("\n")
        if sig_label[sol] != position3[i] :
            c = c + 1
    
    print('error(test): {}'.format(c/len(position3)))
    metricsop.write('error(test): {}'.format(c/len(position3)))
    #
    #
    #
    #
    #
    data = []
    with open(trainip,'r') as tsv:
        newdata = [line.strip().split('\t') for line in tsv]
    
    for i in range (len(newdata)):
        if newdata[i] ==['']:
            newdata[i] = ['EOS','Tawde']
        else:
            newdata[i] = newdata[i]
    
    data = np.transpose(newdata)
    
    ind_space = []
    
    features = data[0]
    labels = data[1]
    
    for i in range (len(features)):
        if(features[i]=='EOS'):
            ind_space.append(i)
        else:
            continue
    #
    #
    #
    #
    data = []
    with open(testip,'r') as tsv:
        newdata = [line.strip().split('\t') for line in tsv]
    
    for i in range (len(newdata)):
        if newdata[i] ==['']:
            newdata[i] = ['EOS','Tawde']
        else:
            newdata[i] = newdata[i]
    
    data = np.transpose(newdata)
    
    ind_space1 = []
    
    features = data[0]
    labels = data[1]
    
    for i in range (len(features)):
        if(features[i]=='EOS'):
            ind_space1.append(i)
        else:
            continue
        
    for i in range(len(ind_space)):
        trainprintmod1.insert(ind_space[i],'\n')
        
    
    for i in range(len(ind_space1)):
        testprintmod1.insert(ind_space1[i],'\n')
        
    for i in range(len(trainprintmod1)):
        if (trainprintmod1[i] == '\n'):
            trainop.write('{}'.format(trainprintmod1[i]))
        else:
            trainop.write('{}\n'.format(trainprintmod1[i]))
    
    for i in range(len(testprintmod1)):
        if (testprintmod1[i] == '\n'):
            testop.write('{}'.format(testprintmod1[i]))
        else:
            testop.write('{}\n'.format(testprintmod1[i]))
            
testop.close()
trainop.close()
metricsop.close()