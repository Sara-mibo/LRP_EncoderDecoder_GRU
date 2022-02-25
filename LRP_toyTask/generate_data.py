import pandas as pd
import numpy as np
import yaml
from types import SimpleNamespace


def generate_data(std_flag):
    """
    Generate data for the toy task
    Parameters
    - std_flag:        when the flag is one, data will be standardized
    Returns
    - (traindata,trainfdata,trainldata):     train dataset (historical, forecast, and labels) 
    - (valdata,valfdata,valldata):           validation dataset (historical, forecast, and labels)
    - (testdata,testfdata,testldata):        test dataset (historical, forecast, and labels)
    - cfg:                                   configurations            
    """    
    ## random values [-1,3]
    np.random.seed(seed=1111)
    data = np.random.randint(1, high=5, size=(100000,5,2))-2
    fdata = np.random.randint(1, high=5, size=(100000,3,2))-2
    labels=[]
    ##generate labels
    for i in range(len(data)):
        l1=data[i].sum()/2
        lf=np.expand_dims(np.transpose(fdata[i].sum(axis=1)), axis=1)
        l2=np.empty((len(lf),1),dtype=float)
        for j in range(len(lf)):
            if j>0:
                l2[j,0]=l2[j-1,0]+lf[j,0]
            else:
                l2[j,0]= l1+lf[j,0]
        labels.append(l2)


    ldata=np.stack(labels,axis=0)

    ##train,validation,test set
    ## data standardization
    if std_flag==1:
        ldata=(ldata-ldata.mean(axis=0))/ldata.std(axis=0)
        fdata=(fdata-fdata.mean(axis=0))/fdata.std(axis=0)
        data=(data-data.mean(axis=0))/data.std(axis=0)
    
    ldata_n=ldata 
    fdata_n=fdata 
    data_n=data

    #######################
    number_of_rows = data.shape[0]
    random_indices = np.random.choice(number_of_rows, size=20000, replace=False)
    valdata=data_n[random_indices]
    valfdata=fdata_n[random_indices]
    valldata=ldata_n[random_indices]


    not_indices=[i for i in np.arange(100000) if i not in random_indices]
    d=data_n[not_indices]
    df=fdata_n[not_indices]
    l=ldata_n[not_indices]

    testdata=d[:1000]
    testfdata=df[:1000]
    testldata=l[:1000]

    traindata=d[1000:]
    trainfdata=df[1000:]
    trainldata=l[1000:]

    print(traindata.shape)
    print(valdata.shape)
    print(testdata.shape)
    
    # load config
    cfg_path = "config.yaml"
    with open(cfg_path, "r") as f:
        d = yaml.safe_load(f)
    cfg = SimpleNamespace(**d)

    return (traindata,trainfdata,trainldata),(valdata,valfdata,valldata),(testdata,testfdata,testldata),cfg
    

if __name__=='__main__':
        generate_data(std_flag)