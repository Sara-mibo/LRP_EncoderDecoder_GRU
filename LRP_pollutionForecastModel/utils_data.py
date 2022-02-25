import pickle
import yaml
from types import SimpleNamespace
import numpy as np
import torch

def load_data(path,device):
    file=open(path,'rb')
    x,xf,tr,st= pickle.load(file)
    x_t,xf_t,tr_t,st_t = torch.tensor(x).double().to(device), torch.tensor(xf).double().to(device), torch.tensor(tr).double().to(device), torch.tensor(st).double().to(device)
    return x_t,xf_t,tr_t,st_t
        
    
def load_minmax():
    file=open('data/train_val_test_minmax_1hotEnc.p','rb')
    min_data,max_data= pickle.load(file)
    return min_data,max_data



def load_configurations():
    """
    load model and data configurations 
    Returns
    - cfg:                 model configurations 
    - mean, std:           mean and standard deviation of dynamic data
    - station_mean, station_std:        mean and standard deviation of static data           
    """ 
    
    # load config
    cfg_path = "model/config_ozone.yaml"
    with open(cfg_path, "r") as f:
        d = yaml.safe_load(f)
    cfg = SimpleNamespace(**d)
    
    ## load meanstd file
    file=open('data/train_val_test_meanstd_1hotEnc.p','rb')
    mean0,std0= pickle.load(file)
    ## binary features should have mean of 0 and std of 1 (no standardization is needed for binary features)
    ## not binary features 
    standardized_cols=['NO_AM1H', 'PM10_GM1H24H', 'NO2_AM1H', 'O3_AM1H', 'TEMP', 'HUM', 'RAIN','SUN','WSPD','year']
    ## binary features
    no_std_cols=list(set(cfg.relevant_headers_encoder)-set(standardized_cols+['crossval']))
    mean_nostd = mean0.copy()
    mean_nostd[no_std_cols]=0
    mean = mean_nostd
    std_nostd= std0.copy()
    std_nostd[no_std_cols]=1
    std = std_nostd
    
    ##stations
    stats=['WESE', 'RODE', 'SOES', 'MSGE', 'WALS', 'HUE2', 'LOER', 'DMD2', 'WULA', 'STYR', 'EIFE', 'LEV2',
       'SOLI', 'BOTT', 'SHW2', 'BORG', 'CHOR', 'ROTH', 'NIED', 'AABU', 'RAT2', 'BIEL']
    for i in range(len(stats)):
        print('{}: {}'.format(i,stats[i]), sep=',', end=', ', flush=True)
    ## stations mean and std
    station_mean = np.mean(range(len(stats)))
    station_std = np.std(range(len(stats)))
    
    return cfg, mean, std, station_mean, station_std