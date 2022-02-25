import torch
import torch.nn as nn
import model

def collect_leaves(module):
    '''Generator function to collect all leaf modules of a module.
    Parameters
    ----------
    module: obj:`torch.nn.Module`
        A module for which the leaves will be collected.
    Yields
    ------
    leaf: obj:`torch.nn.Module`
        Either a leaf of the module structure, or the module itself if it has no children.
    '''
    is_leaf = True

    children = module.children()
    for child in children:
        is_leaf = False
        for leaf in collect_leaves(child):
            yield leaf
    if is_leaf:
        yield module
        

        
def load_model(cfg,device,loading_path):
    '''
    Loading model
    Parameters
    ----------
    cfg:            configurations
    device:         device (cpu or gpu)
    loading_path:   path of the trained model
    Returns
    ------
    net
    '''    
    in_feat_len = cfg.in_feat_len
    out_feat_len = cfg.out_feat_len 

    net = model.EncoderDecoder(in_seq_len = cfg.in_seq_len, out_seq_len = cfg.out_seq_len,
                           in_feat_len = in_feat_len, out_feat_len = out_feat_len, static_feat_len = cfg.static_feat_len,
                           hidden_size = cfg.hidden_size, num_layers=cfg.num_layers, enc_dropout=0,
                           dec_dropout=0, device=device, arch=cfg.arch)

    net.load_state_dict(torch.load(loading_path,map_location=torch.device(device)))

    net = net.double()
    net.to(device)
    return net


def model_parameters(loading_path,device,cfg):
    '''
    Loading model's parameters
    Parameters
    ----------
    loading_path:   path of the trained model
    device:         device (cpu or gpu)
    cfg:            configurations 
    Returns
    ------
    wi_tuples:      weights of GRU layers (input-hidden weights)
    wh_tuples:      weights of GRU layers (hidden-hidden weights)
    bi_tuples:      biases of GRU layers (input-hidden biases)
    bh_tuples:      biases of GRU layers (hidden-hidden biases)
    (wl1,wl2):      weights of linear layers 
    (bl1,bl2):      biases of linear layers
    '''
    ## load model
    model= load_model(cfg,device,loading_path)
    
    ## load model parameters
    weight_array=[]
    bias_array=[]
    for leaf in collect_leaves(model):
        if isinstance(leaf,nn.ReLU) or isinstance(leaf,nn.Dropout):
            pass
        else:
            try:
                weight_array.append(leaf.weight.data)
            except:
                weight_array.append(leaf.all_weights)
     
            bias_array.append(leaf.bias)
        
    ###########encoder###############
    wi_l0_enc=weight_array[0][0][0].double()
    wh_l0_enc=weight_array[0][0][1].double()
    wi_l1_enc=weight_array[0][1][0].double()
    wh_l1_enc=weight_array[0][1][1].double()

    bi_l0_enc=weight_array[0][0][2].double()
    bh_l0_enc=weight_array[0][0][3].double()
    bi_l1_enc=weight_array[0][1][2].double()
    bh_l1_enc=weight_array[0][1][3].double()
    ###########decoder###########
    wi_l0_dec=weight_array[1][0][0].double()
    wh_l0_dec=weight_array[1][0][1].double()
    wi_l1_dec=weight_array[1][1][0].double()
    wh_l1_dec=weight_array[1][1][1].double()

    bi_l0_dec=weight_array[1][0][2].double()
    bh_l0_dec=weight_array[1][0][3].double()
    bi_l1_dec=weight_array[1][1][2].double()
    bh_l1_dec=weight_array[1][1][3].double()
    
    #############################
    wi_tuples =  (wi_l0_enc,wi_l1_enc,wi_l0_dec,wi_l1_dec)
    wh_tuples =  (wh_l0_enc,wh_l1_enc,wh_l0_dec,wh_l1_dec)
    bi_tuples =  (bi_l0_enc,bi_l1_enc,bi_l0_dec,bi_l1_dec)
    bh_tuples =  (bh_l0_enc,bh_l1_enc,bh_l0_dec,bh_l1_dec)
    ###########linear1#############
    wl1=weight_array[2].double()
    bl1=model.decoder_cell.out1.bias.double()
    ##########linear2#############
    wl2=weight_array[3].double()
    bl2=torch.zeros((cfg.out_feat_len)).to(device).double() 
    
    
    
    return wi_tuples,wh_tuples,bi_tuples,bh_tuples,(wl1,wl2),(bl1,bl2)