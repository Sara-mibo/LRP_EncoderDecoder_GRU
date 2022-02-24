import torch
import numpy as np
from numpy import newaxis as na

def gru_cell(input, hidden, w_ih, w_hh, b_ih, b_hh):
    """
    GRU cell
    Returns:
    - hy:                               output hidden state 
    - resetgate,#,newgate:      cell gates
    - pre_gate:                         cell gates before applying activation function
    """
    gi = torch.mm(input, w_ih.t()) + b_ih
    gh = torch.mm(hidden, w_hh.t()) + b_hh
    i_r, i_i, i_n = gi.chunk(3, 1)
    h_r, h_i, h_n = gh.chunk(3, 1)

    resetgate = torch.sigmoid(i_r + h_r)
    updategate = torch.sigmoid(i_i + h_i)
    newgate = torch.tanh(i_n + resetgate * h_n)

    pre_newgate=i_n + resetgate * h_n
    pre_resetgate=i_i + h_i
    pre_resetgate=i_r + h_r

    hy = newgate + updategate * (hidden - newgate)

    pre_gate=torch.cat((pre_resetgate,pre_resetgate,pre_newgate),dim=-1)
    return hy,resetgate,updategate,newgate,pre_gate

def linear(x,wl1,wl2,bl1,bl2):
    """
    Linear layers on top of RNN cells in decoder part
    Args:
    - x:                         historical input sequence
    - wl1,wl2,bl1,bl2:           weights and biases
    Returns:
    - l1:                        output of first linear layer before ReLU 
    - r1:                        output of first linear layer after ReLU
    - l2:                        output of second linear layer
    """
    l1=torch.mm(x, wl1.t())+bl1
    r1=torch.relu(l1)
    l2=torch.mm(r1, wl2.t())+bl2
    return l1,r1,l2

def lrp_linear(hin, w, b, hout, Rout, bias_nb_units, eps, bias_factor=0.0):
    """
    LRP for a linear layer with input dim D and output dim M.
    Args:
    - hin:            forward pass input, of shape (D,)
    - w:              connection weights, of shape (D, M)
    - b:              biases, of shape (M,)
    - hout:           forward pass output, of shape (M,) (unequal to np.dot(w.T,hin)+b if more than one incoming layer!)
    - Rout:           relevance at layer output, of shape (M,)
    - bias_nb_units:  total number of connected lower-layer units (onto which the bias/stabilizer contribution is redistributed for sanity check)
    - eps:            stabilizer (small positive number)
    - bias_factor:    set to 1.0 to check global relevance conservation, otherwise use 0.0 to ignore bias/stabilizer redistribution (recommended)
    Returns:
    - Rin:            relevance at layer input, of shape (D,)
    """

    sign_out = np.where(hout[na,:]>=0, 1., -1.) # shape (1, M)

    numer    = (w * hin[:,na]) + ( bias_factor * (b[na,:]*1. + eps*sign_out*1.) / bias_nb_units ) # shape (D, M)
    #print('numer\n (D,M)',numer.shape)
    # Note: here we multiply the bias_factor with both the bias b and the stabilizer eps since in fact
    # using the term (b[na,:]*1. + eps*sign_out*1.) / bias_nb_units in the numerator is only useful for sanity check
    # (in the initial paper version we were using (bias_factor*b[na,:]*1. + eps*sign_out*1.) / bias_nb_units instead)

    denom    = hout[na,:] + (eps*sign_out*1.)   # shape (1, M)

    message  = (numer/denom) * Rout[na,:]       # shape (D, M)

    Rin      = message.sum(axis=1)              # shape (D,)

    return Rin

def gru_lrp(updategate,newgate,resetgate,hs_previous,x,wi,wh,bi,bh,hs,out3,R_out,eps, bias_factor):
    """
    LRP GRU
    Args:
    - updategate,newgate,resetgate:            gru gates
    - hs_previous:                            previous step's hidden state(h(t-1))   
    - x:                                      input 
    - wi,wh,bi,bh:                       weights and biases
    - hs:                                     output hidden state(h(t))
    - out3:                                   output of gate before applying activation function
    - R_out:                                  relevance at layer output
    - eps:                                    stabilizer (small positive number)
    - bias_factor:                            set to 1.0 to check global relevance conservation, otherwise use 0.0 to ignore bias/stabilizer redistribution (recommended)
    Returns:
    - Rh_previous:                            R(h(t-1)) 
    - Rx:                                     relevance at layer input
    """
    hidden_size=hs.size
    O=np.zeros((hidden_size)) #zero(bias)
    Id=np.identity(hidden_size) #identity(weights)
        
    R_newgate= lrp_linear((1-updategate)*newgate, Id, O, hs, R_out, hs_previous.size+newgate.size, eps, bias_factor)
    Rh_previous =lrp_linear(updategate*hs_previous, Id, O, hs, R_out, hs_previous.size+newgate.size, eps, bias_factor)
    Rx= lrp_linear(x, wi.T, bi+resetgate*bh, out3, R_newgate, x.size+hs_previous.size,eps, bias_factor)
    Rh_previous +=lrp_linear(hs_previous,resetgate*wh.T,resetgate*bh + bi , out3, 
                                      R_newgate, x.size+hs_previous.size, eps, bias_factor)

    return Rh_previous,Rx


def initialization(len_static,num_layers,in_seq_len,out_seq_len,out_feat_len,hidden_size):
    
    #####hidden states initialization#####
    hs_encoder=np.zeros((num_layers,in_seq_len,hidden_size))
    hs_encoder=torch.tensor(hs_encoder).double()
    hs_decoder=np.zeros((num_layers,out_seq_len,hidden_size))
    hs_decoder=torch.tensor(hs_decoder).double()
    #####gates and pre_gates intialization#####
        
    resetgate_enc, updategate_enc, newgate_enc=np.zeros((num_layers,in_seq_len,hidden_size)), np.zeros((num_layers,in_seq_len,hidden_size)), np.zeros((num_layers,in_seq_len,hidden_size))
    resetgate_enc, updategate_enc, newgate_enc=torch.tensor(resetgate_enc).double(), torch.tensor(updategate_enc).double(), torch.tensor(newgate_enc).double()
    resetgate_dec, updategate_dec, newgate_dec=np.zeros((num_layers,out_seq_len,hidden_size)), np.zeros((num_layers,out_seq_len,hidden_size)), np.zeros((num_layers,out_seq_len,hidden_size))
    resetgate_dec, updategate_dec, newgate_dec=torch.tensor(resetgate_dec).double(), torch.tensor(updategate_dec).double(), torch.tensor(newgate_dec).double()

    pre_gate_enc=np.zeros((num_layers,in_seq_len,3*hidden_size))
    pre_gate_enc=torch.tensor(pre_gate_enc).double()
    pre_gate_dec=np.zeros((num_layers,out_seq_len,3*hidden_size))
    pre_gate_dec=torch.tensor(pre_gate_dec).double()
    ##### linear layers, relue, and outputs initialization#####
    y=torch.zeros([out_seq_len,out_feat_len]).double()
    l1=torch.zeros([out_seq_len,hidden_size]).double()
    r1=torch.zeros([out_seq_len,hidden_size]).double()
    #####output decder initialization######
    output=torch.zeros([out_seq_len,hidden_size+len_static]).double()
    return hs_encoder,hs_decoder,resetgate_enc, updategate_enc, newgate_enc,resetgate_dec, updategate_dec, newgate_dec, pre_gate_enc, pre_gate_dec, y,l1, r1, output


def init_decoder(len_static,num_layers,out_seq_len,hidden_size,xfShape,yShape):
    ####relevance matrice initializations linears#####
    R_l2=np.zeros((out_seq_len,hidden_size )).astype(np.float128, copy=False)
    R_l1_st=np.zeros((out_seq_len,hidden_size+len_static)).astype(np.float128, copy=False)
    R_l1=np.zeros((out_seq_len,hidden_size )).astype(np.float128, copy=False)
    Rstatic=np.zeros((out_seq_len,len_static)).astype(np.float128, copy=False)
    #####relevance matrice initializations decoder#####
    Rh_d=np.zeros((num_layers,out_seq_len,hidden_size )).astype(np.float128, copy=False)
    #R_newgate_d=np.zeros((num_layers,out_seq_len,hidden_size )).astype(np.float128, copy=False)
    Rx_d= np.zeros((out_seq_len,hidden_size )).astype(np.float128, copy=False)
    Rxfy=np.zeros((out_seq_len,xfShape[-1]+yShape[-1])).astype(np.float128, copy=False)
    Rxf=np.zeros((out_seq_len,xfShape[-1])).astype(np.float128, copy=False)
    Ry=np.zeros((out_seq_len,yShape[-1])).astype(np.float128, copy=False)
    return R_l2,R_l1_st,R_l1,Rh_d,Rx_d,Rxfy,Rxf,Ry,Rstatic
    

def init_encoder(num_layers,in_seq_len,hidden_size,xShape):
    Rh_e=np.zeros((num_layers,in_seq_len,hidden_size )).astype(np.float128, copy=False)
    R_newgate_e=np.zeros((num_layers,in_seq_len,hidden_size )).astype(np.float128, copy=False)
    Rx=np.zeros((in_seq_len,xShape[-1])).astype(np.float128, copy=False)
    Rx_e= np.zeros((in_seq_len,hidden_size )).astype(np.float128, copy=False)
    return Rh_e,R_newgate_e,Rx,Rx_e

    
def moveToNumpy(array):
     array=array.detach().cpu().numpy().astype(np.float128, copy=False) 
    
     return array