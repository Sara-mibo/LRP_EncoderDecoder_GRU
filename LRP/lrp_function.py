import torch
import numpy as np
from numpy import newaxis as na
from utils_lrp import gru_cell,linear,lrp_linear,gru_lrp,initialization,init_decoder,moveToNumpy,init_encoder



class lrp_decoder_encoder:
    def __init__(self, cfg,len_static, wi_tuples,wh_tuples,bi_tuples,bh_tuples,wl,bl):
        self.num_layers=cfg.num_layers
        self.in_seq_len=cfg.in_seq_len
        self.out_seq_len=cfg.out_seq_len
        self.hidden_size=cfg.hidden_size
        self.out_feat_len = cfg.out_feat_len
        self.len_static=len_static
        #######initializations #################
        self.hs_encoder,self.hs_decoder,self.resetgate_enc, self.updategate_enc, self.newgate_enc,self.resetgate_dec, self.updategate_dec, self.newgate_dec, self.pre_gate_enc, self.pre_gate_dec, self.y,self.l1, self.r1, self.output = initialization(self.len_static,self.num_layers,self.in_seq_len,self.out_seq_len,self.out_feat_len,self.hidden_size) 

        ####weights and biases#####
        self.wi_l0_enc,self.wi_l1_enc,self.wi_l0_dec,self.wi_l1_dec = wi_tuples[0],wi_tuples[1],wi_tuples[2],wi_tuples[3]
        self.wh_l0_enc,self.wh_l1_enc,self.wh_l0_dec,self.wh_l1_dec = wh_tuples[0],wh_tuples[1],wh_tuples[2],wh_tuples[3]

        self.bi_l0_enc,self.bi_l1_enc,self.bi_l0_dec,self.bi_l1_dec = bi_tuples[0],bi_tuples[1],bi_tuples[2],bi_tuples[3]
        self.bh_l0_enc,self.bh_l1_enc,self.bh_l0_dec,self.bh_l1_dec = bh_tuples[0],bh_tuples[1],bh_tuples[2],bh_tuples[3] 

        self.wl1,self.bl1= wl[0], bl[0]
        self.wl2,self.bl2= wl[1], bl[1]
        

    def forward_pass(self,x,xf,static):
        """
        LRP forward pass
        Args:
        - x:            historical input sequence
        - xf:           forecast input sequence
        - static:    static data (we may not have static data)
        Returns:
        - y:             predictions 
        - r1,l1,output:  intermidiate outputs in decoder layers
        """
        self.hs_init=self.hs_encoder[0,[0]].clone()
        #### forward pass encoder ####
        for t in range(self.in_seq_len):
            if t==0:
                #### first input ####
                ##layer 0
                self.hs_encoder[0,[t]], self.resetgate_enc[0,[t]], self.updategate_enc[0,[t]], self.newgate_enc[0,[t]], self.pre_gate_enc[0,[t]]= gru_cell(x[0,[t]], self.hs_init, self.wi_l0_enc, self.wh_l0_enc, self.bi_l0_enc, self.bh_l0_enc)
                ##layer 1
                self.hs_encoder[1,[t]], self.resetgate_enc[1,[t]], self.updategate_enc[1,[t]], self.newgate_enc[1,[t]], self.pre_gate_enc[1,[t]]= gru_cell(self.hs_encoder[0,[t]],self.hs_init, self.wi_l1_enc, self.wh_l1_enc, self.bi_l1_enc,self.bh_l1_enc)
            else:
                ##layer 0
                self.hs_encoder[0,[t]], self.resetgate_enc[0,[t]], self.updategate_enc[0,[t]], self.newgate_enc[0,[t]], self.pre_gate_enc[0,[t]]= gru_cell(x[0,[t]], self.hs_encoder[0,[t-1]], self.wi_l0_enc, self.wh_l0_enc, self.bi_l0_enc, self.bh_l0_enc)
                ##layer 1
                self.hs_encoder[1,[t]], self.resetgate_enc[1,[t]], self.updategate_enc[1,[t]], self.newgate_enc[1,[t]], self.pre_gate_enc[1,[t]]= gru_cell(self.hs_encoder[0,[t]], self.hs_encoder[1,[t-1]], self.wi_l1_enc, self.wh_l1_enc, self.bi_l1_enc, self.bh_l1_enc)

        #### forward pass decoder ####
        for t in range(self.out_seq_len):
            if t==0:
                y_prev = x[:, -1:, :self.out_feat_len]   #initialize first pollution prediction outputs with the last measured values of pollution
                x_init_dec=torch.cat((y_prev,xf[:,[t]]), dim=2)[0]
                ##layer 0
                self.hs_decoder[0,[t]], self.resetgate_dec[0,[t]], self.updategate_dec[0,[t]], self.newgate_dec[0,[t]], self.pre_gate_dec[0,[t]]= gru_cell(x_init_dec,self.hs_encoder[0,[-1]], self.wi_l0_dec, self.wh_l0_dec, self.bi_l0_dec, self.bh_l0_dec)
                ##layer 1
                self.hs_decoder[1,[t]], self.resetgate_dec[1,[t]], self.updategate_dec[1,[t]], self.newgate_dec[1,[t]], self.pre_gate_dec[1,[t]]= gru_cell(self.hs_decoder[0,[t]], self.hs_encoder[1,[-1]], self.wi_l1_dec, self.wh_l1_dec, self.bi_l1_dec, self.bh_l1_dec)
                ## linear layers with ReLU activation
                if self.len_static!=0:
                    self.output[[t]]=torch.cat((self.hs_decoder[1,[t]], static), axis=-1) #static data is added in linear layers
                else:
                    self.output[[t]]=self.hs_decoder[1,[t]]
                self.l1[t],self.r1[t],self.y[t]=linear(self.output[[t]],self.wl1,self.wl2,self.bl1,self.bl2)

            else:
                y_prev=self.y[[t-1]].unsqueeze(1)  #prediction output of previous step
                x_dec=torch.cat((y_prev,xf[:,[t]]), dim=2)[0]
                ##layer 0
                self.hs_decoder[0,[t]], self.resetgate_dec[0,[t]], self.updategate_dec[0,[t]], self.newgate_dec[0,[t]], self.pre_gate_dec[0,[t]]= gru_cell(x_dec, self.hs_decoder[0,[t-1]], self.wi_l0_dec, self.wh_l0_dec, self.bi_l0_dec, self.bh_l0_dec)
                ##layer 1
                self.hs_decoder[1,[t]], self.resetgate_dec[1,[t]], self.updategate_dec[1,[t]], self.newgate_dec[1,[t]], self.pre_gate_dec[1,[t]]= gru_cell(self.hs_decoder[0,[t]], self.hs_decoder[1,[t-1]], self.wi_l1_dec, self.wh_l1_dec, self.bi_l1_dec, self.bh_l1_dec)
                ## linear layers with ReLU activation
                if self.len_static!=0:
                    self.output[[t]]=torch.cat((self.hs_decoder[1,[t]], static), axis=-1)
                else:
                    self.output[[t]]=self.hs_decoder[1,[t]]
                self.l1[t],self.r1[t],self.y[t]=linear(self.output[[t]],self.wl1,self.wl2,self.bl1,self.bl2)   
        
        return self.y
    
    
    
    def lrp_decoder(self,x,xf,mask_output,eps=0.001,bias_factor=1.0):
        """
        LRP for the decoder module.
        Args:
        - x:            historical input sequence
        - xf:           forecast input sequence
        - static_t1:    static data
        - mask_output:  LRP mask          
        - eps:      stabilizer (small positive number)
        - bias_factor:  set to 1.0 to check global relevance conservation, otherwise use 0.0 to ignore bias/stabilizer redistribution (recommended)
        Returns:
        - Ry,Rxf:                 relevance scores of decoder inputs
        - Rh_d_first_l1,Rh_d_first_l1:     relevance scores of initial hidden states of decoder module(they are needed to compute lrp of encoder)
        """
        h=self.hidden_size
        #### initialization decoder arrays##########
        R_l2,R_l1_st,R_l1,Rh_d,Rx_d,Rxfy,self.Rxf,self.Ry,self.Rstatic = init_decoder(self.len_static,self.num_layers, self.out_seq_len, self.hidden_size, xf.shape,self.y.shape) 
        #### move numpy#####
        a=[self.l1, self.y, self.r1, self.wl2, self.bl2, self.wl1, self.bl1, self.hs_decoder, self.updategate_dec, self.newgate_dec, self.resetgate_dec, self.pre_gate_dec, self.hs_encoder, self.wi_l0_dec, self.wi_l1_dec, self.wh_l1_dec, self.wh_l0_dec, self.wh_l0_dec, self.bi_l0_dec, self.bi_l1_dec, self.bh_l0_dec, self.bh_l1_dec, self.output,xf,x]

        [l1, y, r1, wl2, bl2, wl1, bl1, hs_decoder, updategate_dec, newgate_dec, resetgate_dec, pre_gate_dec, hs_encoder, wi_l0_dec, wi_l1_dec, wh_l1_dec, wh_l0_dec, wh_l0_dec, bi_l0_dec, bi_l1_dec, bh_l0_dec, bh_l1_dec, output, xf, x ]=list(map(moveToNumpy,a))
        
        
        ##### LRP mask #####
        ##tuple of hour of output and pollutant ['NO_AM1H', 'PM10_GM1H24H', 'NO2_AM1H','O3_AM1H']
        m = mask_output 
        target=m[0]*y.shape[1]+m[1]
        r = y.shape[0]*y.shape[1]
        m = np.arange(r)
        mask_t=np.where((m==target),1.,0.).reshape([y.shape[0],y.shape[1]])

        self.Rout=y*mask_t

        #### weight chunks #####
        wi_l1_d=np.split(wi_l1_dec,3) ##layer 1
        bi_l1_d=np.split(bi_l1_dec,3)
        wh_l1_d=np.split(wh_l1_dec,3) 
        bh_l1_d=np.split(bh_l1_dec,3)

        wh_l0_d=np.split(wh_l0_dec,3) ##layer 0
        bh_l0_d=np.split(bh_l0_dec,3)
        wi_l0_d=np.split(wi_l0_dec,3)
        bi_l0_d=np.split(bi_l0_dec,3) 
        ############################

        
        for t in reversed(range(self.out_seq_len)):
            if t>0:
                ##relevance propagation(linear layers)
                R_l2[t]=lrp_linear(r1[t],wl2.T,bl2,y[t],self.Rout[t]+self.Ry[t],r1[t].size,eps, bias_factor)

                R_l1_st[[t]]=lrp_linear(output[t],wl1.T,bl1,l1[t], R_l2[t],output[t].size,eps,bias_factor)
                if self.len_static!=0:
                    R_l1[t] = R_l1_st[t][:-self.len_static]
                    self.Rstatic[t]= R_l1_st[t][-self.len_static:]
                else:
                    R_l1[t] = R_l1_st[t]

                ##relevance propagation-gru(decoder layer 1)
                Rh_d[1,[t-1]], Rx_d[t]= gru_lrp(updategate_dec[1,t], newgate_dec[1,t], resetgate_dec[1,t], hs_decoder[1,t-1], hs_decoder[0,t], wi_l1_d[2], wh_l1_d[2], bi_l1_d[2], bh_l1_d[2], hs_decoder[1,t], pre_gate_dec[1,t][2*h:], R_l1[t]+Rh_d[1,t], eps, bias_factor)
                

                ##relevance propagation-gru(decoder layer 0)
                X_decoder=np.concatenate((y[[t-1]],xf[:,t]), axis=1)
                X_decoder=np.squeeze(X_decoder,axis=0)                
                
                Rh_d[0,[t-1]], Rxfy[t]= gru_lrp(updategate_dec[0,t], newgate_dec[0,t], resetgate_dec[0,t], hs_decoder[0,t-1], X_decoder, wi_l0_d[2], wh_l0_d[2], bi_l0_d[2], bh_l0_d[2], hs_decoder[0,t], pre_gate_dec[0,t][2*h:], Rx_d[t]+Rh_d[0,t], eps, bias_factor)

                self.Rxf[t]=Rxfy[t,y.shape[-1]:]
                self.Ry[t-1]=Rxfy[t,:y.shape[-1]]   #relevance scores of predictions of previous step

            else:
                ##input of first cells come from encoder part
                ##relevance propagation(linear layers)
                R_l2[t]=lrp_linear(r1[t],wl2.T,bl2,y[t],self.Rout[t]+self.Ry[t],r1[t].size,eps, bias_factor)

                R_l1_st[[t]]=lrp_linear(output[t],wl1.T,bl1,l1[t],R_l2[t],output[t].size,eps, bias_factor)

                if self.len_static!=0:
                    R_l1[t] = R_l1_st[t][:-self.len_static]
                    self.Rstatic[t]= R_l1_st[t][-self.len_static:]
                else:
                    R_l1[t] = R_l1_st[t]
                

                ##relevance propagation-gru(decoder first step layer 1)               
                self.Rh_d_first_l1, Rx_d[t]= gru_lrp(updategate_dec[1,t], newgate_dec[1,t], resetgate_dec[1,t], hs_encoder[1,-1], hs_decoder[0,t], wi_l1_d[2], wh_l1_d[2], bi_l1_d[2], bh_l1_d[2], hs_decoder[1,t], pre_gate_dec[1,t][2*h:], R_l1[t]+Rh_d[1,t], eps, bias_factor)
                

                ##relevance propagation-gru(decoder first step layer 0)
                y_0 = x[:, -1:, :self.out_feat_len] ##last output of encoder part
                X_decoder=np.concatenate((y_0,xf[:,[0]]), axis=2)[0]
                X_decoder=np.squeeze(X_decoder,axis=0)
                
                self.Rh_d_first_l0, Rxfy[t]= gru_lrp(updategate_dec[0,t], newgate_dec[0,t], resetgate_dec[0,t], hs_encoder[0,-1], X_decoder, wi_l0_d[2], wh_l0_d[2], bi_l0_d[2], bh_l0_d[2], hs_decoder[0,t], pre_gate_dec[0,t][2*h:], Rx_d[t]+Rh_d[0,t], eps, bias_factor)
                
                self.Rxf[t]=Rxfy[t,y.shape[-1]:]
                self.Ry_0=Rxfy[t,:y.shape[-1]]

        if bias_factor==1:
            if self.len_static!=0:
                 print('sanity decoder\n', self.Rout.sum(),'in comparison with', self.Rxf.sum()+self.Rh_d_first_l1.sum()+self.Rh_d_first_l0.sum()+self.Ry_0.sum()+self.Rstatic.sum(),
             '\nsanityCheckResult', np.allclose(self.Rout.sum(), self.Rxf.sum()+self.Rh_d_first_l1.sum()+self.Rh_d_first_l0.sum()+self.Ry_0.sum()+self.Rstatic.sum()))   
            else:
                print('sanity decoder\n', self.Rout.sum(),'in comparison with', self.Rxf.sum()+self.Rh_d_first_l1.sum()+self.Rh_d_first_l0.sum()+self.Ry_0.sum(),
             '\nsanityCheckResult', np.allclose(self.Rout.sum(), self.Rxf.sum()+self.Rh_d_first_l1.sum()+self.Rh_d_first_l0.sum()+self.Ry_0.sum()))
        
        
    def lrp_encoder(self,x,eps=0.001,bias_factor=1.0):
        """
        LRP for the encoder module.
        Args:
        - x:            historical input sequence       
        - eps:      stabilizer (small positive number)
        - bias_factor:  set to 1.0 to check global relevance conservation, otherwise use 0.0 to ignore bias/stabilizer redistribution (recommended)
        Returns:
        - RX:           relevance scores of encoder inputs
        """
        h=self.hidden_size
        ####relevance matrice initializations encoder#####
        Rh_e,R_newgate_e,self.Rx,Rx_e= init_encoder(self.num_layers,self.in_seq_len,self.hidden_size,x.shape)
        #### move to numpy #####
        a1=[self.hs_encoder, self.updategate_enc, self.newgate_enc, self.resetgate_enc, self.pre_gate_enc, self.wi_l0_enc, self.wi_l1_enc, self.wh_l1_enc, self.wh_l0_enc, self.bi_l0_enc ,self.bi_l1_enc , self.bh_l0_enc ,self.bh_l1_enc,x]
        
        [hs_encoder, updategate_enc, newgate_enc, resetgate_enc, pre_gate_enc, wi_l0_enc, wi_l1_enc, wh_l1_enc, wh_l0_enc, bi_l0_enc ,bi_l1_enc , bh_l0_enc ,bh_l1_enc,x]= list(map(moveToNumpy,a1))
        

        ####weight chuncks#####
        T=self.in_seq_len-1
        wi_l1_e=np.split(wi_l1_enc,3) ##layer 1
        bi_l1_e=np.split(bi_l1_enc,3)
        wh_l1_e=np.split(wh_l1_enc,3) 
        bh_l1_e=np.split(bh_l1_enc,3)

        wh_l0_e=np.split(wh_l0_enc,3) ##layer 0
        bh_l0_e=np.split(bh_l0_enc,3)
        wi_l0_e=np.split(wi_l0_enc,3)
        bi_l0_e=np.split(bi_l0_enc,3)


        
        
        ##relevance propagation-gru(encoder last step layer 1)
        Rh_e[1,[T-1]], Rx_e[T]= gru_lrp(updategate_enc[1,T], newgate_enc[1,T], resetgate_enc[1,T], hs_encoder[1,T-1], hs_encoder[0,T], wi_l1_e[2], wh_l1_e[2], bi_l1_e[2], bh_l1_e[2], hs_encoder[1,T], pre_gate_enc[1,T][2*h:], self.Rh_d_first_l1, eps, bias_factor)
        

        ##relevance propagation-gru(encoder last step layer 0)
        X_encoder=x[0,[T]]
        X_encoder=np.squeeze(X_encoder,axis=0)

        Rh_e[0,[T-1]], self.Rx[T]= gru_lrp(updategate_enc[0,T], newgate_enc[0,T], resetgate_enc[0,T], hs_encoder[0,T-1], X_encoder, wi_l0_e[2], wh_l0_e[2], bi_l0_e[2], bh_l0_e[2], hs_encoder[0,T], pre_gate_enc[0,T][2*h:], Rx_e[T]+self.Rh_d_first_l0, eps, bias_factor)
        

        for t in reversed(range(self.in_seq_len-1)):
            if t>0:
                ##relevance propagation-gru(encoder layer 1)
                Rh_e[1,[t-1]], Rx_e[t]= gru_lrp(updategate_enc[1,t], newgate_enc[1,t], resetgate_enc[1,t], hs_encoder[1,t-1], hs_encoder[0,t], wi_l1_e[2], wh_l1_e[2], bi_l1_e[2], bh_l1_e[2], hs_encoder[1,t], pre_gate_enc[1,t][2*h:], Rh_e[1,t], eps, bias_factor)

                ##relevance propagation-gru(encoder layer 0)
                X_encoder=x[0,[t]]
                X_encoder=np.squeeze(X_encoder,axis=0)
                
                Rh_e[0,[t-1]], self.Rx[t]= gru_lrp(updategate_enc[0,t], newgate_enc[0,t], resetgate_enc[0,t], hs_encoder[0,t-1], X_encoder, wi_l0_e[2], wh_l0_e[2], bi_l0_e[2], bh_l0_e[2], hs_encoder[0,t], pre_gate_enc[0,t][2*h:],Rh_e[0,t]+Rx_e[t] , eps, bias_factor)


            else:
                ##relevance propagation-gru(encoder first step layer 1)
                hs_encoder_l1_init=self.hs_init.detach().cpu().numpy().astype(np.float128, copy=False) 
                hs_encoder_l0_init=self.hs_init.detach().cpu().numpy().astype(np.float128, copy=False) 
                
                Rh_e_first_l1, Rx_e[t]= gru_lrp(updategate_enc[1,t], newgate_enc[1,t], resetgate_enc[1,t], hs_encoder_l1_init, hs_encoder[0,t], wi_l1_e[2], wh_l1_e[2], bi_l1_e[2], bh_l1_e[2], hs_encoder[1,t], pre_gate_enc[1,t][2*h:], Rh_e[1,t], eps, bias_factor) 
                

                ##relevance propagation-gru(encoder first step layer 0)
                X_encoder=x[0,[t]]
                X_encoder=np.squeeze(X_encoder,axis=0)
                
                Rh_e_first_l0, self.Rx[t]= gru_lrp(updategate_enc[0,t], newgate_enc[0,t], resetgate_enc[0,t], hs_encoder_l0_init, X_encoder, wi_l0_e[2], wh_l0_e[2], bi_l0_e[2], bh_l0_e[2], hs_encoder[0,t], pre_gate_enc[0,t][2*h:],Rh_e[0,t]+Rx_e[t] , eps, bias_factor)
        if bias_factor==1:
            if self.len_static!=0:
                print('sanity check encoder\n',self.Rout.sum(),'in comparison with', self.Rx.sum()+self.Rxf.sum()+Rh_e_first_l0.sum()+Rh_e_first_l1.sum()+self.Ry_0.sum()+self.Rstatic.sum(),
             '\nsanityCheckResult', np.allclose(self.Rx.sum()+self.Rxf.sum()+Rh_e_first_l0.sum()+Rh_e_first_l1.sum()+self.Ry_0.sum()+self.Rstatic.sum(), self.Rout.sum()))
            else:
                print('sanity check encoder\n',self.Rout.sum(),'in comparison with', self.Rx.sum()+self.Rxf.sum()+Rh_e_first_l0.sum()+Rh_e_first_l1.sum()+self.Ry_0.sum(),
             '\nsanityCheckResult', np.allclose(self.Rx.sum()+self.Rxf.sum()+Rh_e_first_l0.sum()+Rh_e_first_l1.sum()+self.Ry_0.sum(), self.Rout.sum()))


    
    def lrp_seq2seq(self,x,xf,static,mask_output,epsilon,bias_factor):
        """
        LRP flows
        Returns:
        - y:            pollution forecast for next 48 hours
        - Rx:           relevance scores of historical input sequence
        - Rxf:          relevance scores of forecast input sequence
        - Ry:           relevance scores of prediction outputs in previous steps
        - Rstatic:      relevance scores of static data(when we have static data)
        """
        self.forward_pass(x,xf,static)
        self.lrp_decoder(x,xf,mask_output,epsilon,bias_factor)
        self.lrp_encoder(x,epsilon,bias_factor)
        return self.y,self.Rxf,self.Rx,self.Ry,self.Rstatic