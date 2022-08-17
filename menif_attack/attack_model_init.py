# -*- coding: utf-8 -*-


import torch
from torch import nn
from create_encoder import encoder

class attack_model_init(nn.Module):
    def __init__(self,attackinputs,encoderinputs_size,attackiputs_size=0):
         super(attack_model_init, self).__init__()
         

         self.len_attackinputs = len(attackinputs)
         for i in range(self.len_attackinputs):
             locals()["block" + str(i)] = attackinputs[i]            

         for i in range(self.len_attackinputs):
             strname = "Block" + str(i)          
             setattr(self,strname,locals()["block" + str(i)])
         self.encoder = encoder(encoderinputs_size) 
         self.attackiputs_size = attackiputs_size                   
             
    def forward(self,inputArray):
        for i in range(len(inputArray)):
            locals()["input" + str(i)] = inputArray[i]            
        
        
        outputlist = []
        for i in range(len(inputArray)):
            strname = "Block" + str(i)
            module_block = getattr(self,strname)
            result = module_block(locals()["input" + str(i)])
            outputlist.append(result)   
    
        encoder_input = torch.cat((outputlist),1)
        attackoutput = self.encoder(encoder_input)
        return attackoutput  
             

         
         