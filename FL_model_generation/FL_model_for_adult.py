# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import copy


x_train_all = pd.read_csv('../dataset/adult/x_train_adult_all.csv')
y_train_all = pd.read_csv('../dataset/adult/y_train_adult_all.csv')

x_test_all = pd.read_csv('../dataset/adult/x_test_adult_all.csv')
y_test_all = pd.read_csv('../dataset/adult/y_test_adult_all.csv')



class Net(torch.nn.Module):
    def __init__(self, n_feature):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_feature,500)
        self.fc2 = nn.Linear(500,300)
        self.fc3 = nn.Linear(300,2)
        
        self.grad_list = []
        self.layer_list = []        
        self.layer_output_list = []
        

    def forward(self, x):      
        x = F.relu(self.fc1(x))        
        x = F.relu(self.fc2(x))        
        x = self.fc3(x)         
        return x

    def eval_layer_output(self, x):
        x = F.relu(self.fc1(x))
        self.layer_output_list.append(x)
        x = F.relu(self.fc2(x))
        self.layer_output_list.append(x)
        x = self.fc3(x)
        self.layer_output_list.append(x)
        return self.layer_output_list, x

    def get_grad(self, list=[]):        
        return self.grad_list

net_g = Net(14)
    

num_client = 10


g_iteration_count = 10

l_iteration_count = 5

num_model_layer =  3  
 
for g_epoch in range(g_iteration_count):
    cur_all_client = []
    
    for client_i in range(num_client):
         print('######each client start to download global net_g ######')
         print('######this is client_{} in g_count_{}######'.format(client_i,g_epoch))    
         cur_net = copy.deepcopy(net_g)
         
         
         x_train = pd.read_csv('../dataset/adult/x_train_adult_{}.csv'.format(client_i))
         y_train = pd.read_csv('../dataset/adult/y_train_adult_{}.csv'.format(client_i))

         x_test = pd.read_csv('../dataset/adult/x_test_adult_{}.csv'.format(client_i))
         y_test = pd.read_csv('../dataset/adult/y_test_adult_{}.csv'.format(client_i))
         
         
         x_train = np.array(x_train)
         x_train = torch.FloatTensor(x_train)
        
         y_train = np.array(y_train)
         y_train = torch.LongTensor(y_train).squeeze()
        
        
         torch_dataset = torch.utils.data.TensorDataset(x_train, y_train)
         train_loader = torch.utils.data.DataLoader(torch_dataset, 
                                                   batch_size=128, 
                                                  shuffle=False)     
         len_cur_x_train = len(x_train)

         loss_func = torch.nn.CrossEntropyLoss()
         model_opt = optim.SGD(params=cur_net.parameters(),lr=0.005)


         for epoch in range(l_iteration_count):
             for step, data in enumerate(train_loader, start=0):
                 x_feature, y_feature = data 
                 output = cur_net(x_feature)     # input x and predict based on x
                 loss = loss_func(output, y_feature)     # must be (1. nn output, 2. target)        

                 model_opt.zero_grad()   # clear gradients for next train
                 loss.backward()         # backpropagation, compute gradients
                 model_opt.step()        # apply gradients
        
                 if step % (int(len_cur_x_train/128)) == (int(len_cur_x_train/128))-1: # 
                     prediction = torch.max(output, 1)[1]
                     pred = prediction.data.numpy()
                     target = y_feature.data.numpy()
                     accuracy = float((pred == target).astype(int).sum()) / float(target.size)
                     print("epoch={},step={} train_result model{} = {}".format(epoch, step, client_i, accuracy))
         
         print('save current client_{}'.format(client_i))                
         x_test = np.array(x_test)
         x_test = torch.FloatTensor(x_test)
        
         y_test = np.array(y_test)
         y_test = torch.LongTensor(y_test).squeeze() 
         
         out_test = cur_net(x_test)     # input x and predict based on x
         prediction = torch.max(out_test, 1)[1]
         pred_y = prediction.data.numpy()
         target_y = y_test.data.numpy()
         accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
         print('test_result_currrent_client_model =', accuracy)
         
         torch.save(cur_net, '../model/adult/client_{}_g_count_{}.pkl'.format(client_i,g_epoch))  # save entire net
         # torch.save(cur_net.state_dict(),'../model/adult/client_{}_g_count_{}_params.pkl'.format(client_i, g_epoch))
     
         cur_all_client.append(cur_net)

    print('##### start global_model_once_in_g_count_{}'.format(g_epoch))
    

    item_name_arr = []
    for i in range(num_model_layer):
        item_name_arr.append('fc{}.weight'.format(i+1))
    for i in range(num_model_layer):
        item_name_arr.append('fc{}.bias'.format(i+1))

    model_dict  = net_g.state_dict()
    

    for i in range(len(item_name_arr)):
        cur_name = item_name_arr[i]

        
        for j in range(len(cur_all_client)):
            net_cur_ele = cur_all_client[j]        
            if j == 0 :
                sum_item = net_cur_ele.state_dict()[cur_name]
            else:
                sum_item = sum_item + net_cur_ele.state_dict()[cur_name]
                

        model_dict[cur_name] = sum_item/(num_client*1.0)

    net_g.load_state_dict(model_dict)
    print('##### finish global_model_once_in_g_count_{}'.format(g_epoch)) 
    
    torch.save(net_g, '../model/adult/global_model_g_count_{}.pkl'.format(g_epoch))  # save entire net
    # torch.save(net_g.state_dict(),'../model/adult/global_model_g_count_{}_params.pkl'.format(g_epoch))
          

    x_test_g = np.array(x_test_all)
    x_test = torch.FloatTensor(x_test_g)
    y_test_g = np.array(y_test_all)
    y_test = torch.LongTensor(y_test_g).squeeze()    
    
    out_test = net_g(x_test)     # input x and predict based on x
    prediction = torch.max(out_test, 1)[1]
    pred_y = prediction.data.numpy()
    target_y = y_test.data.numpy()
    accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
    print("################# test_result_net_g_model_in_the_g_count_{} = ".format(g_epoch),accuracy)


        
        
    
    