# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from sklearn import metrics
import random
from torch.optim import lr_scheduler

x_train_all = pd.read_csv('../dataset/adult/x_train_adult_all.csv')
y_train_all = pd.read_csv('../dataset/adult/y_train_adult_all.csv')

x_test_all = pd.read_csv('../dataset/adult/x_test_adult_all.csv')
y_test_all = pd.read_csv('../dataset/adult/y_test_adult_all.csv')


data_size = 3000

x_target_dataset = x_train_all.iloc[:2*data_size]
y_target_dataset = y_train_all.iloc[:2*data_size]

# D_shadow
x_shadow_dataset = x_train_all.iloc[2*data_size:4*data_size]
y_shadow_dataset = y_train_all.iloc[2*data_size:4*data_size]
x_shadow_dataset_arr = np.array(x_shadow_dataset)
x_shadow_all = torch.FloatTensor(x_shadow_dataset_arr)

y_shadow_dataset_arr = np.array(y_shadow_dataset)
y_shadow_all = torch.LongTensor(y_shadow_dataset_arr).squeeze()

# D_test
x_test_dataset = x_test_all.iloc[:2*data_size]
y_test_dataset = y_test_all.iloc[:2*data_size]
x_test_dataset_arr = np.array(x_test_dataset)
x_test_dataset = torch.FloatTensor(x_test_dataset_arr)

y_shadow_dataset_arr = np.array(y_test_dataset)
y_test_dataset = torch.LongTensor(y_shadow_dataset_arr).squeeze()


len_shadow_train = int(len(x_shadow_all)/2)

x_shadow_train  = x_shadow_all[0:len_shadow_train]
y_shadow_train = y_shadow_all[0:len_shadow_train]

x_shadow_out = x_shadow_all[len_shadow_train:]
y_shadow_out = y_shadow_all[len_shadow_train:]




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

net_shadow = Net(n_feature = 14)


torch_dataset = torch.utils.data.TensorDataset(x_shadow_train, y_shadow_train)
train_loader = torch.utils.data.DataLoader(torch_dataset, 
                                           batch_size=100, 
                                          shuffle=False)

loss_func = torch.nn.CrossEntropyLoss()
la_opt = optim.Adam(params=net_shadow.parameters(),lr=0.005)
iteration_count = 7

len_data_len_shadow_train = len(x_shadow_train)

for epoch in range(iteration_count):
    for step, data in enumerate(train_loader, start=0):
        x_a, y_a = data 
        out_a = net_shadow(x_a)     # input x and predict based on x
        loss_a = loss_func(out_a, y_a)     # must be (1. nn output, 2. target)        
        la_opt.zero_grad()   # clear gradients for next train
        loss_a.backward()         # backpropagation, compute gradients
        la_opt.step()        # apply gradients

        if step % (int(len_data_len_shadow_train/100)) == (int(len_data_len_shadow_train/100)) - 1: 
            prediction_a = torch.max(out_a, 1)[1]
            pred_a = prediction_a.data.numpy()
            target_a = y_a.data.numpy()
            accuracy_a = float((pred_a == target_a).astype(int).sum()) / float(target_a.size)
            print("epoch={}, train_result_shadow_model = {}".format(epoch, accuracy_a))

          
 
x_member_train = x_shadow_train
y_member = torch.LongTensor(torch.ones_like(y_shadow_train))

x_nonmember_train = x_shadow_out
y_nonmember = torch.LongTensor(torch.zeros_like(y_shadow_out))


x_member_all = F.softmax(net_shadow(x_member_train),dim=1)
x_member = []
for i in range(len(x_member_all)):
    x_member.append((x_member_all[i].sort(descending=True)).values[0:2].detach().numpy())
x_member = torch.tensor(x_member)

x_nonmember_all = F.softmax(net_shadow(x_nonmember_train),dim=1)
x_nonmember = []
for i in range(len(x_nonmember_all)):
    x_nonmember.append((x_nonmember_all[i].sort(descending=True)).values[0:2].detach().numpy())
x_nonmember = torch.tensor(x_nonmember)

# ????????????attack model????????????
x_attack_train = torch.cat((x_member,x_nonmember),0) 
y_attack_train = torch.cat((y_member,y_nonmember),0) 


class Net_two_class(torch.nn.Module):
    def __init__(self, n_feature=2, n_hidden=6, n_output=2):
        super(Net_two_class, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.out = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.out(x)
        return x


net_attack_model = Net_two_class(n_feature=2,n_hidden=5,n_output=2)


torch_dataset = torch.utils.data.TensorDataset(x_attack_train, y_attack_train)
train_loader = torch.utils.data.DataLoader(torch_dataset, 
                                           batch_size=256, 
                                          shuffle=False)

loss_func = torch.nn.CrossEntropyLoss()
m_attack_opt = optim.SGD(params=net_attack_model.parameters(),lr=0.0035)
scheduler = lr_scheduler.LambdaLR(optimizer=m_attack_opt, lr_lambda=lambda epoch:0.95**epoch)


iteration_count = 3

len_data_len_attack_train = len(x_attack_train)


for epoch in range(iteration_count):
    for step, data in enumerate(train_loader, start=0):
        x_a, y_a = data 
        out_a = net_attack_model(x_a)     # input x and predict based on x
        loss_a = loss_func(out_a, y_a)     # must be (1. nn output, 2. target)        
#        print(loss_a)
        m_attack_opt.zero_grad()   # clear gradients for next train
        loss_a.backward()         # backpropagation, compute gradients
        m_attack_opt.step()        # apply gradients

        # if step % (int(len_data_len_attack_train/256)) == (int(len_data_len_attack_train/256)) - 1: 
        if step != 0 :
            prediction_a = torch.max(out_a, 1)[1]#?????????????????????????????????,??????
            pred_a = prediction_a.data.numpy()
            target_a = y_a.data.numpy()
            accuracy_a = float((pred_a == target_a).astype(int).sum()) / float(target_a.size)
            print("epoch={},step={} train_result_attack_model = {}".format(epoch,step, accuracy_a))
    scheduler.step() 



################################

x_target_train_arr = np.array(x_target_dataset)
x_target_train = torch.FloatTensor(x_target_train_arr)
y_target_dataset_arr = np.array(y_target_dataset)
y_target_train = torch.LongTensor(y_target_dataset_arr).squeeze()

net_target = torch.load('../model/adult/global_model_g_count_9.pkl')    
y_org_nonmember =  torch.LongTensor(y_test_dataset)
y_nonmember_test = torch.zeros_like(y_org_nonmember)


y_org_member = y_target_train
y_member_test = torch.ones_like(y_org_member)


x_test_member = x_target_train
x_test_nonmember = x_test_dataset


x_member_all = F.softmax(net_target(x_test_member),dim=1)
x_member = []
for i in range(len(x_member_all)):
    x_member.append((x_member_all[i].sort(descending=True)).values[0:2].detach().numpy())
x_member_test = torch.tensor(x_member)


x_nonmember_all = F.softmax(net_target(x_test_nonmember),dim=1)
x_nonmember = []
for i in range(len(x_nonmember_all)):
    x_nonmember.append((x_nonmember_all[i].sort(descending=True)).values[0:2].detach().numpy())
x_nonmember_test = torch.tensor(x_nonmember)
         
x_test_m_attack = torch.cat((x_member_test,x_nonmember_test),0)
y_test_m_attack = torch.cat((y_member_test,y_nonmember_test),0)



out_test = net_attack_model(x_test_m_attack)     # input x and predict based on x
prediction = torch.max(out_test, 1)[1]
pred_y = prediction.data.numpy()
target_y = y_test_m_attack.data.numpy()

classify_report = metrics.classification_report(target_y, pred_y)
print('classify_report about un-protect result:\n',classify_report)

#############################
epsilon = 0.3
mu = 0
sigma = 1/epsilon

pred = net_target(x_test_nonmember) 
[row, col] = pred.size()
added_noise = []
for i in range(row):
    cur_row_noise = []
    for j in range(col):
        cur_row_noise.append(random.gauss(mu,sigma))
    added_noise.append(cur_row_noise)
added_noise = torch.tensor(added_noise)
pred = pred + added_noise

x_nonmember_all = F.softmax(pred,dim=1)
x_nonmember = []
for i in range(len(x_nonmember_all)):
    x_nonmember.append((x_nonmember_all[i].sort(descending=True)).values[0:2].detach().numpy())
x_nonmember_test = torch.tensor(x_nonmember)


pred = net_target(x_test_member)
[row, col] = pred.size()
added_noise = []
for i in range(row):
    cur_row_noise = []
    for j in range(col):
        cur_row_noise.append(random.gauss(mu,sigma))
    added_noise.append(cur_row_noise)
added_noise = torch.tensor(added_noise)
pred = pred + added_noise

x_member_all = F.softmax(pred,dim=1)
x_member = []
for i in range(len(x_member_all)):
    x_member.append((x_member_all[i].sort(descending=True)).values[0:2].detach().numpy())
x_member_test = torch.tensor(x_member)


x_test_m_attack = torch.cat((x_member_test,x_nonmember_test),0)
y_test_m_attack = torch.cat((y_member_test,y_nonmember_test),0)


out_test = net_attack_model(x_test_m_attack)     # input x and predict based on x
prediction = torch.max(out_test, 1)[1]
pred_y = prediction.data.numpy()
target_y = y_test_m_attack.data.numpy()

classify_report = metrics.classification_report(target_y, pred_y)
print('classify_report about DP-scheme result:\n',classify_report)



#############################

pred = net_target(x_test_nonmember)

x_nonmember_all = F.softmax(pred,dim=1)
x_nonmember = []
for i in range(len(x_nonmember_all)):
    cur_ele = (x_nonmember_all[i].sort(descending=True)).values[0:1].detach().numpy()
    p_arr = np.concatenate((cur_ele,[0])) 
    cur_ele = p_arr
    # sad
    x_nonmember.append(cur_ele)
x_nonmember_test = torch.FloatTensor(x_nonmember)


pred = net_target(x_test_member)

x_member_all = F.softmax(pred,dim=1)
x_member = []
for i in range(len(x_member_all)):
    cur_ele = (x_member_all[i].sort(descending=True)).values[0:1].detach().numpy()
    p_arr = np.concatenate((cur_ele,[0]))
    cur_ele = p_arr    
    x_member.append(cur_ele)
x_member_test = torch.FloatTensor(x_member)


x_test_m_attack = torch.cat((x_member_test,x_nonmember_test),0)
y_test_m_attack = torch.cat((y_member_test,y_nonmember_test),0)


#x_test_m_attack = torch.tensor(x_test_m_attack, dtype=torch.float32)
out_test = net_attack_model(x_test_m_attack)     # input x and predict based on x
prediction = torch.max(out_test, 1)[1]
pred_y = prediction.data.numpy()
target_y = y_test_m_attack.data.numpy()

classify_report = metrics.classification_report(target_y, pred_y)
print('classify_report about top1-scheme result:\n',classify_report)

#############################
x_target_train_arr = np.array(x_target_dataset)
x_target_train = torch.FloatTensor(x_target_train_arr)
y_target_dataset_arr = np.array(y_target_dataset)
y_target_train = torch.LongTensor(y_target_dataset_arr).squeeze()


net_target = torch.load('../model/adult_defend/global/protected_global_model_g_count_9_layers123.pkl')    


y_org_nonmember =  torch.LongTensor(y_test_dataset)
y_nonmember_test = torch.zeros_like(y_org_nonmember)


y_org_member = y_target_train
y_member_test = torch.ones_like(y_org_member)

x_test_member = x_target_train
x_test_nonmember = x_test_dataset


x_member_all = F.softmax(net_target(x_test_member),dim=1)
x_member = []
for i in range(len(x_member_all)):
    x_member.append((x_member_all[i].sort(descending=True)).values[0:2].detach().numpy())
x_member_test = torch.tensor(x_member)

x_nonmember_all = F.softmax(net_target(x_test_nonmember),dim=1)
x_nonmember = []
for i in range(len(x_nonmember_all)):
    x_nonmember.append((x_nonmember_all[i].sort(descending=True)).values[0:2].detach().numpy())
x_nonmember_test = torch.tensor(x_nonmember)
         

x_test_m_attack = torch.cat((x_member_test,x_nonmember_test),0)
y_test_m_attack = torch.cat((y_member_test,y_nonmember_test),0)


out_test = net_attack_model(x_test_m_attack)     # input x and predict based on x
prediction = torch.max(out_test, 1)[1]
pred_y = prediction.data.numpy()
target_y = y_test_m_attack.data.numpy()

classify_report = metrics.classification_report(target_y, pred_y)
print('classify_report about ours-scheme result:\n',classify_report)
