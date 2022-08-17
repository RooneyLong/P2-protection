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

#################


x_train_all = pd.read_csv('../dataset/adult/x_train_adult_all.csv')
y_train_all = pd.read_csv('../dataset/adult/y_train_adult_all.csv')

x_test_all = pd.read_csv('../dataset/adult/x_test_adult_all.csv')
y_test_all = pd.read_csv('../dataset/adult/y_test_adult_all.csv')




data_size = 2000
x_target_dataset = x_train_all.iloc[:2*data_size]
y_target_dataset = y_train_all.iloc[:2*data_size]

x_shadow_dataset = x_train_all.iloc[4*data_size:6*data_size]
y_shadow_dataset = y_train_all.iloc[4*data_size:6*data_size]
x_shadow_dataset_arr = np.array(x_shadow_dataset)
x_shadow_dataset = torch.FloatTensor(x_shadow_dataset_arr)

y_shadow_dataset_arr = np.array(y_shadow_dataset)
y_shadow_dataset = torch.LongTensor(y_shadow_dataset_arr).squeeze()

# D_test
x_test_dataset = x_test_all.iloc[2*data_size:4*data_size]
y_test_dataset = y_test_all.iloc[2*data_size:4*data_size]
# 转成Tensor形式
x_test_dataset_arr = np.array(x_test_dataset)
x_test_dataset = torch.FloatTensor(x_test_dataset_arr)

y_shadow_dataset_arr = np.array(y_test_dataset)
y_test_dataset = torch.LongTensor(y_shadow_dataset_arr).squeeze()


d_si_x_set = []
d_si_y_set = []

d_ti_x_set = []
d_ti_y_set = []

num_dataset = 9
d_data_size = data_size

for i in range(num_dataset):    
    cur_start = random.randint(0,d_data_size-1)
    cur_end = cur_start + d_data_size
    # D_si
    cur_sub_shadow_x =  x_shadow_dataset[cur_start:cur_end]
    cur_sub_shadow_y =  y_shadow_dataset[cur_start:cur_end]
    d_si_x_set.append(cur_sub_shadow_x)
    d_si_y_set.append(cur_sub_shadow_y)
    
    # D_ti
    cur_sub_test_x =  x_test_dataset[cur_start:cur_end]
    cur_sub_test_y =  y_test_dataset[cur_start:cur_end]
    d_ti_x_set.append(cur_sub_test_x)
    d_ti_y_set.append(cur_sub_test_y)

##################################
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


net_shadow0 = Net(n_feature = 14)
net_shadow1 = Net(n_feature = 14)
net_shadow2 = Net(n_feature = 14)
net_shadow3 = Net(n_feature = 14)
net_shadow4 = Net(n_feature = 14)
net_shadow5 = Net(n_feature = 14)
net_shadow6 = Net(n_feature = 14)
net_shadow7 = Net(n_feature = 14)
net_shadow8 = Net(n_feature = 14)

net_arr = []
net_arr.append(net_shadow0)
net_arr.append(net_shadow1)
net_arr.append(net_shadow2)
net_arr.append(net_shadow3)
net_arr.append(net_shadow4)
net_arr.append(net_shadow5)
net_arr.append(net_shadow6)
net_arr.append(net_shadow7)
net_arr.append(net_shadow8)

len_data_len_shadow = len(d_si_x_set[0])

for i in range(num_dataset):
    cur_count_shadow = i
    net_shadow = net_arr[i]
    
    torch_dataset = torch.utils.data.TensorDataset(d_si_x_set[cur_count_shadow], d_si_y_set[cur_count_shadow])
    train_loader = torch.utils.data.DataLoader(torch_dataset, 
                                               batch_size=128, 
                                              shuffle=False)
    loss_func = torch.nn.CrossEntropyLoss()
    la_opt = optim.Adam(params=net_shadow.parameters(),lr=0.005)
    iteration_count = 8 
    
    for epoch in range(iteration_count):
        for step, data in enumerate(train_loader, start=0):
            x_a, y_a = data # 解构出特征和标签
            out_a = net_shadow(x_a)     # input x and predict based on x
            loss_a = loss_func(out_a, y_a)     # must be (1. nn output, 2. target)        
            la_opt.zero_grad()   # clear gradients for next train
            loss_a.backward()         # backpropagation, compute gradients
            la_opt.step()        # apply gradients
           
            if step % (int(len_data_len_shadow/128)) == (int(len_data_len_shadow/128)) - 1: #data_size/batch_size 为batch的个数
                prediction_a = torch.max(out_a, 1)[1]#只返回最大值的每个索引,标签
                pred_a = prediction_a.data.numpy()
                target_a = y_a.data.numpy()
                accuracy_a = float((pred_a == target_a).astype(int).sum()) / float(target_a.size)
                print("epoch={},step={}, train_result_shadow_model_{} = {}".format(epoch, step, cur_count_shadow, accuracy_a))

            
#############################################################################
x_attack_train_set = []
y_attack_train_set = []

for i in range(num_dataset):
    
    cur_count_shadow = i
    net_shadow = net_arr[cur_count_shadow]
    
    # D_si为member,D_ti为nonmember          
    x_member_train = d_si_x_set[cur_count_shadow]
    y_member = torch.LongTensor(torch.ones_like(d_si_y_set[cur_count_shadow]))
    
    x_nonmember_train = d_ti_x_set[cur_count_shadow]
    y_nonmember = torch.LongTensor(torch.zeros_like(d_ti_y_set[cur_count_shadow]))
    
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
    
    x_attack_train = torch.cat((x_member,x_nonmember),0) 
    y_attack_train = torch.cat((y_member,y_nonmember),0) 
    
    x_attack_train_set.append(x_attack_train)
    y_attack_train_set.append(y_attack_train)

###############################################################################

class Net_two_class(torch.nn.Module):
    def __init__(self, n_feature=2, n_hidden=6, n_output=2):
        super(Net_two_class, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.out = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.out(x)
        return x

net_attack_model0 = Net_two_class(n_feature=2,n_hidden=5,n_output=2)
net_attack_model1 = Net_two_class(n_feature=2,n_hidden=5,n_output=2)
net_attack_model2 = Net_two_class(n_feature=2,n_hidden=5,n_output=2)
net_attack_model3 = Net_two_class(n_feature=2,n_hidden=5,n_output=2)
net_attack_model4 = Net_two_class(n_feature=2,n_hidden=5,n_output=2)
net_attack_model5 = Net_two_class(n_feature=2,n_hidden=5,n_output=2)
net_attack_model6 = Net_two_class(n_feature=2,n_hidden=5,n_output=2)
net_attack_model7 = Net_two_class(n_feature=2,n_hidden=5,n_output=2)
net_attack_model8 = Net_two_class(n_feature=2,n_hidden=5,n_output=2)

net_attack_arr = []
net_attack_arr.append(net_attack_model0)
net_attack_arr.append(net_attack_model1)
net_attack_arr.append(net_attack_model2)
net_attack_arr.append(net_attack_model3)
net_attack_arr.append(net_attack_model4)
net_attack_arr.append(net_attack_model5)
net_attack_arr.append(net_attack_model6)
net_attack_arr.append(net_attack_model7)
net_attack_arr.append(net_attack_model8)

x_attack_train_all = []
y_attack_train_all = []

for i in range(num_dataset):
    x_attack_train_all.extend(x_attack_train_set[i].numpy())
    y_attack_train_all.extend(y_attack_train_set[i].numpy())

len_data_len = len(x_attack_train_all)
    
x_attack_train_all = torch.FloatTensor(x_attack_train_all)
y_attack_train_all = torch.LongTensor(y_attack_train_all)


for i in range(num_dataset):
    
    cur_count_attack = i
    net_attack_model = net_attack_arr[cur_count_attack]
    
    torch_dataset = torch.utils.data.TensorDataset(x_attack_train_all, y_attack_train_all)
    train_loader = torch.utils.data.DataLoader(torch_dataset, 
                                               batch_size=256, 
                                              shuffle=False)    
    loss_func = torch.nn.CrossEntropyLoss()
    # m_attack_opt = optim.Adam(params=net_attack_model.parameters(),lr=0.0005) 
    m_attack_opt = optim.SGD(params=net_attack_model.parameters(),lr=0.0045)
    scheduler = lr_scheduler.LambdaLR(optimizer=m_attack_opt, lr_lambda=lambda epoch:0.95**epoch)
    iteration_count = 10
    
    for epoch in range(iteration_count):
        for step, data in enumerate(train_loader, start=0):
            x_a, y_a = data # 解构出特征和标签
            out_a = net_attack_model(x_a)     # input x and predict based on x
            loss_a = loss_func(out_a, y_a)     # must be (1. nn output, 2. target)        
            # print(loss_a)
            m_attack_opt.zero_grad()   # clear gradients for next train
            loss_a.backward()         # backpropagation, compute gradients
            m_attack_opt.step()        # apply gradients  
           
            # if step % (int(len_data_len/256)) == (int(len_data_len/256)) - 1: # #size(x_attack_train_all)/batch_size 为batch的个数
            if step != 0:
                prediction_a = torch.max(out_a, 1)[1]
                pred_a = prediction_a.data.numpy()
                target_a = y_a.data.numpy()
                accuracy_a = float((pred_a == target_a).astype(int).sum()) / float(target_a.size)
                print("epoch={},step={}, train_result_attack_model_{} = {}".format(epoch, step,cur_count_attack, accuracy_a))
        scheduler.step() 
    # sad   
        
        
###############################  


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

result = []
for i in range(num_dataset):
    
    cur_count_attack = i
    net_attack_model = net_attack_arr[cur_count_attack]
     
    out_test = net_attack_model(x_test_m_attack)     # input x and predict based on x
    prediction = torch.max(out_test, 1)[1]
    pred_y = prediction.data.numpy()
    
    result.append(pred_y)

count_arr_add = result[0]
for i in range(len(result)-1):
    count_arr_add += result[i+1]

count_arr_add = pd.DataFrame(count_arr_add)    

def voting(x):
    if x>=5:
        return 1
    else:
        return 0

pred_y_final = count_arr_add.applymap(voting)
target_y = y_test_m_attack.data.numpy()

classify_report = metrics.classification_report(target_y, pred_y_final)
print('classify_report about un-protect result:\n',classify_report)

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


result = []
for i in range(num_dataset):
    
    cur_count_attack = i
    net_attack_model = net_attack_arr[cur_count_attack]
     
    out_test = net_attack_model(x_test_m_attack)     # input x and predict based on x
    prediction = torch.max(out_test, 1)[1]
    pred_y = prediction.data.numpy()
    
    result.append(pred_y)

count_arr_add = result[0]
for i in range(len(result)-1):
    count_arr_add += result[i+1]

count_arr_add = pd.DataFrame(count_arr_add)    

def voting(x):
    if x>=5:
        return 1
    else:
        return 0

pred_y_final = count_arr_add.applymap(voting)
target_y = y_test_m_attack.data.numpy()

classify_report = metrics.classification_report(target_y, pred_y_final)
print('classify_report about DP_scheme result:\n',classify_report)

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


result = []
for i in range(num_dataset):
    
    cur_count_attack = i
    net_attack_model = net_attack_arr[cur_count_attack]
     
    out_test = net_attack_model(x_test_m_attack)     # input x and predict based on x
    prediction = torch.max(out_test, 1)[1]
    pred_y = prediction.data.numpy()
    
    result.append(pred_y)

count_arr_add = result[0]
for i in range(len(result)-1):
    count_arr_add += result[i+1]

count_arr_add = pd.DataFrame(count_arr_add)    

def voting(x):
    if x>=5:
        return 1
    else:
        return 0

pred_y_final = count_arr_add.applymap(voting)
target_y = y_test_m_attack.data.numpy()

classify_report = metrics.classification_report(target_y, pred_y_final)
print('classify_report about top1_scheme result:\n',classify_report)


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


result = []
for i in range(num_dataset):
    
    cur_count_attack = i
    net_attack_model = net_attack_arr[cur_count_attack]
     
    out_test = net_attack_model(x_test_m_attack)     # input x and predict based on x
    prediction = torch.max(out_test, 1)[1]
    pred_y = prediction.data.numpy()
    
    result.append(pred_y)

count_arr_add = result[0]
for i in range(len(result)-1):
    count_arr_add += result[i+1]

count_arr_add = pd.DataFrame(count_arr_add)    

def voting(x):
    if x>=5:
        return 1
    else:
        return 0

pred_y_final = count_arr_add.applymap(voting)
target_y = y_test_m_attack.data.numpy()
classify_report = metrics.classification_report(target_y, pred_y_final)
print('classify_report about ours-protect result:\n',classify_report)
