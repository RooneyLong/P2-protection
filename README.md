#  Poisoning as a Post-Protection: Preventing Membership Privacy Leakage From Gradient and Prediction of Federated Models (P2-Protection) 


## About The Project
**P2-protection** is a privacy defense against membership inference attack. We use controled perturbations to poison the FL model. To poison the prediction and gradient of a target FL model's training data, we train the target model through an additional FL training round and embed the poisoned information into the target model. Then P2-Protection could remove the membership information contained in the model's prediction and gradient, thus significantly degrading the performance of existing membership inference attacks. 


## Getting Started
### Prerequisites
**MeFA** requires the following packages: 
- Python 3.8.0
- Pytorch 1.7.1
- Sklearn 0.24.2
- Numpy 1.19.2
- torchvison 0.9.1
- pandas 1.19.2

### File Structure 
```
P2-protection for Adult
├── dataset
│   ├── adult
│   ├──dataset_purchase
│   └── mnist-original.mat
├── defend
│   ├── defend_adult.py
│   └──  defend_adult_aggragator.py
├── FL_model_generation
│   └── FL_model_for_adult.py
├── menif_attack
│   ├──attack_data.py
│   ├── attack_model_init.py
│   ├── attack_utils.py
│   ├── create_cnn.py
│   ├── create_encoder.py
│   ├── create_fcn.py
│   ├── logger.py
│   ├── losses.py
│   ├── member_inf.py
│   ├── membership_attack_adult.py
│   ├── membership_attack_adult_withDP.py
│   ├── optimizers.py
│   └── losses.py
├── model
│   ├── adult
│   └── adult_defend
├── other_attack_and_defend
│   ├── ML_leaks_attack_adult.py
│   └── shokri_attack_adult.py
├── data_Partitioning_adult
├── data_preprocessing.py
└── README.md



There are several parts of the code:
- dataset folder: This folder is used to save the original dataset. In order to reduce the memory space, we just list the links to theset dataset here. 
   -- Adult: https://archive.ics.uci.edu/ml/datasets/Adult
   -- Purchase: https://github.com/privacytrustlab/datasets/blob/master/dataset_purchase.tgz
   -- MNIST: http://yann.lecun.com/exdb/mnist/
   -- FEMNIST: https://github.com/TalwalkarLab/leaf
   -- CIFAR-10: http://www.cs.toronto.edu/~kriz/cifar.html


- data_Partitioning_adult.py: According to the number of FL clients, divide the dataset equally.
- ./FL_model_generation/FL_model_for_adult.py: The whole procedure of FL training, including loading data, defining the model, local training and FL aggregation.
- ./defend/defend_adult.py: Remove the local membership information by poisoning the gradient of the local model and aggregate to get the protected FL global model.
- ./defend/defend_adult_aggragator.py: The main process of P2-Protection, including local membership information removal and updating FL global model.
- ./menif_attack/membership_attack_adult.py: Use membership inference attacks to attack the trained FL model with and without P2-Protection, including gradient attack and shadow attack.
- main.py: The main function of P2-Protection. 



Execute P2-Protection by running the above files.

# Notes
- There is no need to use a GPU for the model training. 
- If you want to change the model setting, please modify the  corresosponding function in the python files.




