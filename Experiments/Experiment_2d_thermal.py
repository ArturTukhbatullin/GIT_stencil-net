import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
pd.set_option('display.float_format', '{:.3e}'.format)

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam,SGD
from torch.optim.lr_scheduler import ExponentialLR
import imageio
import os

import matplotlib.pyplot as plt


main_params=pd.read_excel('experiment_init_params_FIX_PARAMS2.xlsx',dtype={'has_backward':bool})


main_params=main_params.set_index('param')

display(main_params)

#------------------------MAIN PARAMS-----------------------------
# size of filter to be applied
fs = int(main_params.loc['fs'])

# number of timesteps to be predicted during training 
m = int(main_params.loc['m'])

# decaying weights
decay_const = float(main_params.loc['decay_const'])

# epoch_number
epochs=int(main_params.loc['epochs'])

#random_seed
seed = int(main_params.loc['seed'])

#coef для loss функции
l_wd= float(main_params.loc['l_wd'])

# 'RK3' for runge-kutta solver and 'E1' for euler solver
method=str(main_params.loc['method'].values[0]) 

#neurons num in MLPConv
neurons=int(main_params.loc['neurons'])

#learning_rate
lr=float(main_params.loc['lr'])

#train_size
train_split=float(main_params.loc['train_split'])

# on/off bwd
has_backward=eval(main_params.loc['has_backward'].values[0])

#hidden_layers_num
hidden_layers_num=int(main_params.loc['hidden_layers_num'])

N_TRIALS=int(main_params.loc['N_TRIALS'])

#device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device='cpu'
# device='cuda'


def make_param_table(net,my_doc_params,tau,h,n,t_factor,s_factor):

    my_params_dict={
        'fs':[fs],
        'neurons':neurons,
        'hidden_layers_num':hidden_layers_num,
        'act_func':str(net.sig),
        'epoch':epochs,
        'lr':lr,
        'tau':tau,
        'h':h,
        'n':n,
        'decay_const':decay_const,
        'm':m,
        't_factor':t_factor,
        's_factor':s_factor,
        'has_backward':has_backward,
        'train_size':train_split,
        'method':method,
        'layers':str(list(net.layer)),
    }
    
    INITIAL_SHAPE=len(my_params_dict)
    
    my_params_dict.update(my_doc_params)
    my_params_dict['fs']=[my_doc_params['fs']]
    
    assert len(my_params_dict)==INITIAL_SHAPE,'Dict length was changed!'
    
    
    pd.set_option("display.max_rows", None)
    pd.set_option("max_colwidth", None)
    my_params=pd.DataFrame(my_params_dict).T.reset_index()
    my_params.rename(columns={'index':'Parameter',0:'Value'},inplace=True)
    my_params#.set_index('Parameter')
    
    return my_params

### ===========================Класс нейронки===========================
class MLPConv(nn.Module):
    """
    MLPConv unit for STENCIL-NET.
    
    Keyword arguments:
    sizes -- layer sizes
    noise -- initial noise estimate for noisy data (default=None)
    seed -- seed for random network initialization (default=0)
    fs -- size of filters (default=7)
    activation -- activation function to be applied after linear transformations (default=torch.nn.ELU())
    """
    
    def __init__(self, sizes, noise=None, seed=0, fs=7, activation=nn.ELU()):
        super(MLPConv, self).__init__()
        
        torch.manual_seed(seed)
        
        gain = 5/3 if isinstance(activation, nn.Tanh) else 1
        
        self.fs    = fs
        self.sig   = activation
        self.layer = nn.ModuleList()

        for i in range(len(sizes)-1):
            linear = nn.Linear(in_features=sizes[i], out_features=sizes[i+1],bias=True)
            
            print("input", sizes[i], "output", sizes[i+1])
            nn.init.xavier_normal_(linear.weight, gain=gain) #IC заполнение весов нормальны распределением Хавьера
            nn.init.zeros_(linear.bias)
            
            linear.weight.data=linear.weight.data*10
            # dxc=0.05
            # dxc=0.10101010101010102
            # linear.weight.data=torch.tensor([[1/(dxc),-1/(dxc),0]])
            # linear.weight.data=torch.tensor([[100.,-222.,100.]], requires_grad=True)

            # linear.weight.data=torch.tensor([[16.,16.,-64.,16.,16.]])
            # linear.weight.data=torch.tensor([[100.,100.,-400.,100.,100.]])
            # linear.weight.data=torch.tensor([[200.,200.,-800.,200.,200.]])
            # linear.weight.data=torch.tensor([[2500.,2500.,-10_000.,2500.,2500.]])

            self.layer.append(linear)
            
        self.noise = None if noise is None else nn.Parameter(noise, requires_grad=True)

    # def forward(self, x):
    #     x = self._preprocess(x)
    #     for i, layer in enumerate(self.layer):
    #         x = layer(x) # произведение тензора x на тензор весов _preprocess(v_train[:-m]) @ net.layer[0].weight.data.T
    #         if i < len(self.layer) - 1:
    #             x = self.sig(x)
    #     return x.squeeze()

    def forward(self, x):
        # print(x.shape)
        # print('\n','def_forward_1 :','\n',x)
        x = self._preprocess(x)
        # print(x)
        # print(x.shape)
        # print(x[:,1:-1,:].shape)
        # x=x[:,1:-1,:]
        x_copy=x.clone()
        # print('\n','def_forward_2 :','\n',x)
        for i, layer in enumerate(self.layer):
            x = layer(x) # произведение тензора x на тензор весов _preprocess(v_train[:-m]) @ net.layer[0].weight.data.T
            # x = x_copy@layer.weight.data.T
            # print('\n',fr'def_forward_3_i={i} :',layer.weight)
            # print('\n',fr'def_forward_4_i={i} :',layer.bias)
            # print('\n',fr'def_forward_5_i={i} :',x)
            # print('\n',fr'MY_def_forward_5_i={i} :',x_copy@layer.weight.data.T)
            # print('\n',fr'MY2_def_forward_5_i={i} :',np.dot(x_copy.detach().numpy(),layer.weight.data.T))
            if i < len(self.layer) - 1:
                x_copy2=x.clone()
                x = self.sig(x)
                # print('\n',fr'def_forward_6_i={i} :',x)
                x_copy=self.sig(layer(x_copy))
        # print('\n','def_forward_7 :','\n',x)
        # print('\n','def_forward_8 :','\n',x.squeeze())
        res=x.squeeze()
        return res

    def get_value(self,A,i,j):
        n=int(np.sqrt(len(A)))
        if i>=n and j>=n:
            value = A[i-n+n*j-n]
        elif i>=n:
            value = A[i-n+n*j]
        elif j>=n:
            value = A[i+(n*j)-n]
        else:
            value = A[i+n*j]
        return value

    # def get_value(self,A,i,j):
    #     n=int(np.sqrt(len(A)))
    #     if i>=n and j>=n:
    #         value = A[(n-i)+n*(n-j)] #A[i-n+n*j-n]
    #     elif i>=n:
    #         value = A[(n-i)+n*j] #A[i-n+n*j]
    #     elif j>=n:
    #         value = A[i+n*(n-j)] #A[i+(n*j)-n]
    #     else:
    #         value = A[i+n*j]
    #     return value
    
    def make_px(self,x,ik,jk,n):
        
        # res = x.detach().numpy()
        res=np.zeros((x.shape[0],n*n))
        # print(res.shape)

        # print('тест №1_1')
        # self.get_value(x[0,:].detach().numpy(),1,1)
        # print('тест №1_2')
        for t in range(x.shape[0]):
            temp_arr=[]
            for i in range(n):
                for j in range(n):
                    # temp_arr.append(self.get_value(x[t,:].detach().numpy(),i+ik,j+jk))
                    temp_arr.append(self.get_value(x[t,:].detach().numpy(),j+ik,i+jk))
            for i in range(n*n):
                res[t,i]=temp_arr[i]
        
        # print('тест №2')
        res=torch.tensor(res, dtype=torch.float, device=device)
        # print('тест №3')
        res=res.unsqueeze(2)
        # print('тест №4')
        return res

    def _preprocess(self, x):
        """Prepares filters for forward pass."""
        # print(x.shape)
        x  = x.unsqueeze(-1) # Добавляет новую размерность в конец
        px = x.clone()
        px = px.detach().numpy()
        # print(px.shape)
        
        if self.fs%2!=0:
            for i in range(1, int(self.fs/2)+1):
                # r = torch.roll(x, (-1)*i, 1) # сдвигает значения тензора влево
                # l = torch.roll(x, i, 1) # сдвигает значения тензора вправо
                # r2 = torch.roll(x, (-1)*i, 2) # сдвигает значения тензора влево
                # l2 = torch.roll(x, i, 2) # сдвигает значения тензора вправо
                n=int(np.sqrt(x.shape[1]))#+1
                # print(np.sqrt(len(x)))
                # print(n)

                # print('done_1')
                px=self.make_px(x,0,0,n)
                l=self.make_px(x,-1,0,n)
                r=self.make_px(x,1,0,n)
                l2=self.make_px(x,0,-1,n)
                r2=self.make_px(x,0,1,n)
                # print('done_2')

                # print(l)
                # print(l2)
                # print(px)
                # print(r)
                # print(r2)

                # print(len(px),len(l),len(l2),len(r),len(r2))            
                px = torch.cat([l,l2, px, r,r2], -1) # объединяет 3 тензора
                
                # print(px)

        elif self.fs%2==0 and self.fs!=0:
            for i in range(1, int((self.fs+1)/2)+1):
                r = torch.roll(x, (-1)*i, 1) # сдвигает значения тензора влево
                l = torch.roll(x, i, 1) # сдвигает значения тензора вправо
                # px=torch.zeros(px.shape)
                px = torch.cat([l, px, r], -1) # объединяет 3 тензора
                # px = torch.cat([l, r], -1)

            new_indexes=[i for i in range(int(self.fs/2))]+[i for i in range(int(self.fs/2)+1,self.fs+1)]
            # print('px_shape',px.shape)
            px=px[:,:,new_indexes]
            # print('px_shape',px.shape)

        else:
            raise "Non-correct 'fs' parameter"

        return px

### ===========================Функции нейронки===========================
def forward_rk3_error(net, target, dt, m, wd, fc=None, fc_0p5=None, fc_p1=None,
                      bc_type='periodic',bc_values=[None,None]):
    '''завтра с помощью сегодня'''
    
    """
    Computes MSE for predicting solution forward in time using RK3 with possible forcing terms.
    
    Keyword arguments:
    net -- neural network for prediction
    pred -- prediction by neural net on training data
    target -- training data
    dt -- length of the timestep
    m -- number of timesteps to be predicted
    wd -- decaying weights of predictions errors
    noise -- estimated noise of measurement (default = None)
    fc -- forcing terms at current timestep (default = None)
    fc_0p5 -- forcing terms half a timestep into the future (default = None)
    fc_1p -- forcing terms one timestep into the future (default = None)
    """
    
    # initialize noise and compute clean signal based on estimate
    noise  = torch.zeros_like(target) if net.noise is None else net.noise
    pred   = target - noise
    
    # initialize forcing terms
    fc     = fc if fc is not None else torch.zeros_like(target)
    fc_0p5 = fc_0p5 if fc_0p5 is not None else torch.zeros_like(target)
    fc_p1  = fc_p1 if fc_p1 is not None else torch.zeros_like(target)
    
    if bc_type=='dirichlet' and bc_values!=[None,None]:
        # initialize residual and tensor to be predicted
        res    = torch.zeros_like(pred[0:-m,1:-1])
        p_old  = pred[0:-m,:].clone()

        j=0
        for j in range(m-1): # до m-2 включительно
            # print(fr'-------------K1---------------')
            k1    = dt*(net(p_old) + fc[j:-m+j,:])        # dt*f(t,y^n)
            temp  = p_old + 0.5*k1                           # y^n + 0.5*k1
            # print(fr'-------------K2---------------')
            k2    = dt*(net(temp) + fc_0p5[j:-m+j,:])     # dt*f(t+0.5*dt, y^n + 0.5*k1)
            temp  = p_old - k1 + 2.0*k2                      # y^n - k1 + 2.0*k2
            # print(fr'-------------K3---------------')
            k3    = dt*(net(temp) + fc_p1[j:-m+j,:])      # dt*f(t+dt, y^n - k1 + 2.0*k2)
            p_new = p_old + (1./6.)*(k1 + 4.0*k2 + k3)       # y^n + (1./6.)*(k1 + 4.0*k2 + k3)
            p_new[0][:]=bc_values[0]
            p_new[-1][:]=bc_values[-1]
            res   = res + wd[j+1]*((target[j+1:-m+j+1,1:-1] - (p_new[:,1:-1] + noise[j+1:-m+j+1,1:-1]))**2)
            p_old = p_new

        k1    = dt*(net(p_old) + fc[j:-m+j,:])        
        temp  = p_old + 0.5*k1                           
        k2    = dt*(net(temp) + fc_0p5[j:-m+j,:])     
        temp  = p_old - k1 + 2.0*k2                      
        k3    = dt*(net(temp) + fc_p1[j:-m+j,:])      
        p_new = p_old + (1./6.)*(k1 + 4.0*k2 + k3)
        p_new[0][:]=bc_values[0]
        p_new[-1][:]=bc_values[-1]
        res   = res +  wd[m]*((target[m:,1:-1] - (p_new[:,1:-1] + noise[m:,1:-1]))**2)
    elif bc_type=='periodic':
        # initialize residual and tensor to be predicted
        res    = torch.zeros_like(pred[0:-m,:])
        p_old  = pred[0:-m,:].clone()

        j=0
        for j in range(m-1): # до m-2 включительно
            # print(fr'-------------K1---------------')
            k1    = dt*(net(p_old) + fc[j:-m+j,:])        # dt*f(t,y^n)
            temp  = p_old + 0.5*k1                           # y^n + 0.5*k1
            # print(fr'-------------K2---------------')
            k2    = dt*(net(temp) + fc_0p5[j:-m+j,:])     # dt*f(t+0.5*dt, y^n + 0.5*k1)
            temp  = p_old - k1 + 2.0*k2                      # y^n - k1 + 2.0*k2
            # print(fr'-------------K3---------------')
            k3    = dt*(net(temp) + fc_p1[j:-m+j,:])      # dt*f(t+dt, y^n - k1 + 2.0*k2)
            p_new = p_old + (1./6.)*(k1 + 4.0*k2 + k3)       # y^n + (1./6.)*(k1 + 4.0*k2 + k3)
            res   = res + wd[j+1]*((target[j+1:-m+j+1,:] - (p_new + noise[j+1:-m+j+1,:]))**2)
            p_old = p_new

        k1    = dt*(net(p_old) + fc[j:-m+j,:])        
        temp  = p_old + 0.5*k1                           
        k2    = dt*(net(temp) + fc_0p5[j:-m+j,:])     
        temp  = p_old - k1 + 2.0*k2                      
        k3    = dt*(net(temp) + fc_p1[j:-m+j,:])      
        p_new = p_old + (1./6.)*(k1 + 4.0*k2 + k3)
        res   = res +  wd[m]*((target[m:,:] - (p_new + noise[m:,:]))**2)
    else:
        raise AssertionError('This bc_type didnt exist!')
    return torch.mean(res)

def backward_rk3_error(net, target, dt, m, wd, fc=None, fc_0m5=None, fc_m1=None,
                      bc_type='periodic',bc_values=[None,None]):
    '''сегодня с помощью завтра'''
    """
    Computes MSE for predicting solution backward in time using RK3 with possible forcing terms.
    
    Keyword arguments:
    net -- neural network for prediction
    pred -- prediction by neural net on training data
    target -- training data
    dt -- length of the timestep
    m -- number of timesteps to be predicted
    wd -- decaying weights of predictions errors
    noise -- estimated noise of measurement (default = None)
    fc -- forcing terms at current timestep (default = None)
    fc_0m5 -- forcing terms half a timestep into the past (default = None)
    fc_1m -- forcing terms one timestep into the past (default = None)
    """
    
    # initialize noise and compute clean signal based on estimate
    noise  = torch.zeros_like(target) if net.noise is None else net.noise
    pred   = target - noise

    # initialize forcing terms
    fc     = fc if fc is not None else torch.zeros_like(target)
    fc_0m5 = fc_0m5 if fc_0m5 is not None else torch.zeros_like(target)
    fc_m1  = fc_m1 if fc_m1 is not None else torch.zeros_like(target)
        
    if bc_type=='dirichlet' and bc_values!=[None,None]:
        
        # initialize residual and tensor to be predicted
        res    = torch.zeros_like(pred[m:,1:-1])
        p_old  = pred[m:,:].clone()
        
        k1    = -dt*(net(p_old) + fc[m:,:])
        temp  = p_old + 0.5*k1
        k2    = -dt*(net(temp) + fc_0m5[m:,:])
        temp  = p_old - k1 + 2.0*k2
        k3    = -dt*(net(temp) + fc_m1[m:,:])
        p_new = p_old + (1./6.)*(k1 + 4.0*k2 + k3)
        p_new[0][:]=bc_values[0]
        p_new[-1][:]=bc_values[-1]
        res   = res + wd[1]*((target[m-1:-1,1:-1] - (p_new[:,1:-1] + noise[m-1:-1,1:-1]))**2)
        p_old = p_new
        
        for j in range(1,m):
            k1    = -dt*(net(p_old) + fc[m-j:-j,:])       # -dt*f(t,y^n)
            temp  = p_old + 0.5*k1                           # y^n + 0.5*k1
            k2    = -dt*(net(temp) + fc_0m5[m-j:-j,:])    # -dt*f(t+0.5*dt, y^n + 0.5*k1)
            temp  = p_old - k1 + 2.0*k2                      # y^n - k1 + 2*k2
            k3    = -dt*(net(temp) + fc_m1[m-j:-j,:])     # -dt*f(t+dt, y^n - k1 + 2.0*k2)
            p_new = p_old + (1./6.)*(k1 + 4.0*k2 + k3)       # y^n + (1./6.)*(k1 + 4*k2 + k3)
            p_new[0][:]=bc_values[0]
            p_new[-1][:]=bc_values[-1]
            res   = res + wd[j+1]*((target[m-(j+1):-(j+1),1:-1] - (p_new[:,1:-1] + noise[m-(j+1):-(j+1),1:-1]))**2)
            p_old = p_new
    
    elif bc_type=='periodic':

        # initialize residual and tensor to be predicted
        res    = torch.zeros_like(pred[m:,:])
        p_old  = pred[m:,:].clone()
        
        k1    = -dt*(net(p_old) + fc[m:,:])
        temp  = p_old + 0.5*k1
        k2    = -dt*(net(temp) + fc_0m5[m:,:])
        temp  = p_old - k1 + 2.0*k2
        k3    = -dt*(net(temp) + fc_m1[m:,:])
        p_new = p_old + (1./6.)*(k1 + 4.0*k2 + k3)
        res   = res + wd[1]*((target[m-1:-1,:] - (p_new + noise[m-1:-1,:]))**2)
        p_old = p_new
        
        for j in range(1,m):
            k1    = -dt*(net(p_old) + fc[m-j:-j,:])       # -dt*f(t,y^n)
            temp  = p_old + 0.5*k1                           # y^n + 0.5*k1
            k2    = -dt*(net(temp) + fc_0m5[m-j:-j,:])    # -dt*f(t+0.5*dt, y^n + 0.5*k1)
            temp  = p_old - k1 + 2.0*k2                      # y^n - k1 + 2*k2
            k3    = -dt*(net(temp) + fc_m1[m-j:-j,:])     # -dt*f(t+dt, y^n - k1 + 2.0*k2)
            p_new = p_old + (1./6.)*(k1 + 4.0*k2 + k3)       # y^n + (1./6.)*(k1 + 4*k2 + k3)
            res   = res + wd[j+1]*((target[m-(j+1):-(j+1),:] - (p_new + noise[m-(j+1):-(j+1),:]))**2)
            p_old = p_new
    else:
        raise AssertionError('This bc_type didnt exist!')
    
    return torch.mean(res)

#my        
def forward_rk1_error(net, target, dt, m, wd, 
                      fc=None, fc_0p5=None, fc_p1=None,
                      bc_type='periodic',bc_values=[None,None]):
    """
    Computes MSE for predicting solution forward in time using RK1 with possible forcing terms.
    
    Keyword arguments:
    net -- neural network for prediction
    pred -- prediction by neural net on training data
    target -- training data
    dt -- length of the timestep
    m -- number of timesteps to be predicted
    wd -- decaying weights of predictions errors
    noise -- estimated noise of measurement (default = None)
    fc -- forcing terms at current timestep (default = None)
    fc_0p5 -- forcing terms half a timestep into the future (default = None)
    fc_1p -- forcing terms one timestep into the future (default = None)
    """
    
    # initialize noise and compute clean signal based on estimate
    noise  = torch.zeros_like(target) if net.noise is None else net.noise
    pred   = target - noise
    
    # initialize forcing terms
    fc     = fc if fc is not None else torch.zeros_like(target)
    fc_0p5 = fc_0p5 if fc_0p5 is not None else torch.zeros_like(target)
    fc_p1  = fc_p1 if fc_p1 is not None else torch.zeros_like(target)
    

    if bc_type=='dirichlet' and bc_values!=[None,None]:
    
        # initialize residual and tensor to be predicted
        res    = torch.zeros_like(pred[0:-m,:])
        p_old  = pred[0:-m,:].clone()
        
        j=0
        for j in range(m-1):
            p_new=p_old[:,:]+dt*net(p_old)+ dt * fc[j:-m+j,:]

            dbc1=drop_boundary_points1(p_new)
            dbc2=drop_boundary_points2(p_new)
            p_new[dbc1][:]=bc_values[0]
            p_new[dbc2][:]=bc_values[-1]

            p_new=p_old[:,:]+dt*(net(p_old)+ fc[j:-m+j,:])
            target_slice=np.delete(target, list(set(dbc1+dbc2)), axis=0)
            p_new_slice=np.delete(p_new, list(set(dbc1+dbc2)), axis=0)
            
            res   = res + wd[j+1]*((target_slice[j+1:-m+j+1,:] - (p_new_slice[:,:] ))**2) #+ noise[j+1:-m+j+1,:]
            p_old=p_new
        
        p_new=p_old[:,:]+dt*(net(p_old)+ fc[j:-m+j,:])
        dbc1=drop_boundary_points1(p_new)
        dbc2=drop_boundary_points2(p_new)
        p_new[dbc1][:]=bc_values[0]
        p_new[dbc2][:]=bc_values[-1]
        
        # print('done_1')
        p_new_slice=np.delete(p_new.detach().numpy(), list(set(dbc1+dbc2)), axis=0)
        p_new_slice=torch.tensor(p_new_slice)

        target_slice=np.delete(target.detach().numpy(), list(set(dbc1+dbc2)), axis=0)
        target_slice=torch.tensor(target_slice)
        # print('done_2')

        # print(p_new_slice.shape, target_slice.shape,res.shape)
        res=np.delete(res.detach().numpy(), list(set(dbc1+dbc2)), axis=0)
        res = torch.tensor(res)
        # print('done_3')

        res   = res + wd[m]*((target_slice[m:,:] - (p_new_slice[:,:] ))**2) 
        # print('done')
    elif bc_type=='periodic':
    
        # initialize residual and tensor to be predicted
        res    = torch.zeros_like(pred[0:-m,:])
        p_old  = pred[0:-m,:].clone()
        
        j=0
        for j in range(m-1):
            p_new=p_old[:,:]+dt*net(p_old)+ dt * fc[j:-m+j,:]
            
            res   = res + wd[j+1]*((target[j+1:-m+j+1,:] - (p_new[:,:] ))**2) #+ noise[j+1:-m+j+1,:]
            p_old=p_new
                
        res   = res + wd[m]*((target[m:,:] - (p_new[:,:] ))**2) 
    else:
        raise AssertionError('This bc_type didnt exist!')
    
    return torch.mean(res)

def drop_boundary_points1(A_mat):

    try:
        A=A_mat.detach().numpy()
    except:
        A=A_mat.copy()

    n=int(np.sqrt(len(A)))
    
    boundary_id=[i+j*n for i in range(n) for j in range(n) if i==0 or j==0 ]

    return sorted(boundary_id)

def drop_boundary_points2(A_mat):

    try:
        A=A_mat.detach().numpy()
    except:
        A=A_mat.copy()

    n=int(np.sqrt(len(A)))
    
    boundary_id=[i+j*n for i in range(n) for j in range(n)  if i==n-1 or j==n-1]

    return sorted(boundary_id)

#my
def backward_rk1_error(net, target, dt, m, wd, 
                       fc=None, fc_0m5=None, fc_m1=None,
                       bc_type='periodic',bc_values=[None,None]):
    """
    Computes MSE for predicting solution backward in time using RK3 with possible forcing terms.
    
    Keyword arguments:
    net -- neural network for prediction
    pred -- prediction by neural net on training data
    target -- training data
    dt -- length of the timestep
    m -- number of timesteps to be predicted
    wd -- decaying weights of predictions errors
    noise -- estimated noise of measurement (default = None)
    fc -- forcing terms at current timestep (default = None)
    fc_0m5 -- forcing terms half a timestep into the past (default = None)
    fc_1m -- forcing terms one timestep into the past (default = None)
    """
    
    # initialize noise and compute clean signal based on estimate
    noise  = torch.zeros_like(target) if net.noise is None else net.noise
    pred   = target - noise
    
    # initialize forcing terms
    fc     = fc if fc is not None else torch.zeros_like(target)
    fc_0m5 = fc_0m5 if fc_0m5 is not None else torch.zeros_like(target)
    fc_m1  = fc_m1 if fc_m1 is not None else torch.zeros_like(target)
    

    if bc_type=='dirichlet' and bc_values!=[None,None]:
        
        # initialize residual and tensor to be predicted
        res    = torch.zeros_like(pred[m:,:])
        p_old  = pred[m:,:].clone()

        dbc1=drop_boundary_points1(p_old)
        dbc2=drop_boundary_points2(p_old)
        
        p_new=p_old-dt*(net(p_old)+ fc[m:,:])
        p_new[dbc1][:]=bc_values[0]
        p_new[dbc2][:]=bc_values[-1]

        # res   = res + wd[1]*((target[m-1:-1,1:-1] - (p_new[:,1:-1] ))**2) # +noise
        target_slice=np.delete(target, list(set(dbc1+dbc2)), axis=0)
        p_new_slice=np.delete(p_new, list(set(dbc1+dbc2)), axis=0)
        res   = res + wd[1]*((target_slice[m-1:-1,:] - (p_new_slice[:,:] ))**2) # +noise
        
        p_old = p_new
        for j in range(1,m):
            p_new=p_old-dt*(net(p_old)+ fc[m-j:-j,:])

            p_new_slice=np.delete(p_new, list(set(dbc1+dbc2)), axis=0)
                        
            res   = res + wd[j+1]*((target_slice[m-(j+1):-(j+1),:] - (p_new_slice[:,:] ))**2) #+ noise[m-(j+1):-(j+1),:]
            p_old = p_new  

    elif bc_type=='periodic':

        # initialize residual and tensor to be predicted
        res    = torch.zeros_like(pred[m:,:])
        p_old  = pred[m:,:].clone()
        
        p_new=p_old-dt*(net(p_old)+ fc[m:,:])

        res   = res + wd[1]*((target[m-1:-1,:] - (p_new ))**2) # +noise
        p_old = p_new
        for j in range(1,m):
            p_new=p_old-dt*(net(p_old)+ fc[m-j:-j,:])
            
            res   = res + wd[j+1]*((target[m-(j+1):-(j+1),:] - (p_new ))**2) #+ noise[m-(j+1):-(j+1),:]
            p_old = p_new   
    else:
        raise AssertionError('This bc_type didnt exist!')
    return torch.mean(res)


def subsampling(s_factor,t_factor,h,tau,Tsim,n,v_fact,train_split):
    """
    make subsampling of initial data
    """
    dxc = s_factor*h
    dtc = t_factor*tau

    coarse_t = np.arange(0, Tsim, t_factor)
    coarse_x = np.arange(0, n, s_factor)

    v_coarse=np.zeros((len(coarse_x),len(coarse_t)))
    for i,_x in enumerate(coarse_t):
        v_coarse[:,i]=v_fact[coarse_x,coarse_t[i]].real
        
    v_coarse_train = v_coarse[:, :int(v_coarse.shape[1]*train_split)]
    v_coarse_test = v_coarse[:, int(v_coarse.shape[1]*train_split):]
    Lxc, Ltc = v_coarse_train.shape
    print('full_sample',v_coarse.shape)
    print('train',v_coarse_train.shape)
    print('test',v_coarse_test.shape)
    return dxc,dtc,coarse_t,coarse_x,v_coarse,Lxc, Ltc,v_coarse_train,v_coarse_test

def train_net(MLPConv,v_coarse_train,epochs,dtc,
              fs,
              neurons,
              hidden_layers_num,
              lr,
              m,
              has_backward,
              method,
              decay_const,
              force_terms=[None,None,None,None,None], #fc, fc_0p5,fc_p1,fc_0m5,fc_m1
              bc_type='periodic',bc_values=[None,None],
              verbose=False,
              verbose_step=100
             ):
    
    FS2 = 5

    v_train = torch.tensor(v_coarse_train.T, requires_grad=True, dtype=torch.float, device=device)
    print("v_train",v_train.shape)

    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    temp_net_layer=[FS2]+[neurons for i in range(hidden_layers_num)]+[1]
    net       = MLPConv(temp_net_layer, seed=seed, fs=fs, 
                        activation=nn.ELU()).to(device)
    params    = [{'params': net.parameters(), 'lr': lr}]
    optimizer = Adam(params)
    scheduler = ExponentialLR(optimizer, .9998)

    print("#parameters:", sum(p.numel() for p in net.parameters() if p.requires_grad))

    # decaying weights for accumulating prediction error
    output_weights = [decay_const**j for j in range(m+1)] 
    wd = torch.tensor(output_weights, dtype=torch.float32, device=device)
    
    def temp_zero_generator(*params,fc=None,fc_0m5=None,fc_m1=None,
                            bc_type='periodic',bc_values=[None,None]):
        return torch.zeros(1)[0]

    if method=='RK3':
        fwd_func=forward_rk3_error
        if has_backward:
            bwd_func=backward_rk3_error
        else:
            bwd_func=temp_zero_generator

    elif method=='E1':
        fwd_func=forward_rk1_error
        if has_backward:
            bwd_func=backward_rk1_error
        else:
            bwd_func=temp_zero_generator
    else:
        raise 'method error'

    fc=force_terms[0]
    fc_0p5=force_terms[1]
    fc_p1=force_terms[2]
    fc_0m5=force_terms[3]
    fc_m1=force_terms[4]

    loss_lst=[]
    pbar = tqdm(range(epochs))
    for epoch in pbar:
        optimizer.zero_grad()

        # print('v_train2',v_train.shape)
        # compute forward and backward prediction errors
        # print('-----------FWD------------')
        fwd=fwd_func(net, v_train, dtc, m, wd,
                     fc=fc,fc_0p5=fc_0p5,fc_p1=fc_p1,
                     bc_type=bc_type,bc_values=bc_values)
        # print('-----------BWD------------')
        bwd=bwd_func(net, v_train, dtc, m, wd,
                     fc=fc,fc_0m5=fc_0m5,fc_m1=fc_m1,
                     bc_type=bc_type,bc_values=bc_values)

        # compute norm of weights
        res_w = 0
        for i in range(len(net.layer)):
            W = net.layer[i].weight
            W = W.view(W.shape[0]*W.shape[1], -1)
            res_w = res_w + (torch.norm(W, p=2, dim=0)**2)

        loss =  fwd #+ bwd +0.0*res_w #l_wd*res_w


        # loss_lst.append([fwd.cpu().data.numpy(),bwd.cpu().data.numpy(),
                        #  l_wd*res_w.cpu().data.numpy()[0],loss.cpu().data.numpy()[0]])

        loss.backward()

        # Проверка градиентов:
        for name, param in net.named_parameters():
            if param.grad is not None:
                print(name, param.grad.abs().sum().item())
            else:
                print(name, "None")

        optimizer.step()

        if epoch > 15_000:
            scheduler.step()

        if verbose==True:
            if (epoch)%verbose_step==0:
                    print(fr'Веса после {epoch} эпохи:')
                    print(W)
                    print('Лосс :',loss)
            else:pass
                
        else: pass

        pbar.set_postfix(loss=round(loss.item(), 7))


    return net, loss_lst,loss


def linearization(v_fact):
    n=len(v_fact)
    A=np.zeros((n*n))
    for i in range(n):
        for j in range(n):
            A[i+n*j]=v_fact[i,j]
    return A

def linearization_inverse(v_fact):

    n=int(np.sqrt(len(v_fact)))
    v_fact_mat=np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            v_fact_mat[i,j]=get_value(v_fact,i,j)

    return v_fact_mat

def get_value(A,i,j):

    n=int(np.sqrt(len(A)))
    if i>=n and j>=n:
        value = A[i-n+n*j-n]
    elif i>=n:
        value = A[i-n+n*j]
    elif j>=n:
        value = A[i+(n*j)-n]
    else:
        value = A[i+n*j]
    return value

def make_simulation(net,v_coarse,L,Lxc,dtc,method='RK3',
                    bc_type='periodic',bc_values=[None,None]):
    Lx_sim=v_coarse.shape[0]
    x_sim  = np.linspace(0,L,Lx_sim)
    dxs    = x_sim[1] - x_sim[0]
    dts    = dtc
    T_sim = v_coarse.shape[1]

    NN_sim   = np.zeros((Lx_sim,T_sim))
    phase_NN = np.zeros((Lx_sim,T_sim))

    # NN_sim[:,0] = np.exp(-(x_sim-3)**2)
    NN_sim[:,0]=v_coarse[:,0]

    # print(NN_sim[:,0])

    zf   = 0
    time = 0
    # print(method)
    for j in tqdm(range(0,T_sim-1)):
        if method=='RK3':
            tensor = NN_sim[:,j].reshape(1,Lxc)
            torch_tensor = torch.tensor(tensor,dtype=torch.float,device=device)

            ##was
            # phase_NN[:,j] = net(torch_tensor).cpu().data.numpy()
            # k1   =  dts*phase_NN[:,j] #+ dts*forcing
            # temp =  NN_sim[:,j] + 0.5*k1 

            ##my
            k1   =  dts*net(torch_tensor).cpu().data.numpy() #+ dts*forcing
            temp =  NN_sim[:,j] + 0.5*k1

            tensor = temp.reshape(1,Lxc)
            torch_tensor = torch.tensor(tensor,dtype=torch.float,device=device)

            k2   =  dts*net(torch_tensor).cpu().data.numpy() #+ dts*forcing
            temp =  NN_sim[:,j] - k1 + 2.0*k2

            tensor = temp.reshape(1,Lxc)
            torch_tensor = torch.tensor(tensor,dtype=torch.float,device=device)

            k3   =  dts*net(torch_tensor).cpu().data.numpy() #+ dts*forcing

            NN_sim[:,j+1] = NN_sim[:,j] + (1./6.)*(k1 + 4.0*k2 + k3)

            if bc_type=='dirichlet' and bc_values!=[None,None]:
                dbc1=drop_boundary_points1(NN_sim[:,j])
                dbc2=drop_boundary_points2(NN_sim[:,j])
                NN_sim[dbc1,j]=bc_values[0]
                NN_sim[dbc2,j]=bc_values[-1]
            elif bc_type=='periodic':
                pass
            else:
                raise AssertionError('This bc_type didnt exist!')

            time = time + dts

        elif method=='E1':
            tensor = NN_sim[:,j].reshape(1,Lxc)
            torch_tensor = torch.tensor(tensor,dtype=torch.float,device=device)

            # print(tensor.shape, torch_tensor.shape,NN_sim.shape,net(torch_tensor).shape)

            NN_sim[:,j+1] = NN_sim[:,j]+dts*net(torch_tensor).cpu().data.numpy()

            # print(dts*net(torch_tensor))
            
            if bc_type=='dirichlet' and bc_values!=[None,None]:
                dbc1=drop_boundary_points1(NN_sim[:,j+1])
                dbc2=drop_boundary_points2(NN_sim[:,j+1])
                # print(len(dbc1),len(dbc2))
                # print(dbc1)
                # print(dbc2)
                # NN_sim[dbc1,j]=bc_values[0]
                # NN_sim[dbc2,j]=bc_values[-1]
                NN_sim[dbc1,j+1]=bc_values[0]
                NN_sim[dbc2,j+1]=bc_values[-1]

            elif bc_type=='periodic':
                pass
            else:
                raise AssertionError('This bc_type didnt exist!')
            
            time = time + dts

            # plt.figure()
            # plt.imshow(linearization_inverse(NN_sim[:,j+1]))
            # plt.title(str(j+1)+": "+str(NN_sim.shape))
            # plt.show()

        else:
            raise 'method error'
        
    return NN_sim,T_sim,x_sim   


# def plot_err_and_components_of_err(loss_lst:np.array):
#     '''График loss и слагаемых из формулы loss'''
#     names=['fwd_err','bwd_err','l_wd*res','loss']
#     for i in range(len(names)):
#         plt.figure()
#         plt.plot(loss_lst[:,i],'-o',label=names[i])
#         plt.legend()
#         plt.grid()
#     pass

# def view_results(T_sim,x_sim,NN_sim,v_coarse,T,dtc,
# n=5,
# epochs="",
# fix_axes=False,
# save_path=None,
# save_name=None,
# view_flag=True
# ):
#     time_lst=[int(i) for i in np.linspace(0,T_sim-2,n)]
#     fact_time=np.round(np.arange(0,T+dtc,dtc),4)
#     for t in time_lst:
#         plt.figure(figsize=(6,4))
#         plt.title(fr'Epochs_{epochs}|time={fact_time[t]}')
#         plt.plot(x_sim,NN_sim[:,t],'-*',color='blue',label='STENCIL-NET')
#         plt.plot(x_sim,v_coarse[:,t],'-*',color='red', label='FACT')
#         if fix_axes:
#             plt.ylim([min(v_coarse[:,0])-0.1,max(v_coarse[:,0])+0.1])
#         plt.grid()
#         plt.xlabel('x')
#         plt.ylabel('T')
#         plt.legend()
#         if save_path!=None and save_name!=None:
#             plt.savefig(save_path+'/'+save_name+fr'_t={fact_time[t]}'+'.png')
#         if view_flag:
#             plt.show()
#         else:
#             plt.close()

# def make_gif(folder,epochs,format='gif'):
    
#     files = os.listdir(folder)
#     images = [folder+i for i in files if str(epochs) in i and 'Metric' not in i and 'Imshow' not in i] 
#     with imageio.get_writer(folder+fr'gif/Results_epoch={epochs}.{format}', mode='I',fps=5) as writer:
#         for img in images:
#             writer.append_data(imageio.imread(img))  # Добавление каждого изображения в отдельности
#     # writer.close()
        
# def view_result_imshow(NN_sim, v_coarse, T, dtc, L, dxc,\
#      figsize=(7,6), n_xticks=2,\
#      n_yticks=2,
#      aspect=2,
#      save_path='./',
#      save_name='imshow_default_name',
#      fix_colorbar_axes=False, #TODO не работает
#      colorbar_min_max=[-0.1,1.1],
#      cmap='coolwarm',
#      view_flg=True):
    
#     pre_x = [i for i in range(v_coarse.shape[1])]
#     pre_y = [i for i in range(v_coarse.shape[0])]
#     fact_time = np.round(np.arange(0, T + dtc, dtc), 3)
#     fact_x = np.round(np.arange(0, L + dxc, dxc), 3)

#     data_min=colorbar_min_max[0]#min(min(v_coarse[:,0]),min(NN_sim[:,0]))
#     data_max=colorbar_min_max[1]#max(min(v_coarse[:,-1]),min(NN_sim[:,-1]))
    

#     # Проверка на соответствие длины меток и позиций
#     if len(pre_x[1::n_xticks]) != len(fact_time[1::n_xticks]):
#         raise ValueError(f"Количество позиций ({len(pre_x[1::n_xticks])}) не совпадает с количеством меток ({len(fact_time[1::n_xticks])}). Убедитесь, что n_xticks корректно задан.")

#     plt.figure(figsize=figsize)
#     # plt.subplot(3,1,1)
#     plt.title("Динамика исходных данных")
#     plt1 = plt.imshow(v_coarse, cmap=cmap, aspect=aspect)
#     plt.axvline(x=int(train_split * v_coarse.shape[1]), color='yellow', linestyle='-', linewidth=2)
#     cbar3=plt.colorbar(orientation='vertical')
#     if fix_colorbar_axes:
#         cbar3.mappable.set_clim(vmin=data_min, vmax=data_max)  # Устанавливаем пределы для colorbar
#     plt.xlabel('t')
#     plt.ylabel('x')
#     plt.xticks(pre_x[1::n_xticks], fact_time[1::n_xticks])
#     plt.yticks(pre_y[1::n_yticks], fact_x[1::n_yticks])
#     plt.xticks(rotation=70)

#     if save_path!=None and save_name!=None:
#             plt.savefig(save_path+'/'+save_name+'1.png')
#     if view_flg:
#         plt.show()
#     else:
#         plt.close()

#     plt.figure(figsize=figsize)
#     # plt.subplot(3,1,2)
#     plt.title("Динамика данных STENCIL-NET")
#     plt.imshow(NN_sim, cmap=cmap, aspect=aspect)
#     plt.axvline(x=int(train_split * v_coarse.shape[1]), color='yellow', linestyle='-', linewidth=2)
#     cbar3=plt.colorbar(orientation='vertical')
#     if fix_colorbar_axes:
#         cbar3.mappable.set_clim(vmin=data_min, vmax=data_max)  # Устанавливаем пределы для colorbar
#     plt.xlabel('t')
#     plt.ylabel('x')
#     plt.xticks(pre_x[1::n_xticks], fact_time[1::n_xticks])
#     plt.yticks(pre_y[1::n_yticks], fact_x[1::n_yticks])
#     plt.xticks(rotation=70)

#     if save_path!=None and save_name!=None:
#             plt.savefig(save_path+'/'+save_name+'2.png')
#     if view_flg:
#         plt.show()
#     else:
#         plt.close()

#     plt.figure(figsize=figsize)
#     # plt.subplot(3,1,3)
#     plt.title("Разность решений")
#     plt.imshow(v_coarse - NN_sim, cmap=cmap, aspect=aspect)
#     plt.axvline(x=int(train_split * v_coarse.shape[1]), color='yellow', linestyle='-', linewidth=2)
#     cbar3=plt.colorbar(orientation='vertical')
#     if fix_colorbar_axes:
#         cbar3.mappable.set_clim(vmin=data_min, vmax=data_max)  # Устанавливаем пределы для colorbar
#     plt.xlabel('t')
#     plt.ylabel('x')
#     plt.xticks(pre_x[1::n_xticks], fact_time[1::n_xticks])
#     plt.yticks(pre_y[1::n_yticks], fact_x[1::n_yticks])
#     plt.xticks(rotation=70)

#     if save_path!=None and save_name!=None:
#             plt.savefig(save_path+'/'+save_name+'3.png')
#     if view_flg:
#         plt.show()
#     else:
#         plt.close()
        

# def view_result_metric(NN_sim,v_coarse,T,dtc,L,dxc,figsize=(7,6),n_xticks=2,
#     save_path=None,
#     save_name=None,
#     ):
#     pre_x=[i for i in range(v_coarse.shape[1])]
#     fact_time=np.round(np.arange(0,T+dtc,dtc),3)
#     err=np.abs(v_coarse-NN_sim)
#     mae_list=[err[:,i].mean() for i in range(err.shape[1])]
#     plt.figure(figsize=figsize)
#     plt.axvline(x = int(train_split*v_coarse.shape[1]), color = 'yellow', linestyle = '-',linewidth=2)
#     plt.plot(mae_list,'-*')
#     plt.xticks(pre_x[0::n_xticks],fact_time[0::n_xticks])
#     plt.xticks(rotation=70)
#     # plt.xticks(fact_time[1::2])
#     plt.xlabel('t')
#     plt.ylabel('MAE')
#     plt.legend(['train_test_split','MAE'])
#     plt.grid()
#     if save_path!=None and save_name!=None:
#         plt.savefig(save_path+'/'+save_name+'.png')
#     plt.show()


# def make_subplot_graphs(NN_sim, v_coarse, x_sim, T_sim, T, dtc, n, nx=2, ny=2,
#                         figsize=(12,8),
#                         save_flg=False,
#                         save_path='/.',
#                         save_name='SUBPLOT'):
#     """
#     Строит графики на сетке 2x2 для заданных данных.

#     Parameters:
#     - NN_sim: массив симуляций (например, от нейронной сети).
#     - v_coarse: массив грубых данных для сравнения.
#     - x_sim: массив значений x.
#     - T_sim: общее количество временных шагов.
#     - T: конечное время.
#     - dtc: шаг по времени.
#     - n: количество временных шагов для отображения (должно быть кратно 4).
#     - nx: количество строк в сетке (по умолчанию 2).
#     - ny: количество столбцов в сетке (по умолчанию 2).
#     """
#     if n % (nx * ny) != 0:
#         raise AssertionError(f"n должно быть кратно {nx * ny} (размеру сетки сабплотов)")

#     time_lst = [int(i) for i in np.linspace(0, T_sim - 1, n)]
#     fact_time = np.round(np.arange(0, T + dtc, dtc), 4)

#     fig, axes = plt.subplots(nx, ny, figsize=figsize)
#     axes = axes.flatten()  # Преобразуем в 1D-массив для удобства итерации

#     for t, ax in zip(time_lst, axes):
#         ax.set_title(fr't={fact_time[t]}')
#         ax.plot(x_sim, NN_sim[:, t], '-o', color='blue', label='STENCIL-NET')
#         ax.plot(x_sim, v_coarse[:, t], '--*', color='red', label='Конечно-разностная схема')
#         ax.grid()
#         ax.set_xlabel('x')
#         ax.set_ylabel('T')
#         ax.legend(loc='upper right')

#     plt.tight_layout()
#     if save_flg:
#         plt.savefig(save_path+save_name+'.png')
#     plt.show()


    
def make_subplot_graphs_2d(NN_sim, v_coarse, x_sim, T_sim, T, dtc, n,
                        figsize=[10,2],
                        vmin_diff=-1e-05,
                        vmax_diff=1e-05,
                        save_flg=False,
                        save_path='/.',
                        save_name='SUBPLOT'):
    """
    Строит графики на сетке 2x2 для заданных данных.

    Parameters:
    - NN_sim: массив симуляций (например, от нейронной сети).
    - v_coarse: массив грубых данных для сравнения.
    - x_sim: массив значений x.
    - T_sim: общее количество временных шагов.
    - T: конечное время.
    - dtc: шаг по времени.
    - n: количество временных шагов для отображения (должно быть кратно 4).
    - nx: количество строк в сетке (по умолчанию 2).
    - ny: количество столбцов в сетке (по умолчанию 2).
    """
    time_lst = [int(i) for i in np.linspace(0, T_sim - 1, n)]
    n_times=len(time_lst)
    fact_time = np.round(np.arange(0, T + dtc, dtc), 4)

    # Создаем фигуру с подграфиками
    n_times = len(time_lst)
    fig, axes = plt.subplots(n_times, 3, figsize=(figsize[0], figsize[1]*n_times))  # 3 столбца: точное, численное, разность
    fig.suptitle("Сравнение точного и численного решений", fontsize=1)

    # Минимальное и максимальное значения для цветовой шкалы (чтобы графики были согласованы)
    vmin_exact = -1.5
    vmax_exact = 1.5

    for i, t in enumerate(time_lst):
        # Вычисляем решения
        exact = linearization_inverse(v_coarse[:,t])
        numerical = linearization_inverse(NN_sim[:,t])
        difference = numerical - exact

        shrink = 0.8 #0.1
        # Точное решение
        ax = axes[i, 0]
        im = ax.imshow(exact, cmap='coolwarm', origin='upper', vmin=vmin_exact, vmax=vmax_exact) #, extent=[0, 1, 0, 1]
        ax.set_title(f"Точное решение, t = {fact_time[t]}")
        fig.colorbar(im, ax=ax,shrink=shrink)

        # Численное решение
        ax = axes[i, 1]
        im = ax.imshow(numerical, cmap='coolwarm', origin='upper', vmin=vmin_exact, vmax=vmax_exact)
        ax.set_title(f"STENCIL-NET решение, t = {fact_time[t]}")
        fig.colorbar(im, ax=ax,shrink=shrink)

        # Разность (ошибка)
        ax = axes[i, 2]
        im = ax.imshow(difference, cmap='coolwarm', origin='upper', vmin=vmin_diff, vmax=vmax_diff)
        ax.set_title(f"Разность, t = {fact_time[t]}")
        cbar=fig.colorbar(im, ax=ax,shrink=shrink )

    plt.tight_layout()

    if save_flg:
        plt.savefig(save_path+save_name+'.png')

    plt.show()

def metric_by_time(NN_sim,v_coarse,T,dtc,ymax=None,
                   save_flg=False,
                   save_path='./',
                   save_name='MAE_by_time'
                   ):
    
    fact_time = np.round(np.arange(0, T + dtc, dtc), 4)
    mae=[np.mean(np.abs(NN_sim[:,t]-v_coarse[:,t])) for t in range(len(v_coarse[0,:]))]
    # mae=[np.max(np.abs(NN_sim[:,t]-v_coarse[:,t])) for t in range(len(v_coarse[0,:]))]

    assert len(fact_time)==len(mae),fr'{len(fact_time)}__{len(mae)}__shapes'

    

    plt.plot(fact_time,mae,'-*')
    if ymax!=None:
        plt.ylim([-0.2,ymax])
    plt.grid()
    plt.ylabel('MAE')
    plt.xlabel('t')

    if save_flg:
        plt.savefig(save_path+save_name+'.png')

    return mae