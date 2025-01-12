import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam,SGD
from torch.optim.lr_scheduler import ExponentialLR

# main_params=pd.read_excel('experiment_init_params_FIX_PARAMS2_advection.xlsx',dtype={'has_backward':bool})
main_params=pd.read_excel('experiment_init_params_FIX_PARAMS2_advection2.xlsx',dtype={'has_backward':bool})
main_params=main_params.set_index('param')

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

#device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            linear = nn.Linear(in_features=sizes[i], out_features=sizes[i+1],bias=False)
            
            print("input", sizes[i], "output", sizes[i+1])
            nn.init.xavier_normal_(linear.weight, gain=gain) #IC заполнение весов нормальны распределением Хавьера
            # nn.init.zeros_(linear.bias)
            
            # linear.weight.data=linear.weight.data*10
            # dxc=0.05
            dxc=0.10101010101010102
            linear.weight.data=torch.tensor([[1/(dxc),-1/(dxc),0]])
            # linear.weight.data=torch.tensor([[100.,-222.,100.]], requires_grad=True)

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
    
    
    def _preprocess(self, x):
        """Prepares filters for forward pass."""
        x  = x.unsqueeze(-1) # Добавляет новую размерность в конец
        px = x.clone()
        
        if self.fs%2!=0:
            for i in range(1, int(self.fs/2)+1):
                r = torch.roll(x, (-1)*i, 1) # сдвигает значения тензора влево
                l = torch.roll(x, i, 1) # сдвигает значения тензора вправо
                px = torch.cat([l, px, r], -1) # объединяет 3 тензора

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
def forward_rk3_error(net, target, dt, m, wd, fc=None, fc_0p5=None, fc_p1=None):
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
    
    # initialize residual and tensor to be predicted
    res    = torch.zeros_like(pred[0:-m,:])
    p_old  = pred[0:-m,:].clone()

    # print('p_old',p_old)
    
    # print(p_old.shape)
    # print('m',m)
    for j in range(m-1): # до m-2 включительно
        # print(fr'-------------{j}---------------')
        # print('p_old',p_old)
        # print(fr'-------------K1---------------')
        k1    = dt*(net(p_old) + fc[j:-m+j,:])        # dt*f(t,y^n)
        temp  = p_old + 0.5*k1                           # y^n + 0.5*k1
        # print(fr'-------------K2---------------')
        k2    = dt*(net(temp) + fc_0p5[j:-m+j,:])     # dt*f(t+0.5*dt, y^n + 0.5*k1)
        temp  = p_old - k1 + 2.0*k2                      # y^n - k1 + 2.0*k2
        # print(fr'-------------K3---------------')
        k3    = dt*(net(temp) + fc_p1[j:-m+j,:])      # dt*f(t+dt, y^n - k1 + 2.0*k2)
        p_new = p_old + (1./6.)*(k1 + 4.0*k2 + k3)       # y^n + (1./6.)*(k1 + 4.0*k2 + k3)
        # print('indexes',j+1,-m+j+1,target[j+1:-m+j+1,:].shape)
        # max_case : res = res + wd[m-1]*((target[3:-1] - p_new)**2)
        # min_case : res = res + wd[1]*((target[1:-3] - p_new)**2)
        res   = res + wd[j+1]*((target[j+1:-m+j+1,:] - (p_new + noise[j+1:-m+j+1,:]))**2)
        # print('res')
        # print(res)
        p_old = p_new
    
    # print(fr'-------------{j}---------------')
    # print('indexes',m,-1,target[m:,:].shape)
    k1    = dt*(net(p_old) + fc[j:-m+j,:])        
    temp  = p_old + 0.5*k1                           
    k2    = dt*(net(temp) + fc_0p5[j:-m+j,:])     
    temp  = p_old - k1 + 2.0*k2                      
    k3    = dt*(net(temp) + fc_p1[j:-m+j,:])      
    p_new = p_old + (1./6.)*(k1 + 4.0*k2 + k3)
    res   = res +  wd[m]*((target[m:,:] - (p_new + noise[m:,:]))**2)
    # print('res')
    # print(res)
    return torch.mean(res)

def backward_rk3_error(net, target, dt, m, wd, fc=None, fc_0m5=None, fc_m1=None):
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
        # min_case: res=res+w[2]*((target[2:-2]-p_new)**2)
        # max_case: res=res+ wd[4]*((target[0:-4,:] - p_new)**2)
        res   = res + wd[j+1]*((target[m-(j+1):-(j+1),:] - (p_new + noise[m-(j+1):-(j+1),:]))**2)
        p_old = p_new
    
    return torch.mean(res)

#my        
def forward_rk1_error(net, target, dt, m, wd, fc=None, fc_0p5=None, fc_p1=None):
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
    
    # initialize residual and tensor to be predicted
    res    = torch.zeros_like(pred[0:-m,1:-1])
    p_old  = pred[0:-m,:].clone()

    # print('p_old',p_old.shape)
    
    for j in range(m-1):
        p_new=p_old[:,:]+dt*net(p_old)

        # p_new[0][:]=0
        # p_new[-1][:]=1

        # print('----------')
        # print(net(p_old))
        # print(p_new)
        # print('----------')
        
        res   = res + wd[j+1]*((target[j+1:-m+j+1,1:-1] - (p_new[:,1:-1] ))**2) #+ noise[j+1:-m+j+1,:]
        p_old=p_new
    
    p_new=p_old[:,:]+dt*net(p_old)

    # p_new[0][:]=0
    # p_new[-1][:]=1

    # res   = res + wd[m]*((target[m:,:] - (p_new ))**2) #+ noise[j+1:-m+j+1,:]
    # print('target',target.shape)
    # print('wd',wd.shape)
    res   = res + wd[m]*((target[m:,1:-1] - (p_new[:,1:-1] ))**2) 
    
    return torch.mean(res)

#my
def backward_rk1_error(net, target, dt, m, wd, fc=None, fc_0m5=None, fc_m1=None):
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
    
    # initialize residual and tensor to be predicted
    res    = torch.zeros_like(pred[m:,:])
    p_old  = pred[m:,:].clone()
    
    p_new=p_old-dt*net(p_old)
    p_new[0][:]=0
    p_new[-1][:]=0

    res   = res + wd[1]*((target[m-1:-1,:] - (p_new ))**2) # +noise
    p_old = p_new
    for j in range(1,m):
        p_new=p_old-dt*net(p_old)
        p_new[0][:]=0
        p_new[-1][:]=0
        
        res   = res + wd[j+1]*((target[m-(j+1):-(j+1),:] - (p_new ))**2) #+ noise[m-(j+1):-(j+1),:]
        p_old = p_new    
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
              decay_const
             ):
    
    v_train = torch.tensor(v_coarse_train.T, requires_grad=True, dtype=torch.float, device=device)

    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    temp_net_layer=[fs]+[neurons for i in range(hidden_layers_num)]+[1]
    net       = MLPConv(temp_net_layer, seed=seed, fs=fs, 
                        activation=nn.ELU()).to(device)
    params    = [{'params': net.parameters(), 'lr': lr}]
    optimizer = Adam(params)
    # optimizer  = SGD(params)
    # scheduler = ExponentialLR(optimizer, .9998)

    print("#parameters:", sum(p.numel() for p in net.parameters() if p.requires_grad))

    # decaying weights for accumulating prediction error
    output_weights = [decay_const**j for j in range(m+1)] 
    wd = torch.tensor(output_weights, dtype=torch.float32, device=device)
    
    def temp_zero_generator(*params):
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


    loss_lst=[]
    pbar = tqdm(range(epochs))
    for epoch in pbar:
        optimizer.zero_grad()

        # compute forward and backward prediction errors
        # print('-----------FWD------------')
        fwd=fwd_func(net, v_train, dtc, m, wd)
        # print('-----------BWD------------')
        bwd=bwd_func(net, v_train, dtc, m, wd)

        # compute norm of weights
        res_w = 0
        for i in range(len(net.layer)):
            W = net.layer[i].weight
            W = W.view(W.shape[0]*W.shape[1], -1)
            res_w = res_w + (torch.norm(W, p=2, dim=0)**2)

        loss =  fwd + bwd + 0*res_w #l_wd*res_w
        # loss = fwd

        loss_lst.append([fwd.cpu().data.numpy(),bwd.cpu().data.numpy(),
                         l_wd*res_w.cpu().data.numpy()[0],loss.cpu().data.numpy()[0]])

        loss.backward()
        # optimizer.step()

        if epoch > 15_000:
            scheduler.step()

        pbar.set_postfix(loss=round(loss.item(), 7))

    return net, loss_lst,loss



def make_simulation(net,v_coarse,L,Lxc,dtc,method='RK3'):
    Lx_sim=v_coarse.shape[0]
    x_sim  = np.linspace(0,L,Lx_sim)
    dxs    = x_sim[1] - x_sim[0]
    dts    = dtc
    T_sim = v_coarse.shape[1]

    NN_sim   = np.zeros((Lx_sim,T_sim))
    phase_NN = np.zeros((Lx_sim,T_sim))

    # NN_sim[:,0] = np.exp(-(x_sim-3)**2)
    NN_sim[:,0]=v_coarse[:,0]

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

            time = time + dts

        elif method=='E1':
            tensor = NN_sim[:,j].reshape(1,Lxc)
            torch_tensor = torch.tensor(tensor,dtype=torch.float,device=device)

            NN_sim[:,j+1] = NN_sim[:,j]+dts*net(torch_tensor).cpu().data.numpy()

            NN_sim[0,:]=0
            NN_sim[-1,:]=0

            time = time + dts

        else:
            raise 'method error'
        
    return NN_sim,T_sim,x_sim      


def plot_err_and_components_of_err(loss_lst:np.array):
    '''График loss и слагаемых из формулы loss'''
    names=['fwd_err','bwd_err','l_wd*res','loss']
    for i in range(len(names)):
        plt.figure()
        plt.plot(loss_lst[:,i],'-o',label=names[i])
        plt.legend()
        plt.grid()
    pass

def view_results(T_sim,x_sim,NN_sim,v_coarse,T,dtc,
n=5,
epochs="",
fix_axes=False,
save_path=None,
save_name=None,
view_flag=True
):
    time_lst=[int(i) for i in np.linspace(0,T_sim-2,n)]
    fact_time=np.round(np.arange(0,T+dtc,dtc),4)
    for t in time_lst:
        plt.figure(figsize=(6,4))
        plt.title(fr'Epochs_{epochs}|time={fact_time[t]}')
        plt.plot(x_sim,NN_sim[:,t],'-*',color='blue',label='STENCIL-NET')
        plt.plot(x_sim,v_coarse[:,t],'-*',color='red', label='FACT')
        if fix_axes:
            plt.ylim([min(v_coarse[:,0])-0.1,max(v_coarse[:,0])+0.1])
        plt.grid()
        plt.xlabel('x')
        plt.ylabel('T')
        plt.legend()
        if save_path!=None and save_name!=None:
            plt.savefig(save_path+'/'+save_name+fr'_t={fact_time[t]}'+'.png')
        if view_flag:
            plt.show()
        else:
            plt.close()

def make_gif(folder,epochs):
    
    import imageio
    import os
    files = os.listdir(folder)
    images = [folder+i for i in files if str(epochs) in i and 'Metric' not in i] 
    with imageio.get_writer(folder+fr'gif/Results_epoch={epochs}.gif', mode='I',fps=5) as writer:
        for img in images:
            writer.append_data(imageio.imread(img))  # Добавление каждого изображения в отдельности


        
def view_result_imshow(NN_sim,v_coarse,T,dtc,L,dxc,figsize=(7,6),n_xticks=2):
    
    pre_x=[i for i in range(v_coarse.shape[1])]
    pre_y=[i for i in range(v_coarse.shape[0])]
    fact_time=np.round(np.arange(0,T+dtc,dtc),3)
    fact_x=np.round(np.arange(0,L+dxc,dxc),3)

    plt.figure(figsize=figsize)
    plt.title("Real data")
    plt1=plt.imshow(v_coarse,cmap='seismic', aspect=1.8)
    plt.axvline(x = int(train_split*v_coarse.shape[1]), color = 'yellow', linestyle = '-',linewidth=2)
    plt.colorbar(orientation='horizontal')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.xticks(pre_x[1::n_xticks],fact_time[1::n_xticks])
    plt.yticks(pre_y,fact_x)
    plt.xticks(rotation=70)
    plt.show()

    plt.figure(figsize=figsize)
    plt.title("Net data")
    plt.imshow(NN_sim,cmap='seismic', aspect=1.8)
    plt.axvline(x = int(train_split*v_coarse.shape[1]), color = 'yellow', linestyle = '-',linewidth=2)
    plt.colorbar(orientation='horizontal')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.xticks(pre_x[1::n_xticks],fact_time[1::n_xticks])
    plt.yticks(pre_y,fact_x)
    plt.xticks(rotation=70)
    plt.show()

    plt.figure(figsize=figsize)
    plt.title("Error data")
    plt.imshow(v_coarse-NN_sim,cmap='seismic', aspect=1.8)
    plt.axvline(x = int(train_split*v_coarse.shape[1]), color = 'yellow', linestyle = '-',linewidth=2)
    plt.colorbar(orientation='horizontal')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.xticks(pre_x[1::n_xticks],fact_time[1::n_xticks])
    plt.yticks(pre_y,fact_x)
    plt.xticks(rotation=70)
    plt.show()
        

def view_result_metric(NN_sim,v_coarse,T,dtc,L,dxc,figsize=(7,6),n_xticks=2,
    save_path=None,
    save_name=None,
    ):
    pre_x=[i for i in range(v_coarse.shape[1])]
    fact_time=np.round(np.arange(0,T+dtc,dtc),3)
    err=np.abs(v_coarse-NN_sim)
    mae_list=[err[:,i].mean() for i in range(err.shape[1])]
    plt.figure(figsize=figsize)
    plt.axvline(x = int(train_split*v_coarse.shape[1]), color = 'yellow', linestyle = '-',linewidth=2)
    plt.plot(mae_list,'-*')
    plt.xticks(pre_x[0::n_xticks],fact_time[0::n_xticks])
    plt.xticks(rotation=70)
    # plt.xticks(fact_time[1::2])
    plt.xlabel('t')
    plt.grid()
    if save_path!=None and save_name!=None:
        plt.savefig(save_path+'/'+save_name+'.png')
    plt.show()

### ===========================Функции из задачи теплопроводности=========================
def advection_upwind(v,T,kurant,h,n,order=1):
    ''''
    1 and 3 order upwind schemes for advection task 
    '''
    new_v=copy.copy(v)
    t=0
    
    if order==1:
        tau=kurant*h
        print('tau',tau)
        while t<T:
            t+=tau
            for i in range(1,n):
                new_v[0]=0
                new_v[i]=-(v[i]-v[i-1])/h*tau+v[i]
            v=copy.copy(new_v)
    
    elif order==3:
        tau=kurant*(h**2)
        while t<T:
            t+=tau
            for i in range(1,n-1):
                new_v[0]=0
                new_v[-1]=0
                new_v[i]=-tau/6/h*(2*v[i+1]+3*v[i]-6*v[i-1]+v[i-2])+v[i]
            v=copy.copy(new_v)
    
    else:
        raise 'Неверный порядок алгоритма'
    
    return new_v,tau


def generate_data(generate_flg,v,T,L,kurant,h,n,CUSTOM_TAU=None,save_flg=False):
    if CUSTOM_TAU==None:
        tau=advection_upwind(v,T,kurant,h,n)[1]
        print('tau =',tau)
    else:
        tau=CUSTOM_TAU
    time_lst=[i for i in np.arange(0,T+tau,tau)] #FIRST FIX T->T+tau
    if generate_flg:
        print('Генерация данных')
        v_fact=[]
        for t in tqdm(time_lst):
            v_fact.append(advection_upwind(v,t,kurant,h,n)[0])
        v=np.array(v)
        v_fact=np.array(v_fact)
        x_lst=np.linspace(0,L,num=n)
        
        #save
        if save_flg:
            np.savetxt(fr'data/advection_v_fact_tau={tau}_n={n}.csv',v_fact,delimiter=',')

    else:
        print('Чтение уже сгенерированных данных')
        try:
            v_fact=np.array(pd.read_csv(fr'data/advection_v_fact_tau={tau}_n={n}.csv',header=None))
            x_lst=np.linspace(0,L,num=n)
            print('data: Считал с файла')
        except:
            print('Нет файла!')

    print(len(v_fact),len(v_fact[0]))
    v_fact=v_fact.T
    print(len(v_fact),len(v_fact[0]))
    
    return v_fact,x_lst,tau,time_lst