import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy
from tqdm import tqdm
import optuna
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from stencilnet import MLPConv, forward_rk3_error, backward_rk3_error#,backward_rk3_tvd_error,forward_rk3_tvd_error
from stencilnet import forward_rk1_error,backward_rk1_error
from utils import load_simulation_model


#------------------------------MAIN PARAMS----------------------------------
main_params=pd.read_excel('init_params.xlsx')
main_params=main_params.set_index('param')
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

# ---------------------------------------------------------------------------------

# ----------------------PreProcess------------------
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

# -------------------------end PreProcess---------------------




# ------------------------Process-----------------------


# ----------------------end Process---------------------

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
    scheduler = ExponentialLR(optimizer, .9998)

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
        fwd=fwd_func(net, v_train, dtc, m, wd)
        bwd=bwd_func(net, v_train, dtc, m, wd)

        # compute norm of weights
        res_w = 0
        for i in range(len(net.layer)):
            W = net.layer[i].weight
            W = W.view(W.shape[0]*W.shape[1], -1)
            res_w = res_w + (torch.norm(W, p=2, dim=0)**2)

        loss =  fwd + bwd + l_wd*res_w
        loss_lst.append([fwd.cpu().data.numpy(),bwd.cpu().data.numpy(),
                         l_wd*res_w.cpu().data.numpy()[0],loss.cpu().data.numpy()[0]])

        loss.backward()
        optimizer.step()

        if epoch > 15_000:
            scheduler.step()

        pbar.set_postfix(loss=round(loss.item(), 7))

    return net, loss_lst,loss



# ------------------------PostProcess-----------------------
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
    print(method)
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


def view_train_test_graph(x_sim,v_coarse_train,v_coarse_test):
    plt.figure()
    plt.plot(x_sim,v_coarse_train[:,-1],'-',color='green',label='FACT_last_train')
    plt.plot(x_sim,v_coarse_test[:,0],'--',color='orange',label='FACT_first_test')
    plt.legend()
    plt.grid()

# def view_results_old(T_sim,x_sim,NN_sim,v_coarse,n=5):
#     time_lst=[int(i) for i in np.linspace(0,T_sim-1,n)]
#     # time_lst=np.arange(0,T_sim-1,step)
#     for time in time_lst:
#         plt.figure(figsize=(7,6))
#         plt.title(fr'time={time}')
#         plt.plot(x_sim,NN_sim[:,time],'-*',color='blue',label='STENCIL-NET')
#         plt.plot(x_sim,v_coarse[:,time],'-*',color='red', label='FACT')
#         plt.grid()
#         plt.legend()

def view_results(T_sim,x_sim,NN_sim,v_coarse,T,dtc,n=5):
    time_lst=[int(i) for i in np.linspace(0,T_sim-1,n)]
    fact_time=np.round(np.arange(0,T+dtc,dtc),4)
    for t in time_lst:
        plt.figure(figsize=(6,4))
        plt.title(fr'time={fact_time[t]}')
        plt.plot(x_sim,NN_sim[:,t],'-*',color='blue',label='STENCIL-NET')
        plt.plot(x_sim,v_coarse[:,t],'-*',color='red', label='FACT')
        plt.grid()
        plt.legend()


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
        

def view_result_metric(NN_sim,v_coarse,T,dtc,L,dxc,figsize=(7,6),n_xticks=2):
    pre_x=[i for i in range(v_coarse.shape[1])]
    fact_time=np.round(np.arange(0,T+dtc,dtc),3)
    err=np.abs(v_coarse-NN_sim)
    mae_list=[err[:,i].mean() for i in range(err.shape[1])]
    plt.figure(figsize=figsize)
    plt.axvline(x = int(train_split*v_coarse.shape[1]), color = 'yellow', linestyle = '-',linewidth=2)
    plt.plot(mae_list,'-*')
    plt.xticks(pre_x[1::n_xticks],fact_time[1::n_xticks])
    plt.xticks(rotation=70)
    # plt.xticks(fact_time[1::2])
    plt.xlabel('t')
    plt.grid()
    plt.show()

# ------------------------end PostProcess-----------------------
        
        
# ------------------ Function for documentation----------------
def view_weights(net,rounded_order=4):
    print(len(net.layer))
    k=0
    for j in net.layer:
        k+=1
        kk=0
        print(fr'-----{j}------')
        for i in j.weight:
            kk+=1
            kkk=0
            for ii in i:
                kkk+=1
                print(fr'w_{k}{kk}{kkk}',np.round(ii.data.numpy(),rounded_order),'\t',end='')
            print()
            
def view_tensor(v_train):
    print(v_train.shape)
    for i in range(len(v_train[:].data.numpy())):
        for j in v_train[i].data.numpy():
            print(j,end=' ')
        print()