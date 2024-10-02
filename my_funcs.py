import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy
from tqdm import tqdm

def advection_upwind(v,T,kurant,h,n):
    new_v=copy.copy(v)
    t=0
    tau=kurant*h
    while t<T:
        t+=tau
        for i in range(1,n):
            new_v[0]=0
            new_v[i]=-(v[i]-v[i-1])/h*tau+v[i]
                        
        v=copy.copy(new_v)
    
    return new_v,tau

def advetion_central_diff_2(v_orig,T,kurant,h,n):
    
    v=copy.copy(v_orig)
    v_new=[0 for i in v]
    v_old=copy.copy(v)
    
    t=0
    tau=kurant*h
    
    while t<T:
        t+=tau
        for i in range(1,n-1):
            v_new[0]=0
            v_new[-1]=0
            v_new[i]=tau*(v[i-1]-v[i+1])/h+v_old[i]
        v_old=copy.copy(v)
        v=copy.copy(v_new)
    return v_new,tau

# def advetion_central_diff_4(v_orig,T,kurant,h,n): #todo
    
#     v=copy.copy(v_orig)
#     v_new=[0 for i in v]
#     v_old=copy.copy(v)
    
#     t=0
#     tau=kurant*h
    
#     while t<T:
#         t+=tau
#         for i in range(2,n-2):
#             v_new[0]=v[0]
#             v_new[1]=v[1]
#             v_new[-1]=v[-1]
#             v_new[-2]=v[-2]
#             v_new[i]=2*tau*(-v[i-2]/12+2*v[i-1]/3-2*v[i+1]/3+v[i+2]/12)/h+v_old[i]
#         v_old=copy.copy(v)
#         v=copy.copy(v_new)
#     return v_new,tau
    


def generate_data(generate_flg,v,T,L,kurant,h,n,CUSTOM_TAU=None,save_flg=False):
    if CUSTOM_TAU==None:
        tau=advection_upwind(v,T,kurant,h,n)[1]
    else:
        tau=CUSTOM_TAU
    time_lst=[i for i in np.arange(0,T,tau)]
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


def subsampling(s_factor,t_factor,h,tau,Tsim,n,v_fact,train_split):
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

def view_results(T_sim,x_sim,NN_sim,v_coarse):
    time=[int(i) for i in np.linspace(0,T_sim-1,5)]
    for time in time:
        plt.figure(figsize=(4,3))
        plt.title(fr'time={time}')
        plt.plot(x_sim,NN_sim[:,time],'-',color='blue',label='STENCIL-NET')
        plt.plot(x_sim,v_coarse[:,time],color='red', label='FACT')
        plt.grid()
        plt.legend()
        
        
        
# ------------------ Функции для документации----------------
def view_weights(net):
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
                print(fr'w_{k}{kk}{kkk}',np.round(ii.data.numpy(),4),'\t',end='')
            print()
            
def view_tensor(v_train):
    print(v_train.shape)
    for i in range(len(v_train[:].data.numpy())):
        for j in v_train[i].data.numpy():
            print(j,end=' ')
        print()