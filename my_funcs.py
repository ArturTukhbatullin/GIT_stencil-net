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