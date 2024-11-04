import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy
from tqdm import tqdm

def advection_upwind(v,T,kurant,h,n,order=1):
    ''''
    1 and 3 order upwind schemes for advection task 
    '''
    new_v=copy.copy(v)
    t=0
    
    if order==1:
        tau=kurant*h
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

def advetion_central_diff(v_orig,T,kurant,h,n,order=2):
    v=copy.copy(v_orig)
    v_new=[0 for i in v]
    t=0
    
    if order==2:
        tau=kurant*(h**2)
        while t<T:
            t+=tau
            for i in range(1,n-1):
                v_new[0]=0
                v_new[-1]=0
                v_new[i]=tau*(v[i-1]-v[i+1])/2/h+v[i]
            # v_old=copy.copy(v)
            v=copy.copy(v_new)
    
    elif order==4:
        tau=kurant*(h**2)
        while t<T:
            t+=tau
            for i in range(2,n-2):
                v_new[0]=0
                v_new[1]=0
                v_new[-2]=0
                v_new[-1]=0
                v_new[i]=-tau*(-8*v[i-1]+8*v[i+1]+v[i-2]-v[i+2])/12/h+v[i]
            v=copy.copy(v_new)
    
    else:
        raise 'Неверный порядок алгоритма'
        
    return v_new,tau
    


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