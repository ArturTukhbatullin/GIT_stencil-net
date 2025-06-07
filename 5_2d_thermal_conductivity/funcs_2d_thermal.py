import pandas as pd
import numpy as np
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt




### ===========================Функции из задачи теплопроводности=========================
def thermal_yavniy(v,T,kurant,h,n,tau,order=2):
    ''''
    2 and 4 order central-difference schemes for thermal task 
    '''
    new_v=copy.copy(v)
    t=0
    all_v=[v]
    if order==2:
        while t<T:
            t+=tau
            for i in range(1,n-1):
                for j in range(1,n-1):
                    new_v[0,:]=0
                    new_v[:,0]=0
                    new_v[-1,:]=1
                    new_v[:,-1]=1
                    new_v[i,j]=(v[i-1,j]-2*v[i,j]+v[i+1,j])/h/h*tau+\
                        v[i,j]+(v[i,j-1]-2*v[i,j]+v[i,j+1])/h/h*tau
            v=copy.copy(new_v)
            all_v.append(v)        
    else:
        raise 'Неверный порядок алгоритма'
    return new_v,tau,all_v

def generate_data(generate_flg,v,T,L,kurant,h,n,CUSTOM_TAU=None,save_flg=False):
    '''Generate train data for Stencil_net'''
    
    if CUSTOM_TAU==None:
        tau=thermal_yavniy(v,T,kurant,h,n)[1]
    else:
        tau=CUSTOM_TAU
    print('tau =',tau)
    time_lst=[i for i in np.arange(0,T,tau)]
    
    if generate_flg:
        print('Генерация данных')
        # v_fact=[]
        v_fact=thermal_yavniy(v,T,kurant,h,n,tau)[2]
        v=np.array(v)
        v_fact=np.array(v_fact)
        x_lst=np.linspace(0,L,num=n)
        
        #save
        if save_flg:
            np.savetxt(fr'data/thermal_v_fact_tau={tau}_n={n}.csv',v_fact,delimiter=',')

    else:
        print('Чтение уже сгенерированных данных')
        try:
            v_fact=np.array(pd.read_csv(fr'data/thermal_v_fact_tau={tau}_n={n}.csv',header=None))
            x_lst=np.linspace(0,L,num=n)
            print('data: Считал с файла')
        except:
            print('Нет файла!')
            raise

    v_fact=v_fact.T
    print('v_fact.shape =',len(v_fact),len(v_fact[0]))
    
    return v_fact,x_lst,tau,time_lst