import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy
from tqdm import tqdm

folder_to_save_png='data/output/png/'

def thermal_yavniy(v,T,kurant,h,n,order=2):
    ''''
    2 and 4 order central-difference schemes for thermal task 
    '''
    new_v=copy.copy(v)
    t=0
    all_v=[v]
    
    if order==2:
        tau=kurant*(h**2)
        while t<T:
            t+=tau
            for i in range(1,n-1):
                new_v[0]=0
                new_v[-1]=1
                new_v[i]=(v[i-1]-2*v[i]+v[i+1])/h/h*tau+v[i]
            v=copy.copy(new_v)
            all_v.append(v)
    
    elif order==4:
        tau=kurant*(h**2)
        while t<T:
            t+=tau
            for i in range(2,n-2):
                new_v[0]=0
                new_v[-1]=1
                new_v[1]=v[1]
                new_v[-2]=v[-2]
                new_v[i]=(-v[i-2]+16*v[i-1]-30*v[i]+16*v[i+1]-v[i+2])/12/h/h*tau+v[i]
            v=copy.copy(new_v)
        
    
    else:
        raise 'Неверный порядок алгоритма'
    
    return new_v,tau,all_v

def thermal_plot(l,new_v_0_50,new_v_0_25,final_time,n,order,savefig=False):
    '''Plot solution'''
    plt.figure(figsize=(8,6))
    plt.plot(l,l,'--',label='Точное решение : kurant = 1',color='black')
    # plt.plot(l,new_v_0_75,'-*',label=r'Явный : Курант = 0.75',color='orange') ######!!!!! nans
    # plt.plot(l,new_v_0_50,'-*',label=r'Явный : Курант = 0.5',color='red')
    plt.plot(l,new_v_0_25,'-.',label=r'Явный : Курант = 0.25',color='cyan')
    plt.legend()
    plt.grid()
    plt.title(fr'mesh : {n} & time : {final_time}')
    
    if savefig==True:
        plt.savefig(folder_to_save_png+fr'yavniy_{order}order.png')
        
    plt.show()

    


def generate_data(generate_flg,v,T,L,kurant,h,n,CUSTOM_TAU=None,save_flg=False):
    '''Generate train data for Stencil_net'''
    if CUSTOM_TAU==None:
        tau=thermal_yavniy(v,T,kurant,h,n)[1]
    else:
        tau=CUSTOM_TAU
    time_lst=[i for i in np.arange(0,T,tau)]
    if generate_flg:
        print('Генерация данных')
        # v_fact=[]
        v_fact=thermal_yavniy(v,T,kurant,h,n)[2]
        # for t in tqdm(time_lst):
            # v_fact.append(thermal_yavniy(v,t,kurant,h,n)[0])
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

    print(len(v_fact),len(v_fact[0]))
    v_fact=v_fact.T
    print(len(v_fact),len(v_fact[0]))
    
    return v_fact,x_lst,tau,time_lst