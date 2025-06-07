import pandas as pd
import numpy as np
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt

def advection_upwind_fixe_2d(v_orig,T,kurant,h,n,order=1):
    ''''
    1 and 3 order upwind schemes for advection task 
    '''
    v=copy.copy(v_orig)
    v=np.array(v)
    v1=copy.copy(v)

    new_v=[]
    t=0
    if order==1:
        tau=kurant*h
        while t<=T:
            new_v.append(v)
            t+=tau
            for i in range(1,n-1):
                for j in range(1,n-1):
                    v1[i,j]=-(v[i,j]-v[i-1,j])/h*tau-(v[i,j]-v[i,j-1])/h*tau+v[i,j]

            v1[0,:] = v_orig[0,:]  # Первая строка
            v1[-1,:] = v_orig[-1,:]  # Последняя строка
            v1[:,0] = v_orig[:,0]  # Первый столбец
            v1[:,-1] = v_orig[:,-1]  # Последний столбец
            
            v=copy.copy(v1) 
    else:
        raise 'Неверный порядок алгоритма'
    
    return new_v,tau

def generate_data_fixe_2d(generate_flg,v,T,L,kurant,h,n,CUSTOM_TAU=None,save_flg=False):

    if CUSTOM_TAU!=None:
        tau = 0.1

    time_lst=[i for i in np.arange(0,T,tau)] #FIRST FIX T->T+tau
    if generate_flg:
        print('Генерация данных')
        v_fact=advection_upwind_fixe_2d(v,T,kurant,h,n)[0]
        v_fact=np.array(v_fact)
        x_lst=np.linspace(0,L,num=n)
        
    print(len(v_fact),len(v_fact[0]))
    v_fact=v_fact.T
    print(len(v_fact),len(v_fact[0]))
    
    return v_fact,x_lst,time_lst


# my_v,count_1_all=f2da.my_2d_advection_generator(v,n,tau,T,plot=False,verbose=False)

# def my_2d_advection_generator(v,n,tau,T,plot=False,verbose=True):

#     count_1_all=[]
#     v=np.zeros((int(T/tau),n,n))
#     for t in range(0,int(T/tau)):
#         for i in range(0,n):
#             for j in range(0,n):
#                 if i in range(1+t,int(b+t)+1) and j in range(1+t,int(b+t)+1):
#                     v[t,i,j]=1

#         if plot:
#             plt.imshow(v[:,:,t])
#             plt.show()

#         if verbose:
#             count_1=np.sum(v[:,:,t])
#             print('Доля 1:',np.round(100*count_1/n/n,3))
#             count_1_all.append(np.round(100*count_1/n/n,3))

#     return v,count_1_all

#right solution
def exact_solution(nx,ny,t, u, v, Lx, Ly,square=0.1):

    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    x, y = np.meshgrid(x, y)

    """Точное решение - перенос начального условия со скоростью (u,v)"""
    # Новые координаты с учетом переноса
    x0 = (x - u*t) % Lx
    y0 = (y - v*t) % Ly

    x0=np.round(x0,2)
    y0=np.round(y0,2)

    hx=Lx/(nx-1)
    hy=Ly/(ny-1)

    # square=0.1
    
    # Создаем квадрат в новых координатах
    q_exact = np.zeros_like(x0)
    mask = (x0 > hx) & (x0 <= square) & (y0 > hy) & (y0 <= square)
    q_exact[mask] = 1.0
    
    return q_exact

# copy of right solution
# def exact_solution(nx,ny,t, u, v, Lx, Ly):

#     x = np.linspace(0, Lx, nx)
#     y = np.linspace(0, Ly, ny)
#     x, y = np.meshgrid(x, y)

#     print('x')
#     print(x)

#     """Точное решение - перенос начального условия со скоростью (u,v)"""
#     # Новые координаты с учетом переноса
#     x0 = (x - u*t) % Lx
#     y0 = (y - v*t) % Ly

#     x0=np.round(x0,2)
#     y0=np.round(y0,2)

#     print('x0')
#     print(x0)
#     print('y0')
#     print(y0)

#     hx=Lx/(nx-1)
#     hy=Ly/(ny-1)

#     square=0.2
    
#     # Создаем квадрат в новых координатах
#     q_exact = np.zeros_like(x0)
#     mask = (x0 > hx) & (x0 <= square) & (y0 > hy) & (y0 <= square)
#     q_exact[mask] = 1.0
    
#     return q_exact

# try another BC
# def exact_solution(nx, ny, t, u, v, Lx, Ly):
#     x = np.linspace(0, Lx, nx)
#     y = np.linspace(0, Ly, ny)
#     x, y = np.meshgrid(x, y)

#     # Новые координаты с учетом переноса и периодических граничных условий
#     x0 = (x - u * t) % Lx
#     y0 = (y - v * t) % Ly
#     print('x0')
#     print(x0)
#     print('y0')
#     print(y0)

#     # Для избежания ошибок округления лучше не округлять координаты
#     # hx = Lx / (nx - 1)
#     # hy = Ly / (ny - 1)
#     hx=0
#     hy=0

#     square = 0.1
    
#     # Создаем квадрат в новых координатах
#     q_exact = np.zeros_like(x0)
#     mask = (x0 > hx) & (x0 <= square) & (y0 > hy) & (y0 <= square)
#     q_exact[mask] = 1.0
    
#     return q_exact




def minmod(a, b):
    """Ограничитель Minmod"""
    if a * b <= 0:
        return 0
    else:
        return np.sign(a) * min(abs(a), abs(b))
    
def tvd_mimod_2d(u0, vx, vy, h, dt, nt, nx, ny):
    
    u = u0.copy()
    un = np.zeros((nx, ny))
    
    u_output=np.zeros((nt,nx,ny))
    for n in range(nt):
        un = u.copy()
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                # Потоки по x
                # Низкого порядка
                f_low_ip12 = un[i,j]
                f_low_im12 = un[i-1,j]
                
                # Высокого порядка
                f_high_ip12 = 0.5 * (un[i+1,j] + un[i,j])
                f_high_im12 = 0.5 * (un[i,j] + un[i-1,j])
                
                # Вычисление r
                r_ip12 = (un[i,j] - un[i-1,j]) / (un[i+1,j] - un[i,j] + 1e-6)
                r_im12 = (un[i-1,j] - un[i-2,j]) / (un[i,j] - un[i-1,j] + 1e-6)
                
                # Ограниченные потоки
                F_ip12 = f_low_ip12 - minmod(1, r_ip12) * (f_low_ip12 - f_high_ip12)
                F_im12 = f_low_im12 - minmod(1, r_im12) * (f_low_im12 - f_high_im12)
                
                # Потоки по y
                # Низкого порядка
                f_low_jp12 = un[i,j]
                f_low_jm12 = un[i,j-1]
                
                # Высокого порядка
                f_high_jp12 = 0.5 * (un[i,j+1] + un[i,j])
                f_high_jm12 = 0.5 * (un[i,j] + un[i,j-1])
                
                # Вычисление r
                r_jp12 = (un[i,j] - un[i,j-1]) / (un[i,j+1] - un[i,j] + 1e-6)
                r_jm12 = (un[i,j-1] - un[i,j-2]) / (un[i,j] - un[i,j-1] + 1e-6)
                
                # Ограниченные потоки
                F_jp12 = f_low_jp12 - minmod(1, r_jp12) * (f_low_jp12 - f_high_jp12)
                F_jm12 = f_low_jm12 - minmod(1, r_jm12) * (f_low_jm12 - f_high_jm12)
                
                # Обновление решения
                u[i,j] = un[i,j] - (dt/h) * vx * (F_ip12 - F_im12) - (dt/h) * vy * (F_jp12 - F_jm12)

        u_output[n,:,:]=copy.copy(u)

    return u_output
