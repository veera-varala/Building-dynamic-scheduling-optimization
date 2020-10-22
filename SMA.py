# -*- coding: utf-8 -*-
"""
Created on Thu June 12 12:21:00 2020

@author: Veera Varala
"""


#%%
#Importing required packages
import numpy as np
import pandas as pd
from scipy.io import loadmat            #Matalb data importing
from scipy.interpolate import interp1d  #Interpolating data
import matplotlib.pyplot as plt

from pyomo.environ import * 
from pyomo.dae import *
from pyomo.core import *
from pyomo.core import ConstraintList

#%%
#Constants and building parameters

Vz=np.sum(np.array([1100,1100,641,641,1100,1100,1100,641,641,1100,1100,1100,641,641,1100])) #Zones Volume[m^3]
q_zOA_min=2*Vz*10e-5*3600   # Minimum fresh airflow [m^3/hr]
q_ve_up=25*3600   #units : Maximum fresh airflow [m^3/hr]


'''
e:  External building elements
i:  Internal building elements
pl: Plenums
w:  Windows
z:  Zones
'''
rho=1.225            # Air density (kg/m^3)
Cp=1                 # heat capacity (kJ/Kg-K)
h_int=15/1000        #convective heat coefficient (kW/m^2.K)
h_ext=20/1000        #convective heat coefficient (kW/m^2.K)
U_w=3.24/1000        #Window heat transfer coefficient (kW/m^2.K)
sigma_w=0.39         #Solar gain coefficeint of window
T_ac=286             #Supply air temperature K


Vp=np.sum(np.array([2083,2083,2083]))                 #Plenum's volume (m^3)

M_Ein=np.array([6120,6120,4080,4080,253264])   #Thermal mass of element e(internal) (kJ/K)


M_Eblk=np.array([1544,1544,1029,1029,80511])  #Thermal mass of element e(Bulk)    (kJ/K)

M_Eext=np.array([13840,13840,9227,9227,23892]) #Thermal mass of element e(External) (kJ/K)

A_E=np.array([420,420,280,280,1667])           #Area of building element e (m^2)

sigma_E=np.array([0.6,0.6,0.6,0.6,0.7])        #Solar gain coefficient of e

U_E1=np.array([0.465,0.465,0.465,0.465,7.090])/1000 #Internal heat coefficient (interior to Bulk) (kW/m^2.K)
U_E2=np.array([0.852,0.852,0.852,0.852,0.385])/1000 #Internal heat coefficient (Bulk to exterior) (kW/m^2.K)


Masses=[Vp,M_Ein,M_Eblk,M_Eext,A_E,sigma_E,U_E1,U_E2]


data=pd.read_excel(r'Building_inputs.xlsx') #Importing All arrays data from excel
d=data.to_numpy()
A_ZE1=d[:15,:5]              #Surface area b/w z and e (m^2)
A_ZE1=A_ZE1.astype(float)
A_ZE=np.sum(A_ZE1,axis=0)

A_ZW1=d[:15,6:10]
A_ZW=A_ZW1.astype(float)     #Surface area b/w z and w (m^2)
A_ZW=np.sum(A_ZW1,axis=0)
A_ZW=A_ZW.astype(float)

A_PE1=d[:3,11:16]
A_PE1=A_PE1.astype(float)     #Surface area b/w pl and e (m^2)
A_PE=np.sum(A_PE1,axis=0)

M_I1=d[17:,0]
M_I1=M_I1.astype(float)       #Thermal mass of element i (kJ/K)
M_I=[np.sum(M_I1[:24]),np.sum(M_I1[24:39]),np.sum(M_I1[39:])]
M_I=np.asarray(M_I)

A_PI1=d[17:,2:5]
A_PI1=A_PI1.astype(float)     #Surface area b/w pl and i (m^2)
A_PI=[np.sum(A_PI1[:24,:]),np.sum(A_PI1[24:39,:]),np.sum(A_PI1[39:,:])]
A_PI=np.asarray(A_PI)


A_ZI1=d[17:,6:]
A_ZI1=A_ZI1.astype(float)     #Surface area b/w z and i  (m^2)
A_ZI=[np.sum(A_ZI1[:24,:]),np.sum(A_ZI1[24:39,:]),np.sum(A_ZI1[39:,:])]
A_ZI=np.asarray(A_ZI)


Areas=[A_ZE,A_ZW,A_PE,M_I,A_PI,A_ZI]

q_zLeak=0.001*np.sum(A_ZE)   #Zone leakage flow rate of air m^3/s
q_PLeak=0.001*np.sum(A_PE)   #Plenum leakage flow rate of air m^3/s
q=[q_zLeak,q_PLeak]


#%%

#Pyomo variables creation
def create_model(days):
    t_end=24*days  #t_end in hours
    m=ConcreteModel()
    m.t=ContinuousSet(bounds= (0, t_end))       
    m.q_vent=Var(m.t,bounds=(q_zOA_min, q_ve_up)) #Total airflow rate [m^3/hr]

    
    m.q_OA=Var(m.t,bounds=(q_zOA_min, q_ve_up))  #Fresh airflow rate [m^3/hr]

    
    m.q_ret=Var(m.t,bounds=(0, q_ve_up))    #Return airflow rate [m^3/hr]

    #Maximum cooling load is 300 kW[kJ/s]
    m.Q_ac=Var(m.t,bounds=(0,300*3600))         #Cooling load [kJ/hr]

    
    m.P_ac=Var(m.t,bounds=(0,100*3600))         #Cooling power calculation based on COP [kJ/hr]

    
    m.Q_vent=Var(m.t,within=NegativeReals)      #ventilation of air [kW]

    
    m.P_vent=Var(m.t,bounds=(0,31*3600))           #Ventilation power [kW]

    
    m.T_z=Var(m.t)          #Zone temperature [K]
    m.T_z_dt=DerivativeVar(m.T_z,wrt=m.t)
    
    m.T_in=Var(m.t,[0,1,2])  #Internal element temperature [K]
    m.T_in_dt=DerivativeVar(m.T_in,wrt=m.t)
    
    m.T_Ein=Var(m.t,[0,1,2,3,4])        #External element temperature [K]
    m.T_Ein_dt=DerivativeVar(m.T_Ein,wrt=m.t)
    
    m.T_Eb=Var(m.t,[0,1,2,3,4])         #External bulk temperature [K]
    m.T_Eb_dt=DerivativeVar(m.T_Eb,wrt=m.t)
    
    m.T_Ex=Var(m.t,[0,1,2,3,4])         #[K]
    m.T_Ex_dt=DerivativeVar(m.T_Ex,wrt=m.t)
    
    m.obj = Var(m.t)
    m.dobj_dt = DerivativeVar( m.obj , wrt =m.t )
    return m

#%%
#Interpolation of Solar radiation, ambient temperature data
def interpolating_data(m,days):
    n1=24*days  #defining number of days to simulate 
    #Importing Ambient Temperature (Every Hour data recorded inside the file)
    T_dat=loadmat('AmbientTemp.mat')   #[time;Ambient Temp]
    T_dat1=T_dat['myTemp']  #units: K
    T_am=T_dat1[1,:n1]  
    print('Minumum Ambient Temp: ',np.min(T_am)-273.15)
    print('Maximum Ambient Temp: ',np.max(T_am)-273.15)
    t_am=T_dat1[0,:n1] 
    t_am=t_am.astype(int)
    t_am=t_am/3600
    
    
    #Importing Solar radiation on window (H_w) (Every Hour data recorded)
    #Size(timepoints,4(windows))
    H_dat=loadmat('simWallInsol.mat')   # units : W/m^2
    H1=H_dat['myWallInsol']     
    H2=H1[1:,:n1]/1000   #[kW/m^2]
    
    #Importing Solar radiation on roof (H_e) (Every Hour data recorded)
    He_dat=loadmat('simGHI.mat')  #[time;roof radiation] units: W/m^2
    H_e1=He_dat['myGhi']
    H_e2=H_e1[1,:n1]/1000   #[kW/m^2]
    
    ##Importing Heat generation  (Every Minute data recorded)
    Q=pd.read_excel(r'HeatGeneration_9days.xlsx') #units :kW
    Q_gen1=Q.to_numpy() 
    d=n1*60       
    t1=Q_gen1[:d,0]
    t2=t1/3600
    Q_gen2=Q_gen1[:d,1:]  # units : [kW]
    Q_gen=np.sum(Q_gen2,axis=1)
    
    #Interpolating the Ambient Temperature, H and H_e values according to Q_gen
    T_Amb_inter=interp1d(t_am,T_am,kind='linear',fill_value='extrapolate')
    T_amb=T_Amb_inter(m.t)   #Units : K
    
    
    H_w_inter=interp1d(t_am,H2,kind='linear',fill_value='extrapolate')
    H_w=H_w_inter(m.t)
    
    Q_w=np.matmul(H_w.T,A_ZW)*sigma_w
    
    He_inter=interp1d(t_am,H_e2,kind='linear',fill_value='extrapolate')
    H_e3=He_inter(m.t)
    
    Q_gen_inter=interp1d(t2,Q_gen,kind='linear',fill_value='extrapolate')
    Q_gen_f=Q_gen_inter(m.t)
    
    #Stacking solar gain of 5 external elements
    #(H is having 4 elements values and H_e3 is roof) 
    
    H_e=np.row_stack((H_w,H_e3))
    return T_amb,Q_w,Q_gen_f,H_e

#%%

from functools import reduce
import operator
def sum_a(iterable):
    return reduce(operator.add, iterable, 0)

# Formulating constraints and algebriac expressions
def constraints(m,T,prices,T_amb,Q_w,Q_gen_f,H_e):
    m.constraints=ConstraintList()
    n=0  #Countig total time points
    u=0
    j=8
    c=[18+(i*24) for i in range(days+1)]
    prices_h=np.zeros(len(m.t))
    Tz_Low=np.zeros(len(m.t))
    Tz_High=np.zeros(len(m.t))
    t_end=24*days
    
    for t in m.t:
        m.constraints.add(rho*Cp*(Vz+Vp)*m.T_z_dt[t]==(3600*(rho*Cp*(q_zLeak+q_PLeak)*(T_amb[n]-m.T_z[t])
                       +(h_int*sum_a((A_ZE[e]+A_PE[e])*m.T_Ein[t,e] for e in range(5)))
                       -(np.sum(A_ZE)+np.sum(A_PE))*h_int*m.T_z[t]
                       +(h_int*sum_a((A_ZI[k]+A_PI[k])*m.T_in[t,k] for k in range(3)))
                       -(np.sum(A_ZI)+np.sum(A_PI))*h_int*m.T_z[t]
                       +np.sum(A_ZW)*U_w*(T_amb[n]-m.T_z[t])
                       +Q_w[n]+Q_gen_f[n])+m.Q_vent[t]       
                       ))
        
        for i in range(3):
            m.constraints.add(M_I[i]*m.T_in_dt[t,i]==(3600*(h_int*(A_ZI[i]+A_PI[i])*(m.T_z[t]-m.T_in[t,i]))))
    
        for i in range(5):
            m.constraints.add(M_Ein[i]*m.T_Ein_dt[t,i]==(3600*(A_E[i]*U_E1[i]*(m.T_Eb[t,i]-m.T_Ein[t,i])
            +h_int*(A_ZE[i]+A_PE[i])*(m.T_z[t]-m.T_Ein[t,i]))))
    
        for i in range(5):
            m.constraints.add(M_Eblk[i]*m.T_Eb_dt[t,i]==(3600*(A_E[i]*U_E1[i]*(m.T_Ein[t,i]-m.T_Eb[t,i])
            +A_E[i]*U_E2[i]*(m.T_Ex[t,i]-m.T_Eb[t,i]))))
        
        for i in range(5):
            m.constraints.add(M_Eext[i]*m.T_Ex_dt[t,i]==(3600*(A_E[i]*U_E2[i]*(m.T_Eb[t,i]-m.T_Ex[t,i])
            +A_E[i]*h_ext*(T_amb[n]-m.T_Ex[t,i])+sigma_E[i]*A_E[i]*H_e[i,n])))
            
        
        m.constraints.add(m.q_vent[t]== m.q_OA[t]+m.q_ret[t])
        
        
        m.constraints.add(m.Q_vent[t]== (m.q_vent[t]*rho*Cp*(T_ac-297.15)))
        
        m.constraints.add(m.P_vent[t]== (1*m.q_vent[t]))
        
        m.constraints.add(m.Q_ac[t]== (rho*Cp*m.q_OA[t]*(T_amb[n]-T_ac)
                                    +rho*Cp*m.q_ret[t]*(297.15-T_ac)))
        
        m.constraints.add(m.P_ac[t]== ((1/3)*m.Q_ac[t]))    
    
        if (t) >= j and (t)<=c[u]:   
            m.constraints.add(inequality(294.15,m.T_z[t],297.15))
            Tz_Low[n]=21
            Tz_High[n]=24
        elif (t)>=c[u]:
            j=j+24
            u=u+1
            Tz_Low[n]=18
            Tz_High[n]=27    
        else:
            m.constraints.add(inequality(291.15,m.T_z[t],300.15))
            Tz_Low[n]=18
            Tz_High[n]=27        
        
        t_p=0
        while t>(t_p+1):
            t_p=t_p+1
        
        prices_h[n]=prices[t_p]
        #electricity price in [cent/kWh] and P_vent&P_vent in [kW] and divided by 100 to get cent to $    
        m.constraints.add(3600*100*m.dobj_dt[t]== (prices[t_p])*(m.P_vent[t]+m.P_ac[t]))
        n=n+1
    
    m.constraints.add(m.T_z[0] == T['T_z'])
    
    for i in range(5):
        m.constraints.add(m.T_Ein[0,i]==T['T_E'][i])
        m.constraints.add(m.T_Eb[0,i]==T['T_E'][i])
        m.constraints.add(m.T_Ex[0,i]==T['T_E'][i])
    for i in range(3):
        m.constraints.add(m.T_in[0,i]==T['T_I'][i])
    
    m.constraints.add(m.obj[0] == 0)
    
    m.objective = Objective ( expr = m.obj[t_end])
    return prices_h,Tz_Low,Tz_High
    

#%%

count=0
def plotting(m,count):

    T_z=np.zeros(len(m.t))
    T_in=np.zeros((len(m.t),3))
    T_Ein=np.zeros((len(m.t),5))
    T_Eb=np.zeros((len(m.t),5))
    T_Ex=np.zeros((len(m.t),5))
    time=np.zeros(len(m.t))
    Q_vent=np.zeros(len(m.t))
    Q_ac=np.zeros(len(m.t))
    P_ac=np.zeros(len(m.t))
    P_vent=np.zeros(len(m.t))
    q_ve=np.zeros(len(m.t))
    q_Oa=np.zeros(len(m.t))
    q_re=np.zeros(len(m.t))
    
    objective=np.zeros(len(m.t))
    for i in m.t:
        time[count]=i
        for j in range(3):
            T_in[count,j]=value(m.T_in[i,j])-273.15
        for j in range(5):
            T_Ein[count,j]=value(m.T_Ein[i,j])-273.15
        for j in range(5):
            T_Eb[count,j]=value(m.T_Eb[i,j])-273.15
        for j in range(5):
            T_Ex[count,j]=value(m.T_Ex[i,j])-273.15   
        Q_vent[count]=value(m.Q_vent[i])/3600
        Q_ac[count]=value(m.Q_ac[i])/3600
        P_vent[count]=value(m.P_vent[i])/3600
        P_ac[count]=value(m.P_ac[i])/3600
        q_ve[count]=value(m.q_vent[i])/3600
        q_Oa[count]=value(m.q_OA[i])/3600
        q_re[count]=value(m.q_ret[i])/3600
        objective[count]=value(m.obj[i])
        T_z[count]=value(m.T_z[i])
        count=count+1
    
    results={'T_z':T_z,'T_in':T_in,'T_Ein':T_Ein,'T_Eb':T_Eb,'T_Ex':T_Ex,
             'Q_ac':Q_ac,'P_ac':P_ac,'Q_vent':Q_vent,'P_vent':P_vent,
             'q_vent':q_ve,'q_OA':q_Oa,'q_return':q_re}
    #results={'Temp':[T_z,T_in,T_Ein,T_Eb,T_Ex],'Power':[Q_ac,P_ac,Q_vent,P_vent],'Airflowrate':[q_ve,q_Oa,q_re]}
    return time,results,objective
        
#%%
   
def optimization_euler(days,T,prices,nfe):
    m=create_model(days)
    discretizer = TransformationFactory ('dae.finite_difference') 
    discretizer.apply_to( m , wrt =m.t , nfe = nfe , scheme='BACKWARD' )
    T_amb,Q_w,Q_gen_f,H_e=interpolating_data(m,days)
    prices_h,Tz_Low,Tz_High=constraints(m,T,prices,T_amb,Q_w,Q_gen_f,H_e)
    
    solver = SolverFactory('gurobi')
    results = solver.solve(m, tee =True)
    #results.write()
    time,results,objective = plotting(m,count)
    return time,results,objective,prices_h,Tz_Low,Tz_High


hourly_prices=[5,5,5,5,5,5,10,10,10,10,10,10,10,10,15,15,15,15,15,15,10,10,5,5] #[cent/kWh] 24 hours
hourly_prices=hourly_prices  #[cent/kWh]
days=9
prices=[]  #Array containing prices per hour for all days
for i in range(days):
    prices=prices+hourly_prices

data_price=pd.read_csv(r'20200601-20200610 ERCOT Real-time Price.csv') #Importing All arrays data from excel
price_data=data_price.to_numpy()
prices_RP=price_data[:,1][1::8]

    
T_z0=295.65
T_E0=[295.65,295.65,295.65,295.65,295.65]
T_I0=[295.65,295.65,295.65]
T={'T_z':T_z0,'T_E':T_E0,'T_I':T_I0}

dr='dae.finite_difference'
nfe = 24*4*days

t,results,objective,prices_h,Tz_Low,Tz_High=optimization_euler(days,T,prices,nfe)
t_RP,results_RP,objective_RP,prices_h_RP,Tz_Low_RP,Tz_High_RP=optimization_euler(days,T,prices_RP,nfe)
time_euler=t


#%%

def optimization_collocation(days,T,prices,nfe,ncp):
    m1=create_model(days)
    discretizer = TransformationFactory ('dae.collocation') 
    discretizer.apply_to( m1 , wrt =m1.t , nfe = nfe , ncp =ncp,scheme='LAGRANGE-RADAU' )
    T_amb,Q_w,Q_gen_f,H_e=interpolating_data(m1,days)
    prices_h,Tz_Low,Tz_High=constraints(m1,T,prices,T_amb,Q_w,Q_gen_f,H_e)
    solver = SolverFactory('gurobi')
    results = solver.solve(m1, tee =True)
    #results.write()
    time1,results1,objective1 = plotting(m1,count)
    return time1,results1,objective1,prices_h,Tz_Low,Tz_High

ncp = 4

dr1='dae.collocation'

t1,results1,objective1,prices_h1,Tz_Low1,Tz_High1=optimization_collocation(days,T,prices,nfe,ncp)
t1_RP,results1_RP,objective1_RP,prices_h1_RP,Tz_Low1_RP,Tz_High1_RP=optimization_collocation(days,T,prices_RP,nfe,ncp)


#%%
t_hr=t

params = {'mathtext.default': 'regular' , 'font.size':12 }          
plt.rcParams.update(params)


#%%%
plt.figure()
plt.subplot(3,1,1)
plt.plot(t_hr,results['P_ac']+results['P_vent'])
plt.plot(t1,results1['P_ac']+results1['P_vent'])
plt.legend(['Euler','Collocation'],loc='upper right')
plt.ylabel('Total Power [kW]')

plt.subplot(3,1,2)
plt.plot(t_hr,prices_h)
plt.ylabel('Price [cent/kWh]')

plt.subplot(3,1,3)
plt.plot(t_hr,Tz_Low,'g-')
plt.plot(t_hr,Tz_High,'b-')
plt.plot(t_hr,results['T_z']-273.15)
plt.plot(t1,results1['T_z']-273.15)
plt.legend(['L Limit','H Limit',u'$T_{z}$ Euler',u'$T_{z}$ Collocation'],loc='upper right')
plt.ylabel('Temperature [\u00B0C]')
plt.xlabel('Time [h] ')
plt.show()

#%%
plt.figure()
plt.subplot(3,1,1)
plt.plot(t_RP,results_RP['P_ac']+results_RP['P_vent'])
plt.plot(t1_RP,results1_RP['P_ac']+results1_RP['P_vent'])
plt.legend(['Euler','Collocation'],loc='upper right')
plt.ylabel('Total Power [kW]')

plt.subplot(3,1,2)
plt.plot(t_RP,prices_h_RP)
plt.ylabel('Price [cent/kWh]')

plt.subplot(3,1,3)
plt.plot(t_hr,Tz_Low,'g-')
plt.plot(t_hr,Tz_High,'b-')
plt.plot(t_RP,results_RP['T_z']-273.15)
plt.plot(t1_RP,results1_RP['T_z']-273.15)
plt.legend(['L Limit','H Limit',u'$T_{z}$ Euler',u'$T_{z}$ Collocation'],loc='upper right')
plt.ylabel('Temperature [\u00B0C]')
plt.xlabel('Time [h] ')
plt.show()


#%%

fig, ax1 = plt.subplots()
params = {'mathtext.default': 'regular','font.size':16 }          
plt.rcParams.update(params)
color = 'tab:red'
ax1.set_xlabel('time [hr]')
ax1.set_ylabel('Total Power', color=color)
ax1.plot(t_hr,results['P_ac']+results['P_vent'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('price [cent/kWh]', color=color)  # we already handled the x-label with ax1
ax2.plot(t_hr,prices_h, color=color)
ax2.tick_params(axis='y', labelcolor=color)


fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

#%%
fig, ax3 = plt.subplots()
params = {'mathtext.default': 'regular','font.size':16 }          
plt.rcParams.update(params)
color = 'tab:red'
ax3.set_xlabel('time [hr]')
ax3.set_ylabel('Total Power', color=color)
ax3.plot(t_hr,results_RP['P_ac']+results_RP['P_vent'], color=color)
ax3.tick_params(axis='y', labelcolor=color)

ax4 = ax3.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax4.set_ylabel('price [cent/kWh]', color=color)  # we already handled the x-label with ax1
ax4.plot(t_hr,prices_h_RP, color=color)
ax4.tick_params(axis='y', labelcolor=color)


fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()




