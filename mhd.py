import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

#define some global constants
res=50                                         #resolution of the grid
m_proton=1.67e-27                               #mass of proton in kg
q_proton=1.60217663e-16                         #charge of electron
e_mass=1836                                     #protons are 1836 times as massive as a electrons.
kT=0.026                                        #boltzmann constant/factor
Z=1                                             #ionization factor     
AN=1                                            #atomic number
aidx=1.1                                        #adiabatic index
L_scale=1.0e-3                                  #mm length scale
t_scale=1.0e-6                                  #micro second time scale
t_i=0
t_f=500e-6                                      #run to 500 microseconds
dt=9e-9                                         #time step ns
n_scale=6.022e23                                #avogadros const. as density scale
v_scale=L_scale/t_scale                         #vel. scale
P_scale=AN*m_proton*n_scale*v_scale**2          #pressure scale
n_low=1e-6                                      #min density
T_low=kT/P_scale/n_scale/q_proton               #min temp
P_low=n_low/T_low

#class to store data
class plasma:
    def __init__(self,res):
        self.res=res
        self.number_density = np.zeros((res,res))
        self.energy_density = np.zeros((res,res))
        self.electron_energy_density = np.zeros((res,res))
        self.electron_density = np.zeros((res,res))
        self.electron_pressure = np.zeros((res,res))
        self.ion_temp = np.zeros((res,res))
        self.electron_temp = np.zeros((res,res))
        self.resistivity = np.zeros((res,res))
        self.m = np.zeros((res,res,3))
        self.B = np.zeros((res,res,3))
        self.E = np.zeros((res,res,3))
        self.j = np.zeros((res,res,3))
        #set initial conditions
        self.number_density.fill(n_low)
        self.electron_density.fill(AN*n_low)
        self.energy_density.fill(kT/T_low*n_low*(aidx-1))
        self.electron_energy_density.fill(kT/T_low*n_low*(aidx-1))
       
def calc_src(P, sources):
    #sources=plasma(P.res)
    for i in range(P.res):
        for j in range(P.res):
            v_i=P.m[i,j,:]/P.number_density[i,j]
            v_e=P.m[i,j,:]/P.electron_density[i,j]
            P.ion_temp[i,j] = (aidx-1)*P.energy_density[i,j]/P.number_density[i,j] - np.sum(v_i**2)/2
            P.electron_temp[i,j] = (aidx-1)*P.electron_energy_density[i,j]/P.electron_density[i,j] - np.sum(v_e**2)/2
            if P.ion_temp[i,j] < T_low:
                P.ion_temp[i,j] = T_low
            if P.electron_temp[i,j] < T_low:
                P.electron_temp[i,j] = T_low
            P.resistivity[i,j] = 1/np.sqrt(1/P.ion_temp[i,j]**3 + P.number_density[i,j]**2/L_scale**2)
            #calculate sources
            sources.electron_density[i,j] = (Z*P.number_density[i,j] - P.electron_density[i,j])/dt #update source electron density based on ionization
            sources.m[i,j,:]=np.cross(P.j[i,j,:],P.B[i,j,:])
            sources.energy_density[i,j] = P.number_density[i,j]*np.sum(v_i*P.E[i,j,:]) -\
                                          np.sum(v_i*P.j[i,j,:]) +\
                                          (P.number_density[i,j]*P.resistivity[i,j]*(P.electron_temp[i,j] - P.ion_temp[i,j]))/e_mass
            
            sources.electron_energy_density[i,j] = P.electron_density[i,j]*np.sum(v_e*P.E[i,j,:]) -\
                                                   P.resistivity[i,j]*np.sum(v_i*P.j[i,j,:]) +\
                                                   (P.number_density[i,j]*P.resistivity[i,j]*(P.electron_temp[i,j] - P.ion_temp[i,j]))/e_mass

            #Biermann battery term
            pressure=(P.number_density[i,j]*P.ion_temp[i,j] + P.electron_density[i,j]*P.electron_temp[i,j])
            sources.m[i,j,0] = sources.m[i,j,0] + ( sources.m[i,j,2]*v_i[2] + pressure )
            sources.m[i,j,2] = sources.m[i,j,2] - sources.m[i,j,2] * v_i[0]
            sources.j[i,j,0] = 2*v_i[2]*P.j[i,j,2] - P.j[i,j,2]**2
            sources.j[i,j,2] = v_i[0]*P.j[i,j,2] + v_i[2]*P.j[i,j,0] - P.j[i,j,2]*P.j[i,j,0]
            sources.B[i,j,2] = P.E[i,j,1]
            sources.E[i,j,2] = P.B[i,j,1]
    return sources
    
def calc_flux(P):
    #only works with symmetric grids
    flux_x=plasma(P.res)
    flux_y=plasma(P.res)
    for i in range(P.res):
        for j in range(P.res):
            v_i_fy=P.m[i,j,:]/P.number_density[i,j]
            v_e_fy=P.m[i,j,:]/P.electron_density[i,j]
            v_i_fx=P.m[j,i,:]/P.number_density[j,i]
            v_e_fx=P.m[j,i,:]/P.electron_density[j,i]

            b_fy = P.B[i,j,:]
            b_fx = P.B[j,i,:]

            Pi_fy = ( (aidx-1.1)*P.energy_density[i,j] - (np.sum(v_i_fy**2) * P.number_density[i,j])/2 )
            Pe_fy = ( (aidx-1.1)*P.electron_energy_density[i,j] - (np.sum(v_e_fy**2) * P.electron_density[i,j])/2 )
            P_fy = Pi_fy + Pe_fy

            Pi_fx = ( (aidx-1.1)*P.energy_density[j,i] - (np.sum(v_i_fx**2) * P.number_density[j,i])/2 )
            Pe_fx = ( (aidx-1.1)*P.electron_energy_density[j,i] - (np.sum(v_e_fy**2) * P.electron_density[j,i])/2 )
            P_fx = Pi_fx + Pe_fx

            if P_fy < P_low:
                P_fy = P_low
            if P_fx < P_low:
                P_fx = P_low
               
            #scale velocities based on ionization
            v_fy = (v_i_fy/e_mass + Z*v_e_fy)/(e_mass+1) #produce an electron +1
            v_fx = (v_i_fx/e_mass + Z*v_e_fx)/(e_mass+1)

            #build_flux_y class
            flux_y.number_density[i,j] = P.m[i,j,1] #flux out of surface y is equal to momenutm in y

            flux_y.m=P.m*v_fy
            flux_y.m[i,j,1]=flux_y.m[i,j,1]+P_fy #beirmann

            flux_y.energy_density[i,j] = (P.energy_density[i,j]+Pi_fy)*v_fy[1]

            flux_y.B[i,j,:] = np.flip(P.E[i,j,:])
            flux_y.B[i,j,1] = 0.0 
            flux_y.B[i,j,2] = -flux_y.B[i,j,2]
            
            flux_y.E[i,j,:] = np.flip(P.B[i,j,:])
            flux_y.E[i,j,1] = 0.0 
            flux_y.E[i,j,0] = -flux_y.E[i,j,0]

            flux_y.j[i,j,0] = v_fy[1]*P.j[i,j,0] + v_fy[0]*P.j[i,j,1] - P.j[i,j,1]*P.j[i,j,0]/P.energy_density[i,j]
            flux_y.j[i,j,1] = v_fy[1]*P.j[i,j,1] + v_fy[1]*P.j[i,j,1] - P.j[i,j,1]*P.j[i,j,1]/P.energy_density[i,j]
            flux_y.j[i,j,2] = v_fy[1]*P.j[i,j,2] + v_fy[2]*P.j[i,j,1] - P.j[i,j,1]*P.j[i,j,2]/P.energy_density[i,j]

            flux_y.electron_energy_density[i,j] = P.electron_energy_density[i,j] + Pe_fy*v_e_fy[1]

            flux_y.electron_pressure = Pe_fy

            #build the flux_x class
            flux_x.number_density[i,j] = P.m[j,i,0] #flux out of surface y is equal to momenutm in y

            flux_x.m=P.m*v_fy
            flux_x.m[i,j,0]=flux_x.m[i,j,0]+P_fx #beirmann

            flux_x.energy_density[i,j] = (P.energy_density[j,i]+Pi_fx)*v_fx[0]

            flux_x.B[i,j,0] = 0.0
            flux_x.B[i,j,1] = -P.E[j,i,2]
            flux_x.B[i,j,2] = P.E[j,i,1]
            
            flux_x.E[i,j,:] = 0.0
            flux_x.E[i,j,1] = P.B[j,i,2]
            flux_x.E[i,j,0] = -P.B[j,i,1]

            flux_x.j[i,j,0] = v_fx[0]*P.j[j,i,0] + v_fx[0]*P.j[j,i,0] - P.j[j,i,0]*P.j[j,i,0]/P.energy_density[j,i]
            flux_x.j[i,j,1] = v_fx[0]*P.j[j,i,1] + v_fx[1]*P.j[j,i,0] - P.j[j,i,0]*P.j[j,i,1]/P.energy_density[j,i]
            flux_x.j[i,j,2] = v_fx[0]*P.j[j,i,2] + v_fx[2]*P.j[j,i,0] - P.j[j,i,0]*P.j[j,i,2]/P.energy_density[j,i]

            flux_x.electron_energy_density[i,j] = P.electron_energy_density[j,i] + Pe_fx*v_e_fx[0]

            flux_x.electron_pressure = Pe_fx
    return flux_x,flux_y

def integrate(P,Fx,Fy,S):
    for i in range(1,res,1):
        for j in range(1,res,1):
            P.number_density[i,j] = P.number_density[i,j] - dt*(Fx.number_density[i,j]-Fx.number_density[i-1,j]) - dt*(Fy.number_density[i,j]-Fy.number_density[i,j-1])
            #print((Fx.number_density[i,j]-Fx.number_density[i-1,j])*dt)
            P.energy_density[i,j] = P.energy_density[i,j] - dt*(Fx.energy_density[i,j]-Fx.energy_density[i-1,j]) - dt*(Fy.energy_density[i,j]-Fy.energy_density[i,j-1]) + dt*S.energy_density[i,j]
            P.electron_energy_density[i,j] = P.electron_energy_density[i,j] - dt*(Fx.electron_energy_density[i,j]-Fx.electron_energy_density[i-1,j]) -\
                                             dt*(Fy.electron_energy_density[i,j]-Fy.electron_energy_density[i,j-1]) + dt*S.electron_energy_density[i,j]
            P.electron_density[i,j] = P.electron_density[i,j] - dt*(Fx.electron_density[i,j]-Fx.electron_density[i-1,j]) - dt*(Fy.electron_density[i,j]-Fy.electron_density[i,j-1]) + dt*S.electron_energy_density[i,j]
            P.m[i,j,:] = P.m[i,j,:] - dt*(Fx.m[i,j]-Fx.m[i-1,j]) - dt*(Fy.m[i,j]-Fy.m[i,j-1]) + dt*S.m[i,j,:]
            P.B[i,j,:] = P.B[i,j,:] - dt*(Fx.B[i,j]-Fx.B[i-1,j]) - dt*(Fy.B[i,j]-Fy.B[i,j-1]) + dt*S.B[i,j,:] #this will cause issues with del B = 0. assuming this is ok for the demo.
            P.E[i,j,:] = P.E[i,j,:] - dt*(Fx.E[i,j]-Fx.E[i-1,j]) - dt*(Fy.E[i,j]-Fy.E[i,j-1]) + dt*S.E[i,j,:]
            P.j[i,j,:] = P.j[i,j,:] - dt*(Fx.j[i,j]-Fx.j[i-1,j]) - dt*(Fy.j[i,j]-Fy.j[i,j-1]) + dt*S.j[i,j,:]
    return P

H = plasma(res)
SRC = plasma(res)
t=t_i
N_steps=0
while t<t_f:
    H_s=calc_src(H, SRC)
    H_fx,H_fy=calc_flux(H)
    H=integrate(H,H_fx,H_fy,H_s)
    t=t+dt
    N_steps=N_steps+1
    print('STEP',N_steps)
    if N_steps % 50 == 0:
        #plot the changes in number density during integration
        frame=H.number_density[2:res-2,2:res-2]
        plt.imshow(np.rot90(frame,2), cmap='seismic', norm=colors.LogNorm(vmin=frame.min(), vmax=frame.max()))
        plt.colorbar()
        plt.show()
