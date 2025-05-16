# This python code is written by Dr. Roman Anufriev for Nomura lab, IIS, University of Tokyo in 2018-2021
# The code implements Callaway-Holland model which calculates the thermal conductivity of nanostructures.
# Contact me by anufriev.roman@protonmail.com if you have any questions.

from numpy import pi, exp, zeros, savetxt, sqrt, sinh, genfromtxt
import matplotlib.pyplot as plt
from scipy.constants import k, hbar

# PARAMETERS:
N = 1000                    # Number of points
ROUGHNESS_MEM = 0.15e-9      # [nm]
ROUGHNESS_HOLES = 1.5e-9    # [nm]
MATERIAL = 'Si'             # or 'SiC'


def bulk_phonon_dispersion(N):
    '''Return phonon dispersion calculated for N wavevectors over the G-X direction'''
    dispersion = zeros((N,4))

    if MATERIAL == 'Si':                                            # Ref. APL 95 161901 (2009)
        dispersion[:,0] = [k*12e9/(N-1) for k in range(N)]                                                              # Wavevectors
        dispersion[:,1] = [abs(1369.42*k-2.405e-8*(k**2)-9.70e-19*(k**3)) for k in dispersion[:,0]]                     # LA branch
        dispersion[:,2] = [abs(1081.74*k-7.711e-8*(k**2)+5.674e-19*(k**3)+7.967e-29*(k**4)) for k in dispersion[:,0]]   # TA branch
        dispersion[:,3] = dispersion[:,2]                                                                               # TA branch

    if MATERIAL == 'SiC':                                           # https://journals.aps.org/prb/pdf/10.1103/PhysRevB.50.17054
        dispersion[:,0] = [k*14414281503/(N-1) for k in range(N)]                                                       # Wavevectors
        dispersion[:,1] = [abs(-3.48834e-18*(k**3)+1.7604452e-08*(k**2)+1737.36296*k) for k in dispersion[:,0]]         # LA branch
        dispersion[:,2] = [abs(-2.21696e-19*(k**3)-3.4366886e-08*(k**2)+1077.98941*k) for k in dispersion[:,0]]         # TA branch
        dispersion[:,3] = dispersion[:,2]                                                                 # TA branch

    if MATERIAL == 'Diamond':                                   # https://www.sciencedirect.com/science/article/pii/S0008622315003358
        dispersion[:,0] = [k*11707071561.7/(N-1) for k in range(N)]                                                     # Wavevectors
        dispersion[:,1] = [abs(-1.347265e-18*(k**3)-8.855338e-08*(k**2)+4309.95222*k) for k in dispersion[:,0]]     # LA branch
        dispersion[:,2] = [abs(-5.042335e-18*(k**3)-4.104260e-08*(k**2)+3185.66561*k) for k in dispersion[:,0]]     # TA branch
        dispersion[:,3] = dispersion[:,2]                                                                               # TA branch

    # if MATERIAL == 'AlN':                                       # Davydov, PRB 58, 12899 (1998)
        # dispersion[:,0] = [k*12576399382.998995/(N-1) for k in range(N)]                                                     # Wavevectors
        # dispersion[:,1] = [abs(-2.9909771194746976e-18*(k**3)+3.0825871412244506e-08*(k**2)+946.677*k) for k in dispersion[:,0]]     # LA branch
        # dispersion[:,2] = [abs(-3.0479284862841638e-18*(k**3)-2.0881316790638167e-08*(k**2)+1852.2718*k) for k in dispersion[:,0]]     # TA branch
        # dispersion[:,3] = dispersion[:,2]                                                                               # TA branch

    if MATERIAL == 'SiO2':                                            # Ref. APL 95 161901 (2009)
        dispersion[:,0] = [k*12e9/(N-1) for k in range(N)]                                                              # Wavevectors
        dispersion[:,1] = [abs(1369.42*k-2.405e-8*(k**2)-9.70e-19*(k**3)) for k in dispersion[:,0]]                     # LA branch
        dispersion[:,2] = [abs(1081.74*k-7.711e-8*(k**2)+5.674e-19*(k**3)+7.967e-29*(k**4)) for k in dispersion[:,0]]   # TA branch
        dispersion[:,3] = dispersion[:,2]                                                                               # TA branch
    return dispersion


def phonon_properties_assignment_2(j, branch, N, dispersion):
    '''Assign phonon frequency (f) according to provided wavevector and branch, and calculate group velocity from bulk disperion'''
    K = (dispersion[j+1,0]+dispersion[j,0])/2.0                             # Wavevector (we take average in the interval)
    dK = (dispersion[j+1,0]-dispersion[j,0])                                # Delta wavevector
    w = 2.0*pi*abs((dispersion[j+1,branch+1]+dispersion[j,branch+1])/2.0)   # Angular freequency  (we take average in the interval)
    dw = 2.0*pi*abs(dispersion[j+1,branch+1]-dispersion[j,branch+1])        # Delta angular freequency
    # if MATERIAL == "SiO2":
        # speed = 4100 # https://nanoheat.stanford.edu/wp-content/uploads/1994/09/A8.pdf
    # else:
    speed = dw/dK                                                           # Group velocity
    frequency = w/(2*pi)
    polarization = 'LA' if branch == 0 else 'TA'                            # Polarization according to the branch
    return frequency, polarization, speed, w, K, dK


def relaxation_times(K, w, L_parameter, speed, T, polarization):
    '''This function calculates relaxation times and summs them according to the Mathiesen's rule'''
    if MATERIAL == 'Si':
        A = 2.95e-45        # Parameters from bulk fitting
        B = 0.95e-19
        # E = 3.3e-3	    # Crystal boundary
        deb_temp = 152.0    # Debye temperature

        time_i = A*(w**4.0)                           # 1/Impurities scattering
        time_u = B*(w**2.0)*T*exp(-deb_temp/T)        # 1/Umklapp scattering
        #if polarization=='TA':
        #    time_u=((3.28e-19)*(w**2)*T*exp(-140/T))/4			    # ref. JAP 110 034308 (2011)
        #if polarization=='LA':
        #    time_u=((3.28e-19)*(w**2)*T*exp(-140/T))

        # Specularity [APL 106, 133108 (2015)]
        p = exp(-4.0*(K**2.0)*ROUGHNESS_MEM**2.0)
        mem_lim_dimension = 145e-9
        time_mem = (speed/(mem_lim_dimension))*(1.0-p)/(1.0+p)    # 1/Scattering membrane top and bottom
        # time_relaxation = 1.0/(time_i+time_u+time_mem)                            # Total relaxation time

        p = exp(-4.0*(K**2.0)*ROUGHNESS_HOLES**2.0)
        time_h = (speed/L_parameter)*(1.0-p)/(1.0+p)                            # scattering in the necks
        time_relaxation = 1.0/(time_i+time_u+time_h)                            # Total relaxation time

    if MATERIAL == 'SiC':
        # Parameters from Joshi et al, JAP 88, 265 (2000
        A = 8.46e-45        # Parameters from bulk fitting
        B = 6.16e-20
        C = 6.9e-23
        # E = 10e-5	        # Crystal boundary
        deb_temp = 1200     # Debye temperature

        time_i = A*(w**4.0)                             # 1/Impurities scattering
        time_u = B*(w**2.0)*T*exp(-deb_temp/T)          # 1/Umklapp scattering

        # Specularity [APL 106, 133108 (2015)]
        p = exp(-4.0*(K**2.0)*ROUGHNESS_HOLES**2.0)
        time_h = (speed/L_parameter)*(1.0-p)/(1.0+p)    # 1/Scattering on boundaries

        # p = exp(-4.0*(K**2.0)*ROUGHNESS_MEM**2.0)
        # mem_lim_dimension = 150e-9
        # time_mem = (speed/(mem_lim_dimension))*(1.0-p)/(1.0+p)    # 1/Scattering membrane top and bottom

        time_4p = C*(T**2)*(w**2)                       # 1/Four phonon processes
        time_relaxation = 1.0/(time_i+time_u+time_h+time_4p)                            # Total relaxation time

    # Dummy parameters:
    if MATERIAL == 'Diamond':
        # Parameters from Joshi et al, JAP 88, 265 (2000
        A = 8.46e-45        # Parameters from bulk fitting
        B = 6.16e-20
        C = 6.9e-23
        # E = 10e-5	        # Crystal boundary
        deb_temp = 1200     # Debye temperature

        time_i = A*(w**4.0)                             # 1/Impurities scattering
        time_u = B*(w**2.0)*T*exp(-deb_temp/T)          # 1/Umklapp scattering

        # Specularity [APL 106, 133108 (2015)]
        p = exp(-4.0*(K**2.0)*ROUGHNESS_HOLES**2.0)
        time_h = (speed/L_parameter)*(1.0-p)/(1.0+p)    # 1/Scattering on boundaries

        # p = exp(-4.0*(K**2.0)*ROUGHNESS_MEM**2.0)
        # mem_lim_dimension = 150e-9
        # time_mem = (speed/(mem_lim_dimension))*(1.0-p)/(1.0+p)    # 1/Scattering membrane top and bottom

        time_4p = C*(T**2)*(w**2)                       # 1/Four phonon processes
        time_relaxation = 1.0/(time_i+time_u+time_h+time_4p)                            # Total relaxation time

    # Dummy parameters:
    if MATERIAL == 'AlN':
        # Parameters from  PHYSICAL REVIEW MATERIALS 4, 044602 (2020)
        time_relaxation = 1/1e9

    if MATERIAL == 'SiO2':
        p = exp(-4.0*(K**2.0)*ROUGHNESS_MEM**2.0)
        SiO2_MFP = 2e-9
        time_internal = (speed/SiO2_MFP) # 1/time
        time_mem = (speed/L_parameter)*(1.0-p)/(1.0+p)    # 1/Scattering on boundaries
        time_relaxation = 1.0/(time_internal+time_mem)                            # Total relaxation time

    return time_relaxation


def main():
    '''The main function that calculates the thermal conductivity'''

    if MATERIAL == 'Si':
        material_density = 2330 #[kg/m^3]
    if MATERIAL == 'SiC':
        material_density = 3160 #[kg/m^3]
    if MATERIAL == 'Diamond':
        material_density = 3500 #[kg/m^3]
    if MATERIAL == 'AlN':
        material_density = 3255 #[kg/m^3]
    if MATERIAL == 'SiO2':
        material_density = 2196 #[kg/m^3]

    dispersion = bulk_phonon_dispersion(N+1)
    temperatures = range(5, 301, 5)
    thermal_conductivities = zeros((len(temperatures),2))
    velocities = zeros((N,4))
    specific_heat_capacities = zeros((len(temperatures),2))
    Lc = [l*1e-9 for l in range(10, 5000, 100)]
    # thermal_conductivities = zeros((len(Lc),2))
    # specific_heat_capacities = zeros((len(Lc),2))

    # This is the limiting dimension, i.e. neck of PnC or NW width etc.
    # L_parameter=1.12*sqrt((145e-9)*(5000e-9))
    # L_parameter=sqrt(2)*150e-9

    for index, T in enumerate(temperatures):
        # for index, L in enumerate(Lc):
        # L_parameter=1.12*sqrt((145e-9)*(L))
        L_parameter = 1.12*220e-9
        thermal_conductivity = 1.0
        total_heat_capacity = 0.0
        for branch in range(3):                                             # For each phonon branch
            for j in range(N):

                # Getting parameters of the phonon branch at this point:
                frequency,polarization,speed,w,K,dK = phonon_properties_assignment_2(j,branch,N,dispersion)
                time_relaxation = relaxation_times(K, w, L_parameter, speed, T, polarization)

                # Calculate heat capacity and thermal conductivity at this branch at this wavevector:
                heat_capacity = k*((hbar*w/(k*T))**2.0)*exp(hbar*w/(k*T))/((exp(hbar*w/(k*T))-1.0)**2.0) # Ref. PRB 88 155318 (2013)

                thermal_conductivity += (1.0/(6.0*(pi**2.0)))*heat_capacity*(speed**2.0)*time_relaxation*K**2.0*dK

                # Total heat capacity integrated over all branches and wavevectors:
                total_heat_capacity += (1/(2*pi**2))*heat_capacity*K**2.0*dK    #[J/K]
                # mfp[j,0]=(w/(2*pi))*1e-12
                # mfp[j,branch+1]=speed*time_relaxation

                velocities[j, 0] = frequency
                velocities[j, branch+1] = speed

        # Record the values at this temperature:
        thermal_conductivities[index,0] = T
        thermal_conductivities[index,1] = thermal_conductivity
        # thermal_conductivities[index,0] = L*1e9

        specific_heat_capacities[index,0] = T
        specific_heat_capacities[index,1] = total_heat_capacity/material_density # [J/kg/K]


    # data_bulk = genfromtxt('Slits/T300K.csv', unpack = True,  delimiter=',', skip_header = 1)
    # data_bulk = genfromtxt('PnC-SiC-T300K.csv', unpack = True,  delimiter=',', skip_header = 1)
    # data_bulk = genfromtxt('SiC-3C K Bulk [8].csv', unpack = True,  delimiter=',', skip_header = 1)
    # plt.plot(data_bulk[0], data_bulk[1], 'o')

    # Plot thermal conductivity:
    plt.plot(thermal_conductivities[1:,0],thermal_conductivities[1:,1])
    plt.ylabel('Thermal conductivity (W/mK)')
    plt.xlabel('Temperature (K)')
    # plt.xlabel('Limiting dimension (nm)')
    plt.show()
    savetxt('thermal_conductivity.csv', thermal_conductivities, delimiter=",")

    # Plot group velocity:
    plt.plot(velocities[:,0], velocities[:,1])
    plt.plot(velocities[:,0], velocities[:,2])
    plt.ylabel('Group velocity (m/s)')
    plt.xlabel('Frequency (Hz)')
    plt.show()

    #plt.loglog(mfp[:,0],mfp[:,1],mfp[:,0],mfp[:,2],mfp[:,0],mfp[:,3])
    #plt.ylabel('Mean Free Path (m)', fontsize=12)
    #plt.xlabel('Frequency (THz)', fontsize=12)
    #plt.show()

    # Plot specific heat:
    plt.plot(specific_heat_capacities[:,0], specific_heat_capacities[:,1])
    plt.ylabel('Specific heat (J/kg/K)', fontsize=12)
    plt.xlabel('Temperature (K)', fontsize=12)
    plt.show()

    # Plot dispersion:
    for i in range(2):
        plt.plot(dispersion[:,0],dispersion[:,i+1])
    plt.ylabel('Frequency (Hz)', fontsize=12)
    plt.xlabel('Wavevector (1/m)', fontsize=12)
    plt.show()


if __name__ == "__main__":
    main()
