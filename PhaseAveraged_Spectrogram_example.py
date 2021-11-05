# =============================================================================
# This code presents an example of use of the PhaseAveraged_Spectrogram functions
# to study the noise generetad by an oscillating airfoil in a turbulent flow.
#
# The data file contains:
# - First column: Angle of attack signal
# - Second column: Far-field acoustic pressure signal
#  
# David Raus (21/11/05)
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import scipy as sp
import scipy.io
from scipy import signal

def main():
    global data_Raw,t_interp,f_PSD,PSD_TSA
    
    # Initialization
    fs = 51200                # Microphone sampling frequency (Hz)
    pref = 20e-6              # Acoustic pressure (Pa)
    
    U = 25                    # Flow speed
    alpha0 = 15               # Mean angle of attack (in degrees)
    alpha1 = 15               # Oscillation amplitude (in degrees)
    f = 1.33                  # Oscillation frequency (Hz)
    tc = 1/f                  # Cycle length (s)
    sens = 3.59e-3            # Microphone sensitivity (V/Pa)
    
    # Loading microphone and angle of attack data
    fstr = str(f)
    fstr = fstr[0]+'p'+fstr[2:4]
    path_data = 'data_Raw_example/data_Dyn_N12_U' + str(U) + '_AOA' + str(alpha0) + '_' + str(alpha1) + '_f' + fstr + '.mat'
    data_Raw = scipy.io.loadmat(path_data)
    data_Micros = data_Raw['data_Mic']
    
    # Create time array
    t = np.arange(0,data_Micros.shape[0])
    t = t/fs                  # (en s)
    
    # Create angle of attack vector
    AOA = np.array(data_Micros[:,1])
    AOA = AOA * alpha0/max(AOA)                 
    AOA = AOA + alpha1
    
    
    # High-pass filter of the far-field acoustic pressure signal in order
    # to limit the background noise of the wind tunnel
    f_HighPass = 50
    sos = signal.butter(2, f_HighPass, 'hp', fs = fs, output = 'sos')
    data_filter = signal.sosfilt(sos, data_Micros[:,0])
    
    # Detect index of the beginning of the cycles
    cycle1 = 11;                     # ignore the first 10 cycles of the oscillation to avoid transcient effects
    ind_min,ind_min_approximate = detection_cycles(AOA,cycle1)
 
    
    # Plot the angle of attack signal
    plt.figure()
    plt.plot(t,AOA)
    plt.plot(t[ind_min_approximate],AOA[ind_min_approximate],"o")
    plt.xlabel('t(s)')
    plt.gca().set_ylabel(r'$\alpha$($^o$)')
    rc('font', **{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)
    #plt.xlim(64,65)
    plt.title('Detect the cycles indexes')
    plt.show()
    
    # Parameters of the fft for the spectrograms
    window = 3500
    overlap = window*0.8
    nfft = 8192    
    
    # %%
    # =============================================================================
    # Phase-averaging spectrograms
    # =============================================================================
    
    # If all the cycles have close durations:
    from PhaseAveraged_Spectrogram_functions import PhaseAveraged_spectrogram_MeanCycleLength
    PSD_CL_mean, t_CL_mean, f_CL_mean = PhaseAveraged_spectrogram_MeanCycleLength(data_filter,ind_min_approximate,window,overlap,nfft,fs,sens)
                          
    # If the cycles have varying durations, this function can give better results:   
    from PhaseAveraged_Spectrogram_functions import PhaseAveraged_spectrogram_TSA
    ind_fmax = 500                    # frequecy limit for the computation of the spectrograms
    PSD_TSA,t_TSA,f_PSD = PhaseAveraged_spectrogram_TSA(data_filter,t,ind_min,ind_fmax,window,overlap,nfft,fs,sens,"Time domain")
    
    # Compute the Overall Sound Pressure Level on the phase-averaged spectrogram
    from PhaseAveraged_Spectrogram_functions import OASPL_fromSpectrogram
    
    f_min_OASPL = 70                  # Frequency limits for the integration
    f_max_OASPL = 700
    
    OASPL_approximate = OASPL_fromSpectrogram(PSD_CL_mean,f_CL_mean,t_CL_mean,f_min_OASPL,f_max_OASPL)
    OASPL_TSA = OASPL_fromSpectrogram(PSD_TSA,f_PSD,t_TSA,f_min_OASPL,f_max_OASPL)
    
    
    # %%
    # =============================================================================
    # Plots
    # =============================================================================
    
    # Plot the phase-averaged Power Spectral Density
    AOA_trace = np.array([5, 14, 25])    # choice of the angles of attack to plot
    
    plt.figure()
    for ii in AOA_trace:
        plt.semilogx(f_CL_mean,10*np.log10(PSD_CL_mean[:,ii]/pref**2))   
    plt.xlabel('f(Hz)')
    plt.ylabel('10 log$_{10}(S_{pp}/p_{\mathrm{ref}}^2)$ (dB/Hz)')
    plt.xlim(40,6000)
    plt.ylim(10,45)
    plt.legend(['Pre-stall','Light-stall','Deep-stall'])
    #plt.savefig("DSP.pdf")
    
    
    # Plot OASPL
    plt.figure()
    plt.plot(t_TSA/tc,10*np.log10(OASPL_TSA/(pref**2)))
    plt.plot(t_CL_mean/tc,10*np.log10(OASPL_approximate/(pref**2)))
    plt.xlabel('$f_{0}t$')
    plt.ylabel('OASPL (dB)')
    plt.legend(['TSA','Approximate cycles length'])
    plt.xlim(0,1)
    plt.ylim(47,64)
    plt.show()
    
    
    # Plot spectrograms
    f_CL_instant, t_CL_instant, PSD_CL_instant = signal.spectrogram(data_filter[ind_min_approximate[5]:ind_min_approximate[5+1]], fs,mode='psd',nperseg=window,noverlap = overlap,window = 'hann',nfft=nfft,axis=- 1)    
    PSD_CL_instant = PSD_CL_instant/(sens**2)     
    
    z_min = -12         # Colorbar limits
    z_max = 45
    
    plt.subplot(131)
    plt.pcolor(t_CL_instant/tc,f_CL_instant, 10*np.log10(PSD_CL_instant/(pref**2)), vmin=z_min, vmax=z_max,shading='auto', cmap='jet') 
    plt.yscale('log')
    plt.xlabel('$f_{0}t$')
    plt.ylabel('f(Hz)')
    plt.ylim(60,f_PSD[ind_fmax])
    plt.xlim(0,1)
    plt.title('Instantaneous', fontsize=10)
    
    plt.subplot(132)
    plt.pcolor(t_CL_mean/tc,f_CL_mean, 10*np.log10(PSD_CL_mean/(pref**2)), vmin=z_min, vmax=z_max,shading='auto', cmap='jet') 
    plt.yscale('log')
    plt.xlabel('$f_{0}t$')
    plt.ylim(60,f_PSD[ind_fmax])
    plt.xlim(0,1)
    plt.title('Phase-averaged: \n Approximate cycles length', fontsize=10)

    plt.subplot(133)
    plt.pcolor(t_TSA/tc,f_PSD[:ind_fmax], 10*np.log10(PSD_TSA/(pref**2)), vmin=z_min, vmax=z_max,shading='auto', cmap='jet') 
    plt.colorbar(orientation = 'vertical')
    plt.yscale('log')
    plt.xlabel('$f_{0}t$')
    plt.ylim(60,f_PSD[ind_fmax])
    plt.xlim(0,1)
    plt.title('Phase-averaged: \n TSA', fontsize=10)
    plt.show()
    
    
# %%
# =============================================================================
# Function to detect the index of the beginning of the oscillation cycles
# AOA: Vector containing the angle of attack
# cycle1: first cycle to detect
# =============================================================================
def detection_cycles(AOA,cycle1):
    
    ind_min,_ = sp.signal.find_peaks(-(AOA-15), height=14,distance=10000)    # Detect minimums of angle of attack
    
    # If the cycles have close durations, one can choose to use a constant cycle length
    # corresponding to the mean cycle length of all the oscillations
    nb = np.arange(0,len(ind_min)-cycle1);
    ind_min_approximate = ind_min[cycle1] + int(np.mean(np.round(np.diff(ind_min[cycle1:]))))*nb;
    ind_min_approximate = [int(x) for x in ind_min_approximate ]                  # indice du debut des cycles (debut du mouvement ascendant)
   
    return ind_min,ind_min_approximate


# %%
if __name__ == '__main__':
    main()
    
