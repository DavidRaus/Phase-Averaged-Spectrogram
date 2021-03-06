# Phase-averaged spectrogram of a signal

Phase-averaging a cyclic signal is defined as averaging the signal over one full cycle. It can be tricky if cycles have varying length (for example the position of a fan blade as it slows down after switchoff.)

When studying the noise generated by an oscillating object, it is interesting to analyze the phase-averaged spectrogram of the radiated noise.

This code computes the phase-average of the spectrogram of a cyclic noise, even if the cycles have varying duration.

## Algorithms 

**Method 1:** If the cycles all have a close duration (function **PhaseAveraged_spectrogram_MeanCycleLength**). 

1. Detect the index corresponding to the beginning of every cycles
2. Compute the spectrogram of the signal on each cycle
3. Average all the spectra

**Method 2:** If the cycles have different duration, the code use the TSA (Time-Synchronous Averaging) algorithm (function **PhaseAveraged_spectrogram_TSA**), implemented in the pyTSA function:

1. Detect the index corresponding to the beginning of every cycles
2. Compute the spectrogram of the entire signal.
3. Thanks to a loop on frequencies, compute the phase-averaging of the signal for each frequency of the spectrogram separatly, using the time-domain or the frequency-domain method:

Time-domain method:
1. Divide the signal into segments corresponding to the different cycles
2. Interpolate the signals in each segment on the same number of sample
3. Compute the average of all the resampled segments

Frequency-domain method:
1. Divide the signal into segments corresponding to the different cycles
2. Compute the fft of each segment
3. Truncate the results on each segment so that all fft have the same length as the one of the shortest cycle
4. Average all the spectra
5. Compute the inverse fft to obtain the phase-averaged signal in the time domain.

## Example

The code **PhaseAveraged_Spectrogram_example.py** presents an example of use of these two methods, to study the noise generetad by an oscillating airfoil in a turbulent flow. 

## Reference
Bechhoefer, Eric, and Michael Kingsley. "A Review of Time-Synchronous Average Algorithms." Proceedings of the Annual Conference of the Prognostics and Health Management Society, San Diego, CA, September-October, 2009.

### David Raus
21/11/02

