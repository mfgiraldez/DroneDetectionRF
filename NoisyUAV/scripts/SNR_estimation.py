"""Estimate the SNR in a raw data pack using the background noise as reference
"""
import numpy as np
from scipy.signal import welch


def generate_frequency_hopping_signal(frequencies, hop_duration, sample_rate, total_duration):
    t = np.arange(0, total_duration, 1 / sample_rate)
    signal = np.zeros_like(t)

    num_hops = int(total_duration / hop_duration)
    for i in range(num_hops):
        start_idx = int(i * hop_duration * sample_rate)
        end_idx = int((i + 1) * hop_duration * sample_rate)
        freq = np.random.choice(frequencies)
        signal[start_idx:end_idx] = np.sin(2 * np.pi * freq * t[start_idx:end_idx])

    return t, signal


def add_awgn_noise(signal, snr):
    """Adds complex AWGN noise to an IQ signal for a target SNR."""
    signal_power = np.mean(np.abs(signal) ** 2)
    noise_power = signal_power / (10 ** (snr / 10))
    
    # Generar ruido complejo: la potencia se divide entre el canal I y el Q
    noise_std = np.sqrt(noise_power / 2)
    noise = (np.random.normal(size=signal.shape) + 1j * np.random.normal(size=signal.shape)) * noise_std
    
    return signal + noise


def psd_snr(datapack, noise, fs, nperseg=1024):

    f_signal, psd_signal = welch(datapack, fs=fs, nperseg=nperseg)
    f_noise, psd_noise = welch(noise, fs=fs, nperseg=nperseg)

    signal_power = np.mean(psd_signal)
    noise_power = np.mean(psd_noise)

    snr = 10 * np.log10(signal_power / noise_power)
    return snr


# Usage-----------------------------------------------------------------------------------------------------------------
def main():
    signal = ''
    background = ""

    fs = 100e6
    signal_data = np.fromfile(signal, dtype=np.float32)
    noise_data =np.fromfile(background, dtype=np.float32)

    snr_value = psd_snr(signal_data, noise_data, fs)
    print("SNR:", snr_value, "dB")


if __name__ == '__main__':
    main()