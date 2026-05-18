import numpy as np
import pywt
import torch
from NoisyUAV.funciones.visualizacion import _tensor_a_numpy

def denoise_wavelet_1d(signal: np.ndarray, wavelet='db4', level=1):
    """
    Aplica Denoising Wavelet (DWT) a un array 1D usando un threshold suave universal.
    """
    # 1. Descomposición Wavelet
    coeffs = pywt.wavedec(signal, wavelet, mode='per', level=level)
    
    # 2. Calcular threshold usando la varianza del nivel de detalle más alto (ruido)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
    
    # 3. Aplicar Soft Thresholding a los coeficientes de detalle
    coeffs[1:] = (pywt.threshold(i, value=uthresh, mode='soft') for i in coeffs[1:])
    
    # 4. Reconstrucción
    return pywt.waverec(coeffs, wavelet, mode='per')

def aplicar_denoising_iq(iq_tensor: torch.Tensor, wavelet='db4', level=2) -> np.ndarray:
    """
    Recibe un tensor flotante de I/Q [2, M] y aplica denoising DWT
    a cada canal de forma independiente, devolviendo Arrays Numpy.
    """
    I, Q = _tensor_a_numpy(iq_tensor)
    
    I_denoised = denoise_wavelet_1d(I, wavelet=wavelet, level=level)
    Q_denoised = denoise_wavelet_1d(Q, wavelet=wavelet, level=level)
    
    return I_denoised, Q_denoised
