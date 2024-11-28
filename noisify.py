import numpy as np

def gaussian_noisify(signal, snr):
    signal_avg = np.mean(signal)
    signal_avg_db = 10 * np.log10(signal_avg)
    noise_avg_db = signal_avg_db - snr
    noise_avg = 10 ** (noise_avg_db / 10)

    noise = np.random.normal(0, np.sqrt(noise_avg), signal.shape)
    noisy_signal = signal + noise

    if noisy_signal[noisy_signal < 0].size > 0 or noisy_signal[noisy_signal > 255].size > 0:
        L = noisy_signal.min()
        #rescale the image to 0-255
        noisy_signal = (255 * (noisy_signal - L) / np.ptp(noisy_signal)).astype(int)


    return noisy_signal

def salt_and_pepper_noise(signal, noise_ratio):
    noise = np.random.binomial(1, noise_ratio, signal.shape)
    noisy_signal = signal.copy()
    noisy_signal[noise == 1] = np.random.choice([0, 255], noisy_signal[noise == 1].shape)

    return noisy_signal


def laplacian_noisify(signal, snr):
    signal_avg = np.mean(signal)
    signal_avg_db = 10 * np.log10(signal_avg)
    noise_avg_db = signal_avg_db - snr
    noise_avg = 10 ** (noise_avg_db / 10)

    noise = np.random.laplace(0, np.sqrt(noise_avg), signal.shape)
    noisy_signal = signal + noise

    return noisy_signal

def noisify(image, noise_type, **kwargs):
    if noise_type == 'gaussian':
        return gaussian_noisify(image, kwargs['snr'])
    elif noise_type == 's&p':
        return salt_and_pepper_noise(image, kwargs['noise_ratio'])
    elif noise_type == 'laplace':
        return laplacian_noisify(image, kwargs['snr'])
    else:
        raise ValueError("Invalid noise type")