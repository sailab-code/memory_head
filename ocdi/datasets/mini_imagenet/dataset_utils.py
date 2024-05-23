import numpy as np

def gaussian_noise(x, noise_params):
    x_noise = x + noise_params.noise_factor * np.random.normal(loc=0.0, scale=noise_params.sig, size=x.shape)
    if noise_params.clip == True:
        x_noise = np.clip(x_noise, 0, 1)
    return x_noise
