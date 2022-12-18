import numpy as np
from numpy import ndarray
from scipy import signal
import matplotlib.pyplot as plt


def gen_time(fs, t):
    return np.array(np.linspace(0, t, int(fs * t), endpoint=False))


def gen_pll(fs, t, f, a):
    t_array = gen_time(fs, t)
    return t_array, a * signal.square(2 * np.pi * f * t_array)


def gen_sine(fs, t, f, a):
    t_array = gen_time(fs, t)
    return t_array, a * np.sin(2 * np.pi * f * t_array)


def awgn(x, snr, seed=7):
    '''
    加入高斯白噪声 Additive White Gaussian Noise
    :param x: 原始信号
    :param snr: 信噪比
    :return: 加入噪声后的信号
    '''
    np.random.seed(seed)  # 设置随机种子
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / len(x)
    npower = xpower / snr
    noise = np.random.randn(len(x)) * np.sqrt(npower)
    return x + noise


def lpf(data, fs, fp, digital=False):
    sos = signal.butter(8, fp, 'lowpass', fs=fs, output='sos')
    if digital:
        return signal.sosfilt(sos, data)
    else:
        return signal.sosfiltfilt(sos, data)


def bpf(data, fs, fc, fp, digital=False):
    fp1 = fc - fp / 2
    fp2 = fc + fp / 2
    sos = signal.butter(8, [fp1, fp2], 'bandpass', fs=fs, output='sos')
    if digital:
        return signal.sosfilt(sos, data)
    else:
        return signal.sosfiltfilt(sos, data)


def hpf(data, fs, fp, digital=False):
    sos = signal.butter(8, fp, 'highpass', fs=fs, output='sos')
    if digital:
        return signal.sosfilt(sos, data)
    else:
        return signal.sosfiltfilt(sos, data)


def mix(data1, data2):
    return data1 * data2


def resample(data: ndarray, down):
    if data.ndim == 1:
        return data[::down]
    elif data.ndim == 2:
        return data[:, ::down]
    else:
        raise TypeError


if __name__ == '__main__':
    t1, s11 = gen_sine(10000, 0.2, 50, 1)
    _, s12 = gen_sine(10000, 0.2, 150, 0.4)
    _, s13 = gen_sine(10000, 0.2, 350, 0.1)
    s1 = s11 + s12 + s13
    m1 = mix(s12, s13)

    t2, s2 = gen_pll(10000, 0.2, 100, 1)
    s3 = awgn(s2, 10)
    t2r = resample(t2, 10)
    s3r = resample(s3, 10)

    s4 = lpf(s1, 10000, 60)
    s4d = lpf(s1, 10000, 60, digital=True)
    s5 = bpf(s1, 10000, 150, 50)
    s6 = hpf(s1, 10000, 300)

    t3, s71 = gen_sine(100000, 0.2, 50, 0.1)
    _, s72 = gen_sine(100000, 0.2, 1000, 1)
    s71 = s71 + 0.3
    s7 = mix(s71, s72)

    ax1 = plt.subplot(611)
    ax1.plot(t1, s1)
    ax2 = plt.subplot(612)
    ax2.plot(t2, s2)
    ax3 = plt.subplot(613)
    ax3.plot(t2, s3)
    ax3.plot(t2r, s3r)
    ax4 = plt.subplot(614)
    ax4.plot(t1, s4)
    ax4.plot(t1, s5)
    ax4.plot(t1, s6)
    ax5 = plt.subplot(615)
    ax5.plot(t2, m1)
    ax6 = plt.subplot(616)
    ax6.plot(t3, s7)
    plt.show()
