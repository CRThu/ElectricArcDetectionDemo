import gen_signal
import gen_ac_current_signal
import matplotlib.pyplot as plt
import numpy as np


def spectrum_circuit(signal, fs, t, flocal: list, fmidc, fmidp):
    # [
    # row1[],
    # row2[]
    # ]
    fout_data = np.empty(shape=[0, len(signal)])
    for flo in flocal:
        _, flo_data = gen_signal.gen_sine(fs, t, flo, 1)
        fmix_data = gen_signal.mix(signal, flo_data)
        fmid_data = gen_signal.bpf(fmix_data, fs, fmidc, fmidp)

        fmid_data_temp = np.expand_dims(fmid_data, axis=0)
        fout_data = np.append(fout_data, fmid_data_temp, axis=0)
    return fout_data


if __name__ == '__main__':
    t1, sf100a1 = gen_signal.gen_sine(25000000, 0.2, 105000, 1)
    _, sf200a2 = gen_signal.gen_sine(25000000, 0.2, 197000, 2)
    _, sf300a3 = gen_signal.gen_sine(25000000, 0.2, 303000, 3)
    _, sf005a01dc03 = gen_signal.gen_sine(25000000, 0.2, 30, 0.1)*(t1*6)
    sf005a01dc03 = sf005a01dc03 + 0.3
    sig = gen_signal.mix(sf005a01dc03, sf100a1 + sf200a2 + sf300a3)
    sig = gen_signal.awgn(sig, 20)

    fout_data = spectrum_circuit(sig, 25000000, 0.2, [100000, 200000, 300000], 400000, 100000)

    # adc采样 2.5Msps, 删除滤波器不稳定部分
    t1_sample = gen_signal.resample(t1, 10)[100:-100]
    fout_data_sample = gen_signal.resample(fout_data, 10)[:,100:-100]

    # 分时采样不同频段
    adc_sample = 83
    adc_ch = 30
    [row, col] = np.shape(fout_data_sample)
    sample_len = col - col % (adc_sample * adc_ch)
    fout_data_adc_sampled = np.empty(shape=[0, int(sample_len / adc_ch)])
    for i in range(row):
        f = fout_data_sample[i, 0:sample_len]
        f = f.reshape(int(sample_len / adc_sample), adc_sample)
        f = f[::adc_ch]
        f = f.reshape(1, np.size(f))
        fout_data_adc_sampled = np.append(fout_data_adc_sampled, f, axis=0)

    # 生成400K调制信号后通过信号平方后经过低通滤波器还原包络
    # 或使用Hilbert变换
    fout_pow2 = np.power(fout_data_adc_sampled, 2)
    fout_software_pow2_lpf = gen_signal.lpf(fout_pow2, 2500000/30, 5000)
    fout_software_lpf = np.power(fout_software_pow2_lpf, 0.5)

    ax0 = plt.subplot(411)
    ax0.plot(fout_software_lpf[0,:])
    ax0.plot(sf005a01dc03[::300])
    ax1 = plt.subplot(412)
    ax1.plot(fout_data_adc_sampled[0,:])
    ax2 = plt.subplot(413)
    ax2.plot(fout_data_adc_sampled[1, :])
    ax3 = plt.subplot(414)
    ax3.plot(fout_data_adc_sampled[2, :])
    plt.show()
