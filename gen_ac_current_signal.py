import random

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import gen_signal


def gen_ac_current(fs, t, a):
    return gen_signal.gen_sine(fs, t, 50, a)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def gen_arc(data_in, width, jitter):
    data_internal = np.array(data_in)

    avg = (np.max(data_internal) + np.min(data_internal)) / 2
    avg = find_nearest(data_internal, avg)
    data_avg = np.where(data_internal - avg >= 0, 1, 0)
    data_avg_2 = np.delete(data_avg, 0)
    data_avg_2 = np.append(data_avg_2, 0)
    data_avg_3 = np.where(np.abs(data_avg - data_avg_2) > 0, 1, 0)

    counter = 0
    gen_arc_cnt = 0
    arc_current = 0
    while counter < len(data_internal):
        rand_jitter = random.randint(-jitter, jitter)

        if gen_arc_cnt > width:
            arc_current = data_internal[counter]
            gen_arc_cnt -= 1
        elif gen_arc_cnt > 0:
            data_internal[counter] = arc_current
            gen_arc_cnt -= 1
        elif data_avg_3[counter] > 0:
            gen_arc_cnt = rand_jitter + width
            arc_current = data_internal[counter]

        counter += 1

    return data_internal


if __name__ == '__main__':
    t, data = gen_ac_current(10000, 0.2, 1)
    arc1 = gen_arc(data, 10, 4)
    arc2 = gen_arc(data, 5, 4)
    arc3 = gen_arc(data, 5, 8)
    arc4 = gen_arc(data, 10, 10)

    ax1 = plt.subplot(411)
    ax1.plot(t, arc1)
    ax2 = plt.subplot(412)
    ax2.plot(t, arc2)
    ax3 = plt.subplot(413)
    ax3.plot(t, arc3)
    ax4 = plt.subplot(414)
    ax4.plot(t, arc4)
    plt.show()
