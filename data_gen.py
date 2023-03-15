import matplotlib.pyplot as plt
import numpy as np

from gen_ac_current_signal import *
from gen_signal import *
import matplotlib.pyplot as plt


def normalized(data):
    coff_nomalize = 2.0 / (np.max(data) - np.min(data))
    return data * coff_nomalize


def slice1(data):
    s = np.zeros((50, 200))
    s_index = 0

    cross_zero = False
    start_index = 0
    stop_index = 0
    for i in range(len(data) - 1):
        # 过零点与持续150点以上同时判断切片位置
        if data[i] <= 0 <= data[i + 1] and i > start_index + max(stop_index - start_index, 200) * 0.75:
            if cross_zero == False:
                cross_zero = True
                start_index = i + 1
            else:
                cross_zero = False
                stop_index = i
                temp = data[start_index:i]
                if len(temp > 200):
                    temp = temp[:200]
                s[s_index][0:len(temp)] = temp
                s_index = s_index + 1
    return s[:s_index + 1]


def slice2(data):
    return np.reshape(data[:int(len(data) / 200) * 200], (int(len(data) / 200), 200))


if __name__ == '__main__':
    t, data = gen_ac_current(10000, 1, 100)
    coff1 = np.linspace(1, 2.2, np.size(data))
    data = np.multiply(coff1, data)

    s = slice2(data.copy())
    for i in range(np.size(s, axis=0)):
        s[i] = awgn(s[i], 40)
        s[i] = normalized(s[i])

    for ss in s:
        plt.plot(ss)

    plt.ylabel('raw wave')
    plt.show()

    arc2 = gen_arc(data.copy(), 19, 7)
    sarc = slice2(arc2)

    for i in range(np.size(sarc, axis=0)):
        sarc[i] = awgn(sarc[i], 40)
        sarc[i] = normalized(sarc[i])

    for ssarc in sarc:
        plt.plot(ssarc)

    plt.ylabel('arc wave')
    plt.show()
