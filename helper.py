import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt
from numpy import array, sign, zeros
from scipy.interpolate import interp1d

def parslogfile(file_to_parse,channel_num):
    f = open(file_to_parse, "r")
    read_buffer = f.readline()

    # splitting file_data with pattern"A55A02"
    # Packet protocol given below
    # // Writepacket
    # header and footer
    # TXBuf[0] = 0xa5; // Sync0
    # TXBuf[1] = 0x5a; // Sync1
    # TXBuf[2] = 2; // Protocol version
    # TXBuf[3] = 0; // Packet counter
    # TXBuf[4] = 0x02; // CH1 High Byte
    # TXBuf[5] = 0x00; // CH1 Low Byte
    # TXBuf[6] = 0x02; // CH2 High Byte
    # TXBuf[7] = 0x00; // CH2 Low Byte
    # TXBuf[8] = 0x02; // CH3 High Byte
    # TXBuf[9] = 0x00; // CH3 Low Byte
    # TXBuf[10] = 0x02; // CH4 High Byte
    # TXBuf[11] = 0x00; // CH4 Low Byte
    # TXBuf[12] = 0x02; // CH5 High Byte
    # TXBuf[13] = 0x00; // CH5 Low Byte
    # TXBuf[14] = 0x02; // CH6 High Byte
    # TXBuf[15] = 0x00; // CH6 Low Byte
    # TXBuf[2 * NUMCHANNELS + HEADERLEN] = 0x01;
    packets = read_buffer.split("A55A02")

    #deleting last and first packets as they can be corrupted
    del packets[0]
    del packets[-1]


    packets_id = []
    channels_raw = [[] for i in range(channel_num)]
    for i in range(len(packets)):
        packets_id.append(int(packets[i][0:2],16))

        for j in range(channel_num):
            num = packets[i][(j*2)+2*(j+1):6*(j+1)-(j*2)]
            channels_raw[j].append(int(num, 16))

    return channels_raw

def read_file(file_path):
    file_data = []

    with open(file_path, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            file_data.append(row)

    return file_data


def get_data_from_column(data, column_num):
    print(data)
    print(len(data))

    list1 = []

    for i in range(len(data)):
        list1.append(int(data[i][column_num]))

    return list1


def get_fft(signal, Fs):

    n = len(signal)  # length of the signal
    k = np.arange(n)
    T = n / Fs

    # print("Signal length in sec: %i" % T)

    frq = k / T  # two sides frequency range
    frq = frq[range(int(n / 2))]  # one side frequency range

    Y = np.fft.fft(signal) / n  # fft computing and normalization
    Y = abs(Y[range(int(n / 2))])

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(np.arange(n), signal)
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Amplitude')
    ax[1].plot(frq[1:len(frq)], Y[1:len(frq)], 'r')  # plotting the spectrum
    ax[1].set_xlabel('Freq (Hz)')
    ax[1].set_ylabel('|Y(freq)|')
    # plt.show()


    return frq, Y


def power_line_filter(data, Fs):
    N, Wn = signal.buttord(wp=[48 / Fs, 53 / Fs], ws=[46 / Fs, 55 / Fs],
                                 gpass=0.1, gstop=10.0, analog=False)
    b, a = signal.butter(N, Wn, 'bandstop', False)

    w, h = signal.freqz(b, a)
    fig = plt.figure()
    plt.title('Digital filter frequency response')
    ax1 = fig.add_subplot(111)

    plt.plot(w, 20 * np.log10(abs(h)), 'b')
    plt.ylabel('Amplitude [dB]', color='b')
    plt.xlabel('Frequency [rad/sample]')

    ax2 = ax1.twinx()
    angles = np.unwrap(np.angle(h))
    plt.plot(w, angles, 'g')
    plt.ylabel('Angle (radians)', color='g')
    plt.grid()
    plt.axis('tight')
    # plt.show()

    y = signal.lfilter(b, a, data)

    return y

def get_filter(Fs, power_line_fr):

    fs = Fs  # Sample frequency (Hz)
    f0 = power_line_fr  # Frequency to be removed from signal (Hz)
    Q = 20.0  # Quality factor
    w0 = f0 / (fs / 2)  # Normalized Frequency
     # Design notch filter
    b, a = signal.iirnotch(w0, Q)

    return b,a

def moving_mean(signal,N):
    sum=0

    min_sig = np.min(signal)
    signal = signal - min_sig

    result=[0 for i in signal]
    for i in range(0, N):
        sum=sum+signal[i]
        result[i]=sum/(i+1)

    for i in range(N, len(signal)):
        sum=sum-signal[i-N]+signal[i]
        result[i]=sum/N

    return result




def moving_mean_smoothing(signal,points_to_average):

    N=0
    mean_sum = 0

    smoothed_signal=[0 for i in signal]
    start_p = 0
    stop_p = 0
    half_win_len = int(points_to_average/2)

    for i in range(1, len(signal)):
        if i<points_to_average:
            start_p=0
            stop_p=i
            N = i
        else:
            start_p=i-points_to_average
            stop_p=i
            N = points_to_average

        # N = (stop_p - start_p)+1
        # mean_sum = np.sum([signal[k] for k in range(i - points_to_average, i + points_to_average, 1)]) / N

        for k in range(start_p, stop_p, 1):
            mean_sum += signal[k]
        mean_sum /= N
        smoothed_signal[i]=smoothed_signal[i-1]+mean_sum
        mean_sum = 0

    return smoothed_signal

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def generate_butter_filt(cutoff, fs, order=5, type="low"):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype=type, analog=False)
    return b, a

def filter_signal(data=None, cutoff=1, fs=0, order=5, filter_type="low"):
    b, a = generate_butter_filt(cutoff, fs, order=order, type=filter_type)
    filtered_signal = filtfilt(b, a, data)

    return filtered_signal


def envelope_finder(seq):
    s = array(seq)
    q_u = zeros(s.shape)
    q_l = zeros(s.shape)

    # Prepend the first value of (s) to the interpolating values. This forces the model to use the same starting point for both the upper and lower envelope models.

    u_x = [0, ]
    u_y = [s[0], ]

    l_x = [0, ]
    l_y = [s[0], ]

    # Detect peaks and troughs and mark their location in u_x,u_y,l_x,l_y respectively.

    for k in range(1, len(s) - 1):
        if (sign(s[k] - s[k - 1]) == 1) and (sign(s[k] - s[k + 1]) == 1):
            u_x.append(k)
            u_y.append(s[k])

        if (sign(s[k] - s[k - 1]) == -1) and ((sign(s[k] - s[k + 1])) == -1):
            l_x.append(k)
            l_y.append(s[k])

    # Append the last value of (s) to the interpolating values. This forces the model to use the same ending point for both the upper and lower envelope models.

    u_x.append(len(s) - 1)
    u_y.append(s[-1])

    l_x.append(len(s) - 1)
    l_y.append(s[-1])

    # Fit suitable models to the data. Here I am using cubic splines, similarly to the MATLAB example given in the question.

    u_p = interp1d(u_x, u_y, kind='cubic', bounds_error=False, fill_value=0.0)
    l_p = interp1d(l_x, l_y, kind='cubic', bounds_error=False, fill_value=0.0)

    # Evaluate each model over the domain of (s)
    for k in range(0, len(s)):
        q_u[k] = u_p(k)
        q_l[k] = l_p(k)
    return q_u,q_l