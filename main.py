import matplotlib.pyplot as plt
import numpy as np
import helper as hl
import Parser as pr
from scipy import signal

file_P = 'C:/Users/001/Documents/Wrist_Prothesis/wrist-prothesis-/data/Motor.txt'

parser_data = hl.parslogfile(file_P, 6)
signal_to_check = parser_data[0]
parser_new_data = [0 for i in range(len(signal_to_check))]
parser_checker = False

for i in range(len(signal_to_check)):
    if signal_to_check[i] > 600:
        if parser_checker:
            parser_checker = False
        else:
            parser_checker = True
    if parser_checker:
        parser_new_data[i] = 1
    # if signal_to_check[i] > 600:
    #     parser_new_data[i]=1
    # elif signal_to_check[i] == 600:
    #     parser_new_data[i] = 1
    # elif signal_to_check[i] < 600:
    #     parser_new_data[i] = 0

# fs = 256
# hl.butter_highpass(10, fs, order=5)
# fig, ax = plt.subplots(2, 1)
# ax[0].plot(np.arange(len(signal_to_check)), signal_to_check)
# ax[0].set_xlabel('Time')
# ax[0].set_ylabel('Amplitude')
# ax[1].plot(np.arange(len(parser_new_data)),parser_new_data)  # plotting the spectrum
# ax[1].set_xlabel('Time')
# ax[1].set_ylabel('Amplitude')
# plt.show()

# for i in range(6):
#     hl.get_fft(signal=parser_data[i][:], Fs=256)

signal = signal_to_check
fs = 256

cutoff = 20
order = 6
conditioned_signal = hl.butter_highpass_filter(signal, cutoff, fs, order)

# smoothing signal
for i in range(300, 400, 100):
    smoothed_sig = hl.moving_mean(signal=conditioned_signal, N=i)
q_u, q_l = hl.envelope_finder(conditioned_signal)

s=q_u
s_max = 0
s_max_new = 0
s_min=0
s_meaner=[]
delta_fs = 256
fs_first = 0
delta_s = []
check=round(len(s)/delta_fs)
for i in range(len(s)-delta_fs):
    s_min=np.mean(s[i:delta_fs+i])
    s_meaner.append(s_min)
for i in range(1,round(len(s)/delta_fs)):
    s_max = np.mean(s[(i-1)*delta_fs:delta_fs*i])
    s_max_new = np.mean(s[i * delta_fs:delta_fs * (i + 1)])
    if s_max_new < s_max:
        delta_s.append(s_max-s_max_new)
    if s_max < s_max_new:
        delta_s.append(s_max_new - s_max)
# for i in range(len(s)):
#     s[i]=s_max_second-s_max_first

plt.figure()
plt.subplot(411)
plt.plot(np.arange(len(signal_to_check)), signal_to_check)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.subplot(412)
plt.plot(np.arange(len(conditioned_signal)), conditioned_signal)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.subplot(413)
plt.plot(np.arange(len(smoothed_sig)), smoothed_sig)  # plotting the spectrum
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.subplot(414)
plt.plot(np.arange(len(q_u)),q_u)
plt.xlabel('Time')
plt.ylabel('Amplitude')
for i in range(len(delta_s)):
    plt.plot([i*delta_fs, delta_fs*(i + 1)],[delta_s[i], delta_s[i]], "ro-")
plt.xlabel('Time')
plt.ylabel('Amplitude')



plt.show()
# ax[4].plot(np.arange(len(s)), s)
# ax[4].set_xlabel('Time')
# ax[4].set_xlabel('Ampitude')


# ax[4].plot(np.arange(len(q_l)), q_l)  # plotting the spectrum
# ax[4].set_xlabel('Time')
# ax[4].set_xlabel('Ampitude')


# data = hl.read_file(file_path=file_P)
#
# data_new = hl.get_data_from_column(data=data, column_num=0)
#
# data_new = data_new[500:1500]
#
#
#
# filtered_sig=data_new
# fq_remove=[52]
#
# for i in fq_remove:
#     b, a = hl.get_filter(1000, i)
#     filtered_sig = signal.filtfilt(b, a, filtered_sig)
#
# # filtered_data = hl.power_line_filter(data_new, Fs=1000)
#
#
# hl.get_fft(signal=filtered_sig, Fs=1000)
# signal = signal_to_check
# fs = 256
#
# cutoff = 20
# order = 6
# conditioned_signal = hl.butter_highpass_filter(signal, cutoff, fs, order)

# hl.get_fft(signal=signal_to_check, Fs=fs)

# fig, ax = plt.subplots(2, 1)
# ax[0].plot(np.arange(len(signal_to_check)), signal_to_check)
# ax[0].set_xlabel('Time')
# ax[0].set_ylabel('Amplitude')
# ax[1].plot(np.arange(len(conditioned_signal)),conditioned_signal)  # plotting the spectrum
# ax[1].set_xlabel('Time')
# ax[1].set_ylabel('Amplitude')

# hl.get_fft(signal=conditioned_signal, Fs=fs)

# upper_envelope,lower_envelope=hl.envelope_finder(conditioned_signal)
#
# plt.figure()
# plt.plot(conditioned_signal)
# plt.plot(upper_envelope,'r')
# plt.plot(lower_envelope,'g')
# plt.grid(True)

plt.show()
