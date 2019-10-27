import numpy as np
import helper as hl
import matplotlib.pyplot as plt
import os
import random
from scipy.interpolate import interp1d
from scipy import signal

def threshold(low_pass_signal, th_window):
    th_hold_max = low_pass_signal[0]
    th_hold_min = 0
    for i in range(len(low_pass_signal)):
        if(low_pass_signal[i] > th_hold_max):
            th_hold_max = low_pass_signal[i]
    th_hold_min = th_hold_max-th_window
    if(th_hold_min<=0):
        print("Bad signal")
    return th_hold_max, th_hold_min

def low_pass_filter(signal, koef):
    filtered_s=[]
    out=0
    for i in range(len(signal)):
        out=out+abs(signal[i])*koef
        out=out*(1-koef)
        filtered_s.append(out)
    return filtered_s


def derivative(signal):
    balanced_sig = []
    for i in range(len(signal)):
        S_new = signal[i]
        S_old = signal[i-1]
        balanced_sig.append(S_new-S_old)
    return balanced_sig


def motor_control(signal, koef, w_min, w_max):
    temp_der = derivative(signal)
    temp_low_pass = low_pass_filter(temp_der,koef)
    b_s=binary_signal(temp_low_pass, window_min=w_min,window_max=w_max)
    return temp_der, temp_low_pass, b_s

def signal_create(N,amp1,amp2,amp3):
    randomed_signal_arr = []

    for i in range(int(N/3)):
        randomed_signal_arr.append(500+random.randint(1, amp1))
    for i in range(int(N/3)):
        randomed_signal_arr.append(500+random.randint(-(amp2/2), amp2/2))
    for i in range(int(N/3)):
        randomed_signal_arr.append(500+random.randint(1, amp3))

    return randomed_signal_arr

def graphic(f_signal, s_signal, t_signal, w_min, w_max):
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.subplot(311)
    plt.plot(np.arange(len(f_signal)), f_signal)
    plt.title(file_name)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.subplot(312)
    plt.plot(np.arange(len(s_signal)), s_signal)
    plt.title(file_name)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.subplot(313)
    plt.plot(np.arange(len(t_signal)), t_signal, np.arange(len(t_signal)), [w_min]*len(t_signal),np.arange(len(t_signal)), [w_max]*len(t_signal))
    plt.title(file_name)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()

def binary_signal(signal, window_min, window_max):
    safe_threshold = (window_max-window_min)*0.3 #0.3 = 30% of the window amplitude
    final_s = []
    for i in range(len(signal)):
        if(signal[i]>=window_min+safe_threshold and signal[i]<=window_max):
            final_s.append(1)
        else:
            final_s.append(0)
    return final_s
def calibr(signal, koef):
    sig_der = derivative(signal)
    sig_filt = low_pass_filter(sig_der, koef)
    calibr_max = sig_filt[0]
    for i in range(len(sig_filt)):
        if(sig_filt[i]>calibr_max):
            calibr_max=sig_filt[i]


    return calibr_max

def latency_point(signal_filt, l_on, l_off, calibr_max, calibr_min):
    latenced_sig = []
    clock = 0
    allow_change=True
    motor_state=0
    for i in range(len(signal_filt)):
        if(allow_change):
            if(signal_filt[i]>calibr_min and signal_filt[i]<calibr_max):
                motor_state = 1
            else:
                motor_state = 0
            allow_change = False
            clock=0
        else:
            clock+=1
            if(motor_state==1 and clock>l_on):
                allow_change=True
            if(motor_state==0 and clock>l_off):
                allow_change = True
        latenced_sig.append(motor_state)
    return latenced_sig


if __name__ == "__main__":
    directory = os.getcwd()
    file_P = directory + '/data/Motor.txt'
    parser_data = hl.parslogfile(file_P, 6)
    signal_to_check = parser_data[0]
    file_name = file_P.rsplit("/", 1)[-1]

    calibration_sig=signal_create(400,50,500,50)
    # b,a=motor_control(calibration_sig, 0.05)
    th_min, th_max = threshold(calibration_sig, 300)
    randomed_signal=signal_create(400,50,500,50)
    noize_sig = signal_create(400, 100, 120, 150)
    randomed_signal_test = signal_create(400, 100, 400, 150)


    calibr_max_win = calibr(randomed_signal, koef=0.02)
    calibr_min_win = calibr(noize_sig, koef=0.02)
    balanced_s, smooth_sig, binary_s = motor_control(randomed_signal_test, 0.05, w_max=calibr_max_win, w_min=calibr_min_win)
    latenced_s= latency_point(smooth_sig, 10, 10, calibr_max_win, calibr_min_win)

    graphic(randomed_signal_test, latenced_s, smooth_sig, w_min=calibr_min_win, w_max=calibr_max_win)

    # fs = 256
    # cutoff = 20
    # order = 6
    # conditioned_signal = hl.butter_highpass_filter(signal_to_check, cutoff, fs, order)
    #
    #
    # b, a = hl.get_filter(Fs = 256, power_line_fr = 50)
    # filtered_sig = signal.filtfilt(b, a, conditioned_signal)
    # bt_filtered_sig = hl.filter_signal(data=conditioned_signal, cutoff=40, fs=256, filter_type = 'low', order=10)
    #
    # u_e, l_e = hl.envelope_finder(bt_filtered_sig)
    #
    # mean_builder, coords_vals, min_x, max_y = mean_value(u_e, Fs=256, displacement=90)
    #
    # power_filtered_spectrum, pw_fil_Y = hl.get_fft(signal=filtered_sig, Fs=256)
    # spectrum, Y = hl.get_fft(conditioned_signal, fs)
    # butter_filtered_spectrum, bt_fil_Y = hl.get_fft(signal=bt_filtered_sig, Fs=256)
    # min_x = "%.2f" %  min_x
    # max_y = "%.2f" % max_y
    # mean_interpolated = interp1d(np.arange(len(mean_builder)), mean_builder, kind='cubic', bounds_error=False, fill_value=0.0)
    # new_time = np.linspace(0, len(conditioned_signal)-1, len(conditioned_signal)-1)
    # mean_vector = mean_interpolated(new_time)
    # mean_interpolated = signal.resample(mean_builder, len(bt_filtered_sig))
    # upper_env, lower_env = hl.envelope_finder(conditioned_signal)
    # upper_env = signal.resample(upper_env, len(conditioned_signal))
    # lower_env = signal.resample(lower_env, len(conditioned_signal))
    # min_upper_e
