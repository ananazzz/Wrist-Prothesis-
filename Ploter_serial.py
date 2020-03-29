import serial
import matplotlib.pyplot as plt
import csv
from struct import pack, unpack  # for binary pack
from pylab import *

parsed_data_arr = []
ser_data_arr = []

ser = serial.Serial()

def serial_init(port, ser):
    ser.port = port
    ser.baudrate = 115200
    ser.bytesize = serial.EIGHTBITS
    ser.parity = serial.PARITY_NONE
    ser.stopbits = serial.STOPBITS_ONE
    ser.timeout = None
    ser.timeout = 5
    ser.xonxoff = False
    ser.rtscts = False
    ser.dsrdtr = False
    ser.writeTimeout = 2
    ser.open();


def get_serial_data():
    global parsed_data_arr
    global packed_end_bytes
    global packed_length
    serial_data = ser.read_until(packed_end_bytes, packed_length)  # read up to 0x7D8135 bytes
    parsed_data = unpack('bbbbbbbbbbbbbbbbbbbbbbbb', serial_data);
    parsed_data_arr.append(parsed_data);
    print
    parsed_data
    # Put the rest of your code you want here
    time.sleep(0.001)



def save_data_to_csv(file_name):
    with open(file_name, 'wb') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(['I', 'II', 'III', 'AVR', 'AVL', 'AVF'])  # comments in the end of this file
        csv_out.writerow(ser.read() )
        for row in parsed_data_arr:
            csv_out.writerow(row[1:8])  # comments in the end of this file


def main():


    serial_init("COM4", ser)
    time_s = 5
    while (len(ser_data_arr) < time_s*1000):
        temp_value = int(ser.readline())
        if(temp_value<1024):
            ser_data_arr.append(temp_value)
    # save_data_to_csv('test.csv')
    plt.plot(ser_data_arr)
    plt.show()
if name == "__main__":
    main()
