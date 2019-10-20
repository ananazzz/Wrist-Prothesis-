


file_to_parse = "C:/Users/001/Desktop/prosthetic_stuff/real_code_stuff/27.01.2019/capture_emg19(1).txt"
channel_num = 6

if __name__ == "__main__":
    print("file parser")

    #open file and reading it
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

    print(read_buffer)