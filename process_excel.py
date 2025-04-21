# Excelからデータを読み出すコード
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Process_excel(object):
    def __init__(self, delay):
        self.delay = delay

    def process_excel_files_12CH(self, excel_data):
        t = list()
        for row in excel_data.values:
            t.append([row[0]])

        inSize = 12
        outSize = 1
        u = np.zeros((inSize, len(t)))
        y = np.zeros((outSize, len(t)))

        for i, row in enumerate(excel_data.values):
            u[0, i] = row[1]
            u[1, i] = row[2]
            u[2, i] = row[3]
            u[3, i] = row[4]
            u[4, i] = row[5]
            u[5, i] = row[6]
            u[6, i] = row[7]
            u[7, i] = row[8]
            u[8, i] = row[9]
            u[9, i] = row[10]
            u[10, i] = row[11]
            u[11, i] = row[12]

            y[0, i] = row[14]

        return u, y

    def process_excel_files_16CH(self, excel_data):
        t = list()
        for row in excel_data.values:
            t.append([row[0]])

        inSize = 16
        outSize = 1
        u = np.zeros((inSize, len(t)))
        y = np.zeros((outSize, len(t)))

        for i, row in enumerate(excel_data.values):
            u[0, i] = row[1]
            u[1, i] = row[2]
            u[2, i] = row[3]
            u[3, i] = row[4]
            u[4, i] = row[5]
            u[5, i] = row[6]
            u[6, i] = row[7]
            u[7, i] = row[8]
            u[8, i] = row[9]
            u[9, i] = row[10]
            u[10, i] = row[11]
            u[11, i] = row[12]
            u[12, i] = row[13]
            u[13, i] = row[14]
            u[14, i] = row[15]
            u[15, i] = row[16]

            y[0, i] = row[18]

        return u, y

    def process_excel_files_24CH(self, excel_data):
        t = list()
        for row in excel_data.values:
            t.append([row[0]])

        inSize = 24
        outSize = 1
        u = np.zeros((inSize, len(t)))
        y = np.zeros((outSize, len(t)))

        for i, row in enumerate(excel_data.values):
            u[0, i] = row[1]
            u[1, i] = row[2]
            u[2, i] = row[3]
            u[3, i] = row[4]
            u[4, i] = row[5]
            u[5, i] = row[6]
            u[6, i] = row[7]
            u[7, i] = row[8]
            u[8, i] = row[9]
            u[9, i] = row[10]
            u[10, i] = row[11]
            u[11, i] = row[12]
            u[12, i] = row[13]
            u[13, i] = row[14]
            u[14, i] = row[15]
            u[15, i] = row[16]
            u[16, i] = row[17]
            u[17, i] = row[18]
            u[18, i] = row[19]
            u[19, i] = row[20]
            u[20, i] = row[21]
            u[21, i] = row[22]
            u[22, i] = row[23]
            u[23, i] = row[24]

            y[0, i] = row[26]

        return u, y

    def process_excel_files_36CH(self, excel_data):
        t = list()
        for row in excel_data.values:
            t.append([row[0]])

        inSize = 36
        outSize = 1
        u = np.zeros((inSize, len(t)))
        y = np.zeros((outSize, len(t)))

        for i, row in enumerate(excel_data.values):
            u[0, i] = row[1]
            u[1, i] = row[2]
            u[2, i] = row[3]
            u[3, i] = row[4]
            u[4, i] = row[5]
            u[5, i] = row[6]
            u[6, i] = row[7]
            u[7, i] = row[8]
            u[8, i] = row[9]
            u[9, i] = row[10]
            u[10, i] = row[11]
            u[11, i] = row[12]
            u[12, i] = row[13]
            u[13, i] = row[14]
            u[14, i] = row[15]
            u[15, i] = row[16]
            u[16, i] = row[17]
            u[17, i] = row[18]
            u[18, i] = row[19]
            u[19, i] = row[20]
            u[20, i] = row[21]
            u[21, i] = row[22]
            u[22, i] = row[23]
            u[23, i] = row[24]
            u[24, i] = row[25]
            u[25, i] = row[26]
            u[26, i] = row[27]
            u[27, i] = row[28]
            u[28, i] = row[29]
            u[29, i] = row[30]
            u[30, i] = row[31]
            u[31, i] = row[32]
            u[32, i] = row[33]
            u[33, i] = row[34]
            u[34, i] = row[35]
            u[35, i] = row[36]

            y[0, i] = row[38]

        return u, y

    def preprocess(self, u, y, noise_flag):
        u1 = np.zeros(u.shape)
        y1 = np.zeros(y.shape)
        u1 = np.where(u < 1000000, u, 1000000)

        if noise_flag is True:
            noise_shape = np.shape(u1)
            # np.random.seed(1000)
            noise = np.random.normal(loc=0,
                                     scale=300, size=noise_shape)
            u1 = u1 + noise
            u1 = np.where(u1 > 0, u1, 1000)

        u1 = np.log10(u1) - 4.5
        y1 = y

        return u1, y1

    def collect_half_cycle(self, u, y):
        inSize = 12
        outSize = 1
        batchSize = 50
        u_tdm = np.zeros((1, inSize, batchSize))
        y_tdm = np.zeros((outSize, 1))
        u_next = np.zeros((1, inSize, batchSize))
        y_next = np.zeros((outSize, 1))

        first_batch_flag = True
        list_of_start_all = [0]
        list_of_start_a = [0]
        list_of_start_b = [0]
        loop_cnt = 0
        for i in range(int(len(u[0, :]))):
            if i == 0:
                pre_val1 = u[11, i]
                pre_val2 = u[9, i]
                continue

            val1 = u[11, i]
            val2 = u[9, i]

            if val1 < 0 and pre_val1 > 0:
                if i - list_of_start_a[-1] > 30:
                    loop_cnt += 1
                    list_of_start_all.append(i)
                    list_of_start_a.append(i)
                    batch_start = i - 1
                    # delay = 0で現在の凹凸高さ
                    batch_end = batch_start + batchSize
                    if first_batch_flag is True:
                        u_tdm[0, :, :] = u[:, batch_start:batch_end]
                        y_tdm[0, 0] = y[0, batch_start + 24]
                        first_batch_flag = False
                    elif batch_end >= int(len(u[0, :])):
                        break
                    elif loop_cnt == 130:
                        break
                    else:
                        u_next[0, :, :] = u[:, batch_start:batch_end]
                        y_next[0, 0] = y[0, batch_start + 24]
                        u_tdm = np.vstack((u_tdm, u_next))
                        y_tdm = np.hstack((y_tdm, y_next))

            elif val2 < 0 and pre_val2 > 0:
                if i - list_of_start_b[-1] > 70:
                    loop_cnt += 1
                    list_of_start_all.append(i)
                    list_of_start_b.append(i)
                    batch_start = i - 1
                    batch_end = batch_start + batchSize
                    if first_batch_flag is True:
                        u_tdm[0, :, :] = u[:, batch_start:batch_end]
                        y_tdm[0, 0] = y[0, batch_start + 24]
                        first_batch_flag = False
                    elif batch_end >= int(len(u[0, :])):
                        break
                    elif loop_cnt == 130:
                        break
                    else:
                        u_next[0, :, :] = u[:, batch_start:batch_end]
                        y_next[0, 0] = y[0, batch_start + 24]
                        u_tdm = np.vstack((u_tdm, u_next))
                        y_tdm = np.hstack((y_tdm, y_next))

            pre_val1 = val1
            pre_val2 = val2

        return u_tdm, y_tdm

    def collect_one_cycle(self, u, y):
        inSize = 12
        outSize = 1
        batchSize = 10
        u_tdm = np.zeros((1, inSize, batchSize))
        y_tdm = np.zeros((outSize, 1))
        u_next = np.zeros((1, inSize, batchSize))
        y_next = np.zeros((outSize, 1))

        first_batch_flag = True
        list_of_start_all = [0]
        loop_cnt = 0
        for i in range(int(len(u[0, :]))):
            if i == 0:
                pre_val1 = u[9, i]  # feb27, march22
                # pre_val1 = u[4, i]  # march10
                continue

            val1 = u[9, i]  # feb27, march22
            # val1 = u[4, i]  # march10

            # feb27では周期の終わりを基準にしていたが、march10では周期の頭を基準にする
            if val1 > 0 and pre_val1 < 0:  # feb27, march22
            # if val1 < 0 and pre_val1 < 0:  # march10
                if i - list_of_start_all[-1] > 70 or list_of_start_all[-1] == 0:  # march10, march22
                # if i - list_of_start_all[-1] > 30 or list_of_start_all[-1] == 0:  #feb27
                    loop_cnt += 1
                    list_of_start_all.append(i)
                    batch_start = i - 3  # feb27, march22
                    # batch_start = i - 5  # march10
                    # delay = 0で現在の凹凸高さ
                    batch_end = batch_start + batchSize * 10
                    if first_batch_flag is True:
                        u_tdm[0, :, :] = u[:, batch_start:batch_end:10]
                        y_tdm[0, 0] = y[0, batch_start + 49]
                        first_batch_flag = False
                    elif batch_end >= int(len(u[0, :])):
                        break
                    elif loop_cnt == 130:
                        break
                    elif u[9, batch_end] < 0:
                        break
                    else:
                        u_next[0, :, :] = u[:, batch_start:batch_end:10]
                        y_next[0, 0] = y[0, batch_start + 49]
                        u_tdm = np.vstack((u_tdm, u_next))
                        y_tdm = np.hstack((y_tdm, y_next))

            pre_val1 = val1

        return u_tdm, y_tdm

    def collect_one_cycle_12CH(self, u, y):
        inSize = 12
        outSize = 1
        batchSize = 100
        u_tdm = np.zeros((1, inSize, batchSize))
        y_tdm = np.zeros((outSize, 1))
        u_next = np.zeros((1, inSize, batchSize))
        y_next = np.zeros((outSize, 1))

        first_response_flag = True
        list_of_start = []
        loop_cnt = 0
        for i in range(int(len(u[0, :]))):
            if i == 0:
                pre_val = u[9, i]
                continue
            val = u[9, i]

            if val > 0 and pre_val < 0:
                if first_response_flag is True:
                    list_of_start.append(i)
                    first_response_flag = False
                    batch_start = i - 3
                    batch_end = batch_start + batchSize
                    u_tdm[0, :, :] = u[:, batch_start:batch_end]
                    y_tdm[0, 0] = y[0, batch_start + 49]

            if first_response_flag is False and i == list_of_start[-1] + 86:
                loop_cnt += 1
                list_of_start.append(i)
                batch_start = i - 3
                batch_end = batch_start + batchSize
                if batch_end >= int(len(u[0, :])):
                    break
                elif loop_cnt == 130:
                    break
                elif u[9, batch_end] < 0:
                    break
                else:
                    u_next[0, :, :] = u[:, batch_start:batch_end]
                    y_next[0, 0] = y[0, batch_start + 49]
                    u_tdm = np.vstack((u_tdm, u_next))
                    y_tdm = np.hstack((y_tdm, y_next))

            pre_val = val

        return u_tdm, y_tdm

    def collect_one_cycle_16CH(self, u, y):
        inSize = 16
        outSize = 1
        batchSize = 100
        u_tdm = np.zeros((1, inSize, batchSize))
        y_tdm = np.zeros((outSize, 1))
        u_next = np.zeros((1, inSize, batchSize))
        y_next = np.zeros((outSize, 1))

        first_response_flag = True
        list_of_start = []
        loop_cnt = 0
        for i in range(int(len(u[0, :]))):
            if i == 0:
                pre_val = u[9, i]
                continue
            val = u[9, i]

            if val > 0 and pre_val < 0:
                if first_response_flag is True:
                    list_of_start.append(i)
                    first_response_flag = False
                    batch_start = i - 3
                    batch_end = batch_start + batchSize
                    u_tdm[0, :, :] = u[:, batch_start:batch_end]
                    y_tdm[0, 0] = y[0, batch_start + 49]

            if first_response_flag is False and i == list_of_start[-1] + 86:
                loop_cnt += 1
                list_of_start.append(i)
                batch_start = i - 3
                batch_end = batch_start + batchSize
                if batch_end >= int(len(u[0, :])):
                    break
                elif loop_cnt == 130:
                    break
                elif u[9, batch_end] < 0:
                    break
                else:
                    u_next[0, :, :] = u[:, batch_start:batch_end]
                    y_next[0, 0] = y[0, batch_start + 49]
                    u_tdm = np.vstack((u_tdm, u_next))
                    y_tdm = np.hstack((y_tdm, y_next))

            pre_val = val

        return u_tdm, y_tdm

    def collect_one_cycle_24CH(self, u, y):
        inSize = 24
        outSize = 1
        batchSize = 100
        u_tdm = np.zeros((1, inSize, batchSize))
        y_tdm = np.zeros((outSize, 1))
        u_next = np.zeros((1, inSize, batchSize))
        y_next = np.zeros((outSize, 1))

        first_response_flag = True
        list_of_start = []
        loop_cnt = 0
        for i in range(int(len(u[0, :]))):
            if i == 0:
                pre_val = u[17, i]
                continue
            val = u[17, i]

            if val > 0 and pre_val < 0:
                if first_response_flag is True:
                    list_of_start.append(i)
                    first_response_flag = False
                    batch_start = i - 3
                    batch_end = batch_start + batchSize
                    u_tdm[0, :, :] = u[:, batch_start:batch_end]
                    y_tdm[0, 0] = y[0, batch_start + 49]

            if first_response_flag is False and i == list_of_start[-1] + 86:
                loop_cnt += 1
                list_of_start.append(i)
                batch_start = i - 3
                batch_end = batch_start + batchSize
                if batch_end >= int(len(u[0, :])):
                    break
                elif loop_cnt == 130:
                    break
                elif u[17, batch_end] < 0:
                    break
                else:
                    u_next[0, :, :] = u[:, batch_start:batch_end]
                    y_next[0, 0] = y[0, batch_start + 49]
                    u_tdm = np.vstack((u_tdm, u_next))
                    y_tdm = np.hstack((y_tdm, y_next))

            pre_val = val

        return u_tdm, y_tdm

    def collect_one_cycle_36CH(self, u, y):
        inSize = 36
        outSize = 1
        batchSize = 100
        u_tdm = np.zeros((1, inSize, batchSize))
        y_tdm = np.zeros((outSize, 1))
        u_next = np.zeros((1, inSize, batchSize))
        y_next = np.zeros((outSize, 1))

        first_response_flag = True
        list_of_start = []
        loop_cnt = 0
        for i in range(int(len(u[0, :]))):
            if i == 0:
                pre_val = u[29, i]
                continue
            val = u[29, i]

            if val > 0 and pre_val < 0:
                if first_response_flag is True:
                    list_of_start.append(i)
                    first_response_flag = False
                    batch_start = i - 3
                    batch_end = batch_start + batchSize
                    u_tdm[0, :, :] = u[:, batch_start:batch_end]
                    y_tdm[0, 0] = y[0, batch_start + 49]

            if first_response_flag is False and i == list_of_start[-1] + 86:
                loop_cnt += 1
                list_of_start.append(i)
                batch_start = i - 3
                batch_end = batch_start + batchSize
                if batch_end >= int(len(u[0, :])):
                    break
                elif loop_cnt == 130:
                    break
                elif u[29, batch_end] < 0:
                    break
                else:
                    u_next[0, :, :] = u[:, batch_start:batch_end]
                    y_next[0, 0] = y[0, batch_start + 49]
                    u_tdm = np.vstack((u_tdm, u_next))
                    y_tdm = np.hstack((y_tdm, y_next))

            pre_val = val

        return u_tdm, y_tdm

    def collect_36CH_avg(self, u, y):
        inSize = 36
        outSize = 1
        batchSize = 100
        u_avg = np.zeros((1, inSize, 1))
        y_tdm = np.zeros((outSize, 1))
        u_next = np.zeros((1, inSize, 1))
        y_next = np.zeros((outSize, 1))

        first_response_flag = True
        list_of_start = []
        loop_cnt = 0
        for i in range(int(len(u[0, :]))):
            if i == 0:
                pre_val = u[29, i]
                continue
            val = u[29, i]

            if val > 0 and pre_val < 0:
                if first_response_flag is True:
                    list_of_start.append(i)
                    first_response_flag = False
                    batch_start = i - 3
                    batch_end = batch_start + batchSize
                    u_avg[0, :, 0] = np.mean(u[:, batch_start:batch_end], axis=1)
                    y_tdm[0, 0] = y[0, batch_start + 49]

            if first_response_flag is False and i == list_of_start[-1] + 86:
                loop_cnt += 1
                list_of_start.append(i)
                batch_start = i - 3
                batch_end = batch_start + batchSize
                if batch_end >= int(len(u[0, :])):
                    break
                elif loop_cnt == 130:
                    break
                elif u[29, batch_end] < 0:
                    break
                else:
                    u_next[0, :, 0] = np.mean(u[:, batch_start:batch_end], axis=1)
                    y_next[0, 0] = y[0, batch_start + 49]
                    u_avg = np.vstack((u_avg, u_next))
                    y_tdm = np.hstack((y_tdm, y_next))

            pre_val = val

        return u_avg, y_tdm

    def collect_avg_eachdiv(self, u, y):
        channel_size = u.shape[0]

        match channel_size:
            case 36:
                ref = 29
            case 24:
                ref = 17
            case 16:
                ref = 9
            case 12:
                ref = 9

        inSize = channel_size * 10
        outSize = 1
        batchSize = 100
        u_avg = np.zeros((1, inSize, 1))
        y_tdm = np.zeros((outSize, 1))
        u_next = np.zeros((1, inSize, 1))
        y_next = np.zeros((outSize, 1))

        first_response_flag = True
        list_of_start = []
        loop_cnt = 0
        for i in range(int(len(u[0, :]))):
            if i == 0:
                pre_val = u[ref, i]
                continue
            val = u[ref, i]

            if val > 0 and pre_val < 0:
                if first_response_flag is True:
                    list_of_start.append(i)
                    first_response_flag = False
                    batch_start = i - 3
                    batch_end = batch_start + batchSize
                    u_div1 = np.mean(u[:, batch_start:batch_start+10], axis=1).reshape((channel_size, 1))
                    u_div2 = np.mean(u[:, batch_start+10:batch_start+20], axis=1).reshape((channel_size, 1))
                    u_div3 = np.mean(u[:, batch_start+20:batch_start+30], axis=1).reshape((channel_size, 1))
                    u_div4 = np.mean(u[:, batch_start+30:batch_start+40], axis=1).reshape((channel_size, 1))
                    u_div5 = np.mean(u[:, batch_start+40:batch_start+50], axis=1).reshape((channel_size, 1))
                    u_div6 = np.mean(u[:, batch_start+50:batch_start+60], axis=1).reshape((channel_size, 1))
                    u_div7 = np.mean(u[:, batch_start+60:batch_start+70], axis=1).reshape((channel_size, 1))
                    u_div8 = np.mean(u[:, batch_start+70:batch_start+80], axis=1).reshape((channel_size, 1))
                    u_div9 = np.mean(u[:, batch_start+80:batch_start+90], axis=1).reshape((channel_size, 1))
                    u_div10 = np.mean(u[:, batch_start+90:batch_start+100], axis=1).reshape((channel_size, 1))
                    u_avg[0, :, :] = np.vstack((u_div1, u_div2, u_div3, u_div4, u_div5, u_div6, u_div7, u_div8, u_div9, u_div10))
                    y_tdm[0, 0] = y[0, batch_start + 49]

            if first_response_flag is False and i == list_of_start[-1] + 86:
                loop_cnt += 1
                list_of_start.append(i)
                batch_start = i - 3
                batch_end = batch_start + batchSize
                if batch_end >= int(len(u[0, :])):
                    break
                elif loop_cnt == 130:
                    break
                elif u[ref, batch_end] < 0:
                    break
                else:
                    u_div1 = np.mean(u[:, batch_start:batch_start+10], axis=1).reshape((channel_size, 1))
                    u_div2 = np.mean(u[:, batch_start+10:batch_start+20], axis=1).reshape((channel_size, 1))
                    u_div3 = np.mean(u[:, batch_start+20:batch_start+30], axis=1).reshape((channel_size, 1))
                    u_div4 = np.mean(u[:, batch_start+30:batch_start+40], axis=1).reshape((channel_size, 1))
                    u_div5 = np.mean(u[:, batch_start+40:batch_start+50], axis=1).reshape((channel_size, 1))
                    u_div6 = np.mean(u[:, batch_start+50:batch_start+60], axis=1).reshape((channel_size, 1))
                    u_div7 = np.mean(u[:, batch_start+60:batch_start+70], axis=1).reshape((channel_size, 1))
                    u_div8 = np.mean(u[:, batch_start+70:batch_start+80], axis=1).reshape((channel_size, 1))
                    u_div9 = np.mean(u[:, batch_start+80:batch_start+90], axis=1).reshape((channel_size, 1))
                    u_div10 = np.mean(u[:, batch_start+90:batch_start+100], axis=1).reshape((channel_size, 1))
                    u_next[0, :, :] = np.vstack((u_div1, u_div2, u_div3, u_div4, u_div5, u_div6, u_div7, u_div8, u_div9, u_div10))
                    y_next[0, 0] = y[0, batch_start + 49]
                    u_avg = np.vstack((u_avg, u_next))
                    y_tdm = np.hstack((y_tdm, y_next))

            pre_val = val

        return u_avg, y_tdm

    def combine_data_augment(self, dir_name, list_of_num, train_flag, k):
        if train_flag is True:
            for j in range(0, 1):
                if j == 0:
                    noise_flag = False
                else:
                    noise_flag = True
                for i in list_of_num:
                    excel_file = dir_name + "sensor_voltage_2025march28_{0}_all_brushes.xlsx".format(i)
                    excel_data = pd.read_excel(excel_file)
                    u, y = self.process_excel_files_36CH(excel_data)
                    u_scaled, y_scaled = self.preprocess(u, y, noise_flag)
                    if j == 0 and i == 1:
                        u_tmp, y_tmp = self.collect_avg_eachdiv(u_scaled, y_scaled)
                        y_len = y_tmp.shape[1]
                        u1 = u_tmp[self.delay:, :, :]
                        y1 = y_tmp[:, self.delay-k:y_len-k]
                    else:
                        u_tmp, y_tmp = self.collect_avg_eachdiv(u_scaled, y_scaled)
                        y_len = y_tmp.shape[1]
                        u_tmp = u_tmp[self.delay:, :, :]
                        y_tmp = y_tmp[:, self.delay-k:y_len-k]
                        u1 = np.vstack((u1, u_tmp))
                        y1 = np.hstack((y1, y_tmp))
        else:
            for i in list_of_num:
                excel_file = dir_name + "sensor_voltage_2025march28_{0}_all_brushes.xlsx".format(i)
                excel_data = pd.read_excel(excel_file)
                u, y = self.process_excel_files_36CH(excel_data)
                u_scaled, y_scaled = self.preprocess(u, y, noise_flag=False)
                if i == 24:
                    u_tmp, y_tmp = self.collect_avg_eachdiv(u_scaled, y_scaled)
                    y_len = y_tmp.shape[1]
                    u1 = u_tmp[self.delay:, :, :]
                    y1 = y_tmp[:, self.delay-k:y_len-k]
                else:
                    u_tmp, y_tmp = self.collect_avg_eachdiv(u_scaled, y_scaled)
                    y_len = y_tmp.shape[1]
                    u_tmp = u_tmp[self.delay:, :, :]
                    y_tmp = y_tmp[:, self.delay-k:y_len-k]
                    u1 = np.vstack((u1, u_tmp))
                    y1 = np.hstack((y1, y_tmp))

        return u1, y1


# 検証用コード
if __name__ == '__main__':
    dir_name = "/home/a24nitta/work/measurements/data/data_2025march28/"
    list_of_num = list(range(1, 81))
    test = Process_excel(delay=0)
    cycle_flag = True  # One cycle: True, Half cycle: False
    u_train, y_train = \
        test.combine_data_augment(dir_name=dir_name, list_of_num=list_of_num,
                                  train_flag=True, k=0)

    print(u_train.shape)
    print(y_train.shape)

    # fig1 = plt.figure(figsize=(10, 6))
    # ax = fig1.add_subplot(1, 1, 1)
    # ax.plot(u_train[4, 0, :], label='ch1')
    # ax.plot(u_train[4, 1, :], label='ch2')
    # ax.plot(u_train[4, 2, :], label='ch3')
    # ax.plot(u_train[4, 3, :], label='ch4')
    # ax.plot(u_train[4, 4, :], label='ch5')
    # ax.plot(u_train[4, 5, :], label='ch6')
    # ax.plot(u_train[4, 6, :], label='ch7')
    # ax.plot(u_train[4, 7, :], label='ch8')
    # ax.plot(u_train[4, 8, :], label='ch9')
    # ax.plot(u_train[4, 9, :], label='ch10')
    # ax.plot(u_train[4, 10, :], label='ch11')
    # ax.plot(u_train[4, 11, :], label='ch12')
    # ax.legend(loc=1)
    # ax.set_xlabel("time step")
    # ax.set_ylabel("sensor output[preprocssed]")
    # ax.tick_params(labelsize=16)
    # plt.tight_layout()
    # plt.show()
