import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import pandas as pd
# import openpyxl


# 削れる速度を求めるタスク。前のバッチの高さと現在のバッチの高さの差をもちいて平均の変化率を求める。
# 1バッチを1周期100timestepにしたバージョン
# これまでは窓の動かし方がセンサの反応があったかどうかだが、それだと差分を取るときに、
# 毎回幅が異なる。それはよくないので86ずつずらしていくという動かし方の固定。
# ガウシアンノイズを加える
class Process_excel(object):
    def process_excel_files(self, excel_data):
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
            np.random.seed(1000)
            noise = np.random.normal(loc=0,
                                     scale=300, size=noise_shape)
            u1 = u1 + noise
            u1 = np.where(u1 > 0, u1, 1000)

        u1 = np.log10(u1) - 4.5
        y1 = y

        return u1, y1

    def collect(self, u, y, delay):
        inSize = 36
        outSize = 1
        batchSize = 100
        u_tdm = np.zeros((1, inSize, batchSize))
        y_tdm = np.zeros((outSize, 1))
        u_next = np.zeros((1, inSize, batchSize))
        y_next = np.zeros((outSize, 1))

        first_response_flag = True
        first_batch_flag = True
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
                    # batch_start = i - 5
                    # batch_end = batch_start + batchSize
                    # if first_batch_flag is True:
                    #     u_tdm[0, :, :] = u[:, batch_start:batch_end]
                    #     y_tdm[0, 0] = y[0, batch_start + 49 - delay]
                    #     first_batch_flag = False

            if first_response_flag is False:
                if i == list_of_start[-1] + 86:
                    loop_cnt += 1
                    list_of_start.append(i)
                    batch_start = i - 3
                    pre_batch_start = list_of_start[-2] - 3
                    batch_end = batch_start + batchSize
                    if first_batch_flag is True:
                        u_tdm[0, :, :] = u[:, batch_start:batch_end]
                        y_tdm[0, 0] = \
                            (y[0, pre_batch_start + 49 - delay] - y[0, batch_start + 49 - delay]) / ((batch_start - pre_batch_start) * 0.05)
                        first_batch_flag = False
                    elif batch_end >= int(len(u[0, :])):
                        break
                    elif loop_cnt == 130:
                        break
                    elif u[29, batch_end] < 0:
                        break
                    else:
                        u_next[0, :, :] = u[:, batch_start:batch_end]
                        y_next[0, 0] = \
                            (y[0, pre_batch_start + 49 - delay] - y[0, batch_start + 49 - delay]) / ((batch_start - pre_batch_start) * 0.05)
                        u_tdm = np.vstack((u_tdm, u_next))
                        y_tdm = np.hstack((y_tdm, y_next))

            pre_val = val

        return u_tdm, y_tdm

    def combine_data(self, dir_name, delay):
        for k in range(0, 10):
            if k == 0:
                noise_flag = False
            else:
                noise_flag = True
            for i in range(1, 81):
                excel_file = dir_name + "sensor_voltage_2025march28_{0}_all_brushes.xlsx".format(i)
                excel_data = pd.read_excel(excel_file)
                u, y = self.process_excel_files(excel_data)
                u1, y1 = self.preprocess(u, y, noise_flag)
                if k == 0 and i == 1:
                    u_train, y_train = self.collect(u1, y1, delay)
                else:
                    u_next, y_next = self.collect(u1, y1, delay)
                    u_train = np.vstack((u_train, u_next))
                    y_train = np.hstack((y_train, y_next))

        for i in [81]:
            excel_file = dir_name + "sensor_voltage_2025march28_{0}_all_brushes.xlsx".format(i)
            excel_data = pd.read_excel(excel_file)
            u, y = self.process_excel_files(excel_data)
            u1, y1 = self.preprocess(u, y, noise_flag=False)
            if i == 81:
                u_test, y_test = self.collect(u1, y1, delay)
            else:
                u_next, y_next = self.collect(u1, y1, delay)
                u_test = np.vstack((u_test, u_next))
                y_test = np.hstack((y_test, y_next))

        return u_train, y_train, u_test, y_test


class Physical_reservoir(Process_excel):
    def __init__(self, dir_name, delay):
        self.u1_train, self.y1_train, self.u1_test, self.y1_test = self.combine_data(dir_name, delay)

        self.channelSize = self.u1_train.shape[1]
        self.batchSize = self.u1_train.shape[2]
        print(self.u1_train.shape)

    def train(self):
        trainLen = self.u1_train.shape[0]
        X = np.zeros((self.channelSize * self.batchSize, trainLen))
        Yt = self.y1_train[0, :trainLen].reshape((1, trainLen))

        for t in range(trainLen):
            u_current = self.u1_train[t, :, :].reshape((self.channelSize * self.batchSize))
            X[:, t] = u_current

        # Ridge Regression
        reg = 1e-4

        Wout = linalg.solve(np.dot(X, X.T) + reg*np.eye(self.channelSize * self.batchSize), np.dot(X, Yt.T)).T

        # Linear Regression
        # Wout = linalg.solve(np.dot(X, X.T), np.dot(X, Yt.T)).T

        print(np.linalg.norm(Wout, ord=2))
        print(Wout.shape)
        print(np.linalg.matrix_rank(np.dot(X, X.T)))
        return Wout

    def test(self, Wout):
        testLen = self.u1_test.shape[0]
        outSize = self.y1_test.shape[0]
        X = np.zeros((self.channelSize * self.batchSize, testLen))
        Y = np.zeros((outSize, testLen))

        for t in range(testLen):
            u_current = self.u1_test[t, :, :].reshape((-1, 1))
            y_current = np.dot(Wout, u_current)
            X[:, t] = u_current[:, 0].reshape(self.channelSize * self.batchSize)
            Y[0, t] = y_current[0].item()

        y_true1 = self.y1_test[0, :]
        y_pred1 = Y[0, :]
        # labels = [0, 1, 2, 3]
        # mse = np.sqrt(sum(np.square(y_true[:] - y_pred[:])) / (testLen - self.initLen))
        rmse1 = np.sqrt(sum(np.square(y_true1[:] - y_pred1[:])) / (testLen))
        mae1 = (sum(np.absolute(y_true1[:] - y_pred1[:])) / (testLen))
        max_error = np.amax(np.absolute(y_true1[:] - y_pred1[:]))
        print("rmse:", rmse1)
        print("mae:", mae1)
        print("max error:", max_error)
        result_values = np.array((rmse1, mae1), dtype=float)

        c1,c2 = 'green', 'blue'
        l1,l2 = "target", "predict"

        xl = "time step"
        xl2 = "Batch"

        fig1 = plt.figure(figsize=(10, 6), facecolor='lightblue')
        #グラフを描画するsubplot領域を作成。
        ax1 = fig1.add_subplot(1, 1, 1)
        #各subplot領域にデータを渡す
        ax1.plot(y_true1[:], color=c1, label=l1)
        #各subplot領域にデータを渡す
        ax1.plot(y_pred1[:], color=c2, label=l2)
        #各subplotにxラベルを追加
        ax1.set_xlabel(xl2, fontsize=24, fontweight='bold')
        #各subplotにx軸設定
        # ax1.set_xlim(8000, 9000)
        #各subplotにyラベルを追加
        ax1.set_ylabel('Polishing speed [mm/s]', fontsize=24, fontweight='bold')

        ax1.tick_params(labelsize=20)
        ax1.legend(loc=1, fontsize=24)
        ax1.set_xticks([0, 1, 2, 3])

        # fig1 = plt.figure(figsize=(10, 6), facecolor='lightblue')
        # ax1 = fig1.add_subplot(1, 1, 1)
        # ax1.stem(y_pred1[:], linefmt='g:', markerfmt='b')
        # ax1.set_ylabel('height [mm]', fontsize=16)
        # ax1.set_xlabel(xl2, fontsize=16)
        # ax1.tick_params(labelsize=16)

        plt.tight_layout()
        # plt.show()

        # fig2 = plt.figure(figsize=(10, 6), facecolor='lightblue')
        # ax2 = fig2.add_subplot(1, 1, 1)
        # ax2.plot(self.u1_test[0, 0, :], label="ch1")
        # ax2.plot(self.u1_test[0, 1, :], label="ch2")
        # ax2.plot(self.u1_test[0, 2, :], label="ch3")
        # ax2.plot(self.u1_test[0, 3, :], label="ch4")
        # ax2.plot(self.u1_test[0, 4, :], label="ch5")
        # ax2.plot(self.u1_test[0, 5, :], label="ch6")
        # ax2.plot(self.u1_test[0, 6, :], label="ch7")
        # ax2.plot(self.u1_test[0, 7, :], label="ch8")
        # ax2.plot(self.u1_test[0, 8, :], label="ch9")
        # ax2.plot(self.u1_test[0, 9, :], label="ch10")
        # ax2.plot(self.u1_test[0, 10, :], label="ch11")
        # ax2.plot(self.u1_test[0, 11, :], label="ch12")
        # ax2.set_xlabel(xl, fontsize=16)
        # ax2.set_ylabel('processed value', fontsize=16)
        # ax2.tick_params(labelsize=16)
        # ax2.legend(loc=1)
        # plt.tight_layout()

        fig3 = plt.figure(figsize=(10, 6), facecolor='lightblue')
        ax2 = fig3.add_subplot(1, 1, 1)
        ax2.plot(Wout[0, :])
        ax2.set_ylabel('Weight', fontsize=16)
        ax2.tick_params(labelsize=16)
        plt.tight_layout()

        return result_values
        # print(result_values)

        # np.savez(npz_file, y_true1=y_true1, y_true2=y_true2, y_pred1=y_pred1, y_pred2=y_pred2, result_values=result_values)


dir_name = "/home/a24nitta/work/measurements/data/data_2025march28/"
test = Physical_reservoir(dir_name, delay=0)
Wout = test.train()
test.test(Wout)

plt.show()
