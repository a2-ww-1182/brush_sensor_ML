import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import pandas as pd
# import openpyxl


class Process_excel(object):
    def process_excel_files(self, excel_data):
        t = list()
        for row in excel_data.values:
            t.append([row[0]])

        inSize = 12
        outSize = 1
        batchSize = 100
        u = np.zeros((inSize, len(t)))
        y = np.zeros((outSize, len(t)))
        u_tdm = np.zeros((1, inSize, batchSize))
        y_tdm = np.zeros((1, 1))
        u_next = np.zeros((1, inSize, batchSize))
        y_next = np.zeros((1, 1))

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

        first_response_flag = True
        first_batch_flag = True
        list_of_start = []
        loop_cnt = 0
        for i in range(int(len(t))):
            if i == 0:
                pre_val = u[4, i]
                continue
            val = u[4, i]

            if val < 70000 and pre_val > 70000:
                if first_response_flag is True:
                    list_of_start.append(i)
                    first_response_flag = False
                    batch_start = i - 5
                    batch_end = batch_start + batchSize
                    if first_batch_flag is True:
                        u_tdm[0, :, :] = u[:, batch_start:batch_end]
                        y_tdm[0, 0] = y[0, batch_start]
                        first_batch_flag = False

                elif i - list_of_start[-1] > 70:
                    loop_cnt += 1
                    list_of_start.append(i)
                    batch_start = i - 5
                    batch_end = batch_start + batchSize
                    if first_batch_flag is True:
                        u_tdm[0, :, :] = u[:, batch_start:batch_end]
                        y_tdm[0, 0] = y[0, batch_start]
                        first_batch_flag = False
                    elif batch_end >= len(t):
                        break
                    elif loop_cnt == 130:
                        break
                    else:
                        u_next[0, :, :] = u[:, batch_start:batch_end]
                        y_next[0, 0] = y[0, batch_start]
                        u_tdm = np.vstack((u_tdm, u_next))
                        y_tdm = np.hstack((y_tdm, y_next))
                else:
                    continue

            pre_val = val

        return u_tdm, y_tdm

    def preprocess(self, u, y, delay):
        u1 = np.zeros(u.shape)
        y1 = np.zeros(y.shape)
        u1 = np.where(u < 1000000, u, 1000000)
        u1 = np.log10(u1) - 4.5

        y1[0, :delay] = 0
        y1[0, delay:] = y[0, :y.shape[1] - delay]

        return u1, y1

    def combine_data(self, dir_name, delay):
        for i in [1, 3, 5]:
            excel_file = dir_name + "sensor_voltage_2025january16_{0}_column2.xlsx".format(i)
            excel_data = pd.read_excel(excel_file)
            u_tdm, y_tdm = self.process_excel_files(excel_data)
            if i == 1:
                u_train, y_train = self.preprocess(u_tdm, y_tdm, delay)
            else:
                u1, y1 = self.preprocess(u_tdm, y_tdm, delay)
                u_train = np.vstack((u_train, u1))
                y_train = np.hstack((y_train, y1))

        for i in [10]:
            excel_file = dir_name + "sensor_voltage_2025january16_{0}_column2.xlsx".format(i)
            excel_data = pd.read_excel(excel_file)
            u_tdm, y_tdm = self.process_excel_files(excel_data)
            if i == 10:
                u_test, y_test = self.preprocess(u_tdm, y_tdm, delay)
            else:
                u1, y1 = self.preprocess(u_tdm, y_tdm, delay)
                u_test = np.vstack((u_test, u1))
                y_test = np.hstack((y_test, y1))

        return u_train, y_train, u_test, y_test


class Physical_reservoir(Process_excel):
    def __init__(self, dir_name, delay):
        self.u1_train, self.y1_train, self.u1_test, self.y1_test = self.combine_data(dir_name, delay)

        self.channelSize = self.u1_train.shape[1]
        self.batchSize = self.u1_train.shape[2]

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
        xl2 = "batch"

        fig1 = plt.figure(figsize=(10, 6), facecolor='lightblue')
        #グラフを描画するsubplot領域を作成。
        ax1 = fig1.add_subplot(1, 1, 1)
        #各subplot領域にデータを渡す
        # ax1.plot(y_true1[:], color=c1, label=l1)
        #各subplot領域にデータを渡す
        ax1.plot(y_pred1[:], color=c2, label=l2)
        #各subplotにxラベルを追加
        ax1.set_xlabel(xl2, fontsize=16)
        #各subplotにx軸設定
        # ax1.set_xlim(8000, 9000)
        #各subplotにyラベルを追加
        ax1.set_ylabel('height [mm]', fontsize=16)

        ax1.tick_params(labelsize=16)
        ax1.legend(loc=1)

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
        # ax2.plot(self.u1_test[0, 0, :])
        # ax2.plot(self.u1_test[0, 1, :])
        # ax2.plot(self.u1_test[0, 2, :])
        # ax2.plot(self.u1_test[0, 3, :])
        # ax2.plot(self.u1_test[0, 4, :])
        # ax2.plot(self.u1_test[0, 5, :])
        # ax2.plot(self.u1_test[0, 6, :])
        # ax2.plot(self.u1_test[0, 7, :])
        # ax2.plot(self.u1_test[0, 8, :])
        # ax2.plot(self.u1_test[0, 9, :])
        # ax2.plot(self.u1_test[0, 10, :])
        # ax2.plot(self.u1_test[0, 11, :])
        # ax2.set_xlabel(xl, fontsize=16)
        # ax2.set_ylabel('processed value', fontsize=16)
        # ax2.tick_params(labelsize=16)
        # plt.tight_layout()

        # fig3 = plt.figure(figsize=(10, 6), facecolor='lightblue')
        # ax2 = fig3.add_subplot(1, 1, 1)
        # ax2.plot(self.u1_test[86, 0, :])
        # ax2.plot(self.u1_test[86, 1, :])
        # ax2.plot(self.u1_test[86, 2, :])
        # ax2.plot(self.u1_test[86, 3, :])
        # ax2.plot(self.u1_test[86, 4, :])
        # ax2.plot(self.u1_test[86, 5, :])
        # ax2.plot(self.u1_test[86, 6, :])
        # ax2.plot(self.u1_test[86, 7, :])
        # ax2.plot(self.u1_test[86, 8, :])
        # ax2.plot(self.u1_test[86, 9, :])
        # ax2.plot(self.u1_test[86, 10, :])
        # ax2.plot(self.u1_test[86, 11, :])
        # ax2.set_xlabel(xl, fontsize=16)
        # ax2.set_ylabel('input resistance [ohm]', fontsize=16)
        # ax2.tick_params(labelsize=16)
        # plt.tight_layout()

        return result_values
        # print(result_values)

        # np.savez(npz_file, y_true1=y_true1, y_true2=y_true2, y_pred1=y_pred1, y_pred2=y_pred2, result_values=result_values)


dir_name = "/home/a24nitta/work/measurements/data/data_2025january16/"

test = Physical_reservoir(dir_name, delay=0)
Wout = test.train()
test.test(Wout)

plt.show()
