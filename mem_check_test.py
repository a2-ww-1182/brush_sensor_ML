import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import pandas as pd
# import openpyxl


# デバイスにどれだけメモリがあるのかを確かめるプログラム
class Process_excel(object):  # Excelからデータを取得するクラス
    def process_excel_files(self, excel_data):  # Excelからデータを取得する
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

    def preprocess(self, u, y):  # データを前処理
        u1 = np.zeros(u.shape)
        y1 = np.zeros(y.shape)
        u1 = np.where(u < 1000000, u, 1000000)
        u1 = np.log10(u1) - 4.5

        y1 = y

        return u1, y1

    def collect(self, u, y, delay):
        start = 0

        for i in range(int(len(u[0, :]))):
            if i == 0:
                pre_val = u[11, 0]
            else:
                val = u[11, i]
                if val < 0.5 and pre_val > 0.5:
                    start = i - 5
                    break
                pre_val = val

        batchSize = 100
        inSize = 12
        outSize = 1

        dataLen = int((u.shape[1] - start) / batchSize)
        u_tdm = np.zeros((dataLen, inSize, batchSize))
        y_tdm = np.zeros((outSize, dataLen))

        batch_start = start
        batch_end = start + batchSize
        for i in range(dataLen):
            u_tdm[i, :, :] = u[:, batch_start:batch_end]
            y_tdm[0, i] = y[0, batch_start - delay]
            batch_start = batch_end
            batch_end = batch_start + batchSize
            if batch_end > int(len(u[0, :])):
                break

        return u_tdm, y_tdm


class Physical_reservoir(Process_excel):  # 機械学習するクラス
    def __init__(self, training_data, test_data, delay):
        # 訓練データとテストデータを抽出
        train_data = pd.read_excel(training_data)
        test_data = pd.read_excel(test_data)
        self.u_train, self.y_train = self.process_excel_files(train_data)
        self.u_test, self.y_test = self.process_excel_files(test_data)
        self.u1_train, self.y1_train = self.preprocess(self.u_train,
                                                       self.y_train)
        self.u_test, self.y_test = self.preprocess(self.u_test,
                                                     self.y_test)
        self.u1_train, self.y1_train = self.collect(self.u1_train,
                                                    self.y1_train, delay)
        self.u1_test, self.y1_test = self.collect(self.u_test,
                                                  self.y_test, delay)
        mushi, self.y_without_delay = self.collect(self.u_test,
                                                   self.y_test, 0)

        self.inSize = self.u1_train.shape[1]  # 入力するChの数
        self.batchSize = self.u1_train.shape[2]  # 多重化した1バッチのサイズ
        self.outSize = self.y1_train.shape[0]  # 出力するChの数
        # 学習する重み
        self.Wout = np.zeros((self.outSize, self.inSize * self.batchSize))

        self.y_true = self.y1_test  # テストデータの出力
        self.y_pred = np.zeros(self.y1_test.shape)  # 予測を格納するアレイの初期化

    def train(self):  # 訓練
        trainLen = self.u1_train.shape[0]  # 訓練データの長さ
        X = np.zeros((self.inSize * self.batchSize, trainLen))
        Yt = self.y1_train[0, :trainLen].reshape((1, trainLen))

        for t in range(trainLen):
            u_current = self.u1_train[t, :, :].reshape((self.inSize *
                                                        self.batchSize))
            X[:, t] = u_current

        # Ridge Regression
        reg = 1e-4  # リッジパラメータ

        self.Wout = linalg.solve(np.dot(X, X.T) +
                                 reg*np.eye(self.inSize * self.batchSize),
                                 np.dot(X, Yt.T)).T

        # Linear Regression
        # Wout = linalg.solve(np.dot(X, X.T), np.dot(X, Yt.T)).T

    def test(self, print_flag):  # テスト
        testLen = self.u1_test.shape[0]  # テストデータ長
        X = np.zeros((self.inSize * self.batchSize, testLen))
        Y = np.zeros((self.outSize, testLen))

        for t in range(testLen):  # 各時刻の予測
            u_current = self.u1_test[t, :, :].reshape((-1, 1))
            y_current = np.dot(self.Wout, u_current)
            X[:, t] = u_current[:, 0].reshape(self.inSize * self.batchSize)
            Y[0, t] = y_current[0].item()

        self.y_pred = Y[0, :]  # 予測
        # labels = [0, 1, 2, 3]
        # mse = np.sqrt(sum(np.square(y_true[:] - y_pred[:])) /
        #               (testLen - self.initLen))
        # rmseの算出
        rmse1 = np.sqrt(np.sum(np.square(self.y_true - self.y_pred)) / testLen)
        # maeの算出
        mae1 = (np.sum(np.absolute(self.y_true - self.y_pred)) / testLen)
        # errorの最大値の算出
        max_error = np.amax(np.absolute(self.y_true - self.y_pred))

        # 各指標の表示
        if print_flag is True:
            print("rmse:", rmse1)
            print("mae:", mae1)
            print("max error:", max_error)

        result_values = np.array((rmse1, mae1), dtype=float)

        return result_values

    def determination_coefficient(self):
        covariance_y = np.cov(self.y_true[0, :], self.y_pred[:])
        variance_input = np.var(self.y_without_delay[0, :])
        variance_model = np.var(self.y_pred[:])

        r_2 = (covariance_y[0, 1])**2 / (variance_input * variance_model)

        return r_2

    def show_result(self):
        c1, c2 = 'green', 'blue'
        l1, l2 = "target", "predict"

        # xl = "time step"
        xl2 = "batch"

        fig1 = plt.figure(figsize=(10, 6), facecolor='lightblue')
        # グラフを描画するsubplot領域を作成。
        ax1 = fig1.add_subplot(1, 1, 1)
        # 各subplot領域にデータを渡す
        ax1.plot(self.y_true[0, :], color=c1, label=l1)
        # 各subplot領域にデータを渡す
        ax1.plot(self.y_pred[:], color=c2, label=l2)
        # 各subplotにxラベルを追加
        ax1.set_xlabel(xl2, fontsize=16)
        # 各subplotにx軸設定
        # ax1.set_xlim(8000, 9000)
        # 各subplotにyラベルを追加
        ax1.set_ylabel('height [mm]', fontsize=16)

        ax1.tick_params(labelsize=16)
        ax1.legend(loc=1)

        plt.tight_layout()

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
        # ax2.set_ylabel('input resistance [ohm]', fontsize=16)
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
        plt.show()


training_data = "/home/a24nitta/work/measurements/data/data_2025january16/" \
            "sensor_voltage_2025january16_8_memcheck.xlsx"
test_data = "/home/a24nitta/work/measurements/data/data_2025january16/" \
            "sensor_voltage_2025january16_9_memcheck.xlsx"

for k in range(1, 30):
    test = Physical_reservoir(training_data, test_data, delay=k)
    Wout = test.train()
    test.test(False)
    det_coff = test.determination_coefficient()
    print(det_coff)
    if k == 1:
        det_coff_series = np.zeros(1)
        det_coff_series[0] = det_coff
    else:
        det_coff_series = np.hstack((det_coff_series, det_coff))
    # test.show_result()

fig = plt.figure(figsize=(10, 6))
plt.stem(det_coff_series)
plt.tight_layout()
plt.show()
