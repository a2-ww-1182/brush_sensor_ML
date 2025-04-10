# physical reservoirって言ってるけど入力をただのリッジ回帰
import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt


class Physical_reservoir():
    def __init__(self, u_train, y_train, u_test, y_test):
        self.u1_train = u_train
        self.y1_train = y_train
        self.u1_test = u_test
        self.y1_test = y_test

        self.channelSize = u_train.shape[1]
        self.batchSize = self.u1_train.shape[2]

        self.Wout = None
        self.y_pred = None
        self.y_true = None

    def train(self):
        trainLen = self.u1_train.shape[0]
        X = np.zeros((self.channelSize * self.batchSize, trainLen))
        Yt = self.y1_train

        for t in range(trainLen):
            u_current = self.u1_train[t, :, :].reshape((self.channelSize * self.batchSize))
            X[:, t] = u_current

        # ridge param
        reg = 1e-4
        self.Wout = linalg.solve(np.dot(X, X.T) +
                                 reg*np.eye(self.channelSize * self.batchSize),
                                 np.dot(X, Yt.T)).T

    def test(self):
        testLen = self.u1_test.shape[0]
        outSize = self.y1_test.shape[1]
        X = np.zeros((self.channelSize * self.batchSize, testLen))
        Y = np.zeros((outSize, testLen))

        for t in range(testLen):
            u_current = self.u1_test[t, :, :].reshape((-1, 1))
            y_current = np.dot(self.Wout, u_current)
            X[:, t] = u_current[:, 0].reshape(self.channelSize*self.batchSize)
            Y[0, t] = y_current[0].item()

        y_true = self.y1_test[0, :]
        y_pred = Y[0, :]
        self.y_true = self.y1_test[0, :]
        self.y_pred = Y[0, :]

        rmse1 = np.sqrt(sum(np.square(y_true[:] - y_pred[:])) / (testLen))
        mae1 = (sum(np.absolute(y_true[:] - y_pred[:])) / (testLen))
        max_error = np.amax(np.absolute(y_true[:] - y_pred[:]))
        print("rmse:", rmse1)
        print("mae:", mae1)
        print("max error:", max_error)

        return y_true, y_pred

    def result(self):
        c1, c2 = 'green', 'blue'
        l1, l2 = "target", "predict"

        xl = "time step"
        xl2 = "batch"

        fig1 = plt.figure(figsize=(10, 6), facecolor='lightblue')
        # グラフを描画するsubplot領域を作成。
        ax1 = fig1.add_subplot(1, 1, 1)
        # 各subplot領域にデータを渡す
        ax1.plot(self.y_true[:], color=c1, label=l1)
        # 各subplot領域にデータを渡す
        ax1.plot(self.y_pred[:], color=c2, label=l2)
        # 各subplotにxラベルを追加
        ax1.set_xlabel(xl2, fontsize=16)
        # 各subplotにyラベルを追加
        ax1.set_ylabel('height [mm]', fontsize=16)

        ax1.tick_params(labelsize=16)
        ax1.legend(loc=1)

        plt.tight_layout()

        # fig2 = plt.figure(figsize=(10, 6), facecolor='lightblue')
        # ax2 = fig2.add_subplot(1, 1, 1)
        # ax2.plot(self.u1_train[1, 0, :], label="ch1")
        # ax2.plot(self.u1_train[1, 1, :], label="ch2")
        # ax2.plot(self.u1_train[1, 2, :], label="ch3")
        # ax2.plot(self.u1_train[1, 3, :], label="ch4")
        # ax2.plot(self.u1_train[1, 4, :], label="ch5")
        # ax2.plot(self.u1_train[1, 5, :], label="ch6")
        # ax2.plot(self.u1_train[1, 6, :], label="ch7")
        # ax2.plot(self.u1_train[1, 7, :], label="ch8")
        # ax2.plot(self.u1_train[1, 8, :], label="ch9")
        # ax2.plot(self.u1_train[1, 9, :], label="ch10")
        # ax2.plot(self.u1_train[1, 10, :], label="ch11")
        # ax2.plot(self.u1_train[1, 11, :], label="ch12")
        # ax2.plot(self.u1_train[1, 12, :], label="ch12")
        # ax2.plot(self.u1_train[1, 13, :], label="ch12")
        # ax2.plot(self.u1_train[1, 14, :], label="ch12")
        # ax2.plot(self.u1_train[1, 15, :], label="ch12")
        # ax2.plot(self.u1_train[1, 16, :], label="ch12")
        # ax2.plot(self.u1_train[1, 17, :], label="ch12")
        # ax2.plot(self.u1_train[1, 18, :], label="ch12")
        # ax2.plot(self.u1_train[1, 19, :], label="ch12")
        # ax2.plot(self.u1_train[1, 20, :], label="ch12")
        # ax2.plot(self.u1_train[1, 21, :], label="ch12")
        # ax2.plot(self.u1_train[1, 22, :], label="ch12")
        # ax2.plot(self.u1_train[1, 23, :], label="ch12")
        # ax2.plot(self.u1_train[1, 24, :], label="ch12")
        # ax2.plot(self.u1_train[1, 25, :], label="ch12")
        # ax2.plot(self.u1_train[1, 26, :], label="ch12")
        # ax2.plot(self.u1_train[1, 27, :], label="ch12")
        # ax2.plot(self.u1_train[1, 28, :], label="ch12")
        # ax2.plot(self.u1_train[1, 29, :], label="ch12")
        # ax2.plot(self.u1_train[1, 30, :], label="ch12")
        # ax2.plot(self.u1_train[1, 31, :], label="ch12")
        # ax2.plot(self.u1_train[1, 32, :], label="ch12")
        # ax2.plot(self.u1_train[1, 33, :], label="ch12")
        # ax2.plot(self.u1_train[1, 34, :], label="ch12")
        # ax2.plot(self.u1_train[1, 35, :], label="ch12")
        # ax2.set_xlabel(xl, fontsize=16)
        # ax2.set_ylabel('processed value', fontsize=16)
        # ax2.tick_params(labelsize=16)
        # ax2.legend(loc=1)
        plt.tight_layout()

        # fig3 = plt.figure(figsize=(10, 6), facecolor='lightblue')
        # ax3 = fig3.add_subplot(1, 1, 1)
        # ax3.plot(self.u1_train[573, 0, :], label="ch1")
        # ax3.plot(self.u1_train[573, 1, :], label="ch2")
        # ax3.plot(self.u1_train[573, 2, :], label="ch3")
        # ax3.plot(self.u1_train[573, 3, :], label="ch4")
        # ax3.plot(self.u1_train[573, 4, :], label="ch5")
        # ax3.plot(self.u1_train[573, 5, :], label="ch6")
        # ax3.plot(self.u1_train[573, 6, :], label="ch7")
        # ax3.plot(self.u1_train[573, 7, :], label="ch8")
        # ax3.plot(self.u1_train[573, 8, :], label="ch9")
        # ax3.plot(self.u1_train[573, 9, :], label="ch10")
        # ax3.plot(self.u1_train[573, 10, :], label="ch11")
        # ax3.plot(self.u1_train[573, 11, :], label="ch12")
        # ax3.plot(self.u1_train[573, 12, :], label="ch12")
        # ax3.plot(self.u1_train[573, 13, :], label="ch12")
        # ax3.plot(self.u1_train[573, 14, :], label="ch12")
        # ax3.plot(self.u1_train[573, 15, :], label="ch12")
        # ax3.plot(self.u1_train[573, 16, :], label="ch12")
        # ax3.plot(self.u1_train[573, 17, :], label="ch12")
        # ax3.plot(self.u1_train[573, 18, :], label="ch12")
        # ax3.plot(self.u1_train[573, 19, :], label="ch12")
        # ax3.plot(self.u1_train[573, 20, :], label="ch12")
        # ax3.plot(self.u1_train[573, 21, :], label="ch12")
        # ax3.plot(self.u1_train[573, 22, :], label="ch12")
        # ax3.plot(self.u1_train[573, 23, :], label="ch12")
        # ax3.plot(self.u1_train[573, 24, :], label="ch12")
        # ax3.plot(self.u1_train[573, 25, :], label="ch12")
        # ax3.plot(self.u1_train[573, 26, :], label="ch12")
        # ax3.plot(self.u1_train[573, 27, :], label="ch12")
        # ax3.plot(self.u1_train[573, 28, :], label="ch12")
        # ax3.plot(self.u1_train[573, 29, :], label="ch12")
        # ax3.plot(self.u1_train[573, 30, :], label="ch12")
        # ax3.plot(self.u1_train[573, 31, :], label="ch12")
        # ax3.plot(self.u1_train[573, 32, :], label="ch12")
        # ax3.plot(self.u1_train[573, 33, :], label="ch12")
        # ax3.plot(self.u1_train[573, 34, :], label="ch12")
        # ax3.plot(self.u1_train[573, 35, :], label="ch12")
        # ax3.set_xlabel(xl, fontsize=16)
        # ax3.set_ylabel('processed value', fontsize=16)
        # ax3.tick_params(labelsize=16)
        # ax2.legend(loc=1)
        plt.tight_layout()

        plt.show()
