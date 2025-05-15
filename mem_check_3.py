# memory check task (delay task)
# try to detect height at previous one cycle (100 timestep unit)
# import numpy as np
from process_excel import Process_excel
from physical_model import Physical_reservoir
import glob
import os


if __name__ == '__main__':
    delay = 1
    test = Process_excel(delay)
    train_dir = "/home/a24nitta/work/measurements/data/data_matome/training"
    list_of_train = glob.glob(os.path.join(train_dir, '*.xlsx'))
    test_dir = "/home/a24nitta/work/measurements/data/data_matome/test"
    list_of_test = glob.glob(os.path.join(test_dir, '*.xlsx'))
    cycle_flag = True

    y_current = None

    for k in range(0, delay + 1):
        u_train_k, y_train_k = \
            test.combine_data_augment(file_list=list_of_train,
                                      train_flag=True,
                                      k=k)
        u_test_k, y_test_k = \
            test.combine_data_augment(file_list=list_of_test,
                                      train_flag=False,
                                      k=k)
        print(u_train_k.shape)

        model = Physical_reservoir(u_train_k, y_train_k, u_test_k, y_test_k)
        model.train()
        y_true_k, y_pred_k = model.test()
        model.result()
