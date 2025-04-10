# memory check task (delay task)
# try to detect height at previous one cycle (100 timestep unit)
# import numpy as np
from process_excel import Process_excel
from physical_model import Physical_reservoir


if __name__ == '__main__':
    delay = 1
    test = Process_excel(delay)
    dir_name = "/home/a24nitta/work/measurements/data/data_2025march28/"
    list_of_train = list(range(1, 81))
    list_of_test = [81]
    cycle_flag = True

    y_current = None

    for k in range(0, delay + 1):
        u_train_k, y_train_k = \
            test.combine_data_augment(dir_name=dir_name,
                                      list_of_num=list_of_train,
                                      train_flag=True,
                                      k=k)
        u_test_k, y_test_k = \
            test.combine_data_augment(dir_name=dir_name,
                                      list_of_num=list_of_test,
                                      train_flag=False,
                                      k=k)
        print(u_train_k.shape)

        model = Physical_reservoir(u_train_k, y_train_k, u_test_k, y_test_k)
        model.train()
        y_true_k, y_pred_k = model.test()
        model.result()
