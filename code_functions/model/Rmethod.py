import numpy as np
import pandas as pd
import time
from code_functions.EduCDM import EMDINA as DINA
from code_functions.data_generate.generate import attribute_pattern


def R_correction(K, tmp_P, tmp_N, Q_wrong_rate, data_set, Q_wrong_set):
    # R语言中的cat()输出在这里替换为print()，便于在Python中展示
    # print(f"正在R修正... K: {K}, P: {tmp_P}, N: {tmp_N}, Q_wrong_rate: {Q_wrong_rate}")

    start_time0 = time.time()

    for size in range(data_set.shape[1]):  # 对于数据集的每一列（即每个数据子集）
        # print(size)
        data = data_set.iloc[:, size]
        Qwrong = Q_wrong_set.iloc[:, size]

        # 基于错误Q矩阵和作答矩阵拟合DINA模型
        d = data['ORP']['dat']
        q = Qwrong['Qwrong']

        myDINA = GDINA(d, q, model='DINA')

        gs = myDINA.coef('gs')
        pa = myDINA.personparm()  # person attribute 缩写

        # 尝试修正
        may_ks = attribute_pattern(K)[:-1, ]
        cur_Q = q.copy()
        cur_gs = gs.copy()
        cur_pa = pa.copy()

        for i in range(q.shape[0]):  # 对于每一个q
            i_R = np.zeros((may_ks.shape[0], 1))
            for j in range(may_ks.shape[0]):  # 对于每种可能的考察模式
                cur_Q[i, :] = may_ks[j,]  # 当前Q
                cur_DINA = GDINA(d, cur_Q, model='DINA')
                cur_gs = cur_DINA.coef('gs')
                cur_pa = cur_DINA.personparm()  # person attribute 缩写
                cur_ans = pd.concat([cur_pa, d.iloc[:, i]], axis=1)
                cur_q_right, cur_q_wrong = q_right_wrong(cur_Q[i,])
                cur_right_n = 0
                cur_wrong_n = 0
                cur_right_r = 0
                cur_wrong_r = 0

                for m in range(cur_q_right.shape[0]):
                    cur_right_n += len(
                        cur_ans[(cur_ans.iloc[:, :-1].apply(lambda x: all(x == cur_q_right[m, :]), axis=1))]) / (
                                               len(cur_q_right[m, :]) + 1)
                    cur_right_r += sum(
                        cur_ans[(cur_ans.iloc[:, :-1].apply(lambda x: all(x == cur_q_right[m, :]), axis=1)), -1])

                for mm in range(cur_q_wrong.shape[0]):
                    cur_wrong_n += len(
                        cur_ans[(cur_ans.iloc[:, :-1].apply(lambda x: all(x == cur_q_wrong[mm, :]), axis=1))]) / (
                                               len(cur_q_wrong[mm, :]) + 1)
                    cur_wrong_r += sum(
                        cur_ans[(cur_ans.iloc[:, :-1].apply(lambda x: all(x == cur_q_wrong[mm, :]), axis=1)), -1])

                i_R[j, 0] = 2 * ((2 * cur_right_r - cur_right_n) * np.log(cur_gs[i, 2] / (1 - cur_gs[i, 2])) +
                                 (cur_wrong_n - 2 * cur_wrong_r) * np.log(cur_gs[i, 1] / (1 - cur_gs[i, 1])))

            index = np.argmin(i_R)
            cur_Q[i, :] = may_ks[index,]  # 当前Q
            # print(f"已修正第{i}个题目")

    end_time0 = time.time()
    elapsed_time0 = end_time0 - start_time0
    print(f"R修正耗时: {elapsed_time0:.2f}秒")
