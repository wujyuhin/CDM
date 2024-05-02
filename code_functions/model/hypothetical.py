import math
import numpy as np
from code_functions.data_generate.generate import attribute_pattern
from itertools import combinations
import numpy as np
from scipy.special import comb
import pandas as pd
from code_functions.EduCDM import EMDINA as DINA


class Hypothetical():
    def __init__(self, q_m, R, stu_num, prob_num, know_num, mode='loop'):
        """
        delta方法
        :param q_m: Q矩阵
        :param R:  学生作答矩阵R
        :param stu_num:  学生数
        :param prob_num:  题目数
        :param know_num:  知识点数
        :param mode:  'loop' or 'no_loop' 循环修正或者不循环修正
        :param alpha:  alpha显著性水平
        """
        self.q_m = q_m
        self.R = R
        self.stu_num = stu_num
        self.prob_num = prob_num
        self.know_num = know_num
        self.modify_q_m = q_m.copy()
        self.mode = mode  # 'loop' or 'no_loop'

    def modify_Q(self, **kwargs):
        # 非循环修正
        # 设置显著性水平
        if 'alpha' in kwargs:
            alpha = kwargs['alpha']  # 显著性水平
        else:
            alpha = 0.05
        # 计算guess和slip参数
        cdm = DINA(self.R, self.modify_q_m, self.stu_num, self.prob_num, self.know_num, skip_value=-1)
        cdm.train(epoch=2, epsilon=1e-3)
        g, s = cdm.guess, cdm.slip  # 题目的guess、slip参数
        # 对每一个Q向量(即每道题目)进行第一类错误的判断
        for i in range(self.modify_q_m.shape[0]):
            q0 = self.modify_q_m[i]  # 检验该第i道题目的q向量是否需要修正
            studs = cdm.all_states[cdm.theta, :]  # 获取所有学生的掌握模式
            studs_q0 = studs[np.all(studs == q0, axis=1), :]  # 获取该掌握模式q0的所有学生
            answer_q0 = self.R[np.all(studs == q0, axis=1), :]  # 获取该掌握模式q0的所有学生的作答情况
            n = studs_q0.shape[0]  # 掌握模式q0的所有学生数
            r = n - np.sum(answer_q0[:, i])  # 掌握模式q0的所有学生其中做第i道题目做错的学生数
            # H1:q≠q0 判断q向量是错误的
            if self.is_refuse(n, s[i], r, alpha):
                # 如果拒绝原假设，则进行修正
                q_up = self.q_up_down(q0)['q_up']
                # 将up按考察知识点数量大小排序
                q_up = sorted(q_up, key=lambda x: sum(x), reverse=False)
                if q_up is not None:
                    for m, q_up_item in enumerate(q_up):
                        # 对q_up中的每一个q向量进行检验
                        studs_q_up = studs[np.all(studs == q_up_item, axis=1), :]  # 获取该掌握模式q_up的所有学生
                        answer_q_up = self.R[np.all(studs == q_up_item, axis=1), :]  # 获取该掌握模式q_up的所有学生的作答情况
                        up_n = studs_q_up.shape[0]  # 掌握模式q_up的所有学生数
                        up_r = np.sum(answer_q_up[:, i])  # 其中做第i道题目做对的学生数
                        # H1:q=q_up 判断是否采取q_up的q向量作为修正q向量
                        if self.is_refuse(up_n,  g[i],up_r, alpha):
                            self.modify_q_m[i] = q_up_item
                            break
            # H0:q=q0 判断q向量是正确的
            else:
                q_down = self.q_up_down(q0)['q_down']
                q_down = sorted(q_down, key=lambda x: sum(x), reverse=False)
                if q_down is not None:
                    for m, q_down_item in enumerate(q_down):
                        studs_q_down = studs[np.all(studs == q_down_item, axis=1), :]  # 获取该掌握模式q_down的所有学生
                        answer_q_down = self.R[np.all(studs == q_down_item, axis=1), :]  # 获取该掌握模式q_down的所有学生的作答情况
                        down_n = studs_q_down.shape[0]  # 掌握模式q_down的所有学生数
                        down_r = np.sum(answer_q_down[:, i])  # 其中做第i道题目做对的学生数
                        # H1:q=q_down 判断是否采取q_down的q向量作为修正q向量
                        if self.is_refuse(down_n, g[i],down_r,  alpha):
                            self.modify_q_m[i] = q_down_item
                            break
        return self.modify_q_m

        # cur_Q = np.array(self.modify_q_m)
        # cur_gs = np.array(gs)
        # cur_pa = np.array(pa)
        # temp_Q = np.zeros_like(self.modify_q_m)
        # times = 0
        #

        # while not np.all(self.modify_q_m == temp_Q) and times <= 10:
        #     # cur_Q和temp_Q不完全相等，且迭代次数不超过10次
        #     times += 1
        #     temp_Q = self.modify_q_m.copy()
        #
        #     for i in range(self.modify_q_m.shape[0]):
        #         X = self.modify_q_m[i]
        #         cur_g, cur_s = cur_gs[i]
        #         cur_ans = np.column_stack((cur_pa, self.R[:, i]))  # 属性参数和作答矩阵的合并
        #
        #         n = len(cur_ans[np.all(cur_ans[:, :-1] == X, axis=1)]) / (len(X) + 1)
        #         r = n - np.sum(cur_ans[np.all(cur_ans[:, :-1] == X, axis=1), -1])

        # X = cur_Q[i]
        # cur_g, cur_s = cur_gs[i]
        # cur_ans = np.column_stack((cur_pa, d[:, i]))  # 属性参数和作答矩阵的合并,cur_pa是属性参数，d是作答矩阵
        # 属性参数是一个矩阵，每一行是一个学生的属性参数，作答矩阵是一个矩阵，每一行是一个学生的作答情况
        # np.column_stack作用是将两个矩阵按列合并，即将属性参数和作答矩阵合并
        # 例如：cur_pa = [[1, 0, 1], [0, 1, 1], [1, 1, 0]], d = [[1, 0, 0], [0, 1, 1], [1, 1, 0]]
        # cur_ans = [[1, 0, 1, 1, 0, 0], [0, 1, 1, 0, 1, 1], [1, 1, 0, 1, 1, 0]]

        #
        # n = len(cur_ans[np.all(cur_ans[:, :-1] == X, axis=1)]) / (len(X) + 1)
        # r = n - np.sum(cur_ans[np.all(cur_ans[:, :-1] == X, axis=1), -1])

        # if self.is_refuse(n, r, cur_s, a):
        #     q_up = self.q_up_down(X)['q_up']
        #     if q_up is not None:
        #         for m, q_up_item in enumerate(q_up):
        #             up_n = len(cur_ans[np.all(cur_ans[:, :-1] == q_up_item, axis=1)]) / (len(X) + 1)
        #             up_r = np.sum(cur_ans[np.all(cur_ans[:, :-1] == q_up_item, axis=1), -1])
        #             if self.is_refuse(up_n, up_r, cur_g, a):
        #                 cur_Q[i] = q_up_item
        #                 cur_DINA_results = update_model(self.R, cur_Q)
        #                 cur_gs, cur_pa = cur_DINA_results['gs'], cur_DINA_results['pa']
        #                 break
        # else:
        #     q_down = q_up_down(X)['q_down']
        #     if q_down is not None:
        #         for m, q_down_item in enumerate(q_down):
        #             down_n = len(cur_ans[np.all(cur_ans[:, :-1] == q_down_item, axis=1)]) / (len(X) + 1)
        #             down_r = np.sum(cur_ans[np.all(cur_ans[:, :-1] == q_down_item, axis=1), -1])
        #             if self.is_refuse(down_n, down_r, cur_g, a):
        #                 cur_Q[i] = q_down_item
        #                 cur_DINA_results = update_model(self.R, cur_Q)
        #                 cur_gs, cur_pa = cur_DINA_results['gs'], cur_DINA_results['pa']
        #                 break
        # return 1

    # 非类的函数

    def is_refuse(self, n, p, r, alpha=0.05):
        """
        拒绝域判断
        :param n:  试验次数
        :param r:   拒绝域的上界
        :param p:  原假设为真的情况下，二项分布的概率参数
        :param alpha:  显著性水平
        :return:  1表示拒绝原假设，0表示接受原假设

        example:
        is_refuse(10, 0.5, 5, 0.05),表示在试验次数为10次，拒绝域的上界为5，原假设为真的概率为0.5，显著性水平为0.05的情况下，是否拒绝原假设
        return 1
        """
        # 累计概率
        cumulate_p = 0
        for i in range(r, n + 1):  # 注意Python的range不包括结束值，因此需要加1
            cumulate_p += math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
        return 1 if np.isclose(cumulate_p, alpha, atol=1e-9) or cumulate_p < alpha else 0

    def q_up_down(self, q):
        """
        计算q向量的上下界
        :param q:  q向量
        :return:  返回q向量的上界和下界

        example:
        q_up_down([1, 0, 0, 1])
        return {'q': [1, 0, 0, 1],
                'q_up': [[1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 1]],
                'q_down': [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]]}
        """
        KS = attribute_pattern(len(q))  # Exclude the first row which represents all zeros
        q_up, q_down = [], []
        KS_temp = KS[np.any(KS != np.array(q), axis=1), :]  # Rows where any attribute differs from q
        for i in range(KS_temp.shape[0]):
            if np.all(KS_temp[i, :] >= np.array(q)):  # If all attributes are greater than or equal to q
                q_up.append(list(KS_temp[i, :]))
            elif np.all(KS_temp[i, :] <= np.array(q)):  # If all attributes are less than or equal to q
                q_down.append(list(KS_temp[i, :]))
        return {'q': q, 'q_up': q_up, 'q_down': q_down}


def update_model(cur_Q):
    # 这里模拟GDINA模型的更新，实际应用中需要替换为真实的GDINA模型处理
    # 返回系数gs和属性参数pa
    pass


if __name__ == "__main__":
    # is_refuse(10, 0.5, 9, 0.05)
    # q = [1, 0, 0, 1]
    # q_up_down([1, 0, 0, 1])
    # 注意：上述代码依赖于is_refuse和q_up_down函数的具体实现，以及update_model函数模拟GDINA模型的更新逻辑。
    # 请根据实际情况进行替换，以保证代码的正确性。
    q_m = np.loadtxt("../../data/math2015/simulation/simulation_q.csv", dtype=int, delimiter=',')
    prob_num, know_num = q_m.shape[0], q_m.shape[1]
    R = np.array(pd.read_csv("../../data/math2015/simulation/simulation_data.csv", index_col=0))
    stu_num = R.shape[0]
    # ================= 假设检验法 ==================================
    hypothetical = Hypothetical(q_m, R, stu_num, prob_num, know_num)
    gs = np.random.rand(prob_num, 2)
    pa = np.random.rand(stu_num, know_num)
    a = hypothetical.modify_Q(alpha=0.05)
