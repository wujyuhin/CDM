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

    def no_loop_modify_Q(self, **kwargs):
        # 非循环修正
        alpha = kwargs['alpha'] if 'alpha' in kwargs else 0.05  # 显著性水平
        modify_q_m = kwargs['modify_q_m'] if 'modify_q_m' in kwargs else self.q_m.copy()  # 待修改的Q矩阵
        # 计算guess和slip参数
        cdm = DINA(self.R, modify_q_m, self.stu_num, self.prob_num, self.know_num, skip_value=-1)
        cdm.train(epoch=2, epsilon=1e-3)
        g, s = cdm.guess, cdm.slip  # 题目的guess、slip参数
        # 对每一个Q向量(即每道题目)进行第一类错误的判断
        flag = -1
        for i in range(modify_q_m.shape[0]):
            # debug使用
            flag +=1
            if flag == 31:
                print('debug')
            q0 = modify_q_m[i]  # 检验该第i道题目的q向量是否需要修正
            studs = cdm.all_states[cdm.theta, :]  # 获取所有学生的掌握模式
            studs_q0 = studs[np.all(studs == q0, axis=1), :]  # 获取该掌握模式q0的所有学生
            # 如果没有学生掌握该模式，则不进行修正
            if studs_q0.shape[0] == 0:
                continue
            answer_q0 = self.R[np.all(studs == q0, axis=1), :]  # 获取该掌握模式q0的所有学生的作答情况
            n = studs_q0.shape[0]  # 掌握模式q0的所有学生数
            r = n - np.sum(answer_q0[:, i])  # 掌握模式q0的所有学生其中做第i道题目做错的学生数
            a = self.is_refuse(n, s[i], r, 0.01)
            b = self.is_refuse(n, s[i], r, 0.05)
            c = self.is_refuse(n, s[i], r, 0.1)
            # H1:q≠q0 判断q向量是错误的
            if self.is_refuse(n, s[i], r, alpha):
                # 如果拒绝原假设，则进行修正
                q_up = q_up_down(q0)['q_up']
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
                        a = self.is_refuse(n, s[i], r, 0.01)
                        b = self.is_refuse(n, s[i], r, 0.05)
                        c = self.is_refuse(n, s[i], r, 0.1)
                        if self.is_refuse(up_n, g[i], up_r, alpha):
                            modify_q_m[i] = q_up_item
                            break
            # H0:q=q0 判断q向量是正确的
            else:
                q_down = q_up_down(q0)['q_down']
                q_down = sorted(q_down, key=lambda x: sum(x), reverse=False)
                if q_down is not None:
                    for m, q_down_item in enumerate(q_down):
                        studs_q_down = studs[np.all(studs == q_down_item, axis=1), :]  # 获取该掌握模式q_down的所有学生
                        answer_q_down = self.R[np.all(studs == q_down_item, axis=1), :]  # 获取该掌握模式q_down的所有学生的作答情况
                        down_n = studs_q_down.shape[0]  # 掌握模式q_down的所有学生数
                        down_r = np.sum(answer_q_down[:, i])  # 其中做第i道题目做对的学生数
                        # H1:q=q_down 判断是否采取q_down的q向量作为修正q向量
                        if self.is_refuse(down_n, g[i], down_r, alpha):
                            modify_q_m[i] = q_down_item
                            break
        self.modify_q_m = modify_q_m
        return self.modify_q_m

    def loop_modify_Q(self, **kwargs):
        # 循环修正
        q_modify_old = self.q_m.copy()
        # 因为在函数里面对modify_q_m进行了修改，所以需要copy一份，保证不影响原来的q_modify_old
        q_modify_new = self.no_loop_modify_Q(modify_q_m=q_modify_old.copy(), **kwargs)
        flag = 0
        while not np.all(q_modify_old == q_modify_new):
            q_modify_old = q_modify_new.copy()
            q_modify_new = self.no_loop_modify_Q(modify_q_m=q_modify_old.copy(), **kwargs)
            flag += 1
            if flag > 20:
                break
        # print(f"循环修正次数：{flag}")
        return self.modify_q_m

    def modify_Q(self, **kwargs):
        self.mode = kwargs['mode'] if 'mode' in kwargs else self.mode
        if self.mode == 'loop':
            return self.loop_modify_Q(**kwargs)
        elif self.mode == 'no_loop':
            return self.no_loop_modify_Q(**kwargs)
        else:
            raise ValueError("mode should be 'loop' or 'no_loop'")

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


def q_up_down(q):
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
    # hypothetical = Hypothetical(q_m, R, stu_num, prob_num, know_num, mode='loop')
    # a = hypothetical.loop_modify_Q(alpha=0.05)
