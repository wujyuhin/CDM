# hypothesis_skill是以知识点单位的假设检验
# 孙假设检验是以题目考察模式即q向量未单位的假设检验
import random
import pandas as pd
from scipy.stats import norm
from codes.data_generate.generate import generate_Q, generate_wrong_R, generate_wrong_Q
import numpy as np
from codes.data_generate.generate import attribute_pattern, state_sample, state_answer, state_answer_gs
import math
from codes.EduCDM import EMDINA as DINA
from scipy.stats import binom
from codes.model.metric import PMR, AMR, TPR, FPR
from codes.model.delta import Delta
from codes.model.gamma import Gamma
from tqdm import tqdm
import pandas as pd


def approx_binomial_probability_corrected(n, p, k):
    # 计算二项分布的均值和标准差
    mean = n * p
    std_dev = (n * p * (1 - p)) ** 0.5

    # 进行连续性校正，k减去0.5
    k_corrected = k - 0.5

    # 标准化k到Z分数，使用校正后的k值
    z = (k_corrected - mean) / std_dev

    # 使用正态分布的CDF计算P(X <= k_corrected)，然后转换为P(X > k)
    prob_greater_than_k_corrected = 1 - norm.cdf(z)

    return prob_greater_than_k_corrected


class Hypothetical_skill():
    def __init__(self, q_m, R, stu_num, prob_num, know_num, alpha=0.01, mode='loop'):
        self.q_m = q_m
        self.R = R
        self.stu_num = stu_num
        self.prob_num = prob_num
        self.know_num = know_num
        self.modify_q_m = q_m.copy()
        self.mode = mode
        self.alpha = alpha

    def modify_Qj(self, j, alpha):
        """
        对第j题的q向量进行检验
        :param j:  第j题
        :param alpha:  显著性水平
        :return:  修改后的q向量，并更新对象的modify_q_m属性
        """
        # 参数估计
        Q_tilde = np.delete(self.q_m, j, axis=0)  # 删除第j题的q向量,降低其他题目的cdm估计误差
        R_tilde = np.delete(self.R, j, axis=1)  # 删除第j题的作答情况
        cdm = DINA(R_tilde, Q_tilde, self.stu_num, self.prob_num - 1, self.know_num, skip_value=-1)
        cdm.train(epoch=2, epsilon=1e-3)
        g, s = cdm.guess, cdm.slip  # 题目的guess、slip参数
        beta = cdm.all_states[cdm.theta, :]  # 学生掌握模式估计
        # 假设检验
        q = self.modify_q_m[j, :].copy()
        for k in range(self.know_num):
            r_j = self.R[:, j]  # 所有学生对j题的作答情况
            # q = self.q_m[j, :].copy()  # 第j题的q向量
            # q = self.modify_q_m[j, :].copy()  # 第j题的q向量【边发现边修改】
            if q[k] == 0:
                # q[k] = 0说明可能缺失1，所以检验缺失情况，
                studs_select, r_select = T_sample(beta, q, k, r_j)  # 筛选出所有掌握模式所有分量都大于等于q对应分量，且第k个知识点为0的学生
                n = studs_select.shape[0]  # 掌握模式q0的所有学生数
                r = n - np.sum(r_select)  # 计算统计量：错误数量
                p = np.mean(s)  # 用其他题目的估计的s平均作为第j题的s
                # 检验是否拒绝原假设
                if is_refuse(n, p, r, alpha):
                    # 如果拒绝原假设，则修改q向量
                    self.modify_q_m[j, k] = 1  # 用self.modify_q_m 表示一边发现一边修改
            elif (q[k] == 1) and sum(q) > 1:  # 至少有两个1才有可能冗余
                qq = q.copy()  # 因为修改q,然后进行学生筛选，所以需要备份q
                qq[k] = 0  # 根据不掌握k知识点的学生进行检验，如果做对多了，说明k也是无关紧要的
                studs_select, r_select = T_sample(beta, qq, k, r_j)  # 筛选出所有掌握模式所有分量都大于等于q对应分量，且第k个知识点为0的学生
                n = studs_select.shape[0]  # 掌握模式q0的所有学生数
                # q[k] = 1说明可能冗余1，所以检验冗余情况
                r = np.sum(r_select)  # 计算统计量：正确数量
                # g的中位数
                p = np.mean(g)
                # 检验是否拒绝原假设
                if is_refuse(n, p, r, alpha):
                    # 如果拒绝原假设，则修改q向量
                    self.modify_q_m[j, k] = 0
            else:
                # print(f'修改过程中q向量只考察第{j}题的第{k}个知识点，不需检验冗余')
                continue
        # 需要判断修改后的q向量是否全为0！
        # if np.all(self.modify_q_m[j, :] == 0):
        #     # 则把所有原本是1的都计算显著性检验，取p值最大的，即最不显著的
        #     q_prob = dict()  # q向量每个元素对应的p值
        #     for k in range(self.know_num):
        #         if q[k] == 1:
        #             qq = q.copy()
        #             qq[k] = 0
        #             studs_select, r_select = T_sample(beta, qq, k, r_j)
        #             n = studs_select.shape[0]
        #             r = np.sum(r_select)
        #             p = np.mean(g)
        #             prob = is_refuse(n, p, r, alpha, prob=True)
        #             q_prob[k] = prob
        #     # 取p值最大的，即最不显著的q向量对应元素
        #     max_p = max(q_prob.values())
        #     for k, v in q_prob.items():
        #         if v == max_p:
        #             self.modify_q_m[j, k] = 1
        if np.all(self.modify_q_m[j, :] == 0):
            # 如果全为0，取p值最大的，即最不显著的
            for i in range(self.know_num):
                prob = self.modify_Qjk(cdm, j, i, alpha)
                if prob <= alpha:
                    self.modify_q_m[j, i] = 1
        return self.modify_q_m[j, :]

    def modify_qvector(self, item, **kwargs):
        """ 用作实例验证，对第item道题目的q向量进行检验
        修改题目item的q向量
        :param item:  第item道题目
        :param alpha:  显著性水平
        :return:  修改后的q向量，并更新对象的modify_q_m属性
        """
        alpha = kwargs['alpha'] if 'alpha' in kwargs else self.alpha  # 显著性水平
        mode = kwargs['mode'] if 'mode' in kwargs else self.mode  # 修正模式
        if mode == 'no_loop':
            self.modify_Qj(item, alpha)
        elif mode == 'loop':
            q_old = -1 * np.ones(self.know_num)  # 初始化上一次的q向量
            stop = 5  # 最大循环次数
            while stop > 0:
                stop -= 1
                q = self.modify_q_m[item, :].copy()
                q_new = self.modify_Qj(item, alpha)
                if np.all(q == q_new):
                    break
                elif np.all(q_old == q_new):
                    # 修改后的q向量与上一次修改的q向量不同,但与上上一次修改q向量相同,说明进入了往返循环
                    if np.all(np.array(q_new) >= np.array(q)):
                        self.modify_Qj(item, alpha)  # 若修改后的q向量比原q向量大,则再次修改,取简单的q向量
                        break
                    else:
                        break
                else:
                    q_old = q
                    # 修改后的q向量 与上一次\上上次q向量不同,说明应该正常再进行修改
                    continue
        return self.modify_q_m

    def modify_Q(self, **kwargs):
        """
        对所有题目的q向量进行检验
        :param kwargs:  alpha显著性水平
        :return:  修改后的q向量，并更新对象的modify_q_m属性
        """
        alpha = kwargs['alpha'] if 'alpha' in kwargs else self.alpha  # 显著性水平
        mode = kwargs['mode'] if 'mode' in kwargs else self.mode  # 修正模式
        if mode == 'no_loop':
            # 没到题目的q向量进行检验
            for j in range(self.prob_num):
                self.modify_Qj(j, alpha)
        # elif mode == 'loop':
        #     # 循环修正
        #     stop = 5 # 最大循环次数
        #     flag = 1
        #     for j in range(self.prob_num):
        #         while flag and stop > 0:
        #             flag = 0
        #             stop -= 1
        #             q = self.modify_q_m[j, :].copy()  # 第j题的q向量
        #             q_new = self.modify_Qj(j, alpha)  # 修改后的q向量
        #             if not np.all(q == q_new):  # q==q_new说明没有修改,否则说明修改了,需要再次循环
        #                 flag = 1
        elif mode == 'loop':
            # 循环修正
            q_old = -1 * np.ones(self.know_num)  # 初始化上一次的q向量
            for j in range(self.prob_num):  # 对所有题目分别检验
                stop = 4  # 最大循环次数
                while stop > 0:
                    stop -= 1
                    q = self.modify_q_m[j, :].copy()  # 第j题的q向量
                    q_new = self.modify_Qj(j, alpha)  # 修改后的q向量
                    if np.all(q == q_new):
                        break
                    elif np.all(q_old == q_new):
                        # 修改后的q向量与上一次修改的q向量不同,但与上上一次修改q向量相同,说明进入了往返循环
                        if np.all(np.array(q_new) >= np.array(q)):
                            self.modify_Qj(j, alpha)  # 若修改后的q向量比原q向量大,则再次修改,取简单的q向量
                            break
                        else:
                            break
                    else:
                        q_old = q
                        # 修改后的q向量 与上一次\上上次q向量不同,说明应该正常再进行修改
                        continue
        return self.modify_q_m

    def modify_Qjk(self, cdm_deletej, j, k, alpha):
        """ 对第j题的第k个知识点的q向量进行检验
        :param cdm_deletej:  删除了第j题的QR计算的cdm模型
        :param j:  第j题
        :param k:  第k个知识点
        :param alpha:  显著性水平
        :return:  修改后的q向量，并更新对象的modify_q_m属性和拒绝原假设概率
        """
        # 参数估计
        g, s = cdm_deletej.guess, cdm_deletej.slip  # 题目的guess、slip参数
        beta = cdm_deletej.all_states[cdm_deletej.theta, :]  # 学生掌握模式估计
        # 假设检验
        q_vector = self.modify_q_m[j, :].copy()
        if q_vector[k] == 0:
            # q[k] = 0说明可能缺失1，所以检验缺失情况，
            studs_select, r_select = T_sample(beta, q_vector, k, self.R[:, j])
            n = studs_select.shape[0]  # 掌握模式q0的所有学生数
            r = n - np.sum(r_select)  # 计算统计量：错误数量
            p = np.mean(s)  # 用其他题目的估计的s平均作为第j题的s
            # 拒绝原假设概率
            return is_refuse(n, p, r, alpha, prob=True)
        elif q_vector[k] == 1:
            # q[k] = 1说明可能冗余1，所以检验冗余情况
            qq = q_vector.copy()
            qq[k] = 0
            studs_select, r_select = T_sample(beta, qq, k, self.R[:, j])
            n = studs_select.shape[0]
            r = np.sum(r_select)  # 计算统计量：正确数量
            p = np.mean(g)
            # 拒绝原假设概率
            return is_refuse(n, p, r, alpha, prob=True)
        else:
            raise ValueError("q向量中的元素存在非01元素，不符合假设检验条件，不进行假设检验")

    def modify_Qj_method2(self, j, alpha, **kwargs):
        """ 对第j题的q向量进行检验, 想了第二种修正1题的方法(修正顺序与方法1不同)
        :param j:  第j题
        :param alpha:  显著性水平
        :return:  修改后的q向量，并更新对象的modify_q_m属性
        """
        if 'model' in kwargs:
            cdm = kwargs['model']
        else:
            Q_tilde = np.delete(self.q_m.copy(), j, axis=0)  # 删除第j题的q向量,降低其他题目的cdm估计误差
            R_tilde = np.delete(self.R.copy(), j, axis=1)  # 删除第j题的作答情况
            cdm = DINA(R_tilde, Q_tilde, self.stu_num, self.prob_num - 1, self.know_num, skip_value=-1)
            cdm.train(epoch=2, epsilon=1e-3)
        # 先对q向量为0的元素进行检验
        q_modify_from0 = self.modify_q_m[j, :].copy()  # 待修正的q向量,先对q向量为0的元素进行检验
        zero_index = np.where(q_modify_from0 == 0)[0]  # 先对q向量为0的元素索引
        ht_zero = dict()  # 存储检验结果
        for k in zero_index:
            prob = self.modify_Qjk(cdm, j, k, alpha)
            if prob <= alpha:
                ht_zero[k] = prob
        q_modify_from0[list(ht_zero.keys())] = 1  # 将检验拒绝的元素修改为1
        # 再对q向量为1的元素进行检验
        q_modify_from1 = q_modify_from0.copy()  # 待修正的q向量,再对q向量为1的元素进行检验
        one_index = np.where(q_modify_from0 == 1)[0]
        ht_one = dict()
        for k in one_index:
            prob = self.modify_Qjk(cdm, j, k, alpha)
            if prob <= alpha:
                ht_one[k] = prob
        q_modify_from1[list(ht_one.keys())] = 0
        # 判断是否全为0
        ht_zero = dict()
        if np.all(q_modify_from1 == 0):
            # 如果全为0，取p值最大的，即最不显著的
            for i in range(self.know_num):
                ht_zero[i] = self.modify_Qjk(cdm, j, i, alpha)
            max_p = max(ht_zero.values())
            for k, v in ht_zero.items():
                if v == max_p:
                    q_modify_from1[k] = 1
        self.modify_q_m[j, :] = q_modify_from1
        return q_modify_from1

    def modify_Qj_method2_loop(self, j, alpha):
        """ 对第j题的q向量进行检验
        :param j:  第j题
        :param alpha:  显著性水平
        :return:  修改后的q向量，并更新对象的modify_q_m属性
        """
        Q_tilde = np.delete(self.q_m.copy(), j, axis=0)  # 删除第j题的q向量,降低其他题目的cdm估计误差
        R_tilde = np.delete(self.R.copy(), j, axis=1)  # 删除第j题的作答情况
        cdm = DINA(R_tilde, Q_tilde, self.stu_num, self.prob_num - 1, self.know_num, skip_value=-1)
        cdm.train(epoch=2, epsilon=1e-3)
        stop = 5  # 最大循环次数
        q_old = -1 * np.ones(self.know_num)  # 初始化上一次的q向量
        while stop > 0:
            stop -= 1
            q = self.modify_q_m[j, :].copy()
            q_new = self.modify_Qj_method2(j, alpha, model=cdm)
            if np.all(q == q_new):
                break
            elif np.all(q_old == q_new):
                # 修改后的q向量与上一次修改的q向量不同,但与上上一次修改q向量相同,说明进入了往返循环
                if np.all(np.array(q_new) >= np.array(q)):
                    self.modify_Qj_method2(j, alpha, model=cdm)
                    break
                else:
                    break
            else:
                q_old = q
                continue
        return self.modify_q_m[j, :]

    def modify_Q_method2(self, **kwargs):
        """
        对所有题目的q向量进行检验
        :param kwargs:  alpha显著性水平
        :return:  修改后的q向量，并更新对象的modify_q_m属性
        """
        alpha = kwargs['alpha'] if 'alpha' in kwargs else self.alpha
        mode = kwargs['mode'] if 'mode' in kwargs else self.mode
        if mode == 'no_loop':
            for j in range(self.prob_num):
                self.modify_Qj_method2(j, alpha)
        elif mode == 'loop':
            for j in range(self.prob_num):
                self.modify_Qj_method2_loop(j, alpha)
        else:
            raise ValueError("mode must be no_loop or loop")
        return self.modify_q_m

    def modify_Qjk_method3(self, cdm_deletej, j, k, alpha):
        """ 对第j题的q向量进行检验 输出第j题0->1概率和1>0概率
        :param cdm_deletej:  删除了第j题的QR计算的cdm模型
        :param j:  第j题
        :param k:  第k个知识点
        :param alpha:  显著性水平
        :return:  修改后的q向量，并更新对象的modify_q_m属性和拒绝原假设概率
        """
        # 参数估计
        g, s = cdm_deletej.guess, cdm_deletej.slip  # 题目的guess、slip参数
        beta = cdm_deletej.all_states[cdm_deletej.theta, :]  # 学生掌握模式估计
        # 筛选样本（规则是一致的）
        q_vector = self.modify_q_m[j, :].copy()
        q_vector[k] = 0
        studs_select, r_select = T_sample(beta, q_vector, k, self.R[:, j])
        n = studs_select.shape[0]  # 掌握模式q0的所有学生数
        # q向量k知识点为0的情况
        r = n - np.sum(r_select)  # 计算统计量：错误数量
        p = np.mean(s)  # 用其他题目的估计的s平均作为第j题的s
        prob0 = is_refuse(n, p, r, alpha, prob=True)  # p<0.01，说明k知识点修正为1
        # k知识点为1的情况
        r = np.sum(r_select)  # 计算统计量：正确数量
        p = np.mean(g)
        prob1 = is_refuse(n, p, r, alpha, prob=True)  # p<0.01，说明k知识点修正为0
        return prob0, prob1

    def modify_Qj_method3(self, j, alpha, **kwargs):
        """ 对第j题的q向量进行检验
        :param j:  第j题
        :param alpha:  显著性水平
        """
        # 如果输入了model=cdm就用输入的，没有输入就重新训练一遍
        if 'model' in kwargs:
            cdm = kwargs['model']
        else:
            Q_tilde = np.delete(self.q_m.copy(), j, axis=0)
            R_tilde = np.delete(self.R.copy(), j, axis=1)
            cdm = DINA(R_tilde, Q_tilde, self.stu_num, self.prob_num - 1, self.know_num, skip_value=-1)
            cdm.train(epoch=2, epsilon=1e-3)
        # 对每个知识点进行检验
        for k in range(self.know_num):
            prob0, prob1 = self.modify_Qjk_method3(cdm, j, k, alpha)
            # 0->1,1->0都小于alpha，说明两者都显著，取p值小的
            if prob0 <= alpha and prob1 <= alpha:
                #  ===============  取简单的改法(1->0)  ===================a
                # if prob0/prob1 <= 0.1:  # 0->1的概率足够小，比1->0还要小
                #     self.modify_q_m[j, k] = 1
                # else:
                #     self.modify_q_m[j, k] = 0
                # self.modify_q_m[j, k] = 0
                # ================  取概率小的改法(p0->1,p1->0)  ===================
                if prob0 < prob1:
                    self.modify_q_m[j, k] = 1
                else:
                    self.modify_q_m[j, k] = 0
            # 0->1显著，1->0不显著，说明0->1显著
            elif prob0 <= alpha:
                self.modify_q_m[j, k] = 1
            # 1->0显著，0->1不显著，说明1->0显著
            elif prob1 <= alpha:
                self.modify_q_m[j, k] = 0
            # 两者都不显著，不修改
            else:
                continue
        ht_zero = dict()
        if np.all(self.modify_q_m[j, :] == 0):
            # 如果全为0，取p值最小的，即最显著为1
            for i in range(self.know_num):
                ht_zero[i] = self.modify_Qjk(cdm, j, i, alpha)
            # 取ht_zero字典最小值的键
            min_key = min(ht_zero, key=ht_zero.get)
            self.modify_q_m[j, min_key] = 1
        return self.modify_q_m

    def modify_Qj_method3_loop(self, j, alpha):
        """ 对第j题的q向量进行检验
        :param j:  第j题
        :param alpha:  显著性水平
        :return:  修改后的q向量，并更新对象的modify_q_m属性
        """
        Q_tilde = np.delete(self.q_m.copy(), j, axis=0)
        R_tilde = np.delete(self.R.copy(), j, axis=1)
        # ======================= 以下两行代码可以放到while循环内，
        cdm = DINA(R_tilde, Q_tilde, self.stu_num, self.prob_num - 1, self.know_num, skip_value=-1)
        cdm.train(epoch=2, epsilon=1e-3)
        stop = 5  # 最大循环次数
        q_old = -1 * np.ones(self.know_num)  # 初始化上一次的q向量
        while stop > 0:
            stop -= 1
            q = self.modify_q_m[j, :].copy()
            q_new = self.modify_Qj_method3(j, alpha, model=cdm)
            if np.all(q == q_new):
                break
            elif np.all(q_old == q_new):
                # 修改后的q向量与上一次修改的q向量不同,但与上上一次修改q向量相同,说明进入了往返循环
                if np.all(np.array(q_new) >= np.array(q)):
                    self.modify_Qj_method3(j, alpha, model=cdm)
                    break
                else:
                    break
            else:
                q_old = q
                continue
        return self.modify_q_m

    def modify_Q_method3(self, **kwargs):
        """
        对所有题目的q向量进行检验
        :param kwargs:  alpha显著性水平
        :return:  修改后的q向量，并更新对象的modify_q_m属性
        """
        alpha = kwargs['alpha'] if 'alpha' in kwargs else self.alpha
        mode = kwargs['mode'] if 'mode' in kwargs else self.mode
        if mode == 'no_loop':
            for j in range(self.prob_num):
                self.modify_Qj_method3(j, alpha)
        elif mode == 'loop':
            for j in range(self.prob_num):
                self.modify_Qj_method3_loop(j, alpha)
        else:
            raise ValueError("mode must be no_loop or loop")
        return self.modify_q_m


# 先编写一个检验一个知识点的方法
def is_refuse(n, p, r, alpha, prob=None):
    """
    :param n:  总数
    :param p:  成功概率
    :param r:  成功次数
    :param alpha:  显著性水平
    :return:  是否拒绝原假设
    """

    cumulate_p = 0
    # 如果p=1，r=n，那么不需要检验
    # 因为样本全部都成功，对于缺失情况来说，所有学生都不会做对，说明该知识点不会，说明假设q不考这个知识点是错误的
    if np.isclose(p, 1, atol=1e-9) and int(r) == int(n):
        if prob is None:
            return 1
        elif prob:
            return 1
    # p<1正常检验
    else:
        # 正态逼近逼近
        if n >= 1000:
            cumulate_p = approx_binomial_probability_corrected(n, p, r)
            if prob is None:
                return 1 if np.isclose(cumulate_p, alpha, atol=1e-9) or cumulate_p <= alpha else 0
            elif prob:
                return cumulate_p
        else:
            for i in range(int(r), int(n + 1)):  # 注意Python的range不包括结束值，因此需要加1
                cumulate_p += math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
            if prob is None:
                return 1 if np.isclose(cumulate_p, alpha, atol=1e-9) or cumulate_p <= alpha else 0
            elif prob:
                return cumulate_p


# 根据输入的q向量筛选样本
def T_sample(beta, q, k, r):
    """ 筛选掌握模式beta中第k个知识点为0，其他知识点分量≥q的学生，返回这些学生的作答情况r
    :param beta: dina模型对学生的掌握模式估计 example: beta = [[1,1,1],[0,1,1],[0,1,0]]
    :param q: “指定题目”的q向量 example: q = [0,1,1]
    :param k:  指定题目的知识点 example: k = 0
    :param r:  所有学生的对“指定题目”的作答情况 example: R = [0
                                                           1
                                                           1]
    :return:  返回指定筛选的学生的掌握模式，和作答第j道题目情况 example: [[1,1,0,1]]

    example:
    beta = [[1,1,1],[0,1,1],[0,1,0]]
    q = [0,1,1]
    k = 0
    r = [1,1,0]
    T_sample(beta, q, k, r)
    return [[1,1,1],[0,1,1]],[1,1] # 返回了两个学生的掌握模式和作答情况
    """
    # if isinstance(k,int or float):
    # 合并学生掌握模式和作答情况
    beta_r = np.concatenate((beta, r.reshape(-1, 1)), axis=1)  # debug 需要两个array都是有行列！！！
    # 获取所有学生的掌握模式分量大于等于q分量的学生掌握迷失
    studs = beta_r[np.all(beta_r[:, 0:-1] >= q, axis=1), :]  # axis=1表示列，3列压缩成1列
    studs_q = studs[studs[:, k] == 0, :]  # 获取studs中原假设第k个知识点为0的学生
    return studs_q[:, 0:-1], studs_q[:, -1]
    # elif isinstance(k,list):  # k =[0,1]
    #     beta_r = np.concatenate((beta, r.reshape(-1, 1)), axis=1)  # debug 需要两个array都是有行列！！！
    #     studs = beta_r[np.all(beta_r[:, 0:-1] >= q, axis=1), :]  # axis=1表示列，3列压缩成1列
    #     studs_copy = studs.copy()
    #     # 将k中指定的知识点的掌握模式分量为0的学生筛选出来，如果k=[0,1]，则筛选出来的学生是第0和第1个知识点都不会的学生
    #     for i in k:
    #         studs_copy = studs_copy[studs_copy[:, i] == 0, :]  # 获取studs中原假设第k个知识点为0的学生
    #     studs_q = studs_copy
    #     return studs_q[:,0:-1], studs_q[:, -1]
    # else:
    #     raise ValueError("k should be int or list")


def delaTorre(skills, items, students, R, wrong_Q):
    # states_samples_20 = states_samples[np.all(states_samples>=np.array([1,1,1,0,0]),axis=1)]
    ht = Hypothetical_skill(wrong_Q, R, students, items, skills, alpha=0.01)
    delta = Delta(wrong_Q, R, students, items, skills, epsilon=0.05)
    gamma = Gamma(wrong_Q, R, students, items, skills, threshold_g=0.2, threshold_s=0.2, threshold_es=0.2)
    modify_q_m1 = ht.modify_Q(mode='loop', alpha=0.01)
    modify_q_m2 = delta.modify_Q()
    modify_q_m3 = gamma.modify_Q()
    return modify_q_m1, modify_q_m2, modify_q_m3
    # print('原始Q',Q[0,:])
    # print('ht修改',modify_q_m1[0,:])
    # print('delta修改',modify_q_m2[0,:])
    # print('gamma修改',modify_q_m3[0,:])
    # return modify_q_m1[0,:],modify_q_m2[0,:],modify_q_m3[0,:]


if __name__ == '__main__':
    items = 40  # 题目数量
    skills = 5  # 知识点数量
    prob = [0.1, 0.3, 0.4, 0.3, 0.1]  # 题目知识点分布
    # items = 24
    # skills = 3
    # prob = [0.3,0.5,0.3]
    # items = 24
    # skills = 3
    # items = 40
    # skills = 5
    # prob = [0.3, 0.5, 0.3, 0.1, 0.1]

    students = 300  # 学生数量

    Qwrong_rate = [0.1, 0.1]
    Rwrong_rate = [0.2, 0.2]
    # Q = generate_Q(items, skills, probs=prob)  # 生成Q矩阵
    # # Q = np.load('../../data/Q.npy')
    # # wrong_Q = np.load('../../data/wrong_Q.npy')
    # wrong_Q = generate_wrong_Q(Q, wrong_rate=[0.1, 0.1])['Q_wrong']  # 生成错误率的Q矩阵
    # # np.save('../../data/wrong_Q.npy', wrong_Q)
    # states = np.concatenate((np.zeros((1, skills)), attribute_pattern(skills)))  # 所有学生掌握模式的可能情况
    # # states_samples = state_sample(states, num=students, method="uniform_mode")  # 从掌握模式中抽样
    # states_samples = state_sample(states, num=students, method="normal", mu_skills=2, sigma_skills=1)
    # R = np.apply_along_axis(state_answer, axis=1, arr=states_samples, Q=Q)  # 根据掌握模式生成作答情况
    # R = generate_wrong_R(R, wrong_rate=[0.2, 0.2])['R_wrong']  # 设置题目质量,高质量应该gs更小，低质量应该gs更大
    # np.save('../../data/R.npy', R)
    # R = np.load('../../data/R.npy')
    # ============================  手动修改Q:缺失情况
    # j = 23  # 指定题目修改q向量
    # wrong_Q = Q.copy()  # 生成错误率的Q矩阵
    # wrong_Q[j, 0] = 1  # 生成缺失1个知识点的Q向量
    # wrong_Q[j ,1] = 1
    # wrong_Q[j, 2] = 0
    # =============================   手动修改Q:冗余情况
    # j = 0
    # wrong_Q = Q.copy()
    # wrong_Q[j, 0] = 0  # 生成冗余1个知识点的Q向量
    # wrong_Q[j, 1] = 1
    #
    # j = 5
    # wrong_Q = Q.copy()
    # wrong_Q[j, 1] = 0
    # wrong_Q[j, 2] = 1
    #
    # # =============================  手动修改Q:缺失和冗余情况
    # j = 5
    # wrong_Q = Q.copy()
    # wrong_Q[j, 0] = 0
    # wrong_Q[j, 1] = 1
    # wrong_Q[j, 2] = 1
    # 检验第j题的第0个知识点是否缺失

    #
    # amr, pmr, tpr, fpr = [], [], [], []
    # np.random.seed(0)
    # for i in tqdm(range(20)):
    #     Q = generate_Q(items, skills, probs=None)  # 生成Q矩阵
    #     # Q = np.load('../../data/Q.npy')
    #     # wrong_Q = np.load('../../data/wrong_Q.npy')
    #     wrong_Q = generate_wrong_Q(Q, Qwrong_rate)['Q_wrong']  # 生成错误率的Q矩阵
    #     # np.save('../../data/wrong_Q.npy', wrong_Q)
    #     states = np.concatenate((np.zeros((1, skills)), attribute_pattern(skills)))  # 所有学生掌握模式的可能情况
    #     # states_samples = state_sample(states, num=students, method="uniform_mode")  # 从掌握模式中抽样
    #     states_samples = state_sample(states, num=students, method="normal", mu_skills=3, sigma_skills=1.5)
    #     R = np.apply_along_axis(state_answer, axis=1, arr=states_samples, Q=Q)  # 根据掌握模式生成作答情况
    #     # R = np.apply_along_axis(state_answer_gs, axis=1, arr=states_samples, Q=Q,g=0.2,s=0.2)  # 根据掌握模式生成作答情况
    #     R = generate_wrong_R(R, Rwrong_rate)['R_wrong']  # 设置题目质量,高质量应该gs更小，低质量应该gs更大
    #
    #     ht = Hypothetical_skill(wrong_Q, R, students, items, skills, alpha=0.01)
    #     delta = Delta(wrong_Q, R, students, items, skills, epsilon=0.05)
    #     gamma = Gamma(wrong_Q, R, students, items, skills, threshold_g=0.2, threshold_s=0.2, threshold_es=0.2)
    #     modify_q_m1 = ht.modify_Q()
    #     modify_q_m2 = delta.modify_Q()
    #     modify_q_m3 = gamma.modify_Q()
    #     pmr.append([PMR(Q,wrong_Q),PMR(Q, modify_q_m1), PMR(Q, modify_q_m2), PMR(Q, modify_q_m3)])
    #     amr.append([AMR(Q,wrong_Q),AMR(Q, modify_q_m1), AMR(Q, modify_q_m2), AMR(Q, modify_q_m3)])
    #     tpr.append([TPR(Q, wrong_Q, modify_q_m1), TPR(Q, wrong_Q, modify_q_m2), TPR(Q, wrong_Q, modify_q_m3)])
    #     fpr.append([FPR(Q, wrong_Q, modify_q_m1), FPR(Q, wrong_Q, modify_q_m2), FPR(Q, wrong_Q, modify_q_m3)])
    #
    # print("参数为：")
    # print("学生数量：", students)
    # print("知识点数量：", skills)
    # print("题目数量：", items)
    # print("知识点分布：", prob)
    # print("Q矩阵错误率：", Qwrong_rate)
    # print("R矩阵错误率：", Rwrong_rate)
    # print('修改Q矩阵估计的PMR参数：', np.mean(pmr, axis=0))
    # print('修改Q矩阵估计的AMR参数：', np.mean(amr, axis=0))
    # print('修改Q矩阵估计的TPR参数：', np.mean(tpr, axis=0))
    # print('修改Q矩阵估计的FPR参数：', np.mean(fpr, axis=0))
    # ===========================================================================================================

    # q_modify = ht.modify_Q(j, alpha=0.01)
    # print('真实q向量', Q[j, :])
    # print('错误q向量', wrong_Q[j, :])
    # print('修改后q向量', q_modify[j, :])

    # wrong_Q[j, 2] = 0
    # 计算剔除第j题后的Q矩阵和作答矩阵R，即Q_tilde和answer_tilde，使用DINA模型进行训练
    # Q_tilde = np.delete(Q, j, axis=0)
    # R_tilde = np.delete(R, j, axis=1)
    # dina模型参数估计
    # cdm = DINA(R_tilde, Q_tilde, 1000, 24-1, 3, skip_value=-1)
    # cdm.train(epoch=5, epsilon=1e-3)
    # g, s = cdm.guess, cdm.slip  # 题目的guess、slip参数
    # beta = cdm.all_states[cdm.theta, :]  # 学生掌握模式估计

    # q真 = [1,1,1]
    # q错 = [0,0,1]
    # 在已知j题的q有缺失问题，但实际上是不知道哪个知识点缺失的
    # 对于每一个知识点，进行检验
    # r = R[:, j]  # 所有学生对j题的作答情况
    # q = wrong_Q[j,:]  # 最j题的q向量
    # k = [1]  # 缺失的知识点
    # studs_select,r_select = T_sample(beta, q, k, r)  # 筛选出所有掌握模式所有分量都大于等于q对应分量，且第k个知识点为0的学生

    # 假设检验
    # n = studs_select.shape[0]  # 掌握模式q0的所有学生数
    # r = n-sum(r_select)
    # p = 1-np.mean(s)
    # studs_select中的学生不应该做对第j题，所以做对数量应该很小

    # boo = is_refuse(n, 1-np.mean(s), n-np.sum(r_select), 0.01,prob=True)
    # print(boo)
    # binom.ppf(0.95, n, p)
    # print(r)

    # =============================== de la Torre2008  ==========================================================
    # ===============  数据初始化 ===================
    import random

    Q = pd.read_excel('../../data/delaTorre/q_m.xlsx', header=None).values  # 真实Q矩阵
    students = 5000  # 学生数量
    items = Q.shape[0]  # 题目数量
    skills = Q.shape[1]  # 知识点数量
    gs = [0.2, 0.2]

    states = np.concatenate((np.zeros((1, skills)), attribute_pattern(skills)))  # 所有学生掌握模式的可能情况
    states_samples = state_sample(states, num=students, method="uniform_mode", mu_skills=3, sigma_skills=1)  # 从掌握模式中抽样
    np.random.seed(0)
    R = np.apply_along_axis(state_answer_gs, axis=1, arr=states_samples, Q=Q, g=gs[0], s=gs[1])  # 根据掌握模式生成作答情况
    wrong_Q = Q.copy()

    # debug
    # wrong_Q[0, 0] = 0
    # wrong_Q[0, 1] = 1
    # ht = Hypothetical_skill(wrong_Q, R, students, items, skills, alpha=0.01)
    # delta = Delta(wrong_Q, R, students, items, skills, epsilon=0.05)
    # gamma = Gamma(wrong_Q, R, students, items, skills, threshold_g=0.2, threshold_s=0.2, threshold_es=0.2)
    # modify_q_m1 = ht.modify_Q(mode='loop', alpha=0.01)
    # modify_q_m2 = delta.modify_Q()
    # modify_q_m3 = gamma.modify_Q()
    # print('原始Q', Q[0, :])
    # print('wrong_Q', wrong_Q[0, :])
    # print('ht修改', modify_q_m1[0, :])
    # print('delta修改', modify_q_m2[0, :])
    # print('gamma修改', modify_q_m3[0, :])

    print("======================= item1 ========================")
    wrong_Q[0, 0] = 0
    wrong_Q[0, 1] = 1
    item1 = delaTorre(skills, items, students, R, wrong_Q)
    print('原始Q', Q[0, :])
    print('wrong_Q', wrong_Q[0, :])
    print('ht修改', item1[0][0, :])
    print('delta修改', item1[1][0, :])
    print('gamma修改', item1[2][0, :])

    print("======================= item2 ========================")
    wrong_Q = Q.copy()
    wrong_Q[0, 1] = 1
    item2 = delaTorre(skills, items, students, R, wrong_Q)
    print('原始Q', Q[0, :])
    print('wrong_Q', wrong_Q[0, :])
    print('ht修改', item2[0][0, :])
    print('delta修改', item2[1][0, :])
    print('gamma修改', item2[2][0, :])

    # item3
    print("======================= item3 ========================")
    wrong_Q = Q.copy()
    wrong_Q[10, 0] = 0
    wrong_Q[10, 2] = 1
    item3 = delaTorre(skills, items, students, R, wrong_Q)
    print('原始Q', Q[10, :])
    print('wrong_Q', wrong_Q[10, :])
    print('ht修改', item3[0][10, :])
    print('delta修改', item3[1][10, :])
    print('gamma修改', item3[2][10, :])

    # item4
    print("======================= item4 ========================")
    wrong_Q = Q.copy()
    wrong_Q[10, 0] = 0
    item4 = delaTorre(skills, items, students, R, wrong_Q)
    print('原始Q', Q[10, :])
    print('wrong_Q', wrong_Q[10, :])
    print('ht修改', item4[0][10, :])
    print('delta修改', item4[1][10, :])
    print('gamma修改', item4[2][10, :])

    # item5
    print("======================= item5 ========================")
    wrong_Q = Q.copy()
    wrong_Q[10, 2] = 1
    item5 = delaTorre(skills, items, students, R, wrong_Q)
    print('原始Q', Q[10, :])
    print('wrong_Q', wrong_Q[10, :])
    print('ht修改', item5[0][10, :])
    print('delta修改', item5[1][10, :])
    print('gamma修改', item5[2][10, :])

    # item6
    print("======================= item6 ========================")
    wrong_Q = Q.copy()
    wrong_Q[20, 0] = 0
    item6 = delaTorre(skills, items, students, R, wrong_Q)
    print('原始Q', Q[20, :])
    print('wrong_Q', wrong_Q[20, :])
    print('ht修改', item6[0][20, :])
    print('delta修改', item6[1][20, :])
    print('gamma修改', item6[2][20, :])

    # item7
    print("======================= item7 ========================")
    wrong_Q = Q.copy()
    wrong_Q[20, 0] = 0
    wrong_Q[20, 1] = 0
    item7 = delaTorre(skills, items, students, R, wrong_Q)
    print('原始Q', Q[20, :])
    print('wrong_Q', wrong_Q[20, :])
    print('ht修改', item7[0][20, :])
    print('delta修改', item7[1][20, :])
    print('gamma修改', item7[2][20, :])

    # item8
    print("======================= item8 ========================")
    wrong_Q = Q.copy()
    wrong_Q[20, 0] = 0
    wrong_Q[20, 3] = 1
    item8 = delaTorre(skills, items, students, R, wrong_Q)
    print('原始Q', Q[20, :])
    print('wrong_Q', wrong_Q[20, :])
    print('ht修改', item8[0][20, :])
    print('delta修改', item8[1][20, :])
    print('gamma修改', item8[2][20, :])

    # item9
    print("======================= item9 ========================")
    wrong_Q = Q.copy()
    wrong_Q[20, 0] = 0
    wrong_Q[20, 1] = 0
    wrong_Q[20, 3] = 1
    item9 = delaTorre(skills, items, students, R, wrong_Q)
    print('原始Q', Q[20, :])
    print('wrong_Q', wrong_Q[20, :])
    print('ht修改', item9[0][20, :])
    print('delta修改', item9[1][20, :])
    print('gamma修改', item9[2][20, :])

    # item10
    print("======================= item10 ========================")
    wrong_Q = Q.copy()
    wrong_Q[20, 0] = 0
    wrong_Q[20, 1] = 0
    wrong_Q[20, 3] = 1
    wrong_Q[20, 4] = 1
    item10 = delaTorre(skills, items, students, R, wrong_Q)
    print('原始Q', Q[20, :])
    print('wrong_Q', wrong_Q[20, :])
    print('ht修改', item10[0][20, :])
    print('delta修改', item10[1][20, :])
    print('gamma修改', item10[2][20, :])

    # item11
    print("======================= item11 ========================")
    wrong_Q = Q.copy()
    wrong_Q[0, 1] = 1
    wrong_Q[10, 1] = 0
    wrong_Q[10, 2] = 0
    wrong_Q[20, 0] = 0
    wrong_Q[20, 3] = 1
    item11 = delaTorre(skills, items, students, R, wrong_Q)
    print('原始Q_0', Q[0, :])
    print('wrong_Q', wrong_Q[0, :])
    print('ht修改', item11[0][0, :])
    print('delta修改', item11[1][0, :])
    print('gamma修改', item11[2][0, :])

    print('原始Q_10', Q[10, :])
    print('wrong_Q', wrong_Q[10, :])
    print('ht修改', item11[0][10, :])
    print('delta修改', item11[1][10, :])
    print('gamma修改', item11[2][10, :])

    print('原始Q_20', Q[20, :])
    print('wrong_Q', wrong_Q[20, :])
    print('ht修改', item11[0][20, :])
    print('delta修改', item11[1][20, :])
    print('gamma修改', item11[2][20, :])
