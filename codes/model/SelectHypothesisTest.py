# hypothesis_skill是以选择特定样本，对单个知识点进行假设检验
# 孙假设检验是以题目考察模式即q向量未单位的假设检验
from scipy.stats import norm
from codes.data_generate.generate import generate_Q, generate_wrong_R, generate_wrong_Q
import numpy as np
from codes.data_generate.generate import attribute_pattern, state_sample, state_answer, state_answer_gs
import math
from codes.EduCDM import EMDINA as DINA


def approx_binomial_probability_corrected(n, p, k):
    """ 使用正态分布的连续性校正来近似二项分布的概率"""

    mean = n * p
    std_dev = (n * p * (1 - p)) ** 0.5

    # 进行连续性校正，k减去0.5
    k_corrected = k - 0.5

    # 标准化k到Z分数，使用校正后的k值
    z = (k_corrected - mean) / std_dev

    # 使用正态分布的CDF计算P(X <= k_corrected)，然后转换为P(X > k)
    prob_greater_than_k_corrected = 1 - norm.cdf(z)

    return prob_greater_than_k_corrected


class SelectHypothesisTest():
    def __init__(self, q_m, R, stu_num, prob_num, know_num, alpha=0.01, mode='loop'):
        self.q_m = q_m
        self.R = R
        self.stu_num = stu_num
        self.prob_num = prob_num
        self.know_num = know_num
        self.modify_q_m = q_m.copy()
        self.mode = mode
        self.alpha = alpha

    def modify_Qjk(self, cdm_deletej, j, k, alpha):
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

    def modify_Qj(self, j, alpha, **kwargs):
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
            prob0, prob1 = self.modify_Qjk(cdm, j, k, alpha)
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

    def modify_Qj_loop(self, j, alpha):
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
            q_new = self.modify_Qj(j, alpha, model=cdm)
            if np.all(q == q_new):
                break
            elif np.all(q_old == q_new):
                # 修改后的q向量与上一次修改的q向量不同,但与上上一次修改q向量相同,说明进入了往返循环
                if np.all(np.array(q_new) >= np.array(q)):
                    self.modify_Qj(j, alpha, model=cdm)
                    break
                else:
                    break
            else:
                q_old = q
                continue
        return self.modify_q_m

    def modify_Q(self, **kwargs):
        """
        对所有题目的q向量进行检验
        :param kwargs:  alpha显著性水平
        :return:  修改后的q向量，并更新对象的modify_q_m属性
        """
        alpha = kwargs['alpha'] if 'alpha' in kwargs else self.alpha
        mode = kwargs['mode'] if 'mode' in kwargs else self.mode
        if mode == 'no_loop':
            for j in range(self.prob_num):
                self.modify_Qj(j, alpha)
        elif mode == 'loop':
            for j in range(self.prob_num):
                self.modify_Qj_loop(j, alpha)
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


# def delaTorre(skills, items, students, R, wrong_Q):
#     # states_samples_20 = states_samples[np.all(states_samples>=np.array([1,1,1,0,0]),axis=1)]
#     ht = Hypothetical_skill(wrong_Q, R, students, items, skills, alpha=0.01)
#     delta = Delta(wrong_Q, R, students, items, skills, epsilon=0.05)
#     gamma = Gamma(wrong_Q, R, students, items, skills, threshold_g=0.2, threshold_s=0.2, threshold_es=0.2)
#     modify_q_m1 = ht.modify_Q(mode='loop', alpha=0.01)
#     modify_q_m2 = delta.modify_Q()
#     modify_q_m3 = gamma.modify_Q()
#     return modify_q_m1, modify_q_m2, modify_q_m3
# print('原始Q',Q[0,:])
# print('ht修改',modify_q_m1[0,:])
# print('delta修改',modify_q_m2[0,:])
# print('gamma修改',modify_q_m3[0,:])
# return modify_q_m1[0,:],modify_q_m2[0,:],modify_q_m3[0,:]


if __name__ == '__main__':
    items = 40  # 题目数量
    skills = 5  # 知识点数量
    prob = [0.1, 0.3, 0.4, 0.3, 0.1]  # 题目知识点分布
    students = 300  # 学生数量
    Q_wrong_rate = [0.1, 0.1]
    R_wrong_rate = [0.2, 0.2]
    Q = generate_Q(items, skills, probs=prob)  # 生成Q矩阵
    wrong_Q = generate_wrong_Q(Q, wrong_rate=Q_wrong_rate)['Q_wrong']  # 生成错误率的Q矩阵
    states = np.concatenate((np.zeros((1, skills)), attribute_pattern(skills)))  # 所有学生掌握模式的可能情况
    states_samples = state_sample(states, num=students, method="frequency", mu_skills=2, sigma_skills=1)
    R = np.apply_along_axis(state_answer, axis=1, arr=states_samples, Q=Q)  # 根据掌握模式生成作答情况
    R = generate_wrong_R(R, wrong_rate=R_wrong_rate)['R_wrong']  # 设置题目质量,高质量应该gs更小，低质量应该gs更大
    ht = SelectHypothesisTest(wrong_Q, R, students, items, skills, alpha=0.01)
    ht.modify_Q(mode='loop')  # 对同一题多次迭代，直到q向量不再变化
