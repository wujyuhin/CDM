# 先导入库
import math
import logging

logging.getLogger().setLevel(logging.INFO)
import pandas as pd
import numpy as np
# from EduCDM import EMDINA as DINA
from code_functions.EduCDM import EMDINA as DINA

class Gamma():
    def __init__(self, q_m, R, stu_num, prob_num, know_num,threshold_g, threshold_s, threshold_es):
        self.q_m = q_m
        self.R = R
        self.stu_num = stu_num
        self.prob_num = prob_num
        self.know_num = know_num
        self.modify_q_m = q_m.copy()
        self.threshold_g = threshold_g
        self.threshold_s = threshold_s
        self.threshold_es = threshold_es
        # 训练cdm模型，根据阈值挑选参数g和s较大的题目
        # 不同的实例参数会有不同的cmd、g_big、s_big
        self.cdm,self.g_big,self.s_big = self.pick(R, q_m, stu_num, prob_num, know_num, threshold_g, threshold_s)

    # ================================   第2步：挑选参数g和s较大的项目（不依赖于实例状态，只依赖传入参数）  ==================
    def pick(cls, R, q_m, stu_num, prob_num, know_num, threshold_g, threshold_s):
        """ 挑选参数g和s较大的题目列表
        :param R:  作答矩阵
        :param q_m:  Q矩阵
        :param stu_num:  学生数
        :param prob_num:  题目数
        :param know_num:  知识点数
        :param threshold_g:  g的临界值
        :param threshold_s:  s的临界值
        :return:  超过临界值的g和s的题目索引，例如g_big=[1,2,3,4,5,6,7,8],s_big=[8,9,10]
        """
        # 训练cdm
        cdm = DINA(R, q_m, stu_num, prob_num, know_num, skip_value=-1)
        cdm.train(epoch=2, epsilon=1e-3)
        # 挑选g、s较大的题目
        g = pd.Series(cdm.guess)
        g_big = g[g > threshold_g].index  # 取出对应g值的题目索引
        s = pd.Series(cdm.slip)
        s_big = s[s > threshold_s].index  # 取出对应s值的题目索引
        return cdm, g_big, s_big

    # ================================  第3步 对任一属性k划分掌握与未掌握的人群  ========================================
    def divide(self, knowledge):
        """ 划分掌握与未掌握的人群列表
        :param knowledge:  知识点索引
        :return:  掌握知识点的人和未掌握知识点的人
        例如：understand=[1,2,3,4,5,6,7,8,9,10],disunderstand=[11,12,13,14,15,16,17,18,19,20]
        """
        understand = []
        disunderstand = []
        for id in range(0, self.cdm.R.shape[0]):
            state_id = self.cdm.theta[id]
            state = self.cdm.all_states[state_id, :]
            if state[knowledge] == 0:
                disunderstand.append(id)
            elif state[knowledge] == 1:
                understand.append(id)
        return understand, disunderstand

    # ========================   第4步，计算题目item的ES 计算掌握和未掌握组在项目j上的差异性大小  ============================
    def cal_average(self, understand, item):
        """计算平均数
        understand: 掌握人的id
        item: 第j个项目(题目)
        Returns: 正确率
        """
        score_understand = [self.R[stu, item] for stu in understand if self.R[stu, item] >= 0]
        return sum(score_understand) / len(score_understand)

    def cal_s(self, understand, disunderstand, item):
        """ 计算样本合并标准差
        understand : 掌握人的id
        disunderstand : 掌握人的id
        j : 第j个项目/题目

        Returns 合并标准差
        """
        score1 = [self.R[stu, item] for stu in understand if self.R[stu, item] >= 0]
        score2 = [self.R[stu, item] for stu in disunderstand if self.R[stu, item] >= 0]
        s1 = math.sqrt(sum((score1 - sum(score1) / len(score1)) ** 2) / (len(score1) - 1))
        s2 = math.sqrt(sum((score2 - sum(score2) / len(score2)) ** 2) / (len(score2) - 1))
        s = math.sqrt(((len(score1) - 1) * (s1 ** 2) + (len(score2) - 1) * (s2 ** 2)) / (len(score1) + len(score2) - 2))
        return s

    def ES(self, understand, disunderstand, item):
        es = (self.cal_average(understand, item) - self.cal_average(disunderstand, item)) / self.cal_s(understand,
                                                                                                       disunderstand,
                                                                                                       item)
        return es

    # =============================== 第5步：修改q矩阵  =========================================
    def Gmodify(self,knowledge):
        """ 实例中对偏大的g的题目对应的知识点knowledge 进行修改（1->0）
        :param knowledge:  知识点索引
        :return: 修改后的q矩阵
        """
        # 1.先判断有无偏大的g(g>0.2)
        # 2.若有，则遍历所有的g
        # 3.每个g都可计算es
        # 4.根据es(es<0.2)大小判断是否要修改q矩阵
        # q_copy = self.q_m.copy()
        understand, disunderstand = self.divide(knowledge)  # 划分人群
        if len(self.g_big) == 0:
            return self.modify_q_m
        else:
            for j in self.g_big:
                es = self.ES(understand, disunderstand, j)
                if es < self.threshold_es:
                    if self.modify_q_m[j, knowledge] == 1:
                        self.modify_q_m[j, knowledge] = 0
            return self.modify_q_m

    def Smodify(self, knowledge):
        """

        :param knowledge:
        :return:
        """
        # 1.先判断有无偏大的s(s>0.2)
        # 2.若有，则遍历所有的s
        # 3.每个s都可计算es
        # 4.根据es(es>=0.2)大小判断是否要修改q矩阵
        # q_copy = self.q_m.copy()
        understand, disunderstand = self.divide(knowledge)
        if len(self.s_big) == 0:
            return self.modify_q_m
        else:
            for j in self.s_big:
                es = self.ES(understand, disunderstand, j)
                if es >= self.threshold_es:
                    if self.modify_q_m[j, knowledge] == 0:
                        self.modify_q_m[j, knowledge] = 1
            return self.modify_q_m

    def modify_Q(self):
        for k in range(self.know_num):
            self.modify_q_m = self.Gmodify(k)
            self.modify_q_m = self.Smodify(k)
        return self.modify_q_m


#
#
#
# # ================================   第2步：挑选参数g和s较大的项目  ========================================
# def pick(threshold_g, threshold_s):
#     # threshold = 0.2  # 界定临界值为0.2
#     g = pd.Series(cdm.guess)
#     g_big = g[g > threshold_g].index  # 取出对应g值的题目索引
#     s = pd.Series(cdm.slip)
#     s_big = s[s > threshold_s].index  # 取出对应s值的题目索引
#     return g_big, s_big
#
#
# # ================================  第3步 对任一属性k划分掌握与未掌握的划分  ========================================
# def divide(cdm, knowledge):
#     """ 划分掌握与未掌握
#     Parameters
#     ----------
#     cdm : object 训练好的cdm模型
#     knowledge:int 第k个知识点，知识点索引
#     Return:list1掌握第k个知识点的学生id，list2未掌握知识点k的学生id
#     -------
#     """
#     # 将掌握的和未掌握的分为两个列表
#     understand = []
#     disunderstand = []
#     # 如果学生答题状况对属性k是0则分到disunderstang里  反之则放到understand
#     for id in range(0, cdm.R.shape[0]):
#         state_id = cdm.theta[id]  # 指状态id，注：状态分别是0，1-..255
#         state = cdm.all_states[state_id, :]  # 上述学生的状态state的具体表现00110（学生对知识点的掌握情况）
#         if state[knowledge] == 0:  # 具体第j个是指点掌握情况
#             disunderstand.append(id)
#         elif state[knowledge] == 1:
#             understand.append(id)
#     return understand, disunderstand
#
#
# # ========================   第4步，计算掌握和未掌握组在项目j上的差异性大小  ========================================
#
# def cal_average(understand, item):
#     """计算平均数
#     understand: 掌握人的id
#     item: 第j个项目(题目)
#
#     Returns: 正确率
#     """
#     score_understand = [R[stu, item] for stu in understand if R[stu, item] >= 0]
#     return sum(score_understand) / len(score_understand)
#
#
# def cal_s(understand, disunderstand, j):
#     """ 计算样本合并标准差
#     understand : 掌握人的id
#     disunderstand : 掌握人的id
#     j : 第j个项目/题目
#
#     Returns 合并标准差
#     """
#     score1 = [R[stu, j] for stu in understand if R[stu, j] >= 0]  # 先是掌握的学生作答情况
#     score2 = [R[stu, j] for stu in disunderstand if R[stu, j] >= 0]  # 后是未掌握的学生作答情况
#     s1 = math.sqrt(sum((score1 - sum(score1) / len(score1)) ** 2) / (len(score1) - 1))  # 样本标准差
#     s2 = math.sqrt(sum((score2 - sum(score2) / len(score2)) ** 2) / (len(score2) - 1))
#     # s1 = st.stdev(score1)  # st.stdev 标准差，但是需要list类型数据
#     # s2 = st.stdev(score2)
#     s = math.sqrt(((len(score1) - 1) * (s1 ** 2) + (len(score2) - 1) * (s2 ** 2)) / (len(score1) + len(score2) - 2))
#     return s
#
#
# # def ES(understand, disunderstand, g_big):
# #     if len(g_big)>=1:
# #         for item in g_big:
# #             es = (cal_average(understand, item) - cal_average(disunderstand, item)) / cal_s(understand, disunderstand, item)
# #         return es
# #     elif len(g_big)==0:
# #         return -1
#
# def ES(understand, disunderstand, item):
#     es = (cal_average(understand, item) - cal_average(disunderstand, item)) / cal_s(understand, disunderstand, item)
#     return es
#
#
# # =============================== 第五步：修改q矩阵  =========================================
# def Gmodify(q, g_big, threshold_es, knowledge, understand, disunderstand):
#     """ 修改q矩阵
#     Parameters
#     ----------
#     g_big
#     threshold_g: g的临界值
#     es
#     threshold_es
#     Returns : cdm 模型类
#     -------
#
#     """
#     # 1.先判断有无偏大的g(g>0.2)
#     # 2.若有，则遍历所有的g
#     # 3.每个g都可计算es
#     # 4.根据es(es<0.2)大小判断是否要修改q矩阵
#     q_copy = q.copy()
#     if len(g_big) == 0:
#         return q_copy
#     else:
#         for j in g_big:
#             es = ES(understand, disunderstand, j)
#             if es < threshold_es:
#                 if q_copy[j, knowledge] == 1:
#                     q_copy[j, knowledge] = 0
#         return q_copy
#
#
# def Smodify(q, s_big, threshold_es, knowledge, understand, disunderstand):
#     """ 修改q矩阵
#     Parameters
#     ----------
#     s_big
#     threshold_s: g的临界值
#     es
#     threshold_es
#     Returns : cdm 模型类
#     -------
#
#     """
#     # 1.先判断有无偏大的s(s>0.2)
#     # 2.若有，则遍历所有的s
#     # 3.每个s都可计算es
#     # 4.根据es(es>=0.2)大小判断是否要修改q矩阵
#     q_copy = q.copy()
#     if len(s_big) == 0:
#         return q_copy
#     else:
#         for j in s_big:  # 每个偏大s的id就是对应的题目id
#             es = ES(understand, disunderstand, j)  # 计算对应题目的ES(j)
#             if es >= threshold_es:
#                 if q_copy[j, knowledge] == 0:
#                     q_copy[j, knowledge] = 1
#         return q_copy


if __name__ == '__main__':
    # ============================  数据部分  ==========================================================
    q_m = np.loadtxt("../../data/math2015/simulation/simulation_q.csv", dtype=int, delimiter=',')  # Q 矩阵
    R = np.array(pd.read_csv("../../data/math2015/simulation/simulation_data.csv", index_col=0))  # R 作答矩阵
    prob_num, know_num = q_m.shape[0], q_m.shape[1]  # 题目数、知识点数
    stu_num = R.shape[0]    # 学生数
    threahold_g = 0.2       # guess界定临界值为0.2
    threahold_s = 0.2       # slip界定临界值为0.2
    threahold_es = 0.2      # ES界定临界值为0.2
    # =================================================  gamma  =================================
    logging.getLogger().setLevel(logging.INFO)
    gamma_model = Gamma(q_m, R, stu_num, prob_num, know_num, threahold_g, threahold_s, threahold_es)
    gamma_model.modify_Q()
