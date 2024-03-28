# 先导入库
import math
import logging

logging.getLogger().setLevel(logging.INFO)
import pandas as pd
import numpy as np
import json
from EduCDM import EMDINA as DINA
import statistics as st




# ================================   第2步：挑选参数g和s较大的项目  ========================================
def pick(threshold_g, threshold_s):
    # threshold = 0.2  # 界定临界值为0.2
    g = pd.Series(cdm.guess)
    g_big = g[g > threshold_g].index  # 取出对应g值的题目索引
    s = pd.Series(cdm.slip)
    s_big = s[s > threshold_s].index  # 取出对应s值的题目索引
    return g_big, s_big







# ================================  第3步 对任一属性k划分掌握与未掌握的划分  ========================================
def divide(cdm, knowledge):
    """ 划分掌握与未掌握
    Parameters
    ----------
    cdm : object 训练好的cdm模型
    knowledge:int 第k个知识点，知识点索引
    Return:list1掌握第k个知识点的学生id，list2未掌握知识点k的学生id
    -------
    """
    # 将掌握的和未掌握的分为两个列表
    understand = []
    disunderstand = []
    # 如果学生答题状况对属性k是0则分到disunderstang里  反之则放到understand
    for id in range(0, cdm.R.shape[0]):
        state_id = cdm.theta[id]  # 指状态id，注：状态分别是0，1-..255
        state = cdm.all_states[state_id, :]  # 上述学生的状态state的具体表现00110（学生对知识点的掌握情况）
        if state[knowledge] == 0:  # 具体第j个是指点掌握情况
            disunderstand.append(id)
        elif state[knowledge] == 1:
            understand.append(id)
    return understand, disunderstand





# ========================   第4步，计算掌握和未掌握组在项目j上的差异性大小  ========================================

def cal_average(understand, item):
    """计算平均数
    understand: 掌握人的id
    item: 第j个项目(题目)

    Returns: 正确率
    """
    score_understand = [R[stu, item] for stu in understand if R[stu, item] >= 0]
    return sum(score_understand) / len(score_understand)


def cal_s(understand, disunderstand, j):
    """ 计算样本合并标准差
    understand : 掌握人的id
    disunderstand : 掌握人的id
    j : 第j个项目/题目

    Returns 合并标准差
    """
    score1 = [R[stu, j] for stu in understand if R[stu, j] >= 0]  # 先是掌握的学生作答情况
    score2 = [R[stu, j] for stu in disunderstand if R[stu, j] >= 0]  # 后是未掌握的学生作答情况
    s1 = math.sqrt(sum((score1-sum(score1)/len(score1)) ** 2)/(len(score1)-1))   # 样本标准差
    s2 = math.sqrt(sum((score2-sum(score2)/len(score2)) ** 2)/(len(score2)-1))
    # s1 = st.stdev(score1)  # st.stdev 标准差，但是需要list类型数据
    # s2 = st.stdev(score2)
    s = math.sqrt(((len(score1) - 1) * (s1 ** 2) + (len(score2) - 1) * (s2 ** 2)) / (len(score1) + len(score2) - 2))
    return s







# def ES(understand, disunderstand, g_big):
#     if len(g_big)>=1:
#         for item in g_big:
#             es = (cal_average(understand, item) - cal_average(disunderstand, item)) / cal_s(understand, disunderstand, item)
#         return es
#     elif len(g_big)==0:
#         return -1

def ES(understand, disunderstand,item):
    es = (cal_average(understand, item) - cal_average(disunderstand, item)) / cal_s(understand, disunderstand, item)
    return es




# =============================== 第五步：修改q矩阵  =========================================
def Gmodify(q, g_big,threshold_es,knowledge,understand,disunderstand):
    """ 修改q矩阵
    Parameters
    ----------
    g_big
    threshold_g: g的临界值
    es
    threshold_es
    Returns : cdm 模型类
    -------

    """
    # 1.先判断有无偏大的g(g>0.2)
    # 2.若有，则遍历所有的g
    # 3.每个g都可计算es
    # 4.根据es(es<0.2)大小判断是否要修改q矩阵
    q_copy = q.copy()
    if len(g_big)==0:
        return q_copy
    else:
        for j in g_big:
            es = ES(understand,disunderstand,j)
            if es < threshold_es:
                if q_copy[j, knowledge] == 1:
                    q_copy[j, knowledge] = 0
        return q_copy

def Smodify(q, s_big,threshold_es,knowledge,understand,disunderstand):
    """ 修改q矩阵
    Parameters
    ----------
    s_big
    threshold_s: g的临界值
    es
    threshold_es
    Returns : cdm 模型类
    -------

    """
    # 1.先判断有无偏大的s(s>0.2)
    # 2.若有，则遍历所有的s
    # 3.每个s都可计算es
    # 4.根据es(es>=0.2)大小判断是否要修改q矩阵
    q_copy = q.copy()
    if len(s_big)==0:
        return  q_copy
    else:
        for j in s_big:  # 每个偏大s的id就是对应的题目id
            es = ES(understand,disunderstand,j)  # 计算对应题目的ES(j)
            if es >= threshold_es:
                if q_copy[j, knowledge] == 0:
                    q_copy[j, knowledge] = 1
        return q_copy



# for j in s_big:
#     ES = (cal_average(understand, j) - cal_average(disunderstand, j)) / cal_s(understand, disunderstand, j)
#     if j > threshold and ES >= 0.2:
#         if cdm.q_m[j, k] == 0:
#             cdm.q_m[j, k] = 1

if __name__ == '__main__':
    # ============================ 数据部分（Q and R）  ==========================================================
    q_m = np.loadtxt("../../data/math2015/simulation/simulation_q.csv", dtype=int, delimiter=',')  # Q 矩阵
    R = np.array(pd.read_csv("../../data/math2015/simulation/simulation_data.csv", index_col=0))  # R 作答矩阵
    # q_m[11,0]=0
    # q_m[11,1]=1
    # q_m[11,2]=0
    q_new= q_m.copy()
    prob_num, jnow_num = q_m.shape[0], q_m.shape[1]  # 行数与列数=题目数、知识点数
    stu_num = R.shape[0]  # 学生数
    # ================================================= 训练cdm模型 =================================
    logging.getLogger().setLevel(logging.INFO)
    cdm = DINA(R, q_m, stu_num, prob_num, jnow_num, skip_value=-1)
    cdm.train(epoch=2, epsilon=1e-3)
    # ================================   第2步：挑选参数g和s较大的项目  ========================================
    g_big, s_big = pick(0.2, 0.2)
    # ================================  第3步 对任一属性k划分掌握与未掌握的划分  ========================================
    for k in range(q_m.shape[1]):
        understand, disunderstand = divide(cdm, knowledge=k)
    #     # ========================   第4步，计算掌握和未掌握组在项目j上的差异性大小  ========================================
    #     # 四五步合并
    #     # =============================== 第五步：修改q矩阵  =========================================
    #     # 以下两个函数只会执行一个，因为es满足了>=0.2，就不会满足<0.2，所以只执行其中一个
        q_new = Gmodify(q_new, g_big, 0.2,k,understand,disunderstand)  # 1 -> 0
        q_new = Smodify(q_new,s_big,0.2,k,understand,disunderstand)   # 0 -> 1
