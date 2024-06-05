import random

import numpy as np
import itertools
from scipy.stats import norm
from tqdm import tqdm
import time
import pickle

"""
想要生成不同类型的Q矩阵
1.最简单的是生成随机的Q矩阵，即Q矩阵中的每个元素都是随机生成的
2.生成的Q矩阵中只考察单个属性的考察模式
3.生成的Q矩阵中不包含单位矩阵，即不考察单个属性的考察模式
4.生成的Q矩阵中，至少一个，最多考察两个属性的考察模式
以此类推
# ================================= 生成Q矩阵规则  =======================================
当确定了生成的Q矩阵考察的知识点数量K
对一个属性、两个属性、三个属性、...、K个属性的掌握q向量按一定规则生成
如对于K=3，生成的Q矩阵中的q向量为
考察一个属性的q向量：[1,0,0],[0,1,0],[0,0,1]
考察两个属性的q向量：[1,1,0],[1,0,1],[0,1,1]
考察三个属性的q向量：[1,1,1]
按概率生成上述q向量，如考察一个属性的q向量的概率为0.3，考察两个属性的q向量的概率为0.5，考察三个属性的q向量的概率为0.2
"""


# def initial_all_knowledge_state(know_num):
#     state_num = 2 ** know_num
#     all_states = np.zeros((state_num, know_num))
#     for i in range(state_num):
#         k, quotient, residue = 1, i // 2, i % 2
#         while True:
#             all_states[i, know_num - k] = residue
#             if quotient <= 0:
#                 break
#             quotient, residue = quotient // 2, quotient % 2
#             k += 1
#     return all_states
#
# a = initial_all_knowledge_state(3)

# 生成所有可能的掌握模式
def attribute_pattern(skills):
    """
    基于k个属性生成2^k-1种掌握模式，返回除第一行外的所有模式（即去除全0模式）
    :param skills: 属性数量
    :return:  2^k种掌握模式
    例如：skills=3 返回：
    [[0 0 1]
     [0 1 0]
     [0 1 1]
     [1 0 0]
     [1 0 1]
     [1 1 0]
     [1 1 1]]
    """
    powerset = np.array(list(map(list, itertools.product([0, 1], repeat=skills))))
    return powerset[1:]  # 去除全0模式


# 指定只考察n个属性的考察模式
def attribute_pattern_n(skills, n):
    """
    指定只考察n个属性的考察模式
    :param skills: 属性数量
    :param n: 考察的属性数量
    :return:  2^k种掌握模式
    例如：skills=3 n=2 返回：
    [[0 1 1]
     [1 0 1]
     [1 1 0]]
    """
    powerset = np.array(list(map(list, itertools.product([0, 1], repeat=skills))))
    index = np.where(np.sum(powerset, axis=1) == n)
    return powerset[index]


def generate_Q(items, skills, probs: [list or str or float] = None):
    """
    生成Q矩阵的方法，skills表示属性个数，items表示题目个数，probs表示生成考察模式的概率
    :param items: 题目个数
    :param skills: 属性个数
    :param probs: 指定生成考察模式的概率，probs的和应该小于1.
    1.默认为None,所有考察模式以均匀分布生成
        如3个知识点的掌握模式有2^3-1=7种,，生成的Q矩阵中的每个元素都是随机生成的，probs=[1/7,1/7,1/7,1/7,1/7,1/7,1/7]
    2.如果probs为一个浮点数，则生成每种考察模式的概率为[probs,probs,probs,...]
        如3个知识点,probs=0.3，则自动转换成 probs=[0.3,0.3,0.3],考察一个属性的概率为0.3，考察两个属性的概率为0.3，考察三个属性的概率为0.3
    3.如果probs为一个列表，则生成每种考察模式的概率为probs
        如3个知识点,,probs=[0.3,0.5,0.2]，则考察一个属性的概率为0.3，考察两个属性的概率为0.5，考察三个属性的概率为0.2
    4.如果probs为'single'，则返回所有可能的考察模式，即KS
        如3个知识点的掌握模式有2^3-1=7种,则返回所有的7种掌握模式
    5.如果probs为'frequency'，则返回所有掌握模式中掌握知识点的个数的频率，
        如3个知识点的掌握模式有2^3-1=7种,掌握一个知识点的有3种，掌握两个知识点的有3种，掌握三个知识点的有1种
    example: items=10, skills=3. probs=[0.3,0.5,0.2]，则考察一个属性的概率为0.3，考察两个属性的概率为0.5，考察三个属性的概率为0.2
    result example:
    [[0 1 0]
     [1 0 0]
     [1 0 0]
     [0 1 0]
     [0 1 0]
     [0 1 1]
     [0 1 1]
     [1 0 1]
     [0 1 1]
     [1 1 1]]
    """
    KS = attribute_pattern(skills)  # 生成所有可能的考察模式
    # 若probs为None，则生成随机抽取可能的考察模式组成Q矩阵
    if probs is None:
        Q = np.zeros((items, skills))  # 初始化Q矩阵，生成 items 行 K列的全0矩阵
        while np.any(np.sum(Q, axis=0) == 0):
            Q = KS[np.random.choice(np.arange(1, KS.shape[0]), items, replace=True), :]
        return Q
    # 若probs==single，则返回所有可能的考察模式，即KS
    elif isinstance(probs, str) and probs == 'single':
        return KS
    # 若probs为一个浮点数，则生成每种考察模式的概率为[probs,probs,probs,...]，生成Q矩阵
    # 若probs为一个列表，则生成每种考察模式的概率为probs，生成Q矩阵
    elif isinstance(probs, (float, list)):
        probs = (np.array(probs) / np.sum(np.array(probs))) * items  # 将probs转换为数量
        probs = np.round(probs).astype(int)  # 将probs转换为整数
        # 如果probs的和大于items，则随机选一个有数的减到和为items
        if sum(probs) > items:
            while sum(probs) != items:
                index = np.random.choice(range(len(probs)))
                if probs[index] > 0:
                    probs[index] -= 1
        if sum(probs) < items:
            while sum(probs) != items:
                index = np.random.choice(range(len(probs)))
                probs[index] += 1
        Q = np.zeros((items, skills))  # 初始化Q矩阵，生成 items 行 K列的全0矩阵
        while np.any(np.sum(Q, axis=0) == 0):  # 当Q的任何一列存在0时进入循环，即Q中至少各个属性都有题目对应被考察到
            Q_mode = []
            for k in range(1, skills + 1):  # 遍历所有属性
                KS = attribute_pattern_n(skills, k)  # 例如：skills=3 k=2 返回：[[0 1 1] [1 0 1] [1 1 0]]
                Q_mode.append(
                    KS[np.random.choice(np.arange(KS.shape[0]), int(probs[k - 1]), replace=True)])  # 生成考察k个属性的模式
            Q = np.concatenate(Q_mode, axis=0)
        return Q
    elif isinstance(probs, str) and probs == 'frequency':
        KS_sample = np.array([]).reshape(0, skills)
        for i in range(1, skills + 1):
            # print(f"生成考察{i}个知识点的模式...")
            KS_i = KS[np.where(np.sum(KS, axis=1) == i)]  # 生成考察i个知识点的模式
            freq = KS_i.shape[0] / KS.shape[0]  # 计算考察i个知识点的模式的频率
            # 保证要有至少一个考察i个知识点的模式
            # ================================== 需要抽样之后查看是否<1,如果<1则向上取整，如果>1则向下取整 ==============================
            sample_num = np.ceil(items * freq) if (freq < 1) and (freq - np.floor(freq) < 0.5) else np.floor(
                items * freq)
            index = np.random.choice(range(KS_i.shape[0]), int(sample_num), replace=True)
            KS_sample = np.concatenate((KS_sample, KS_i[index]), axis=0)
        # 如果抽样的数量小于items，则随机抽样补全
        if KS_sample.shape[0] < items:
            while KS_sample.shape[0] != items:
                KS_sample = np.concatenate((KS_sample, KS[np.random.choice(range(KS.shape[0]), 1, replace=True)]),
                                           axis=0)
        # 如果抽样的数量大于items，则随机抽样减少
        if KS_sample.shape[0] > items:
            while KS_sample.shape[0] != items:
                KS_sample = np.delete(KS_sample, np.random.choice(range(KS_sample.shape[0])), axis=0)
        return KS_sample
    else:
        raise ValueError("probs参数输入错误！")
    # print(f"题目数量为{items}，属性数量为{skills},考察模式的概率为{probs}")
    # print("生成的Q矩阵为：")
    # for i in range(Q.shape[1]):
    #     print(f"考察{i+1}个知识点的有{sum(np.sum(Q, axis=1) == i+1)}个")
    # print("考察2个知识点的有", sum(np.sum(Q, axis=1) == 2), "个")
    # print("考察3个知识点的有", sum(np.sum(Q, axis=1) == 3), "个")
    # print(Q)


def index_set(Q):
    """
    生成Q矩阵中0或者1的所有坐标(x,y)
    :param Q:  Q矩阵
    :return:  返回Q矩阵中0或者1的所有坐标(x,y)

    example:
    Q = np.array([[0, 1, 0, 0, 1],
                  [1, 0, 1, 0, 0],
                  [0, 0, 1, 1, 0]])
    result:
    {'set_0': [[0, 0], [0, 2], [0, 3], [1, 3], [2, 0], [2, 1], [2, 4]],
     'set_1': [[0, 1], [0, 4], [1, 0], [1, 2], [2, 2], [2, 3]]}
    """
    # print("生成Q矩阵中0或者1的所有坐标(x,y)...")
    set_0 = []
    set_1 = []
    for i in range(Q.shape[0]):
        for j in range(Q.shape[1]):
            if Q[i, j] == 0:
                set_0.append([i, j])
            else:
                set_1.append([i, j])
    # print("生成Q矩阵中0或者1的所有坐标(x,y)完成...\n", "--------------------------------------")
    return {'set_0': set_0, 'set_1': set_1}


def generate_wrong_Q(Q, wrong_rate: list or float):
    """
    生成设定错误率的Q矩阵
    :param Q:  Q矩阵
    :param wrong_rate:  错误率,可以是一个列表，也可以是一个浮点数,如果是一个列表，则第一个元素表示0的错误率，第二个元素表示1的错误率
                        wrong_rate=[0.2,0.2]表示0的错误率为0.2，1的错误率为0.2
                        wrong_rate=0.2表示0和1的错误率都为0.2
    :return:  返回生成设定错误率的Q矩阵

    example: Q矩阵为3个属性，3个题目，wrong=0.2
    Q = np.array([[0, 1, 0],
                  [1, 0, 1],
                  [0, 0, 1]])
    result:
    {'Q': array([[0, 1, 0],
                  [1, 0, 1],
                  [0, 0, 1]]),
     'Q_wrong': array([[1, 1, 0],
                       [1, 0, 1],
                       [0, 0, 1]]),
     'is_wrong_10': array([[True, False, False],
                           [False, False, False],
                           [False, False, False]]),
     'wrong_set_01': array([[0., 0.],
     'wrong_set_10': array(}
    """
    if isinstance(wrong_rate, (float, int)):
        wrong_rate = [wrong_rate, wrong_rate]
    q_index = index_set(Q)  # 生成Q矩阵中0或者1的所有坐标(x,y)
    set_0 = q_index['set_0']  # Q矩阵中0的坐标,[(x,y),...]
    set_1 = q_index['set_1']  # Q矩阵中1的坐标,[(x,y),...]

    Q_wrong = Q.copy()  # 用作修改的Q矩阵
    is_wrong = np.zeros_like(Q_wrong, dtype=bool)  # 创建一个与Q.wrong相同大小的全零布尔矩阵isWrong，用于记录哪些位置已经被修改过

    sum_errors_0 = int(np.floor(len(set_0) * wrong_rate[0]))  # 计算要设定错误的1元素个数
    sum_errors_1 = int(np.floor(len(set_1) * wrong_rate[1]))  # 计算要设定错误的0元素个数

    wrong_set_10 = np.zeros((sum_errors_1, 2))  # 用于记录修改1为0的坐标(x,y)
    wrong_set_01 = np.zeros((sum_errors_0, 2))  # 用于记录修改0为1的坐标(x,y)

    # ================= 对Q矩阵中的0元素进行修改 =================
    temp = 0  # 记录修改次数
    while temp < sum_errors_0:
        i, j = set_0[np.random.choice(range(len(set_0)))]  # 随机选择一个0元素的坐标
        while is_wrong[i, j]:
            # 规则 修改过的元素不能再次修改
            i, j = set_0[np.random.choice(range(len(set_0)))]
        Q_wrong[i, j] = 1  # 把Q矩阵中的0改为1
        is_wrong[i, j] = True  # 记录修改过的位置
        wrong_set_01[temp, :] = [i, j]  # 记录修改的坐标:(x,y)
        temp += 1

    # while循环太慢了，有无效的循环，每次选中后，都要判断是否已经修改过，可以直接生成不重复的索引
    # 每次选中ij，删除一对
    # temp = 0
    # while temp < sum_errors_0:
    #     i, j = set_0.pop(np.random.choice(range(len(set_0))))
    #     Q_wrong[i, j] = 1
    #     is_wrong[i, j] = True
    #     wrong_set_01[temp, :] = [i, j]
    #     temp += 1

    # ================= 对Q矩阵中的1元素进行修改 =================
    temp = 0  # 初始化一个临时变量temp，用于记录当前已经生成的错误数量
    # 生成is_wrong 中为
    while temp < sum_errors_1:
        i, j = set_1[np.random.choice(range(len(set_1)))]
        while is_wrong[i, j] or ((np.sum(Q_wrong[:, j]) < 2 or np.sum(Q_wrong[i, :]) < 2) and Q[i, j]):
            # 规则 修改过的元素不能再次修改 or 是惟一的1不能修改 or 0元素不能修改
            # is_wrong_10[i, j] == 1 表示该元素已经被修改过, 不用再修改，重新选择
            # 以下三个条件是为了保证生成的Q矩阵中的1元素不是唯一的一个1，如果是唯一一个1则不能修改，重新选择
            # 行只有一个1，列只有一个1，且该元素为1，不能修改
            # np.sum(Q_wrong[:, j]) < 2 表示该元素为所在列唯一的一个1
            # np.sum(Q_wrong[i, :]) < 2 表示该元素为所在行唯一的一个1
            # Q[i, j] 表示该元素为1
            i, j = set_0[np.random.choice(range(len(set_0)))]

        Q_wrong[i, j] = 1 - Q[i, j]  # 把Q矩阵中的1改为0
        is_wrong[i, j] = True
        wrong_set_10[temp, :] = [i, j]
        temp += 1

    wrong_set_10 = None if sum_errors_1 == 0 else wrong_set_10  # 如果sum_errors_1=0，则wrong_set_10=None
    wrong_set_01 = None if sum_errors_0 == 0 else wrong_set_01  # 如果sum_errors_0=0，则wrong_set_01=None
    # print("生成设定错误率的Q矩阵完成...\n", "--------------------------------------")
    return {'Q': Q, 'Q_wrong': Q_wrong, 'wrong_set_01': wrong_set_01,
            'wrong_set_10': wrong_set_10}


def expand_to_center(skills_num: int, specified_value: int):
    """
    将指定的值扩展到中间
    :param skills_num: 知识点个数 1,2,3,4...
    :param specified_value: 指定考察的知识点个数 1,2,3,4...
    :return: 返回扩展到中间的长度

    example:
    expand_to_center(4,1)
    return 7
    """
    natrue_list = list(range(skills_num + 1))  # 生成自然数列表 [0,1,2,3,4]
    index = natrue_list.index(specified_value)  # 指定值的索引
    l1 = len(natrue_list[:index])  # 左边的长度
    l2 = len(natrue_list[index + 1:])  # 右边的长度
    if l1 <= l2:
        return l2 * 2 + 1
    else:
        return l1 * 2 + 1


def state_sample(states, num, method: str = None, mu_skills: int = None, sigma_skills: int = None,
                 set_skills: int = None):
    """
    从掌握模式中抽样
    :param states: ndarray,掌握模式 如[[0,1],[1,0],[1,1]]
    :param num:  抽样数量
    :param method:  抽样方法,uniform_mode:均匀分布,uniform_skill:根据掌握知识点的模式，对每种模式均匀抽样,normal:正态分布,frequency:根据掌握知识点的模式，对每种模式按照频率抽样
    :param mu_skills:  抽样均值，填写掌握是指点的个数，会自动压缩成标准正态分布的分位数,type:int,example:1,2,3... method=normal时使用
    :param sigma_skills:  抽样方差 默认为1,method=normal时使用
    :param set_skills:  指定抽样的掌握模式中的知识点个数 method=uniform_skill时使用
    :return:  ndarray 返回抽样结果

    example1: 2个知识点，抽样3个
    states = np.array([[0, 1], [1, 0], [1, 1]])
    num = 3
    method = "uniform"
    result:
    [[1 0]
     [1 0]
     [1 0]]

    example2: 2个知识点，抽样3个,抽样方法为正态分布
    states = np.array([[0, 1], [1, 0], [1, 1]])
    num = 3
    method = "normal"
    mu_skills = 0
    sigma_skills = 1
    result:
    [[1 0]
     [1 1]
     [1 0]]
    """
    np.random.seed(0)
    if method is None:
        return states[np.random.choice(states.shape[0], num, replace=True)]
    elif method == "uniform_mode":
        # 根据掌握模式均匀抽样
        return states[np.random.choice(states.shape[0], num, replace=True)]
    elif method == "uniform_skill":
        # 根据掌握知识点的模式，对知识点数量，进行均匀抽样
        skills_lists = np.random.randint(0, states.shape[1] + 1,
                                         num)  # 共6个skills,若skills_list=[0,1,2,3]表示抽样只抽掌握了0,1,2,3个知识点的模式
        skills_list = list(set(skills_lists))  # 抽样的掌握知识点的模式
        states_sample = np.array([]).reshape(0, states.shape[1])
        for i in skills_list:
            N = len(np.where(skills_lists == i)[0])  # 对第i中模式抽样的数量
            states_i = states[np.sum(states, axis=1) == i]
            states_sample = np.concatenate(
                [states_sample, states_i[np.random.choice(states_i.shape[0], N, replace=True)]], axis=0)
        return states_sample
    elif method == "normal":
        # mu_skills=1,表示掌握1个知识点
        # 将指定的知识点掌握模式放到中间，例如4个知识点，指定均值为掌握1个知识点，则面积有7块，加上正无穷，正态分位点有7个
        # 这7个分为点对应的掌握知识点数量分别为[0,0,0,1,2,3,4]，可以看出均值是1
        skills_n = states.shape[1]  # 知识点数量
        # 根据均值和掌握的知识点类型，确定分位数
        if mu_skills < 0 and isinstance(mu_skills, int):
            raise ValueError("mu_skills should be int and greater than 0")
        else:
            mu_skills = 1 if mu_skills is None else mu_skills
        length = expand_to_center(skills_n, mu_skills)  # [0,1,2,3,4]将指定的知识点1作为均值时，扩展序列的长度[0,0,0,1,2,3,4]为7
        # 生成指定方差为2的正态分布的分位数
        mode = np.concatenate((norm.ppf(np.arange(1, length) / length), 1e10), axis=None)
        if mu_skills <= skills_n / 2:
            # [0,0]+[0,1,2,3,4]
            mode_to_skill = np.array(list(np.zeros((1, len(mode) - skills_n - 1))[0]) + list(range(skills_n + 1)))
        else:
            # [0,1,2,3,4]+[0,0]
            mode_to_skill = np.array(list(range(skills_n + 1)) + list(np.zeros((1, len(mode) - skills_n - 1))[0]))
        # 加上无穷大，这样每块面积都对应一个分位数
        sigma = 1 if sigma_skills is None else sigma_skills  # 指定正态分布的方差，默认为1
        # 应该截断抽样，[0,0,1,2,3]随机数应该只抽到后面四位，第一个的0不应该抽到,不然就不符合正态分布
        i = np.where(np.random.normal(0, sigma) <= mode)[0][0]
        temp = 0
        # 确定考察不同知识点的掌握模式分别有多少个,如抽2个掌握2个知识点的，抽4个掌握1个知识点的...
        skills_num = []
        while temp < num:
            if mu_skills <= skills_n / 2:
                # [0,0,0,1,2,3,4] 这种情况应该抽样忽略前两个0
                if i < len(mode_to_skill) - skills_n - 1:  # [0,0,0,1,2,3,4]排除除[0,1,2,3,4]外的前2个
                    i = np.where(np.random.normal(0, sigma) <= mode)[0][0]
                    continue
            else:
                # [0,1,2,3,4,0,0] 这种情况应该抽样忽略后两个0
                if i >= skills_n + 1:
                    i = np.where(np.random.normal(0, sigma) <= mode)[0][0]
                    continue
            skills_num.append(i)
            i = np.where(np.random.normal(0, sigma) <= mode)[0][0]
            temp += 1
        # 根据抽样的掌握模式数量，直接抽样
        skills_list = mode_to_skill[list(set(skills_num))]  # 共6个skills,若skills_list=[1,2,3]表示抽样只抽掌握了1,2,3个知识点的模式
        states_sample = np.array([]).reshape(0, states.shape[1])  # 用于存储抽样结果
        for j in skills_list:
            # 第i种模式数量，如Q = [[1,0],[0,1],[1,1]],i=1,则states_i=[[1,0],[0,1]]
            states_j = states[np.sum(states, axis=1) == j]
            N = len(np.where(mode_to_skill[skills_num] == j)[0])
            states_sample = np.concatenate(
                [states_sample, states_j[np.random.choice(states_j.shape[0], N, replace=True)]], axis=0)  # 对第i中模式抽样
        return states_sample
    elif method == "assign":
        if set_skills is None:
            raise ValueError("set_skills is None")
        states_i = states[np.sum(states, axis=1) == set_skills]
        return states_i[np.random.choice(states_i.shape[0], num, replace=True)]
    elif method == "frequency":
        KS_sample = np.array([]).reshape(0, states.shape[1])
        for i in range(0, states.shape[1] + 1):
            # print(f"生成考察{i}个知识点的模式...")
            KS_i = states[np.where(np.sum(states, axis=1) == i)]  # 生成考察i个知识点的模式
            freq = KS_i.shape[0] / states.shape[0]  # 计算考察i个知识点的模式的频率
            # 保证要有至少一个考察i个知识点的模式
            # ================================== 需要抽样之后查看是否<1,如果<1则向上取整，如果>1则向下取整 ==============================
            sample_num = np.ceil(num * freq) if (freq < 1) and (freq - np.floor(freq) < 0.5) else np.floor(num * freq)
            index = np.random.choice(range(KS_i.shape[0]), int(sample_num), replace=True)
            KS_sample = np.concatenate((KS_sample, KS_i[index]), axis=0)
        # 如果抽样的数量小于items，则随机抽样补全
        if KS_sample.shape[0] < num:
            while KS_sample.shape[0] != num:
                KS_sample = np.concatenate(
                    (KS_sample, KS_sample[np.random.choice(range(KS_sample.shape[0]), 1, replace=True)]), axis=0)
        # 如果抽样的数量大于items，则随机抽样减少
        if KS_sample.shape[0] > num:
            while KS_sample.shape[0] != num:
                KS_sample = np.delete(KS_sample, np.random.choice(range(KS_sample.shape[0])), axis=0)
        return KS_sample
    else:
        raise ValueError("method should be 'uniform_mode' or 'uniform_skill' or 'normal' or 'assign' or 'frequency'")


def state_answer(state, Q):
    """ 根据掌握模式与Q矩阵生成作答矩阵R
    输入掌握模式 1*skills 输出items道题目回答1*items
    :param state: 每一种掌握模式 1*skills [0,1,0,1]
    :param Q:  Q矩阵 items*skills
    :return: 返回items道题目回答1*items

    example:
    state = [0,1,0,1,1]
    Q = np.array([[0,1,0,0,1],
                  [1,0,1,0,0],
                  [0,0,1,1,0]])
    result:
    [1,0,0]
    """
    answers = []
    for item in range(Q.shape[0]):
        if sum(np.array(state) >= np.array(Q[item, :])) == len(state):
            answers.append(1)
        else:
            answers.append(0)
    return np.array(answers)


def state_answer_gs(state, Q, g, s):
    """
    根据掌握模式与Q矩阵生成作答矩阵R,和猜测参数g和失误参数s,根据如果state>=Q,则正确回答概率未 p = (1-s)**掌握 + s**不掌握,输出作答
    :param state:  每一种掌握模式 1*skills [0,1,0,1]
    :param Q: Q矩阵 items*skills
    :param g: 猜测参数
    :param s: 失误参数
    :return: 输出items道题目的作答结果
    """
    answers = []
    ran = []
    for item in range(Q.shape[0]):
        if sum(np.array(state) >= np.array(Q[item, :])) == len(state):
            r1 = np.random.rand()
            ran.append(r1)
            if r1 < 1 - s:
                answers.append(1)
            else:
                answers.append(0)
        else:
            r2 = np.random.rand()
            ran.append(r2)
            if r2 < g:
                answers.append(1)
            else:
                answers.append(0)
    return np.array(answers)


# 对作答矩阵进行修正
def generate_wrong_R(R, wrong_rate):
    """
    :param R:  作答矩阵
    :param wrong_rate: 错误率(g,s),可以是一个列表，也可以是一个浮点数,如果是一个列表，则第一个元素表示0的错误率(0->1)，第二个元素表示1的错误率(1->0)
    :return:
    """
    if isinstance(wrong_rate, (float, int)):
        wrong_rate = [wrong_rate, wrong_rate]
    q_index = index_set(R)  # 生成Q矩阵中0或者1的所有坐标(x,y)
    set_0 = q_index['set_0']  # Q矩阵中0的坐标,[(x,y),...]
    set_1 = q_index['set_1']  # Q矩阵中1的坐标,[(x,y),...]
    sum_errors_0 = int(np.floor(len(set_0) * wrong_rate[0]))  # 计算要设定错误的1元素个数
    sum_errors_1 = int(np.floor(len(set_1) * wrong_rate[1]))  # 计算要设定错误的0元素个数
    wrong_set_10 = np.zeros((sum_errors_1, 2))  # 用于记录修改1为0的坐标(x,y)
    wrong_set_01 = np.zeros((sum_errors_0, 2))  # 用于记录修改0为1的坐标(x,y)
    R_wrong = R.copy()  # 用作修改的R矩阵
    # is_wrong = np.zeros_like(R_wrong, dtype=bool)  # 创建一个与R_wrong相同大小的全零布尔矩阵isWrong，用于记录哪些位置已经被修改过
    # ================= 对R矩阵中的0元素进行修改 =================
    for index in range(sum_errors_0):
        i, j = set_0[np.random.choice(range(len(set_0)))]
        R_wrong[i, j] = 1
        # wrong_set_01[index, :] = [i, j]
    # ================= 对R矩阵中的1元素进行修改 =================
    for index in range(sum_errors_1):
        i, j = set_1[np.random.choice(range(len(set_1)))]
        R_wrong[i, j] = 1 - R[i, j]
        # wrong_set_10[index, :] = [i, j]
    # return {'R': R, 'R_wrong': R_wrong,
    #         'wrong_set_01': wrong_set_01, 'wrong_set_10': wrong_set_10}
    return {'R_wrong': R_wrong}


if __name__ == '__main__':
    # skills = 5
    # items = 40
    # Q = generate_Q(items, skills, probs='frequency')
    # 生成Q矩阵
    np.random.seed(0)
    skills = 5
    items = 40
    students = 300
    probs = [0.5, 0.3, 0.2, 0.1, 0]
    wrong = 0.05
    Q = generate_Q(items, skills, probs)
    # # Q = generate_Q(items, skills, probs='frequency')
    # result = generate_wrong_Q(Q, wrong)
    # # # 正态分布抽样
    states = np.concatenate((np.zeros((1, skills)), attribute_pattern(skills)))
    states_samples = state_sample(states, num=students, method="frequency")
    # states_samples = state_sample(states, num=students, method="normal", mu_skills=2,sigma_skills=0.7)  # 从掌握模式中抽样
    # print("===== 正态分布抽样 =====")
    # for i in range(states.shape[1]):
    #     print(f"正态掌握{i}个知识点的有{sum(np.sum(states_samples, axis=1) == i)}个")
    # 指定抽样
    # states_samples2 = state_sample(states, num=students, method="assign",set_skills=1)  # 从掌握模式中抽样
    # print("===== 指定抽样 =====")
    # for i in range(states.shape[1]):
    #     print(f"掌握{i}个知识点的有{sum(np.sum(states_samples2, axis=1) == i)}个")
    # # 按全模式进行均匀分布抽样
    # states_samples3 = state_sample(states, num=students, method="uniform_mode")  # 从掌握模式中抽样
    # print("===== 全模式均匀分布抽样 =====")
    # for i in range(states.shape[1]):
    #     print(f"掌握{i}个知识点的有{sum(np.sum(states_samples3, axis=1) == i)}个")
    # # 按掌握知识点进行均匀分布抽样
    # states_samples4 = state_sample(states, num=students, method="uniform_skill")  # 从掌握模式中抽样
    # print("===== 知识点均匀分布抽样 =====")
    # for i in range(states.shape[1]):
    #     print(f"掌握{i}个知识点的有{sum(np.sum(states_samples4, axis=1) == i)}个")
    # # 根据掌握模式、Q矩阵生成答案
    # answer = np.apply_along_axis(state_answer, axis=1, arr=states_samples, Q=Q)  # 把arr中的每种模式都回答Q矩阵题目
    # answer = generate_wrong_R(answer, 0.1)['R_wrong']

    # np.random.seed(0)
    # students = [300, 500, 1000]  # 生成学生数量
    # skills_items_probs = [[3, 24, [[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]]],
    #                       [4, 32, [[0.2, 0.6, 0.2, 0], [0, 0.2, 0.6, 0.2]]]]# 生成知识点数、题目数、知识点数量分布
    # Q_wrong_rate = [0.05, 0.1, 0.15]  # 生成Q矩阵错误率
    # qualities = [[0, 0], [0.1, 0.1], [0.2, 0.2]]  # 生成题目质量
    # sample_modes = ["uniform_mode", "normal"]
    # sample_modes_para = {"uniform_mode": None, "normal": [2, 1]}  # normal:均值为2，方差为1 assign:指定抽样只掌握2个知识点
    # t1 = time.time()
    # dataset = []
    # data = {}
    # index=0
    # for i in tqdm(range(100)):  # 每类数据生成1次
    #     for student in students:
    #         # t1 = time.time()
    #         for skills, items, probs in skills_items_probs:
    #             for prob in probs:
    #                 for wrong in Q_wrong_rate:
    #                     for quality in qualities:
    #                         for mode in sample_modes:
    #                             np.random.seed(0)
    #                             Q = generate_Q(items, skills, probs=prob)  # 生成Q矩阵
    #                             wrong_Q = generate_wrong_Q(Q, wrong)['Q_wrong']  # 生成错误率的Q矩阵
    #                             states = np.concatenate(
    #                                 (np.zeros((1, skills)), attribute_pattern(skills)))  # 所有学生掌握模式的可能情况
    #                             if mode == "normal":
    #                                 states_samples = state_sample(states, num=student, method=mode,
    #                                                               mu_skills=sample_modes_para[mode][0],
    #                                                               sigma_skills=sample_modes_para[mode][1])
    #                             elif mode == "assign":
    #                                 states_samples = state_sample(states, num=student, method=mode,
    #                                                               set_skills=sample_modes_para[mode])
    #                             else:
    #                                 states_samples = state_sample(states, num=student, method=mode)  # 从掌握模式中抽样
    #                             answer = np.apply_along_axis(state_answer, axis=1, arr=states_samples,
    #                                                          Q=Q)  # 根据掌握模式生成作答情况
    #                             answer = generate_wrong_R(answer, wrong_rate=quality)[
    #                                 'R_wrong']  # 设置题目质量,高质量应该gs更小，低质量应该gs更大
    #                             data[f"data{index}_{student}_{skills}_{items}_{prob}_{wrong}_{quality}_{mode}"] = {
    #                                 "Q": Q,
    #                                 "Q_wrong": wrong_Q,
    #                                 "states": states_samples,
    #                                 "answer": answer}
    #     index += 1
    #     dataset.append(data)
    #
    # t2 = time.time()
    # t = t2 - t1
    # print(f"生成数据集耗时{t / 60:.2f}分钟,合计{t / 3600:.2f}小时,共生成{len(dataset)}个数据集")
    # # 保存dataset
    # file_path = '../../data/dataset_error.pkl'
    # with open(file_path, "wb") as f:
    #     pickle.dump(dataset, f)

    # 读取
    # with open(file_path, "rb") as f:
    #     dataset_read = pickle.load(f)

    # 测试读取数据的速度
    # t1 = time.time()
    # np.random.seed(0)
    # skills = 3
    # items = 24
    # students = 1000
    # quality = [0.2, 0.2]
    # wrong = 0.1
    # Q = generate_Q(items, skills, probs=[0.3, 0.3, 0.4])
    # t2 = time.time()
    # result = generate_wrong_Q(Q, wrong)
    # t3 = time.time()
    # wrong_Q = result['Q_wrong']
    # states = np.concatenate((np.zeros((1, skills)), attribute_pattern(skills)))  # 所有学生掌握模式的可能情况
    # t4 = time.time()
    # states_samples = state_sample(states, num=students, method="uniform_mode")  # 从掌握模式中抽样
    # t5 = time.time()
    # answer = np.apply_along_axis(state_answer, axis=1, arr=states_samples, Q=Q)  # 根据掌握模式生成作答情况
    # t6 = time.time()
    # answer = generate_wrong_R(answer, wrong_rate=quality)['R_wrong']  # 设置题目质量,高质量应该gs更小，低质量应该gs更大
    # t7 = time.time()
    # print(f"生成Q矩阵耗时{t2 - t1}秒")
    # print(f"生成错误率Q矩阵耗时{t3 - t2}秒")
    # print(f"生成掌握模式耗时{t4 - t3}秒")
    # print(f"抽样耗时{t5 - t4}秒")
    # print(f"生成作答情况耗时{t6 - t5}秒")
    # print(f"生成错误率作答情况耗时{t7 - t6}秒")
    # print("总耗时", t7 - t1, "秒")
