import numpy as np
import itertools

"""
想要生成不同类型的Q矩阵
1.最简单的是生成随机的Q矩阵，即Q矩阵中的每个元素都是随机生成的
2.生成的Q矩阵中不包含单位矩阵，即不考察单个属性的考察模式
3.生成的Q矩阵中最多考察单个属性的考察模式
4.生成的Q矩阵中最多考察两个属性的考察模式
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


def attributepattern(skills):
    """
    基于k个属性生成2^k种掌握模式，返回除第一行外的所有模式（即去除全0模式）
    :param skills: 属性数量
    :return:  2^k种掌握模式
    例如：skills=3 返回：
    [[0 0 1]
     [0 1 0]
     [0 1 1]
     [1 0 0]
     [1 0 1]
     [1 1 0]]
    """
    powerset = np.array(list(map(list, itertools.product([0, 1], repeat=skills))))
    return powerset[1:]  # 去除全0模式


# 指定只考察n个属性的考察模式
def attributepattern_n(skills, n):
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


def sim_Q(items, skills, probs:list=None):
    """
    生成Q矩阵的方法，K表示属性个数，J表示题目个数，method表示生成方法
    :param items: 题目个数
    :param skills: 属性个数
    :param probs: 指定生成考察模式的概率，probs的长度为K-1，probs的和应该小于1.默认为None,所有考察模式均匀分布
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
    if probs is None:
        KS = attributepattern(skills)  # 生成所有可能的考察模式
        Q = np.zeros((items, skills))  # 初始化Q矩阵，生成 items 行 K列的全0矩阵
        while np.any(np.sum(Q, axis=0) == 0):
            Q = KS[np.random.choice(np.arange(1, KS.shape[0]), items, replace=True), :]
    else:
        probs = np.array(probs)*items  # 将probs转换为数量
        probs[-1] = items - np.sum(probs[:-1])  # 最后一个属性的数量为总数减去前面属性的数量
        Q = np.zeros((items, skills))  # 初始化Q矩阵，生成 items 行 K列的全0矩阵
        while np.any(np.sum(Q, axis=0) == 0):  # 当Q的任何一列存在0时进入循环，即Q中至少各个属性都有题目对应被考察到
            Q = []
            for k in range(1,skills+1):  # 遍历所有属性
                KS = attributepattern_n(skills,k) # 例如：skills=3 k=2 返回：[[0 1 1] [1 0 1] [1 1 0]]
                Q.append(KS[np.random.choice(np.arange(KS.shape[0]), int(probs[k-1]), replace=True)])
            Q = np.concatenate(Q,axis=0)
    print(f"题目数量为{items}，属性数量为{skills},考察模式的概率为{probs}")
    print("生成的Q矩阵为：")
    print(Q)
    return Q


# 生成Q矩阵
skills = 3
items = 10
probs=[0.5,0.4,0.1]
Q = sim_Q(items,skills,probs)

import numpy as np
import random


def sim_wrong_q_rate(Q, wrong):
    # print("生成设定错误率的Q矩阵中...")
    # print("错误率：", wrong)

    sum_errors = int(np.floor(Q.shape[0] * Q.shape[1] * wrong))  # 计算要设定错误的元素个数
    # print("错误个数：", sum_errors)

    Q_wrong = Q.copy()
    wrong_set = np.zeros((sum_errors, 2))  # 用于记录修改的坐标
    is_wrong = np.zeros_like(Q_wrong, dtype=bool)  # 创建一个与Q.wrong相同大小的全零布尔矩阵isWrong，用于记录哪些位置已经被修改过
    temp = 0  # 初始化一个临时变量temp，用于记录当前已经生成的错误数量

    while temp < sum_errors:
        i = round(random.uniform(1, Q.shape[0]))  # 随机生成i和j，表示要修改的元素在矩阵中的位置
        j = round(random.uniform(1, Q.shape[1]))  # 随机生成i和j，表示要修改的元素在矩阵中的位置

        while is_wrong[i - 1, j - 1] or \
                ((np.sum(Q_wrong[:, j - 1]) < 2 or np.sum(Q_wrong[i - 1, :]) < 2) and not (1 - Q[i - 1, j - 1])):
            # while用于判断刚选的要修改的元素的下标是否合规，若不合规进入循环重新选择，若合规不进入循环 进入while的条件即不合规的条件是:iswrong =
            # 1该元素已经被修改过，或者，该元素为1且该元素为所在行或所在列唯一的一个1，因为如果改行唯一一个1被修改，则表示该题未考察任何属性，若该列唯一一个1被修改1，则表示该属性未被考察
            i = round(random.uniform(1, Q.shape[0]))
            j = round(random.uniform(1, Q.shape[1]))

        Q_wrong[i - 1, j - 1] = 1 - Q[i - 1, j - 1]
        is_wrong[i - 1, j - 1] = True
        temp += 1
        wrong_set[temp - 1, :] = [i, j]
        # print("错误位置", temp, "下标:", i, j, "初始:", Q[i - 1, j - 1], "修改后:", Q_wrong[i - 1, j - 1])

    # print("错误Q矩阵生成完成...\n", "--------------------------------------")
    return {'Q': Q, 'Q_wrong': Q_wrong, 'is_wrong': is_wrong, 'wrong_set': wrong_set}

# 运行sim_wrong_q_rate函数
wrong = 0.2
result = sim_wrong_q_rate(Q, wrong)
print(result['Q'])