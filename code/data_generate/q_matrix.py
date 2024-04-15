import numpy as np
import itertools
import numpy as np
import pandas as pd

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


def generate_Q(items, skills, probs: list = None):
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
        probs = np.array(probs) * items  # 将probs转换为数量
        probs[-1] = items - np.sum(probs[:-1])  # 最后一个属性的数量为总数减去前面属性的数量
        Q = np.zeros((items, skills))  # 初始化Q矩阵，生成 items 行 K列的全0矩阵
        while np.any(np.sum(Q, axis=0) == 0):  # 当Q的任何一列存在0时进入循环，即Q中至少各个属性都有题目对应被考察到
            Q = []
            for k in range(1, skills + 1):  # 遍历所有属性
                KS = attributepattern_n(skills, k)  # 例如：skills=3 k=2 返回：[[0 1 1] [1 0 1] [1 1 0]]
                Q.append(KS[np.random.choice(np.arange(KS.shape[0]), int(probs[k - 1]), replace=True)])
            Q = np.concatenate(Q, axis=0)
    print(f"题目数量为{items}，属性数量为{skills},考察模式的概率为{probs}")
    print("生成的Q矩阵为：")
    print(Q)
    return Q


import numpy as np
import random


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
    {'wrong_set_0': [[0, 0], [0, 2], [0, 3], [1, 3], [2, 0], [2, 1], [2, 4]],
     'wrong_set_1': [[0, 1], [0, 4], [1, 0], [1, 2], [2, 2], [2, 3]]}
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


def wrong_Q_rate(Q, wrong_rate:list or float):
    """
    生成设定错误率的Q矩阵
    :param Q:  Q矩阵
    :param wrong_rate:  错误率,可以是一个列表，也可以是一个浮点数,如果是一个列表，则第一个元素表示0的错误率，第二个元素表示1的错误率
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
     'is_wrong_10': array([[False, False, False],
                           [False, False, False],
                           [False, False, False]]),
     'wrong_set_01': array([[0., 0.],
                            [0., 1.],
                            [0., 2.],
                            [1., 0.],
                            [1., 2.]]),
     'wrong_set_10': array([[0., 0.],
                            [0., 1.],
                            [1., 0.],
                            [1., 2.]])}
    """
    if isinstance(wrong_rate, float):
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
            # 规则 修改过的元素不能再次修改 or 是惟一的1不能修改 or 1元素不能修改
            i, j = set_0[np.random.choice(range(len(set_0)))]
        Q_wrong[i, j] = 1  # 把Q矩阵中的0改为1
        is_wrong[i, j] = True  # 记录修改过的位置
        wrong_set_01[temp, :] = [i, j]  # 记录修改的坐标:(x,y)
        temp += 1

    # ================= 对Q矩阵中的1元素进行修改 =================
    temp = 0  # 初始化一个临时变量temp，用于记录当前已经生成的错误数量
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

    # print("生成设定错误率的Q矩阵完成...\n", "--------------------------------------")
    return {'Q': Q, 'Q_wrong': Q_wrong, 'is_wrong_10': is_wrong, 'wrong_set_01': wrong_set_01, 'wrong_set_10': wrong_set_10}


import numpy as np
from scipy.stats import norm
import random


def sim_ORP_GDINA(P, num, Q, model, distribute):
    # 该方法基于输入的参数生成一个model

    # R语言中的cat()输出在这里替换为print()，便于在Python中展示
    # print("模型生成中...")
    # print(f"被试属性分布模式： {distribute}")
    # print(f"模型： {model}")
    # print(f"属性个数: {Q.shape[1]}")
    # print(f"题目个数: {Q.shape[0]}")
    # print(f"题目和属性个数之比: {Q.shape[0] / Q.shape[1]}")
    # print(f"被试数量: {num}")
    # print(f"题目质量的参数,g和s从该范围取值: Iq = [{P[0]}, {P[1]}]")  # 应该是题目质量

    gs = np.zeros((Q.shape[0], 2))  # 生成题目数行，2列的全0矩阵，用于存储各题的guess和slip参数
    gs[:, 0] = np.random.uniform(P[0], P[1], Q.shape[0])  # 在P[0]-P[1]之间随机生成数字填充gs的第一列
    gs[:, 1] = np.random.uniform(P[0], P[1], Q.shape[0])  # 在P[0]-P[1]之间随机生成数字填充gs的第二列

    ORP_GDINA = None

    if distribute == "mvnorm.random" or distribute == "mvnorm":
        sigma = 0.5
        K = Q.shape[1]  # 属性个数
        cutoffs = norm.ppf(np.arange(1, K + 1) / (K + 1))  # 生成正态分布的分位数
        if distribute == "mvnorm.random":
            cutoffs = norm.ppf(np.random.uniform(1 / (K + 1), K / (K + 1), K))  # 生成随机的正态分布的分位数

        # print(f"sigma: {sigma}")
        # print(f"cutoffs: {cutoffs}")

        m = np.zeros(K)
        vcov = np.eye(K) * sigma
        ORP_GDINA = simGDINA(num, Q, gs_parm=gs, model=model, att_dist="mvnorm",
                             mvnorm_parm={"mean": m, "sigma": vcov, "cutoffs": cutoffs})

    elif distribute == "higher.order":
        theta = np.random.normal(size=num)
        lambda_ = pd.DataFrame({"a": np.random.uniform(1, 2, K),
                                "b": np.linspace(-1.5, 1.5, K)})
        ORP_GDINA = simGDINA(num, Q, gs_parm=gs, model=model, att_dist="higher.order",
                             higher_order_parm={"theta": theta, "lambda": lambda_})

    elif distribute == "uniform":
        ORP_GDINA = simGDINA(num, Q, gs_parm=gs, model=model)  # gs参数控制题目难度，即各题的guess和slip参数

    # print("模型生成完成\n", "--------------------------------------")
    return ORP_GDINA


def sim_data_one(K, J, N, distribute, P, model, Q_method):
    Q = generate_Q(K, J, Q_method)
    ORP_GDINA = sim_ORP_GDINA(P, N, Q, model, distribute)
    LCprob_parm = ORP_GDINA["LCprob_parm"].T  # 该变量是每行为一个题目，每列为一种掌握模式，指：对于每个题目，各种掌握模式答对的概率
    return {"Q": Q, "model": model, "LCprob_parm": LCprob_parm, "P": P, "ORP": ORP_GDINA}





def sim_data(data_size, K, P, distribute, model, N_range, JK_range, Q_method):
    # data_size
    # K表示题目考察的属性个数
    # P表示用于控制题目质量的参数
    # distribute表示被试的属性掌握模式，默认均匀分布
    # model模型，我们应该要用DINA
    # N表示被试数量
    # JK表示题目和属性数量之比
    # Q_method表示Q矩阵的生成方法，完全随机/存在单位矩阵

    KS = attributepattern(K)  # 生成所有类型的掌握模式比如k=2个属性，则生成[0 0,1 0,0 1,1 1]矩阵
    L = len(KS)  # 获取KS的长度,其实就是2^k
    Pattern = []
    data_set = pd.DataFrame()
    Q = None

    for data_num in range(1, data_size + 1):
        # print(f"--------------------- {data_num} ---------------------")

        N_cur = int(np.round(np.random.uniform(N_range[0], N_range[1])))  # runif生成随机数，并四舍五入到整数
        J_cur = int(np.round(np.random.uniform(K * JK_range[0], K * JK_range[1])))  # 生成随机整数

        data = sim_data_one(K, J_cur, N_cur, distribute, P, model, Q_method)
        data_set = pd.concat([data_set, data], axis=1)

    return data_set




if __name__ == '__main__':
    # 生成Q矩阵
    np.random.seed(0)
    skills = 3
    items = 10
    probs = [0.5, 0.4, 0.1]
    Q = generate_Q(items, skills, probs)
    # 运行sim_wrong_q_rate函数
    wrong = 0.2
    result = wrong_Q_rate(Q, wrong)
    print(result['Q_wrong'])

    sim_data_one(3, 10, 100, "uniform", [0.2, 0.8], "GDINA", [0.3, 0.5, 0.2])