import numpy as np
import itertools
from scipy.stats import norm
import numpy as np

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


def generate_Q(items, skills, probs: list = None):
    """
    生成Q矩阵的方法，skills表示属性个数，items表示题目个数，probs表示生成考察模式的概率
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
        KS = attribute_pattern(skills)  # 生成所有可能的考察模式
        Q = np.zeros((items, skills))  # 初始化Q矩阵，生成 items 行 K列的全0矩阵
        while np.any(np.sum(Q, axis=0) == 0):
            Q = KS[np.random.choice(np.arange(1, KS.shape[0]), items, replace=True), :]
    else:
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
                Q_mode.append(KS[np.random.choice(np.arange(KS.shape[0]), int(probs[k - 1]), replace=True)])  # 生成考察k个属性的模式
            Q = np.concatenate(Q_mode, axis=0)
    # print(f"题目数量为{items}，属性数量为{skills},考察模式的概率为{probs}")
    # print("生成的Q矩阵为：")
    # for i in range(Q.shape[1]):
    #     print(f"考察{i+1}个知识点的有{sum(np.sum(Q, axis=1) == i+1)}个")
    # print("考察2个知识点的有", sum(np.sum(Q, axis=1) == 2), "个")
    # print("考察3个知识点的有", sum(np.sum(Q, axis=1) == 3), "个")
    # print(Q)
    return Q


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
    wrong_set_10 = None if sum_errors_1 == 0 else wrong_set_10  # 如果sum_errors_1=0，则wrong_set_10=None
    wrong_set_01 = None if sum_errors_0 == 0 else wrong_set_01  # 如果sum_errors_0=0，则wrong_set_01=None
    # print("生成设定错误率的Q矩阵完成...\n", "--------------------------------------")
    return {'Q': Q, 'Q_wrong': Q_wrong, 'is_wrong': is_wrong, 'wrong_set_01': wrong_set_01,
            'wrong_set_10': wrong_set_10}


# 发现之前正态生成掌握模式有漏洞，现在重新实现
# def state_sample_new(states, num, method: str = None, mu_skills: int = None, sigma_skills: int = None,
#                      set_skills: int = None):
#     """
#     从掌握模式中抽样
#     :param states: ndarray,掌握模式 如[[0,1],[1,0],[1,1]]
#     :param num:  抽样数量
#     :param method:  抽样方法,uniform:均匀分布,normal:正态分布,assign:指定抽样
#     :param mu_skills:  抽样均值，填写掌握是指点的个数，会自动压缩成标准正态分布的分位数,type:int,example:1,2,3...
#     :param sigma_skills:  抽样方差
#     :param set_skills:  指定抽样的掌握模式中的知识点个数
#     :return:  ndarray 返回抽样结果
#
#     example1: 2个知识点，抽样3个
#     states = np.array([[0,1],[1,0],[1,1]])
#     num = 3
#     method = "uniform"
#     result:
#     [[1 0]
#      [1 0]
#      [1 0]]
#
#     example2: 2个知识点，抽样3个,抽样方法为正态分布
#     states = np.array([[0,1],[1,0],[1,1]])
#     num = 3
#     method = "normal"
#     mu_skills = 0
#     sigma_skills = 1
#     result:
#     [[1 0]
#      [1 1]
#      [1 0]]
#     """
#     if method is None:
#         return states[np.random.choice(states.shape[0], num, replace=True)]
#     elif method == "uniform":
#         return states[np.random.choice(states.shape[0], num, replace=True)]
#     elif method == "normal":
#         # 将知识点id压缩成正态分布的分位数，如[1,2]压缩成[-0.43,0.43]
#         mode = norm.ppf(np.arange(0, states.shape[1] + 1) / (states.shape[1]))
#         sigma = 1 if sigma_skills is None else sigma_skills
#         # 知识点id映射到正态分布的分位数，以只掌握一个知识点的情况为均值，实际上是均值是mode[0]
#         mu = 0 if (mu_skills is None) or mu_skills == 0 else mode[mu_skills - 1]
#         rs = np.random.normal(mu, sigma, num)  # 生成正态分布的随机数
#         # 返回考察知识点数量列表，如[1,2,1]，考察1个知识点的模式有两个，考察2个知识点的模式有一个
#         skills_num = np.array([np.where(x < mode)[0][0] + 1 for x in rs])  # 查找随机数属于考察多少个知识点的掌握模式
#         skills_set = set(skills_num)  # 对每种掌握情况抽样
#         states_sample = np.array([]).reshape(0, states.shape[1])  # 用于存储抽样结果
#         for i in skills_set:
#             states_i = states[
#                 np.sum(states, axis=1) == i]  # 第i种模式数量，如Q = [[1,0],[0,1],[1,1]],i=1,则states_i=[[1,0],[0,1]]
#             N = len(np.where(skills_num == i)[0])  # 对第i中模式抽样的数量
#             states_sample = np.concatenate(
#                 [states_sample, states_i[np.random.choice(states_i.shape[0], N, replace=True)]], axis=0)  # 对第i中模式抽样
#         return states_sample
#     elif method == "assign":
#         if set_skills is None:
#             raise ValueError("set_skills is None")
#         states_i = states[np.sum(states, axis=1) == set_skills]
#         return states_i[np.random.choice(states_i.shape[0], num, replace=True)]
#     else:
#         raise ValueError("method should be 'uniform' or 'normal' or 'assign'")


def expand_to_center(skills_num:int, specified_value:int):
    """
    将指定的值扩展到中间
    :param skills_num: 知识点个数 1,2,3,4...
    :param specified_value: 指定考察的知识点个数 1,2,3,4...
    :return: 返回扩展到中间的长度

    example:
    expand_to_center(4,1)
    return 7
    """
    natrue_list = list(range(skills_num+1))  # 生成自然数列表 [0,1,2,3,4]
    index = natrue_list.index(specified_value)  # 指定值的索引
    l1 = len(natrue_list[:index]) # 左边的长度
    l2 = len(natrue_list[index+1:]) # 右边的长度
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
    :param method:  抽样方法,uniform_mode:均匀分布,uniform_skill:根据掌握知识点的模式，对每种模式均匀抽样,normal:正态分布
    :param mu_skills:  抽样均值，填写掌握是指点的个数，会自动压缩成标准正态分布的分位数,type:int,example:1,2,3...
    :param sigma_skills:  抽样方差
    :param set_skills:  指定抽样的掌握模式中的知识点个数
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
    if method is None:
        return states[np.random.choice(states.shape[0], num, replace=True)]
    elif method == "uniform_mode":
        # 根据掌握模式均匀抽样
        return states[np.random.choice(states.shape[0], num, replace=True)]
    elif method == "uniform_skill":
        # 根据掌握知识点的模式，对知识点数量，进行均匀抽样
        skills_lists = np.random.randint(0,states.shape[1]+1, num)  # 共6个skills,若skills_list=[0,1,2,3]表示抽样只抽掌握了0,1,2,3个知识点的模式
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
        if mu_skills<0 and isinstance(mu_skills,int):
            raise ValueError("mu_skills should be int and greater than 0")
        else:
            mu_skills = 1 if mu_skills is None else mu_skills
        length = expand_to_center(skills_n, mu_skills)  # [0,1,2,3,4]将指定的知识点1作为均值时，扩展序列的长度[0,0,0,1,2,3,4]为7
        # 生成指定方差为2的正态分布的分位数
        mode = np.concatenate((norm.ppf(np.arange(1,length)/length),1e10),axis=None)
        if mu_skills <= skills_n/2:
            # [0,0]+[0,1,2,3,4]
            mode_to_skill = np.array(list(np.zeros((1, len(mode) - skills_n - 1))[0]) + list(range(skills_n + 1)))
        else:
            # [0,1,2,3,4]+[0,0]
            mode_to_skill = np.array(list(range(skills_n + 1)) + list(np.zeros((1, len(mode) - skills_n - 1))[0]))
        # 加上无穷大，这样每块面积都对应一个分位数

        sigma = 1 if sigma_skills is None else sigma_skills  # 指定正态分布的方差，默认为1

        # rs = np.random.normal(0, sigma, num)  # 生成正态分布的随机数
        #
        # # 查找rs中随机数属于 考察多少个知识点的掌握模式，返回列表如skills_num = [1,2,1]，考察1个知识点的模式有两个，考察2个知识点的模式有一个
        # skills_num = np.array([mode_to_skill[np.where(x <= mode)[0][0]] for x in rs])
        # skills_set = set(skills_num)  # 考察知识点数量的种类
        # states_sample = np.array([]).reshape(0, states.shape[1])  # 用于存储抽样结果
        # for i in skills_set:
        #     if int(i) == 0:
        #         # 第i中模式数量，若i=0,Q = [[1,0],[0,1],[1,1]],则states_i=[[0,0]]
        #         states_i = np.array([[0] * skills_n])
        #     else:
        #         # 第i种模式数量，如Q = [[1,0],[0,1],[1,1]],i=1,则states_i=[[1,0],[0,1]]
        #         states_i = states[np.sum(states, axis=1) == i]
        #     N = len(np.where(skills_num == i)[0])  # 对第i中模式抽样的数量
        #     states_sample = np.concatenate(
        #         [states_sample, states_i[np.random.choice(states_i.shape[0], N, replace=True)]], axis=0)  # 对第i中模式抽样

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
    else:
        raise ValueError("method should be 'uniform' or 'normal' or 'assign'")


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


# 对作答矩阵进行修正
def generate_wrong_R(R, wrong_rate):
    result = generate_wrong_Q(R, wrong_rate)
    return {'R': R, 'R_wrong': result['Q_wrong'], 'is_wrong': result['is_wrong'],
            'wrong_set_01': result['wrong_set_01'], 'wrong_set_10': result['wrong_set_10']}


if __name__ == '__main__':
    # 生成Q矩阵
    np.random.seed(0)
    skills = 6
    items = 15
    students = 100
    probs = [0.5, 0.4, 0.1, 0, 0, 0]
    wrong = [0.2, 0.2]
    Q = generate_Q(items, skills, probs)
    # 运行sim_wrong_q_rate函数
    result = generate_wrong_Q(Q, wrong)
    # print(result['Q_wrong'])
    # print(result['is_wrong_10'])
    # print(result['wrong_set_01'])

    # cdm = DINA(self.R, modify_q_m, self.stu_num, self.prob_num, self.know_num, skip_value=-1)
    # answer = np.apply_along_axis(state_answer, 1, A, A)  # 生成每种掌握模式下的答案 2^k-1 * 2^k-1
    # 应该加一行掌握模式全为0的答案
    # 生成掌握模式，掌握模式应该比Q矩阵多一行全0的模式
    # 正态分布抽样
    states = np.concatenate((np.zeros((1,skills)),attribute_pattern(skills)))
    states_samples = state_sample(states, num=students, method="normal", mu_skills=5,sigma_skills=0.7)  # 从掌握模式中抽样
    print("===== 正态分布抽样 =====")
    for i in range(states.shape[1]):
        print(f"正态掌握{i}个知识点的有{sum(np.sum(states_samples, axis=1) == i)}个")
    # 指定抽样
    states_samples2 = state_sample(states, num=students, method="assign",set_skills=2)  # 从掌握模式中抽样
    print("===== 指定抽样 =====")
    for i in range(states.shape[1]):
        print(f"掌握{i}个知识点的有{sum(np.sum(states_samples2, axis=1) == i)}个")
    # 按全模式进行均匀分布抽样
    states_samples3 = state_sample(states, num=students, method="uniform_mode")  # 从掌握模式中抽样
    print("===== 全模式均匀分布抽样 =====")
    for i in range(states.shape[1]):
        print(f"掌握{i}个知识点的有{sum(np.sum(states_samples3, axis=1) == i)}个")
    # 按掌握知识点进行均匀分布抽样
    states_samples4 = state_sample(states, num=students, method="uniform_skill")  # 从掌握模式中抽样
    print("===== 知识点均匀分布抽样 =====")
    for i in range(states.shape[1]):
        print(f"掌握{i}个知识点的有{sum(np.sum(states_samples4, axis=1) == i)}个")
    # 根据掌握模式、Q矩阵生成答案
    answer = np.apply_along_axis(state_answer, axis=1, arr=states_samples, Q=Q)  # 把arr中的每种模式都回答Q矩阵题目
    generate_wrong_R(answer, 0.1)
