# ============================ 导入必要的包  ====================================================
import numpy as np
import json
import pandas as pd
import logging
import tqdm
from code_functions.model.delta import Delta
from code_functions.model.gamma import Gamma

logging.getLogger().setLevel(logging.INFO)
from EduCDM import EMDINA as DINA


# ============================  加减法数据  ====================================================
# 数据准备
q_m = np.loadtxt("../data/math2015/FrcSub/q_m.csv", dtype=int, delimiter=',')  # Q矩阵
prob_num, know_num = q_m.shape[0], q_m.shape[1]  # 题目数、属性数
R = np.array(np.loadtxt("../data/math2015/FrcSub/data.csv", dtype=int, delimiter=','))  #作答R矩阵
stu_num = R.shape[0]  # 学生数

# cdm
cdm = DINA(R, q_m, stu_num, prob_num, know_num, skip_value=-1)



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


# 在所有掌握模式中抽样，生成被试掌握模式真值
def simGDINA(N, Q, att_dist="uniform", mvnorm_parm=None, higher_order_parm=None):
    # N表示被试数量
    # Q表示Q矩阵,维度为J*K，K表示属性个数，J表示题目个数
    # att_dist(attribute distribution)表示被试的属性掌握模式，默认均匀分布
    # mvnorm_parm表示多变量正态分布的参数
    # higher_order_parm表示高阶属性的参数
    # 生成被试掌握模式真值
    att = np.zeros((N, Q.shape[1]))  # 生成N行，Q列的全0矩阵，用于存储被试的掌握模式
    if att_dist == "uniform":
        att = np.random.randint(0, 2, (N, Q.shape[1]))  # 生成N行，Q列的随机数矩阵，元素取值为0或1
    elif att_dist == "mvnorm":
        att = np.random.multivariate_normal(mvnorm_parm["mean"], mvnorm_parm["sigma"], N)  # 生成多变量正态分布的随机数
        att = np.apply_along_axis(lambda x: x > mvnorm_parm["cutoffs"], 1, att)  # 生成多变量正态分布的随机数
    elif att_dist == "higher.order":
        att = np.random.binomial(1, 1 / (
                1 + np.exp(-np.dot(higher_order_parm["lambda"].values, higher_order_parm["theta"]))),
                                 N)  # 生成高阶属性的随机数

    # print("生成被试掌握模式真值完成\n", "