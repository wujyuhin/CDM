# 计算评价指标
import numpy as np
from code_functions.data_generate import generate


# 项目模式判准率 pattern match ratio, PMR
def PMR(Q, cur_Q):
    """
    计算掌握模式准确率
    :param Q:  真实Q矩阵
    :param cur_Q: 当前Q矩阵
    :return:
    """
    PMR_right = 0
    for i in range(Q.shape[0]):
        if np.all(Q[i, :] == cur_Q[i, :]):
            PMR_right += 1
    return PMR_right / Q.shape[0]


# 属性判准率 attribute match ratio, AMR
def AMR(Q, cur_Q):
    AMR_right = 0
    for i in range(Q.shape[0]):
        for j in range(Q.shape[1]):
            if Q[i, j] == cur_Q[i, j]:
                AMR_right += 1
    return AMR_right / (Q.shape[0] * Q.shape[1])


# 计算两个矩阵的相同元素与不同元素的bool矩阵
def get_bool_matrix(Q, cur_Q):
    """
    计算两个矩阵的相同元素与不同元素的bool矩阵
    :param Q:  真实Q矩阵
    :param cur_Q:  当前Q矩阵
    :return:  相同元素与不同元素的bool矩阵
    """

    pass


# 正确属性被保留的比例 true positive rate, TPR
def TPR(Qright, Qwrong, Qmodify):
    """
    计算正确属性被保留的比例
    :param Qright:  正确的Q矩阵
    :param Qwrong:  错误的Q矩阵
    :param Qmodify:  修改后的Q矩阵
    :return:
    """
    isWrong = Qright != Qwrong  # 生成错误的bool矩阵,在真实的Q下，错误了哪些
    TPR_right = 0
    # 先循环Q矩阵的行和列，如果Q矩阵的元素与当前Q矩阵的元素相等，且不是错误元素，则TPR_right+1
    for i in range(Qright.shape[0]):
        for j in range(Qright.shape[1]):
            if Qright[i, j] == Qmodify[i, j] and isWrong[i, j] != 1:
                # isWrong[i, j] != 1 表示不是错误元素,即看到的Q[i,j]是真实的Q[i,j]
                # Qright[i, j] == Qmodify[i, j] 表示当前的Q[i,j]是正确的
                TPR_right += 1
    return TPR_right / ((Qright.shape[0] * Qright.shape[1]) - np.sum(isWrong))


# 错误属性被修改的比例 false positive rate, FPR
def FPR(Qright, Qwrong, Qmodify):
    """
    计算错误属性被修改的比例
    :param Qright:  正确的Q矩阵
    :param Qwrong:  错误的Q矩阵
    :param Qmodify:  修改后的Q矩阵
    :return:
    """
    isWrong = Qright != Qwrong  # 生成错误的bool矩阵,在真实的Q下，错误了哪些
    FPR_right = 0
    # 先循环Q矩阵的行和列，如果Q矩阵的元素与当前Q矩阵的元素不相等，且是错误元素，则FPR_right+1
    for i in range(Qright.shape[0]):
        for j in range(Qright.shape[1]):
            if Qright[i, j] != Qmodify[i, j] and isWrong[i, j] == 1:
                # isWrong[i, j] == 1 表示是错误元素,即看到的Q[i,j]不是真实的Q[i,j]
                # Qright[i, j] != Qmodify[i, j] 表示当前的Q[i,j]是错误的
                FPR_right += 1
    return FPR_right / np.sum(isWrong)


if __name__ == '__main__':
    np.random.seed(0)
    # 生成Q矩阵
    Q = generate.generate_Q(items=5, skills=4, probs=[0.2, 0.3, 0.4, 0.5])
    # 改错Q矩阵
    Qwrong_dict = generate.generate_wrong_Q(Q, wrong_rate=[0.2, 0.2])  # 是一个dict，包含Q矩阵和错误矩阵
    # 这是假设已经通过修改Q矩阵的方法修改好了，是一个dict，包含Q矩阵和错误矩阵
    Qmodify_dict = generate.generate_wrong_Q(Qwrong_dict['Q_wrong'], wrong_rate=[0.2, 0.2])
    # 计算PMR
    print(PMR(Q, Qwrong_dict['Q_wrong']))
    # 计算AMR
    print(AMR(Q, Qwrong_dict['Q_wrong']))
    # 计算TPR
    print(TPR(Q, Qwrong_dict['Q_wrong'], Qmodify_dict['Q_wrong']))
    # 计算FPR
    print(FPR(Q, Qwrong_dict['Q_wrong'], Qmodify_dict['Q_wrong']))
    print(FPR(Q, Qwrong_dict['Q_wrong'], Q))
