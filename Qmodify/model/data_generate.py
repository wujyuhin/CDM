import numpy as np


class GenerateData:
    def __init__(self,stu_num,prob_num,know_num):
        self.stu_num = stu_num
        self.prob_num = prob_num
        self.know_num = know_num
        self.q_m = self.generate_all_state(know_num)
        self.r_m = self.generate_r_normal(stu_num,self.q_m)

    # 生成Q矩阵真值
    def generate_all_state(self,know_num):
        state_num = 2 ** know_num  # 状态数
        q_m = np.zeros((state_num, know_num))
        for i in range(state_num):
            k, quotient, residue = 1, i // 2, i % 2  # 除数与余数
            while True:
                q_m[i, know_num - k] = residue
                if quotient <= 0:
                    break
                quotient, residue = quotient // 2, quotient % 2  # 对除数进行取整和取余得到更高一位二进制数是0还是1，直到除数为0
                k += 1
        return q_m

    def classify(cls,q_m):
        """
        对学生分类，根据掌握的知识点数，如掌握1个知识点的学生，掌握2个知识点的学生
        :param q_m:  Q矩阵
        :return:  返回分类结果,例如[[1,2,3],[4,5,6]]
        """
        total = np.sum(q_m, axis=1)
        clas = []
        for i in range(q_m.shape[1]):
            clas.append(np.where(total == i + 1))
        return clas

    def generate_stu_state(self,stu_num,q_m,method='uniform')
        """
        生成均匀分布的学生
        :param stu_num:  学生数 
        :param q_m:  Q矩阵
        :return: 
        """
        if method == 'uniform':
            stu_states = q_m[np.random.randint(1,32,stu_num)]
        if method == 'normal':
            pass
        if method == 'others':
            pass
        return stu_states

    def generate_r(self,stu_num,q_m):
        prob_num = q_m.shape[0]
        r_m = np.zeros((stu_num,prob_num))
        for i in range(stu_num):
            for j in range(prob_num):
                p = np.random.rand()
                if p > 0.5:
                    r_m[i,j] = 1
        return r_m

    def generate_r_normal(self,stu_num,q_m):
        prob_num = q_m.shape[0]
        r_m = np.zeros((stu_num,prob_num))
        for i in range(stu_num):
            for j in range(prob_num):
                p = np.random.rand()
                if p > 0.5:
                    r_m[i,j] = 1
        return r_m


# 生成Q矩阵真值
# def generate_all_state(know_num):
#     state_num = 2 ** know_num  # 状态数
#     q_m = np.zeros((state_num, know_num))
#     for i in range(state_num):
#         k, quotient, residue = 1, i // 2, i % 2  # 除数与余数
#         while True:
#             q_m[i, know_num - k] = residue
#             if quotient <= 0:
#                 break
#             quotient, residue = quotient // 2, quotient % 2  # 对除数进行取整和取余得到更高一位二进制数是0还是1，直到除数为0
#             k += 1
#     return q_m







if __name__ == '__main__':
    stu_num = 1000
    prob_num = 50
    know_num = 3
    data = GenerateData(stu_num,prob_num,know_num)
    print(data.q_m)
    print(data.r_m)
    print(data.classify(data.q_m))