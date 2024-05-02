# CDM
本项目编写了认知诊断模型的相关代码，包括数据生成、DINA等认知诊断模型、Q矩阵修正算法等。

## 一、数据生成
- Q矩阵 
  - 例如三道考察五个知识点的$Q$矩阵： 
  - $$Q = \begin{bmatrix} 1 & 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 & 0 \\ 0 & 0 & 1 & 0 & 1 \end{bmatrix}$$
- 错误的Q矩阵 
  - 在正确Q矩阵基础上，按照一定比例随机改变其中的值，生成错误的Q矩阵，例如：
  - $$Q = \begin{bmatrix} 1 & 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 & 0 \\ 0 & 0 & 1 & 0 & 1 \end{bmatrix}$$
- 根据一定规则抽样学生掌握模式
  - 学生的掌握模式共有$2^k$种，$k$为知识点个数，例如五个知识点的掌握模式有$2^5=32$种，其中一种为$[0,0,1,1,0]$，表示学生掌握了第三和第四个知识点。
  - 生成学生掌握模式的方法有两种：
    - 每种掌握模式均匀生成
    - 根据正态分布生成，例如：
      - $$\mu = 2$$ 、$$\sigma = 1$$
      - 生成的学生掌握模式为：以掌握2个知识点为均值，1个知识点为标准差的正态分布生成掌握模式，如下，因为是均值为2，所以大多数是掌握情况都是掌握2个知识点
      - 如$$attribute = \begin{bmatrix} 1 & 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 & 0 \\ 0 & 0 & 1 & 0 & 1 \end{bmatrix}$$
- 根据掌握模式生成R作答矩阵
  - 跟矩阵$Q$和学生掌握模式$attribute$，生成学生的作答矩阵$R$，例如：
  - Q矩阵：$$Q = \begin{bmatrix} 1 & 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 & 0 \\ 0 & 0 & 1 & 0 & 1 \end{bmatrix}$$
  - 学生掌握模式：$$attribute = \begin{bmatrix} 1 & 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 & 0 \\ 0 & 0 & 1 & 0 & 1 \end{bmatrix}$$
  - 作答矩阵：$$R = \begin{bmatrix} 1 & 1 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$
- 根据猜对和失误的误差修改作答矩阵
  - 生成学生作答矩阵的时候，根据猜对和失误的误差，修改作答矩阵，例如：
  - $$R = \begin{bmatrix} 1 & 1 & 1 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

## 二、认知诊断模型DINA

本项目修改Q矩阵是基于DINA模型估计参数，并进行下一步的修改Q矩阵算法

使用中科大的软件包EduCDM，由于部分代码进行过修改，所以将修改过的EduCDM代码放入本仓库。
### example:运行文件为cdm_test.py

```python
# ============ 导入必要的包  ==========================================
import numpy as np
from code_functions.EduCDM import EMDINA as DINA
# ========  加减法数据进行认知诊断  ====================================
# 数据准备
# Q矩阵
q_m = np.loadtxt("../data/math2015/FrcSub/q_m.csv", dtype=int, delimiter=',')  
# 题目数、属性数
prob_num, know_num = q_m.shape[0], q_m.shape[1]  
#作答R矩阵
R = np.array(np.loadtxt("../data/math2015/FrcSub/data.csv", dtype=int, delimiter=','))  
stu_num = R.shape[0]  # 学生数
# cdm估计参数
cdm = DINA(R, q_m, stu_num, prob_num, know_num, skip_value=-1)
cdm.train(epoch=2, epsilon=1e-3)
# 题目的guess、slip参数
guess = cdm.guess
slip = cdm.slip

```

## 三、Q矩阵修改模型

本项目为认知诊断Q矩阵修正算法复现 Q矩阵修正算法：delta法，gamma法
- delta法（已完成）
- gamma法（已完成）
- R法(待完成)
- 假设检验方法(待完成)

测试例子在example中，delta法输入参数有

- Q矩阵
- R作答矩阵
- 学生数
- 题目数
- 知识点数
- delta法中的$\epsilon$

```python
# ============================ 导入必要的包  ====================================================
import numpy as np
import logging
from code_functions.model.delta import Delta
from code_functions.EduCDM import EMDINA as DINA
logging.getLogger().setLevel(logging.INFO)

# ============================  加减法数据  ====================================================
# 数据准备
q_m = np.loadtxt("../data/math2015/FrcSub/q_m.csv", dtype=int, delimiter=',')  # Q矩阵
prob_num, know_num = q_m.shape[0], q_m.shape[1]  # 题目数、属性数
R = np.array(np.loadtxt("../data/math2015/FrcSub/data.csv", dtype=int, delimiter=','))  #作答R矩阵
stu_num = R.shape[0]  # 学生数

# ============================  边发现边修正  ====================================================
# 模型实例化
delta_model = Delta(q_m, R, stu_num, prob_num, know_num,mode='inherit',epsilon=0.05)
# 对输入的实例进行修正
modify_q_m1 = delta_model.modify_Q() # Q矩阵

# 发现完所有的再修正
delta_model2 = Delta(q_m, R, stu_num, prob_num, know_num,mode='dependence',epsilon=0.05)
modify_q_m2 = delta_model2.modify_Q() # Q矩阵

# 对上述两种Q矩阵使用DINA模型进行参数估计
model1 = DINA(R, modify_q_m1, stu_num, prob_num, know_num, skip_value=-1)
model1.train(epoch=2, epsilon=0.05)
model2 = DINA(R, modify_q_m2, stu_num, prob_num, know_num, skip_value=-1)
model2.train(epoch=2, epsilon=0.05)

print('边发现边修改Q矩阵估计的平均guess参数：',sum(model1.guess)/len(model1.guess))
print('发现完所有的再修改Q矩阵估计的平均guess参数：',sum(model2.guess)/len(model2.guess))


print('边发现边修改Q矩阵估计的平均slip参数：',sum(model1.slip)/len(model1.slip))
print('发现完所有的再修改Q矩阵估计的平均slip参数：',sum(model2.slip)/len(model2.slip))

np.concatenate((np.array([1,2,3,4]),1),axis=None)

```


gamma法的输入参数有

- Q矩阵
- R作答矩阵
- 学生数
- 题目数
- 知识点数
- gamma法中的guess参数阈值（论文设为0.2）
- gamma法中的slip参数阈值（论文设为0.2）
- gamma法中的ES阈值（论文设为0.2）


