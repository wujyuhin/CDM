a1 = data.pisa00R.ct$data$R040Q02
a2 = data.pisa00R.ct$data$R040Q03A
a3 = data.pisa00R.ct$data$R040Q03B
a4 = data.pisa00R.ct$data$R040Q04
a5 = data.pisa00R.ct$data$R040Q06
a6 = data.pisa00R.ct$data$R077Q02
a7 = data.pisa00R.ct$data$R077Q04
a8 = data.pisa00R.ct$data$R077Q06
a9 = data.pisa00R.ct$data$R088Q01
a10 = data.pisa00R.ct$data$R088Q07
a11 = data.pisa00R.ct$data$R110Q01
a12 = data.pisa00R.ct$data$R110Q04
a13 = data.pisa00R.ct$data$R110Q05
a14 = data.pisa00R.ct$data$R110Q06
a15 = data.pisa00R.ct$data$R216Q01
a16 = data.pisa00R.ct$data$R216Q02
a17 = data.pisa00R.ct$data$R216Q03T
a18 = data.pisa00R.ct$data$R216Q04
a19 = data.pisa00R.ct$data$R216Q06
a20 = data.pisa00R.ct$data$R236Q01
pisadata = data.frame(cbind(a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a1,a18,a19,a20))
write.csv(pisadata, file = "data.csv", row.names = FALSE)  # 不保存行名
getwd()

setwd('D:\\Attachments\\00-search\\CDM\\data\\math2015\\FrcSub1')
write.csv(data.fraction1$data, file = "data.csv", row.names = FALSE)  # 不保存行名

library(GDINA)
library(CDM)
data(fraction.subtraction.data)
q <- fraction.subtraction.qmatrix
data <- fraction.subtraction.data
setwd('D:\\Attachments\\00-search\\CDM\\data\\FrcSub\\FrcSub3')
write.csv(data, file = "data.csv", row.names = FALSE)  # 不保存行名
dat <- sim10GDINA$simdat
Q <- matrix(c(1,0,0,
              0,1,0,
              0,0,1,
              1,0,1,
              0,1,1,
              1,1,0,
              1,0,1,
              1,1,0,
              1,1,1,
              1,0,1),byrow = T,ncol = 3)

Q1<- matrix(c(1,1,1,
              0,1,1,
              0,1,0,
              1,0,1,
              0,1,0,
              1,1,1,
              1,0,1,
              1,1,0,
              1,0,1,
              1,1,1),byrow = T,ncol = 3)

# 相对拟合度
est.Q <- GDINA(dat,Q,model="DINA",verbose = 0)
est.Q1 <- GDINA(dat,Q1,model="DINA",verbose = 0)
anova(est.Q,est.Q1)
# 绝对拟合度
modelfit(est.Q)
modelfit(est.Q1)

est.wald <- GDINA(dat, sugQ, model = extract(mc,"selected.model")$models, verbose = 0)
anova(est.sugQ,est.wald)

mc <- modelcomp(est.Q)
mc

objs = list(est.Q,est.Q1)

setwd('D:\\Attachments\\00-search\\CDM\\data\\TIMSS\\TIMSS2003')
write.csv(timss03$data, file = "data.csv", row.names = FALSE)  # 不保存行名

setwd('D:\\Attachments\\00-search\\CDM\\data\\TIMSS\\TIMSS2007')
write.csv(data.timss07.G4.lee$data, file = "data.csv", row.names = FALSE)  # 不保存行名

anova()