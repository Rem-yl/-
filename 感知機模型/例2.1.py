import numpy as np
# 数据点
x1 = np.array((3,3))
x2 = np.array((4,3))
x3 = np.array((1,1))
x_train = [x1,x2,x3]
x_label = [1,1,-1]

# 初始化权重
w = np.array((0,0))
b = np.array((0))
r = 1
flag = 1

while flag:  
    output = []
    for x,y in zip(x_train,x_label):
        if y * (np.dot(w,x) + b) <= 0:
            w = w + r * y * x
            b = b + r * y
    
    # 检测是否有误分类点
    for x,y in zip(x_train,x_label):
        pre = y * (np.dot(w,x) + b)
        output.append(pre)
    if min(output) > 0:
        flag = 0

print(w,b)
