import numpy as np
import matplotlib.pyplot as plt
# 生成的随机数数组100行1列的矩阵，每个随机数都在0到1之间
X = 2*np.random.rand(100, 1)
# 生成服从标准正态分布（均值为0，标准差为1）的随机数的函数
y = 4 + 3*X + np.random.randn(100, 1)

# 数据集拼接一列1。按列连接（concatenate）多个数组
X_b = np.c_[np.ones((100, 1)), X]
# # .dot是矩阵的乘法，不能写*。    inv是求逆矩阵
# theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
# # Out：[[4.00788854], [3.04738413]] 偏置项，权重项。2行1列的表示方式。
#
#
X_new = np.array([[0], [2]])
# # Out：[[0], [2]] 2行1列，第一行行是0，第二行是2
X_new_b = np.c_[np.ones((2, 1)), X_new]
# y_predict = X_new_b.dot(theta_best)
# y_predict
#
# plt.plot(X_new, y_predict, 'r--', X, y, 'b.')
# plt.plot(X, y, "b.")
# plt.axis([0, 2, 0, 15])
# plt.show()
#
#
#
# from sklearn.linear_model import LinearRegression
# lin_reg = LinearRegression()
# lin_reg.fit(X, y)
# print(lin_reg.coef_)  #fit完后，获取权重参数 和偏置项
# print(lin_reg.intercept_)



# 批量梯度下降
# eta = 0.1 #学习率
# n_iterations = 1000
# m = 100
# # 2行1列初始化theta
# theta = np.random.randn(2, 1)
# for iteration in range(n_iterations):
#     gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
#     # 梯度更新
#     theta = theta - eta * gradients
# print(theta)
# y_predict = X_new_b.dot(theta)
# print(y_predict)


# 批量梯度下降 学习率demo
def plot_gradient_descent(theta, eta, theta_path=None):
    m = len(X_b)  # count of samples in X_b
    plt.plot(X, y, "b.")
    n_iterations = 1000
    for iteration in range(n_iterations):
        if iteration < 10:
            y_predict = X_new_b.dot(theta)
            style = "b-" if iteration > 0 else "r--"
            plt.plot(X_new, y_predict, style)  # plot the linear model
        gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y)  # gradient vector of cost in gradient descent
        theta = theta - eta * gradients  # eta:learning rate    theta: cost vector
        if theta_path is not None:
            theta_path.append(theta)
    plt.xlabel("$x_1$", fontsize=18)
    plt.axis([0, 2, 0, 15])
    plt.title(r"$\eta = {}$".format(eta), fontsize=16)


theta_path_bgd = []  # theta list in all iterations
np.random.seed(42)
theta = np.random.randn(2, 1)  # random initialization

plt.figure(figsize=(10, 4))
plt.subplot(131)
plot_gradient_descent(theta, eta=0.02)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.subplot(132)
plot_gradient_descent(theta, eta=0.1, theta_path=theta_path_bgd)
plt.subplot(133)
plot_gradient_descent(theta, eta=0.5)
plt.show()