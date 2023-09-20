from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

m=100
X = 6*np.random.rand(m,1) - 3
Y = 0.5*X**2 + X + 2 + np.random.randn(m,1)
lr=linear_model.LinearRegression()

# 创建一个多项式特征生成器的实例。degree=2 表示要生成二次多项式特征
poly_feacures = PolynomialFeatures(degree=2,include_bias=False)
# 将输入特征矩阵 X 转换为多项式特征矩阵 X_poly。这个操作会将原始特征按照二次多项式进行组合，生成新的特征矩阵。
X_poly = poly_feacures.fit_transform(X)
# 使用多项式特征矩阵 X_poly 和目标变量 Y 来训练线性回归模型
lr.fit(X_poly,Y)

X = np.sort(X,axis=0)
Y = 0.5*X**2 + X + 2 + np.random.randn(m,1)
print(X)

plt.scatter(X,Y)
plt.plot(X,lr.predict(poly_feacures.fit_transform(X)),color='red',linewidth=3)
plt.show()