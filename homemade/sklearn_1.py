# https://blog.csdn.net/the_ZED/article/details/128458861
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)
import scipy.io
mnist = scipy.io.loadmat('../data/mnist-original.mat')
#
# from sklearn.datasets import fetch_mldata, fetch_openml
#
# mnist = fetch_openml()
mnist

# 查看 Mnist 数据集的内容
X = mnist['data']
print("X的内容:\n",X)
print("X的规格:",X.shape)
print()

# Mnist数据集的标签
y = mnist['label']
print("y的内容:\n",y)
print("y的规格:",y.shape)

# 为了方便理解和后续使用，将 X 和 y 进行转置
X = X.T
y = y.T

# 查看执行转置后的两个矩阵
print("x shape" + str(X.shape))
print("y[:10]" + str(y[:10]))
print("y shape" + str(y.shape))

# 为了后续处理数据的方便，对 y 进行拉平
y = y.flatten()
print(y[:10])
print(y.shape)

# %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt

# 从数据集中任意找一个数字
index = 36000
certain_digit = X[index]

# 查看其真实标签
print("该数字为：%d"%y[index])

# 将样本转为28大小的像素矩阵
certain_digit = certain_digit.reshape(28, 28)

# 按‘0’‘1’数值转为灰度图像
# interpolation当小图像放大时,interpolation ='nearest'效果很好，否则用None。
plt.imshow(certain_digit, cmap = matplotlib.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()

# 构建测试数据和训练数据（60000份训练数据、10000份测试数据）
X_train = X[:60000]
X_test = X[60000:]
y_train = y[:60000]
y_test = y[60000:]

# 置乱操作（尽可能让各数字样本置乱，使得分布尽可能随机，也就是理论部分中提到的 “分层采样”）
import numpy as np

# 定义一个随机的索引序列
shuffle_index = np.random.permutation(60000)

# 利用随机的索引序列来重新装填训练集的内容
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

# 查看置乱后的索引序列
print("shuffle_index:")
print(shuffle_index)

# 构建新的标签
y_train_5 = (y_train==5)
y_test_5 = (y_test==5)

# 查看
y_train_5[:10]

from sklearn.linear_model import SGDClassifier

# 设置最大迭代次数 max_iter = 10
# 设置随机种子 random_state = 42
sgd_clf = SGDClassifier(max_iter=10, random_state = 42)

# 开始训练
sgd_clf.fit(X_train,y_train_5)

#用训练好的分类器进行预测
print(y[index])
print(sgd_clf.predict([X[index]]))

# # 使用交叉验证测量准确性
# from sklearn.model_selection import cross_val_score
#
# # 该方法会自动对数据集进行切分并对切分后的样本进行交叉检验
# # cv 表示交叉验证的次数（每次都将数据划分为 cv 份，并采取“留一法”进行验证）
# # scoring 选择 "accuracy" 表示用准确率来衡量
# cross_val_score(sgd_clf,X_train,y_train_5,cv=3,scoring='accuracy')


# 自定义交叉验证过程
# 调用切分数据方法
from sklearn.model_selection import StratifiedKFold

# 为了在每次训练时，使用一个规格一致，而都是未经训练的分类器，这里调用了 clone 库
from sklearn.base import clone

# 将数据集切分 n_splits 份，并进行 n_splits 次交叉实验（每次都采取“留一法”单独用一组数据来验证，并计算得分）
skflods = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)
i = 0;

# 对每份数据重新进行 fit 操作
for train_index, test_index in skflods.split(X_train, y_train_5):
    # 克隆分类器
    clone_clf = clone(sgd_clf);

    # 划分数据集
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_folds = X_train[test_index]
    y_test_folds = y_train_5[test_index]

    # 开始训练
    clone_clf.fit(X_train_folds, y_train_folds)

    # 进行预测
    y_pred = clone_clf.predict(X_test_folds)

    # 对预测结果进行评估
    i += 1
    cnt_correct = sum(y_pred == y_test_folds)

    # 但是这里执行 y_pred==y_test_folds 得到的是一个二维元组
    print("第 %d 次评估的准确率为 %.2f%% " % (i, cnt_correct / len(y_pred) * 100))


from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf,X_train,y_train_5,cv=3)

print(X_train.shape)
print(y_train_pred.shape)
print(y_train_pred[:10])

# 可通过调用 confusion_matrix( ) 方法计算混淆矩阵
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train_5,y_train_pred)
print(cm)
# true negative (被正确分为非5分类）, false positive （被错误分为5）SGDClassifier
# false negative（被错误分为非5）, true positive（被正确分为5）

from sklearn.metrics import precision_score,recall_score,f1_score
print("查准率为 %.2f%%" % (precision_score(y_train_5,y_train_pred)*100))
print("查全率为 %.2f%%" % (recall_score(y_train_5,y_train_pred)*100))
print("F1 SCORE为 %.2f%%" % (f1_score(y_train_5,y_train_pred)*100))

