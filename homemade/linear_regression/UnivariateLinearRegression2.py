import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as graph_objs
plotly.offline.init_notebook_mode()

from linear_regression import LinearRegression

# 多特征demo
# plotly.offline.init_notebook_mode()

if __name__ == '__main__':
    # files_and_dirs = os.listdir("../../")
    data = pd.read_csv('../../data/world-happiness-report-2017.csv')  # 导入数据
    # 得到训练和测试数据，以8：2切分
    train_data = data.sample(frac=0.8)
    test_data = data.drop(train_data.index)

    input_param_name = 'Economy..GDP.per.Capita.'  # 特征features
    input_param_name2 = 'Freedom'
    output_param_name = 'Happiness.Score'  # 标签label

    x_train = train_data[[input_param_name, input_param_name2]].values  # 构建数据
    y_train = train_data[[output_param_name]].values

    x_test = test_data[[input_param_name, input_param_name2]].values
    y_test = test_data[[output_param_name]].values

    plot_training_trace = graph_objs.Scatter3d(x=x_train[:, 0].flatten(), y=x_train[:, 1].flatten(), z=y_train.flatten()
                         , name="Training Set"
                         , mode='markers'
                         , marker={'size': 10,
                                   'opacity': 1,
                                   'line': {
                                       'color': 'rgb(255, 255, 255)',
                                        'width': 1
                                   }
                            }
                         )

    plot_test_trace = graph_objs.Scatter3d(x=x_test[:, 0].flatten(), y=x_test[:, 1].flatten(), z=y_test.flatten()
                                               , name="Test Set"
                                               , mode='markers'
                                               , marker={'size': 10,
                                                         'opacity': 1,
                                                         'line': {
                                                             'color': 'rgb(255, 255, 255)',
                                                             'width': 1
                                                         }
                                                         }
                                               )

    plot_layout = graph_objs.Layout(title='Date Setes', scene={
        'xaxis': {'title': input_param_name},
        'yaxis': {'title': input_param_name2},
        'zaxis': {'title': output_param_name},
    }, margin={'l':0, 'r':0, 'b':0, 't':0})
    plot_data = [plot_training_trace, plot_test_trace]
    plot_config = graph_objs.Figure(data=plot_data, layout=plot_layout)
    plotly.offline.plot(plot_config)


    # 训练线性回归模型
    num_iterations = 500  # 迭代次数
    learning_rate = 0.01  # 学习率

    linear_regression = LinearRegression(x_train, y_train)  # 初始化模型
    (theta, cost_history) = linear_regression.train(learning_rate, num_iterations)

    print('开始时的损失：', cost_history[0])
    print('训练后的损失：', cost_history[-1])

    plt.plot(range(num_iterations), cost_history)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Gradient Descent Progress')
    # plt.show()

    # 生成一系列用于模型预测的输入特征值 x_predictions
    # 测试线性回归模型
    predictions_num = 10  # 生成10x10=100个预测值组合
    x_min = x_train[:, 0].min()
    x_max = x_train[:, 0].max()

    y_min = x_train[:, 1].min()
    y_max = x_train[:, 1].max()

    x_axis = np.linspace(x_min, x_max, predictions_num)
    y_axis = np.linspace(x_min, y_max, predictions_num)

    # 100行1列，预填0
    x_predications = np.zeros((predictions_num * predictions_num, 1))
    y_predications = np.zeros((predictions_num * predictions_num, 1))
    x_y_index = 0
    for x_index, x_value in enumerate(x_axis):
        for y_index, y_value in enumerate(y_axis):
            x_predications[x_y_index] = x_value
            y_predications[x_y_index] = y_value
            x_y_index += 1
    z_predictions = linear_regression.predict(np.hstack((x_predications, y_predications)))
    plot_predictions_trace = graph_objs.Scatter3d(
        x=x_predications.flatten(),
        y=y_predications.flatten(),
        z=z_predictions.flatten(),
        name="Prediction Plane",
        mode='markers',
        marker={'size': 1,},
        opacity=0.8,
        surfaceaxis=2,
    )
    plot_data = [plot_training_trace, plot_test_trace, plot_predictions_trace]
    plot_config = graph_objs.Figure(data=plot_data, layout=plot_layout)
    plotly.offline.plot(plot_config)


