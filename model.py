import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from modAL.models import ActiveLearner
from sklearn.metrics import accuracy_score

df = np.asarray(pd.read_csv('./data.csv'))
test = np.asarray(pd.read_csv('./test.csv'))
# print(df)

X = df[:, 1:8].astype('float64')
y = df[:, 8:].astype('int')
test_X = test[:, :7].astype('float64')
test_y = test[:, 7:].astype('int')
print(X.shape, y.shape)

scaler = MinMaxScaler()  # 实例化


# 归一化
def minMax(scaler, X):
    s = scaler.fit(X)  # fit，在这里本质是生成min(x)和max(x)
    return s.transform(X)


# 对数据进行归一化
X = minMax(scaler, X)
test_X = minMax(scaler, test_X)

# 初始数据集的选取
n_initial = 5  # 最开始随机选取5个标注好的数据集进行训练
initial_idx = np.random.choice(range(len(X)), size=n_initial, replace=False)
X_training, y_training = X[initial_idx], y[initial_idx]


# 把数据添加到csv中
def addXAndy(Xd, yd):
    global X, y
    print(X.shape, y.shape)
    print(Xd.shape, yd.shape)
    X = np.concatenate((X, np.asarray([Xd])))
    y = np.concatenate((y, np.asarray([yd])))
    print(X.shape, y.shape)
    output = np.concatenate((X, y), axis=1)
    df = pd.DataFrame(output)
    df.to_csv('./data.csv')


# 预测数据
def predictData(X):
    y_predict = regressor.predict(X)
    svm = svm_clf.predict(X)
    rf = rf_clf.predict(X)
    dt = dt_clf.predict(X)
    knn = knn_clf.predict(X)
    bagging = bagging_clf.predict(X)

    return {
        'al': y_predict.tolist(),
        'svm': svm.tolist(),
        'rf': rf.tolist(),
        'dt': dt.tolist(),
        'knn': knn.tolist(),
        'bagging': bagging.tolist()
    }


# print(X, y)


# 定义query_stategy应用于请求标注的查询策略
def GP_regression_std(regressor, X):
    std = regressor.predict(X)  # 不确定度度量
    query_idx = np.argmax(std)  # 样本的选取
    return query_idx, X[query_idx]


# print(X_training.shape, y_training.shape)


# 定义ActiveLeaner 主动学习器
regressor = ActiveLearner(
    # estimator=GaussianProcessRegressor(kernel=kernel),
    estimator=DecisionTreeClassifier(),
    query_strategy=GP_regression_std,
    X_training=X_training, y_training=y_training
)

svm_clf = svm.SVC(C=0.6, kernel="rbf", decision_function_shape="ovo")

rf_clf = RandomForestClassifier(n_estimators=500,
                                max_depth=30,
                                random_state=0,
                                min_samples_leaf=1,
                                oob_score=True,
                                min_samples_split=3)

dt_clf = DecisionTreeClassifier(random_state=0)

knn_clf = KNeighborsClassifier()

bagging_clf = BaggingClassifier(KNeighborsClassifier(),
                                max_samples=0.5,
                                max_features=0.5)

# 定义n_queries（要标记数据的数量）进行主动学习
n_queries = 10
al_acc_list = []


# 训练数据
def train_data(n_queries):
    for idx in range(n_queries):
        # 这里是主动学习
        query_idx, query_instance = regressor.query(X)
        # print('!!!', query_idx.reshape(1, -1)[0], query_instance.reshape(1, -1)[0])
        regressor.teach(X[query_idx].reshape(1, -1), y[query_idx].reshape(1, -1))
        y_pred = regressor.predict(test_X)
        y_pred = y_pred.ravel()
        # print(y_pred.astype('int'), y.reshape(1, -1))
        al_acc_list.append(accuracy_score(test_y.reshape(1, -1)[0], y_pred.astype('int')))
        print(accuracy_score(test_y.reshape(1, -1)[0], y_pred.astype('int')))
    return al_acc_list


def else_model():
    # 这里是SVM
    svm_clf.fit(X, y.reshape(1, -1)[0])
    svm_acc = svm_clf.score(test_X, test_y)
    print('svm', svm_acc)
    # 这里是随机森林
    rf_clf.fit(X, y.reshape(1, -1)[0])
    rf_acc = rf_clf.score(test_X, test_y)
    print('rf', rf_acc)
    # 这里是决策树
    dt_clf.fit(X, y.reshape(1, -1)[0])
    dt_acc = dt_clf.score(test_X, test_y)
    print('dt', dt_acc)
    # KNN
    knn_clf.fit(X, y.reshape(1, -1)[0])
    knn_acc = dt_clf.score(test_X, test_y)
    print('knn', knn_acc)
    # Bagging
    bagging_clf.fit(X, y.reshape(1, -1)[0])
    bagging_acc = bagging_clf.score(test_X, test_y)
    print('bagging', bagging_acc)
    return {
        'svm': svm_acc,
        'rf': rf_acc,
        'dt': dt_acc,
        'knn': knn_acc,
        'bagging': bagging_acc
    }

train_data(10)
else_model()
