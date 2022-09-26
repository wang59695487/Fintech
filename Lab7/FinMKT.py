import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score,StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder



# some usable model
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestRegressor
from imblearn.over_sampling import SMOTE



import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def analysis(data):
    data.replace(to_replace='unknown', value=np.NaN, inplace=True)
    missing_values_count = data.isnull().sum()
    attr1 = ['age','job','marital','education','default', 'housing', 'loan' , 'contact']
    values = []
    for k in attr1:
        values.append(missing_values_count[k])
    y_pos = np.arange(len(attr1))
    # 缺失的图
    plt.barh(y_pos ,values, align='center', alpha=0.4)
    plt.yticks(y_pos, attr1)
    plt.xlabel('unknown')
    plt.xlim(0, 10000)
    plt.title('The unknown in the features')
    plt.show()

    # 看连续变量密度图
    numeric_feats = list(data.keys())
    numeric_feats.remove('y')
    non_numeric_feats = list((data.dtypes[data.dtypes == 'object'].index)[:-1])
    for k in non_numeric_feats:
        numeric_feats.remove(k)
    for k in numeric_feats:
      sns.kdeplot(data.loc[data['y'] == 'yes', k], shade=True, color="orchid", label="yes", alpha=.7)
      sns.kdeplot(data.loc[data['y'] == 'no',  k], shade=True, color="deepskyblue", label="no", alpha=.7)
      plt.title(k)
      plt.legend()
      plt.show()

    attr_size = dict()
    for p in data.keys():
        attr_size[p] = len(data[p].value_counts())
    # 看离散变量密度图
    for k in non_numeric_feats:
       mp = data.loc[data['y'] == 'yes', k]
       mp = mp.value_counts()
       x = mp.keys()
       y = []
       for k2 in x:
           y.append(mp[k2])
       mp = data.loc[data['y'] == 'no', k]
       mp = mp.value_counts()
       x = mp.keys()
       y1 = []
       for k2 in x:
           y1.append(mp[k2])
       xp = []
       for i in range(len(x)):
           xp.append(i+1)

       plt.bar(xp, y, align="center", color="#66c2a5", label="yes", tick_label = x)
       plt.bar(xp, y1, align="center", bottom=y, color="#8da0cb", label="no",tick_label = x)

       plt.xlabel(k)
       plt.ylabel("counts")
       plt.legend()
       plt.show()





def data_preprocess(data):
    # your code here
    # example:
    #x = pd.get_dummies(data)
    ############# int:age,duration,campaing,pdays,previous ##################
    #data = pd.read_csv('bank-additional-full.csv', sep=';')
    data.replace(to_replace='unknown', value=np.NaN, inplace=True)
    # 把数值型特征都放到随机森林里面,补全data
    attr1 = ['job','marital','education','default', 'housing', 'loan']
    attr2 = ['emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
    for k in attr1:
       known =  data[data[k].notnull()]
       unknown= data[data[k].isnull()]
       y = known[k]  # y是预测的
       label_encoder = LabelEncoder()
       y = label_encoder.fit_transform(y)
       x = known[attr2]     # x是特征属性值，后面几列
       rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
       # 根据已有数据去拟合随机森林模型
       rfr.fit(x, y)
       # 预测缺失值
       predict = rfr.predict(unknown[attr2])
       # 填补缺失值
       data.loc[data[k].isnull(), k] = predict

    missing_values_count = data.isnull().sum()

    data['age']=pd.cut(data['age'], bins=[0,10,20,30,40,50,60,70,80,90,100])
    data['duration'] = pd.cut(data['duration'],bins=
    [-1, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000])
    data['campaign'] = pd.cut(data['campaign'], bins=[0, 10, 20, 30, 40, 50, 60])
    data['pdays'] = pd.cut(data['pdays'], bins=[-1,100,200,300,400,500,600,700,800,900,1000])
    data['y'] = data['y'].map(lambda x: 0 if x == 'no' else 1)
    ############# str:age,duration,campaing,pdays,previous ##################
    attr = ['age', 'duration', 'campaign', 'pdays','job', 'marital', 'education', 'default', 'housing', 'loan',
            'contact', 'month', 'day_of_week', 'poutcome']
    for t in attr:
        temp_list = data[t].value_counts().to_dict()
        cnt = 0
        for i in temp_list.keys():
            temp_list[i] = cnt
            cnt += 1
        data[t] = data[t].map(lambda x: temp_list[x])
    #TODO

    attr = ['emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
    data[attr] = data[attr].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    data = data.dropna(axis = 0, how = 'any')
    return data
    # your code end
    #return x


def split_data(data):
    data = data_preprocess(data)
    y = data['y']
    x = data.loc[:, data.columns != 'y']

    attr= ['age','job','marital','education','default','housing','loan','contact','month', 'day_of_week','duration','campaign','pdays','previous','poutcome']
    attr_size = dict()
    feature_size = 0

    for p in attr:
        t = len(x[p].value_counts())
        attr_size[p]=t
        feature_size+=t

    attr = ['emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']
    for p in attr:
        attr_size[p]=1
        feature_size+=1

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

    n_train = len(x_train)
    n_test = len(x_test)
    train_feature = np.zeros([n_train,feature_size])
    test_feature = np.zeros([n_test,feature_size])
    train_gt = np.array(list(y_train))
    test_gt = np.array(list(y_test))

    attr_list = x_train.keys()

    for i in range(n_train):
        sample = x_train.iloc[i]
        pos=0
        for s in attr_list:
            size = attr_size[s]
            if size==1:
                train_feature[i][pos]=sample[s]
            else:
                train_feature[i][int(pos+sample[s])]=1
            pos+=size

    for i in range(n_test):
        sample = x_test.iloc[i]
        pos=0
        for s in attr_list:
            size = attr_size[s]
            if size==1:
                test_feature[i][pos]=sample[s]
            else:
                test_feature[i][int(pos+sample[s])]=1
            pos+=size

    x_train = torch.Tensor(train_feature).to(device)
    x_test = torch.Tensor(test_feature).to(device)
    y_train = torch.LongTensor(train_gt).to(device)
    y_test = torch.LongTensor(test_gt)
    


    return x_train, x_test, y_train, y_test


#简单的4层全连接网络
class Qnetwork(torch.nn.Module):
    def __init__(self, n_feature, n_output):
        super(Qnetwork, self).__init__()
        tn = n_feature
        self.hidden_1 = torch.nn.Linear(tn, tn // 2)
        tn //= 2
        self.hidden_2 = torch.nn.Linear(tn, tn // 2)
        tn //= 2
        self.hidden_3 = torch.nn.Linear(tn, tn // 2)
        tn //= 2
        self.predict = torch.nn.Linear(tn, n_output)

    def forward(self, x):
        x = F.relu(self.hidden_1(x))
        x = F.relu(self.hidden_2(x))
        x = F.relu(self.hidden_3(x))
        x = self.predict(x)
        return x

#重采样的神经网络
class DeepNet(torch.nn.Module):
    def __init__(self,n_output,attr_list=None):
        super(DeepNet, self).__init__()
        self.attr_list = attr_list
        self.embed = dict()
        for key,value in attr_list.items():
            self.embed[key] = torch.nn.Linear(value,1).to(device)
        self.feature_size = tn = len(self.embed.keys())
        self.hidden_1 = torch.nn.Linear(tn, tn * 2)
        tn *= 2
        self.hidden_2 = torch.nn.Linear(tn, tn * 2)
        tn *= 2
        self.hidden_3 = torch.nn.Linear(tn, tn * 2)
        tn *= 2
        self.predict = torch.nn.Linear(tn, n_output)

    def forward(self, x):
        batch = []
        dim = len(x)
        for i in range(dim):
            feature = []
            for key, value in self.attr_list:
                temp = self.embed[key](x[i][key].to(device))
                feature.append(temp)
            temp_feature = torch.unsqueeze(torch.cat(feature), dim=0).to(device)
            print(temp_feature)
            batch.append(temp_feature)
        feature = torch.cat(batch, dim=0).to(device)
        x = F.relu(self.hidden_1(feature))
        x = F.relu(self.hidden_2(x))
        x = F.relu(self.hidden_3(x))
        x = self.predict(x)
        return x

#自己搭建的神经网络训练和测试
def predictNet(x_train, x_test, y_train):

    feature_size = len(x_train[0])
    output_size = 2
    #myNet = DeepNet(2,attr_size)
    myNet = Qnetwork(feature_size, output_size)
    myNet = myNet.to(device)

    learning_rate = 0.01
    loss_f = torch.nn.CrossEntropyLoss()
    opt_SGD = torch.optim.SGD(myNet.parameters(), lr=learning_rate)

    n_train = len(x_train)
    n_epoch = 100
    batch_size = 64
    batch_num = n_train // batch_size

    if n_train % batch_size != 0:
        batch_num += 1
    for i in range(n_epoch):
        batch_loss = 0
        for j in range(batch_num):
            st = j * batch_size
            ed = st + batch_size
            if ed <= n_train:
                data = x_train[st:ed]
                gt = y_train[st:ed]
            else:
                data = x_train[st:]
                gt = y_train[st:]

            y = myNet(data)
            loss = loss_f(y, gt)
            opt_SGD.zero_grad()
            loss.backward()
            opt_SGD.step()
            batch_loss += loss

        #print("[Epoch] : %d/%d , Loss = %f "
             # % (i, n_epoch, batch_loss / batch_num))

    y_pred = []
    n_test = len(x_test)
    for i in range(n_test):
        data = x_test[i]
        result = myNet(data)
        if result[0] > result[1]:
             y_pred.append(0)
        else:
             y_pred.append(1)

    return y_pred


#运用库里的向量机调整参数进行的测试
def predictSVM(x_train, x_test, y_train):

    # your code here begin
    # train your model on 'x_train' and 'x_test'
    # predict on 'y_train' and get 'y_pred'
    # grid search start
    best_score = 0
    '''for gamma in [0.001, 0.01, 1, 10, 100]:
        for c in [0.001, 0.01, 1, 10, 100]:
            # 对于每种参数可能的组合，进行一次训练
            svm = SVC(gamma=gamma, C=c)
            svm.fit(x_train, y_train)
            score = svm.score(x_test, y_test)
            # 找到表现最好的参数
            if score > best_score:
                best_score = score
                best_parameters = {'gamma': gamma, "C": c}
                y_pred = svm.predict(x_test)

    print('Best socre:{:.2f}'.format(best_score))
    print('Best parameters:{}'.format(best_parameters))'''
    clf = SVC()
    clf.fit(x_train, y_train)

    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False)

    y_pred = clf.predict(x_test)

    # your code here end

    return y_pred


def predictGB(x_train, x_test, y_train):

    clf = GradientBoostingClassifier(random_state=0)
    clf.fit(x_train, y_train)
    GradientBoostingClassifier(random_state=0)
    y_pred = clf.predict(x_test)
    np.array([1, 0])

    return y_pred

def predictMLPC(x_train, x_test, y_train):
    clf = MLPClassifier(random_state=1, max_iter=300).fit(x_train, y_train)
    clf.predict_proba(x_test)
    y_pred = clf.predict(x_test)

    return y_pred

def predictAB(x_train, x_test, y_train):

    clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)

    return y_pred

def predictBC(x_train, x_test, y_train):
    clf = BaggingClassifier(base_estimator=SVC(),n_estimators = 10, random_state = 0)
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)

    return y_pred

def predictLR(x_train, x_test, y_train):

    model = LogisticRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    return y_pred


def print_result(y_test, y_pred):
    report = confusion_matrix(y_test, y_pred)
    precision = report[1][1] / (report[:, 1].sum())
    recall = report[1][1] / (report[1].sum())
    print('model precision:' + str(precision)[:4] + ' recall:' + str(recall)[:4])
    F1_score = 2*precision*recall / (precision+recall)
    print('F1_score:' + str(F1_score)[:4])

if __name__ == '__main__':

    data = pd.read_csv('bank-additional-full.csv', sep=';')
    analysis(data)

    data = pd.read_csv('bank-additional-full.csv', sep=';')
    x_train, x_test, y_train, y_test = split_data(data)
    y_pred = predictSVM(x_train, x_test, y_train )
    print("\n===SVM===")
    print_result(y_test, y_pred)

    data = pd.read_csv('bank-additional-full.csv', sep=';')
    x_train, x_test, y_train, y_test = split_data(data)
    y_pred = predictLR(x_train, x_test, y_train )
    print("\n===LR===")
    print_result(y_test, y_pred)

    data = pd.read_csv('bank-additional-full.csv', sep=';')
    x_train, x_test, y_train, y_test = split_data(data)
    y_pred = predictGB(x_train, x_test, y_train )
    print("\n===GB===")
    print_result(y_test, y_pred)

    data = pd.read_csv('bank-additional-full.csv', sep=';')
    x_train, x_test, y_train, y_test = split_data(data)
    x_ntrain, y_ntrain = SMOTE().fit_sample(x_train, y_train)
    y_pred = predictNet(x_ntrain, x_test, y_ntrain )
    print("\n===Mynet===")
    print_result(y_test, y_pred)

    data = pd.read_csv('bank-additional-full.csv', sep=';')
    x_train, x_test, y_train, y_test = split_data(data)
    x_ntrain, y_ntrain = SMOTE().fit_sample(x_train, y_train)
    y_pred = predictMLPC(x_ntrain, x_test, y_ntrain )
    print("\n===MLPC===")
    print_result(y_test, y_pred)

    data = pd.read_csv('bank-additional-full.csv', sep=';')
    x_train, x_test, y_train, y_test = split_data(data)
    y_pred = predictAB(x_ntrain, x_test, y_ntrain )
    print("\n===AdaBoost===")
    print_result(y_test, y_pred)

    data = pd.read_csv('bank-additional-full.csv', sep=';')
    x_train, x_test, y_train, y_test = split_data(data)
    y_pred = predictBC(x_ntrain, x_test, y_ntrain )
    print("\n===Bagging===")
    print_result(y_test, y_pred)



