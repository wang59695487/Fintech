'''
risk model
'''
import lightgbm as lgb
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from math import exp

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RiskModel():
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.train, self.test, self.param = self.__construct_dataset()
        self.feature_name = [i for i in self.train.columns if i not in ['Y','id']]
        print('train set:', self.train.shape, ', ', 'test set:', self.test.shape)
        self.lgb_train = lgb.Dataset(data=self.train[self.feature_name],
                                     label=self.train['Y'],
                                     feature_name=self.feature_name)
        self.lgb_test = lgb.Dataset(data=self.test[self.feature_name],
                                    label=self.test['Y'],
                                    feature_name=self.feature_name)
        self.evals_result = {}
        self.gbm = None

    def __construct_dataset(self):
        train = pd.read_csv(self.data_path + 'train_no.csv',index_col=0)
        test = pd.read_csv(self.data_path + 'test_no.csv',index_col=0)
        train = train.astype('float')
        test = test.astype('float')

        param = dict()
        param['objective'] = 'binary'
        param['boosting_type'] = 'gbdt'
        param['metric'] = 'auc'
        param['verbose'] = 0
        param['learning_rate'] = 0.1
        param['max_depth'] = -1
        param['feature_fraction'] = 0.8
        param['bagging_fraction'] = 0.8
        param['bagging_freq'] = 1
        param['num_leaves'] = 15
        param['min_data_in_leaf'] = 64
        param['is_unbalance'] = False
        param['verbose'] = -1

        return train, test, param

    def fit(self):
        self.gbm = lgb.train(self.param,
                             self.lgb_train,
                             early_stopping_rounds=10,
                             num_boost_round=1000,
                             evals_result=self.evals_result,
                             valid_sets=[self.lgb_train, self.lgb_test],
                             verbose_eval=1)


    def evaluate(self):
        test_label = pd.DataFrame(self.test['Y'])
        test1 = []
        for i in test_label.values:
            test1.append(int(i))
        test1 = np.array(test1)
        prob_label = self.gbm.predict(self.test[self.feature_name])

        fpr, tpr, thresholds = roc_curve(test1, prob_label, pos_label=1)
        sklearn_auc = auc(fpr, tpr)
        my_auc = self.Myauc(test1, prob_label)
        return sklearn_auc, my_auc

    def Myauc(self, labels, preds, n_bins=100):
        labels.astype(np.int)
        postive_len = sum(labels)
        negative_len = len(labels) - postive_len
        total_case = postive_len * negative_len
        pos_histogram = [0 for _ in range(n_bins)]
        neg_histogram = [0 for _ in range(n_bins)]
        bin_width = 1.0 / n_bins
        for i in range(len(labels)):
            nth_bin = int(preds[i]/ bin_width)
            if labels[i] == 1:
                pos_histogram[nth_bin] += 1
            else:
                neg_histogram[nth_bin] += 1
        accumulated_neg = 0
        satisfied_pair = 0
        for i in range(n_bins):
            satisfied_pair += (pos_histogram[i] * accumulated_neg + pos_histogram[i] * neg_histogram[i] * 0.5)
            accumulated_neg += neg_histogram[i]

        return satisfied_pair / float(total_case)

    def predict(self,data):
        prob_label = self.gbm.predict(data[self.feature_name])

        return pd.DataFrame(prob_label)

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

def predictNet(x_train, x_test, y_train):

    feature_size = len(x_train[0])
    output_size = 2
    myNet = Qnetwork(feature_size, output_size)
    myNet = myNet.to(device)

    learning_rate = 0.1
    loss_f = torch.nn.CrossEntropyLoss()
    opt_SGD = torch.optim.SGD(myNet.parameters(), lr=learning_rate)

    n_train = len(x_train)
    n_epoch = 100
    batch_size = 50
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
              #% (i, n_epoch, batch_loss / batch_num))

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


def getdata():
    data1 = pd.read_csv('train.csv', index_col=0)
    data2 = pd.read_csv('test.csv', index_col=0)

    x_train = data1.loc[:, data1.columns != 'Y']
    x_test = data2.loc[:, data2.columns != 'Y']
    y_train = data1['Y']
    y_test = data2['Y']


    return x_train, y_train, x_test, y_test

def getdata2():
    data1 = pd.read_csv('train.csv', index_col=0)
    data2 = pd.read_csv('test.csv', index_col=0)

    for k in ['X1_X2','X2']:
      data1.drop([k], axis=1, inplace=True)
      data2.drop([k], axis=1, inplace=True)
      print("drop ",k)

    x_train = data1.loc[:, data1.columns != 'Y']
    x_test = data2.loc[:, data2.columns != 'Y']
    y_train = data1['Y']
    y_test = data2['Y']


    n_train = len(x_train)
    n_test = len(x_test)
    numeric_feats = list((x_train.dtypes[x_train.dtypes != 'object'].index)[:-1])
    if 'X1' in numeric_feats:
        numeric_feats.remove('X1')
    if 'X2' in numeric_feats:
        numeric_feats.remove('X2')
    feature_size = len(numeric_feats)+12

    train_feature = np.zeros([n_train, feature_size])
    test_feature = np.zeros([n_test, feature_size])
    train_gt = np.array(list(y_train))
    test_gt = np.array(list(y_test))


    x_train = torch.Tensor(train_feature).to(device)
    x_test = torch.Tensor(test_feature).to(device)
    y_train = torch.LongTensor(train_gt).to(device)
    y_test = torch.LongTensor(test_gt)

    return x_train, y_train, x_test, y_test

if __name__ == "__main__":

    MODEL = RiskModel(data_path='./')
    MODEL.fit()
    sklearn_auc, my_auc = MODEL.evaluate()
    print('eval auc:', sklearn_auc)
    print('--my auc:', my_auc)

''' data3 = pd.read_csv('./data/test2.csv', index_col=0)
    pred = MODEL.predict(data3)
    y_pred = []
    for k in pred.values:
        if (k-0.5) > 0.00000001 :
            y_pred.append(1)
        else:
            y_pred.append(0)
    y_pred = pd.DataFrame(y_pred)
    y_pred.to_csv('3170100186_pre.csv')

    # Mynet
    x_train, y_train, x_test, y_test = getdata2()
    #x_ntrain, y_ntrain = SMOTE().fit_sample(x_train, y_train)
    #x_train, y_train = SMOTE().fit_sample(x_train, y_train)
    y_pred = predictNet(x_train, x_test, y_train)
    print('My net auc:', roc_auc_score(y_test, y_pred))

    # LR
    x_train, y_train, x_test, y_test = getdata()
    model = LogisticRegression()
    model.fit(x_train, y_train.astype('int'))
    y_pred = model.predict_proba(x_test)[:, 1]
    print('lr train auc:', roc_auc_score(y_test, y_pred))

    # GBDT
    x_train, y_train, x_test, y_test = getdata()
    model = GradientBoostingClassifier()
    model.fit(x_train, y_train.astype('int'))
    y_pred = model.predict_proba(x_test)[:, 1]
    print('gbdt train auc:', roc_auc_score(y_test, y_pred))
    # XGB
    x_train, y_train, x_test, y_test = getdata()
    xgb_clf = xgb.sklearn.XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators = 10, gamma = 0, reg_lambda = 1)
    xgb_clf.fit(x_train, y_train)
    xgb_pred = xgb_clf.predict_proba(x_test)[:, 1]
    print('xgb train auc:', roc_auc_score(y_test, xgb_pred))
'''

