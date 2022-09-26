import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

def analysis(data):
    data.replace(to_replace='unknown', value=np.NaN, inplace=True)
    missing_values_count = data.isnull().sum()

    # 看连续变量密度图
    numeric_feats = list(data.keys())
    if 'Y' in data.keys():
      numeric_feats.remove('Y')
    if 'id' in data.keys():
      numeric_feats.remove('id')
    if 'X2' in data.keys():
      numeric_feats.remove('X2')
    non_numeric_feats = ['X1','X2','id']

    numeric_feats1 =['X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12',
                     'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19', 'X20', 'X21',
                     'X22', 'X23', 'X24', 'X25', 'X26', 'X27', 'X28', 'X29', 'X30']
    numeric_feats2 = ['X31', 'X32', 'X33', 'X34', 'X35', 'X36', 'X37', 'X38', 'X39',
                      'X40', 'X41', 'X42', 'X43', 'X44', 'X45', 'X46', 'X47', 'X48',
                      'X49', 'X50', 'X51', 'X52', 'X53', 'X54', 'X55', 'X56', 'X57']

    numeric_feats3 =['X58', 'X59', 'X60', 'X61','X62', 'X63', 'X64', 'X65', 'X66',
                     'X67', 'X68', 'X69', 'X70','X71', 'X72']

    for k in non_numeric_feats:
      if k in numeric_feats:
        numeric_feats.remove(k)
    for k in numeric_feats2:
      if k in data.keys() :
       sns.kdeplot(data.loc[data['Y'] == 1, k], shade=True, color="orchid", label="yes", alpha=.7)
       sns.kdeplot(data.loc[data['Y'] == 0, k], shade=True, color="deepskyblue", label="no", alpha=.7)
       plt.title(k)
       plt.legend()
       plt.show()

    # 看离散变量密度图
    for k in ['X1']:
     if k in data.keys():
       mp = data.loc[data['Y'] == 1, k]
       mp = mp.value_counts()
       x = mp.keys()
       y = []
       for k2 in x:
           y.append(mp[k2])
       mp = data.loc[data['Y'] == 0, k]
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

class Dataprocess():
    def __init__(self):

        '''
        ptrain = pd.read_csv('train_new.csv')
        ptest = pd.read_csv('test_new.csv')
        data1, f1 = self.dataprocess(ptrain , index = 0)
        data2, f2 = self.dataprocess(ptest , index = 0 , test =1)
        
        num_train = int(data1.shape[0] * 0.8)
        train = data1[:num_train]
        test = data1[num_train:]
        train.to_csv("train_no.csv")
        test.to_csv("test_no.csv")
        data2.to_csv("test2_no.csv")
        

        '''
        ptrain = pd.read_csv('train_new.csv')
        ptest = pd.read_csv('test_new.csv')
        data1, f1 = self.dataprocess(ptrain , index = 1)
        data2, f2 = self.dataprocess(ptest , index = 1,test = 1)

        num_train = int(data1.shape[0] * 0.8)
        train = data1[:num_train]
        test = data1[num_train:]
        train.to_csv("train.csv")
        test.to_csv("test.csv")
        data2.to_csv("test2.csv")

    def dataprocess(self, data , index , test = 0):
        #ptrain = pd.read_csv('train_new.csv')
        #ptest = pd.read_csv('test_new.csv')


        data.dropna(how='all', inplace=True)
        #data.dropna(subset=['Y'], inplace=True)
        data.dropna(thresh=5, inplace=True)

        #处理空值
        missing_values_count = data.isnull().sum()
        n_attr = len(missing_values_count) - 1
        tot = len(data)
        for x in missing_values_count.keys():
            if missing_values_count[x]/tot>=0.75:
                data.drop([x], axis=1, inplace=True)
                print("drop ", x)
                n_attr -= 1

        attr_size = dict()
        for p in data.keys():
            attr_size[p] = len(data[p].value_counts())

        #data.dropna(subset=['X1'], inplace=True)
        #data.dropna(subset=['X2'], inplace=True)
        # 把数值型特征都放到随机森林里面,补全data

        # 查看箱图
        '''
        tm = pd.DataFrame(data)
        r,l =data.shape
        normolized_data = preprocessing.StandardScaler().fit_transform(tm)
        outliers_rows, outliers_columns = np.where(np.abs(normolized_data) > 3)
        re = np.zeros((l, 1))
        for i in outliers_columns:
            re[i] += 1
        '''

        # 填充空值
        missing_values_count = data.isnull().sum()
        for x in missing_values_count.keys():
            if x not in ['X1','X2','Y','id']:
                if x not in ['X3','X13','X46','X63','X65','X66']:
                    data[x].fillna(data[x].mean(), inplace=True)


        # 分析去除：
        '''
        analysis(data)
        '''
        delete = ['X5','X23','X35',
                  'X45','X51','X59','X62','X70','X72']
        for x in delete:
            if x in data.keys():
                data.drop([x], axis=1, inplace=True)
                print("drop ", x)
                n_attr -= 1

        attr_size = dict()
        for p in data.keys():
            attr_size[p] = len(data[p].value_counts())

        missing_values_count = data.isnull().sum()
        # 随机森林法
        attr1 = ['X1','X2','X3','X13','X46','X63','X65','X66']
        attr2 = []
        for x in missing_values_count.keys():
             if x not in ['X1','X2','X13','X63','X65','X66','Y', 'id']:
                    attr2.append(x)
        attr3 = ['X14','X25','X30','X32','X42', 'X49','X50','X52']
        for k in attr1:
            known = data[data[k].notnull()]
            unknown = data[data[k].isnull()]
            y = known[k]  # y是预测的
            x = known[attr3]  # x是特征属性值，后面几列
            if k in ['X1', 'X2']:
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y)
            rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
            # 根据已有数据去拟合随机森林模型
            rfr.fit(x, y)
            # 预测缺失值
            predict = rfr.predict(unknown[attr3])
            # 填补缺失值
            data.loc[data[k].isnull(), k] = predict

        missing_values_count = data.isnull().sum()
        #离散分箱
        #网龄：0-6，相当于离散化

        #特征值构造
        #将X1，X2交叉，由于都是地方属性
        def add_cross_feature(data, feature_1, feature_2):
            comb_index = data[[feature_1, feature_2]].drop_duplicates()
            comb_index[feature_1 + '_' + feature_2] = np.arange(comb_index.shape[0])
            data = pd.merge(data, comb_index, 'left', on=[feature_1, feature_2])
            return data
        data = add_cross_feature(data,'X1','X2')



        #特征选择
        # 查看方差，筛选特征

        '''varianceThreshold = VarianceThreshold(threshold=1)
        numeric_feats = list((data.dtypes[data.dtypes != 'object'].index)[:-1])
        numeric_feats.remove('Y')
        varianceThreshold.fit_transform(data[numeric_feats])
        varianceThreshold.get_support()'''

        data.drop(['X7'], axis=1, inplace=True)


        #回归系数
        selectKBest = SelectKBest(
            f_regression, k = 55
        )
        numeric_feats = list((data.dtypes[data.dtypes != 'object'].index)[:-1])
        if 'Y' in numeric_feats:
            numeric_feats.remove('Y')
            feature = data[numeric_feats]
            bestFeature = selectKBest.fit_transform(
                feature,
                data['Y']
            )
            self.feature = feature.columns[selectKBest.get_support()]

        for x in numeric_feats:
            if x not in self.feature:
                data.drop([x], axis=1, inplace=True)
                print("drop ", x)

        #特征抽取PCA
        '''numeric_feats = list((data.dtypes[data.dtypes != 'object'].index)[:-1])
        numeric_feats.remove('Y')
        pca = PCA (n_components = 26)
        data2 = pca.fit_transform(data[numeric_feats])'''


        numeric_feats = list((data.dtypes[data.dtypes != 'object'].index)[:-1])
        # 归一化 , id 为 1 归一化，0 则没有
        if 'X1' in numeric_feats:
            numeric_feats.remove('X1')
        if 'X2' in numeric_feats:
            numeric_feats.remove('X2')
        if 'Y' in numeric_feats:
            numeric_feats.remove('Y')
        if(index == 1):
            data[numeric_feats] = preprocessing.scale(data[numeric_feats])

        feature_size = len(numeric_feats) + 12
        return data,feature_size

if __name__ == "__main__":
    #数据预处理
    Data =  Dataprocess()


