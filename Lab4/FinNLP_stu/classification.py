import sklearn
import torch
import torch.nn.functional as F
import math as math
import numpy as np
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

class Perceptron(torch.nn.Module):
     def __init__(self, n_feature, n_output):
        super(Perceptron, self).__init__()
        tn = n_feature
        self.hidden_1 = torch.nn.Linear(tn, tn // 4)
        tn //= 4
        self.hidden_2 = torch.nn.Linear(tn, tn // 4)
        tn //= 4
        self.hidden_3 = torch.nn.Linear(tn, tn // 4)
        tn //= 4
        self.predict = torch.nn.Linear(tn, n_output)

     def forward(self,x):
         x = F.relu(self.hidden_1(x))
         x = F.relu(self.hidden_2(x))
         x = F.relu(self.hidden_3(x))
         x = self.predict(x)
         return x





def classification2(processed_df):
    learning_rate = 0.1
    data_num = len(processed_df)
    input_size = len(processed_df[0][0])
    output_size = len(processed_df[0][1])
    myNet = Perceptron(input_size,output_size)

    n_train=math.ceil(data_num*(5/7))
    n_test = data_num - n_train
    print("# Traning Sample : ", n_train)
    print("# Test Sample : ", n_test)

    sample_train=processed_df[:n_train]
    sample_test=processed_df[n_train:]

    opt_SGD= torch.optim.SGD(myNet.parameters(), lr=learning_rate)
    loss_f =F.cross_entropy

    batch_size= 5
    batch_data=[]
    GT_data=[]
    batch_num = n_train//batch_size
    pos=0

    for i in range(batch_num):
        data = sample_train[pos:pos + batch_size]
        temp_feature=[]
        temp_gt=[]
        for j in range(batch_size):
            temp_feature.append(data[j][0])
            temp_gt.append(data[j][1])
        batch_data.append(torch.Tensor(temp_feature))
        GT=temp_gt
        L=[]
        for j in range(batch_size):
           for k in range(output_size):
                if GT[j][k] == 1:
                    L.append(int(k))
                    break
        GT_data.append(torch.LongTensor(L))
        pos+=batch_size

    if pos<n_train:
        batch_num += 1
        res_size = n_train - pos
        data = sample_train[pos:n_train]
        data = np.array(data)
        batch_data.append(torch.Tensor(data[:, 0]))
        GT = data[:, 1]
        L=[]
        for j in range(res_size):
           for k in range(output_size):
               if GT[j][k] == 1:
                    L.append(int(k))
                    break
        GT_data.append(torch.LongTensor(L))

        n_epoch = 100
        for i in range(n_epoch):
            for j in range(batch_num):
                y = myNet(batch_data[j])
                loss = loss_f(y, GT_data[j])
                opt_SGD.zero_grad()
                loss.backward()
                opt_SGD.step()
                print("[Epoch] : %d/%d , [Batch] : %d/%d , Loss = %f" % (i,n_epoch,j,batch_num,loss))

    torch.save(myNet, 'net.pkl')

    ##测试
    cnt_pos_class = np.zeros(output_size)
    cnt_neg_class = np.zeros(output_size)
    cnt_class = np.zeros(output_size)
    pos = 0

    for i in range(n_test):
        data = sample_train[i]
        temp_feature = data[0]
        temp_gt = data[1]
        y = torch.argmax(myNet(torch.Tensor(temp_feature)))
        for j in range(output_size):
             if temp_gt[j] == 1:
                 GTL = j
                 break
        cnt_class[GTL]+=1
        if(y==GTL):
            pos += 1
            cnt_pos_class[y] += 1
        else:
            cnt_neg_class[y] += 1
    print("####### Display Result ########")
    for i in range(output_size):
        per = cnt_pos_class[i] / (cnt_pos_class[i] + cnt_neg_class[i])
        recall = cnt_pos_class[i] / cnt_class[i]
        F1 = 2 * per * recall / (per + recall)
        print("# class %d : percission=%f , recall=%f, f1=%f" % (i,per,recall,F1) )
    print("Global Percision: ", pos / n_test)



def classification(processed_df):
	# split into train and test sets
	kf = KFold(n_splits = 10, shuffle = True, random_state = 2)
	split_result = next(kf.split(processed_df), None)
	# print(split_result)
	train = processed_df.iloc[split_result[0]]
	test = processed_df.iloc[split_result[1]]
	Y = train['industry']
	# TF-IDF
	X_train_tf, X_test_tf = TF_IDF(train, test)

	# classification
	# Your code here
    # Answer begin

	mnb = MultinomialNB()
	mnb.fit(X_train_tf.toarray(), Y)
	# print('class count', mnb.class_count_)

	print('The accuracy of Naive Bayes Classifier is', mnb.score(X_test_tf, test['industry']))



	# Answer end
	return mnb

def TF_IDF(train, test):
	# Your code here
    # Answer begin
    # print(train['pre_data'])
	vectorizer= CountVectorizer()
	# print(type(train['pre_data']))
	# X = vectorizer.fit_transform(train['business_scope'].tolist())
	# print(X)
	transformer = TfidfTransformer()
	X_train_tf = transformer.fit_transform(vectorizer.fit_transform((train['business_scope'].tolist())))
	# print(X_train_tf)
	X_test_tf = transformer.fit_transform(vectorizer.fit_transform((test['business_scope'].tolist())))


	# Answer end
	return X_train_tf, X_test_tf


