import tushare as ts
import pandas as pd
import numpy as np
import jieba
import jieba.analyse

def get_data(token, N):
	# get_data
	pro = ts.pro_api(token)
	pd.set_option('max_colwidth',120)
	df0 = pro.stock_company(exchange='SZSE', fields='ts_code,business_scope')
	df1 = df0.dropna(axis=0, how='any')
	df2 = pro.stock_basic(exchange='', list_status='L', fields='ts_code,industry')
	df3 = df2.dropna(axis=0, how='any')

	# merge
	df = pd.merge(df1,df3, on='ts_code', how='right')
	# filter by number of records
	nonan_df = df.dropna(axis=0, how='any')
	vc  = nonan_df['industry'].value_counts()
	pat = r'|'.join(vc[vc>N].index)          
	merged_df  = nonan_df[nonan_df['industry'].str.contains(pat)]
	
	return merged_df

def text_preprocess(merged_df):
	# word segmentation + extract keywords (using jieba)
	# 启动paddle模式。

    result = []
	######## build lable lsit ######
    vc = merged_df['industry'].value_counts()
    lable_list = vc.to_dict()
    lable_set = lable_list.keys()

    cnt = 0
    for i in lable_set:
       lable_list[i] = cnt
       cnt += 1
    #####################################

	######### Segment + Key + Voc ########
    field=merged_df['business_scope']
    field_list=field.values
    GT=merged_df['industry'].values
    size=len(GT)
    GT_Vector=[]

    for i in range(size):
         gid = lable_list[GT[i]]
         gvec = np.zeros(cnt)
         gvec[gid] = 1
         GT_Vector.append(gvec)

    voc={}
    Seg=[]
    Key=[]
    voc_cnt=0

    for i in range(size):
         scope_str=field_list[i]
         Seg.append((','.join(jieba.cut(scope_str))).split(','))
         for word in Seg[i]:
            if word not in voc:
                   voc[word]=voc_cnt
                   voc_cnt+=1
         Key.append(jieba.analyse.extract_tags(scope_str))

    ##################################
	############## IDF ################
    IDF_Seg = {}
    IDF_Key = {}
    for i in range(size):
        temp = dict()
        for word in Seg[i]:
           temp[word] = 1
        for word in temp:
           if word not in IDF_Seg:
               IDF_Seg[word] = 1
           else:IDF_Seg[word] += 1
    temp = dict()
    for word in Key[i]:
        temp[word] = 1
    for word in temp:
        if word not in IDF_Key:
            IDF_Seg[word] = 1
        else:
            IDF_Seg[word] += 1
    ##################################################

    ######### feature-vec #############
    for i in range(size):
       feature_seg = np.zeros(voc_cnt)
       feature_key = np.zeros(voc_cnt)
       # print(voc_cnt)
       for word in Seg[i]:
            wid = voc[word]
            feature_seg[wid] += 1
       feature_seg/=len(Seg[i])
       for word in Key[i]:
            wid = voc[word]
            feature_key[wid] += 1
       feature_key/=len(Key[i])
       for word in voc:
           tid = voc[word]
           if word in IDF_Key:
              feature_key[tid] *= IDF_Key[word]
           if word in IDF_Seg:
               feature_seg[tid] *= IDF_Seg[word]
       feature = np.concatenate((feature_key, feature_seg))
       result.append([feature,GT_Vector[i]])

    return result