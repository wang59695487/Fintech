#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 23:13:13 2016
@author: Jared
"""
#%%
import pandas as pd
import numpy as np
import random
import matplotlib.pylab as plt
from matplotlib.ticker import MaxNLocator
import pylab as p
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold



import warnings
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None  # default='warn'

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    from sklearn import model_selection
    import xgboost as xgb

#import xgboost as xgb
import operator
import timeit
import scipy.stats as stats



tic0=timeit.default_timer()

#目标列和信息列
target_cols = ['ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1',
               'ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1',
               'ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1',
               'ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1',
               'ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1',
               'ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1',
               'ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1',
               'ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']

info_cols = ['fecha_dato', 'ncodpers', 'ind_empleado', 'pais_residencia', 'sexo', 'age', 'fecha_alta',
             'ind_nuevo', 'antiguedad', 'indrel', 'ult_fec_cli_1t', 'indrel_1mes', 'tiprel_1mes',
             'indresi', 'indext', 'conyuemp', 'canal_entrada', 'indfall', 'tipodom', 'cod_prov',
             'nomprov', 'ind_actividad_cliente', 'renta', 'segmento']

#数据类型转换
dtype_dict = {'ind_ahor_fin_ult1':np.int8,'ind_aval_fin_ult1':np.int8,'ind_cco_fin_ult1':np.int8,
               'ind_cder_fin_ult1':np.int8,'ind_cno_fin_ult1':np.int8,'ind_ctju_fin_ult1':np.int8,
               'ind_ctma_fin_ult1':np.int8,'ind_ctop_fin_ult1':np.int8,'ind_ctpp_fin_ult1':np.int8,
               'ind_deco_fin_ult1':np.int8,'ind_deme_fin_ult1':np.int8,'ind_dela_fin_ult1':np.int8,
               'ind_ecue_fin_ult1':np.int8,'ind_fond_fin_ult1':np.int8,'ind_hip_fin_ult1':np.int8,
               'ind_plan_fin_ult1':np.int8,'ind_pres_fin_ult1':np.int8,'ind_reca_fin_ult1':np.int8,
               'ind_tjcr_fin_ult1':np.int8,'ind_valo_fin_ult1':np.int8,'ind_viv_fin_ult1':np.int8,
               'ind_nomina_ult1':np.int8,'ind_nom_pens_ult1':np.int8,'ind_recibo_ult1':np.int8}


#对加入的渠道进行编码
canal_dict = {'KAI': 35,'KBG': 17,'KGU': 149,'KDE': 47,'KAJ': 41,'KCG': 59,
 'KHM': 12,'KAL': 74,'KFH': 140,'KCT': 112,'KBJ': 133,'KBL': 88,'KHQ': 157,'KFB': 146,'KFV': 48,'KFC': 4,
 'KCK': 52,'KAN': 110,'KES': 68,'KCB': 78,'KBS': 118,'KDP': 103,'KDD': 113,'KBX': 116,'KCM': 82,
 'KAE': 30,'KAB': 28,'KFG': 27,'KDA': 63,'KBV': 100,'KBD': 109,'KBW': 114,'KGN': 11,
 'KCP': 129,'KAK': 51,'KAR': 32,'KHK': 10,'KDS': 124,'KEY': 93,'KFU': 36,'KBY': 111,
 'KEK': 145,'KCX': 120,'KDQ': 80,'K00': 50,'KCC': 29,'KCN': 81,'KDZ': 99,'KDR': 56,
 'KBE': 119,'KFN': 42,'KEC': 66,'KDM': 130,'KBP': 121,'KAU': 142,'KDU': 79,
 'KCH': 84,'KHF': 19,'KCR': 153,'KBH': 90,'KEA': 89,'KEM': 155,'KGY': 44,'KBM': 135,
 'KEW': 98,'KDB': 117,'KHD': 2,'RED': 8,'KBN': 122,'KDY': 61,'KDI': 150,'KEU': 72,
 'KCA': 73,'KAH': 31,'KAO': 94,'KAZ': 7,'004': 83,'KEJ': 95,'KBQ': 62,'KEZ': 108,
 'KCI': 65,'KGW': 147,'KFJ': 33,'KCF': 105,'KFT': 92,'KED': 143,'KAT': 5,'KDL': 158,
 'KFA': 3,'KCO': 104,'KEO': 96,'KBZ': 67,'KHA': 22,'KDX': 69,'KDO': 60,'KAF': 23,'KAW': 76,
 'KAG': 26,'KAM': 107,'KEL': 125,'KEH': 15,'KAQ': 37,'KFD': 25,'KEQ': 138,'KEN': 137,
 'KFS': 38,'KBB': 131,'KCE': 86,'KAP': 46,'KAC': 57,'KBO': 64,'KHR': 161,'KFF': 45,
 'KEE': 152,'KHL': 0,'007': 71,'KDG': 126,'025': 159,'KGX': 24,'KEI': 97,'KBF': 102,
 'KEG': 136,'KFP': 40,'KDF': 127,'KCJ': 156,'KFR': 144,'KDW': 132,-1: 6,'KAD': 16,
 'KBU': 55,'KCU': 115,'KAA': 39,'KEF': 128,'KAY': 54,'KGC': 18,'KAV': 139,'KDN': 151,
 'KCV': 106,'KCL': 53,'013': 49,'KDV': 91,'KFE': 148,'KCQ': 154,'KDH': 14,'KHN': 21,
 'KDT': 58,'KBR': 101,'KEB': 123,'KAS': 70,'KCD': 85,'KFL': 34,'KCS': 77,'KHO': 13,
 'KEV': 87,'KHE': 1,'KHC': 9,'KFK': 20,'KDC': 75,'KFM': 141,'KHP': 160,'KHS': 162,
 'KFI': 134,'KGV': 43}

#对所在的地区进行编码
pais_dict = {'LV': 102,'CA': 2,'GB': 9,'EC': 19,'BY': 64,'ML': 104,'MT': 118,
 'LU': 59,'GR': 39,'NI': 33,'BZ': 113,'QA': 58,'DE': 10,'AU': 63,'IN': 31,
 'GN': 98,'KE': 65,'HN': 22,'JM': 116,'SV': 53,'TH': 79,'IE': 5,'TN': 85,
 'PH': 91,'ET': 54,'AR': 13,'KR': 87,'GA': 45,'FR': 8,'SG': 66,'LB': 81,
 'MA': 38,'NZ': 93,'SK': 69,'CN': 28,'GI': 96,'PY': 51,'SA': 56,'PL': 30,
 'PE': 20,'GE': 78,'HR': 67,'CD': 112,'MM': 94,'MR': 48,'NG': 83,'HU': 106,
 'AO': 71,'NL': 7,'GM': 110,'DJ': 115,'ZA': 75,'OM': 100,'LT': 103,'MZ': 27,
 'VE': 14,'EE': 52,'CF': 109,'CL': 4,'SL': 97,'DO': 11,'PT': 26,'ES': 0,
 'CZ': 36,'AD': 35,'RO': 41,'TW': 29,'BA': 61,'IS': 107,'AT': 6,'ZW': 114,
 'TR': 70,'CO': 21,'PK': 84,'SE': 24,'AL': 25,'CU': 72,'UY': 77,'EG': 74,'CR': 32,
 'GQ': 73,'MK': 105,'KW': 92,'GT': 44,'CM': 55,'SN': 47,'KZ': 111,'DK': 76,
 'LY': 108,'AE': 37,'PA': 60,'UA': 49,'GW': 99,'TG': 86,'MX': 16,'KH': 95,
 'FI': 23,'NO': 46,'IT': 18,'GH': 88, 'JP': 82,'RU': 43,'PR': 40,'RS': 89,
 'DZ': 80,'MD': 68,-1: 1,'BG': 50,'CI': 57,'IL': 42,'VN': 90,'CH': 3,'US': 15,'HK': 34,
 'CG': 101,'BO': 62,'BR': 17,'BE': 12,'BM': 117}
max_val = 0
for key,item in pais_dict.items():
    if item > max_val:
        max_val = item

emp_dict = {'N':0,-1:-1,'A':1,'B':2,'F':3,'S':4}
indfall_dict = {'N':0,-1:-1,'S':1}
sexo_dict = {'V':0,'H':1,-1:-1}
tiprel_dict = {'A':0,-1:-1,'I':1,'P':2,'N':3,'R':4}
indresi_dict = {'N':0,-1:-1,'S':1}
indext_dict = {'N':0,-1:-1,'S':1}
conyuemp_dict = {'N':0,-1:-1,'S':1}
segmento_dict = {-1:-1,'01 - TOP':1,'02 - PARTICULARES':2,'03 - UNIVERSITARIO':3}


tic=timeit.default_timer()
def preprocess(DF,is_DF=True):
    DF.replace(' NA', -1, inplace=True)
    DF.replace('         NA', -1, inplace=True)
    DF.fillna(-1, inplace=True)
    DF['ncodpers'] = DF['ncodpers'].astype(np.int32)
    DF['renta'] = DF['renta'].astype(np.float64)
    DF['renta'] = DF['renta'].astype(np.int64)
    DF['indrel'] = DF['indrel'].map(lambda x: 2 if x == 99 else x).astype(np.int8)
    DF['ind_empleado'] = DF['ind_empleado'].map(lambda x: emp_dict[x]).astype(np.int8)
    DF['sexo'] = DF['sexo'].map(lambda x: sexo_dict[x]).astype(np.int8)
    DF['age'] = DF['age'].astype(np.int16)
    DF['ind_nuevo'] = DF['ind_nuevo'].astype(np.int8)
    DF['antiguedad'] = DF['antiguedad'].map(lambda x: -1 if x == '     NA' else x).astype(int)
    DF['antiguedad'] = DF['antiguedad'].map(lambda x: -1 if x == -999999 else x).astype(np.int16)
    DF['indrel_1mes'] = DF['indrel_1mes'].map(lambda x: -2 if x == 'P' else x).astype(np.float16)
    DF['indrel_1mes'] = DF['indrel_1mes'].astype(np.int8)
    DF['tiprel_1mes'] = DF['tiprel_1mes'].map(lambda x: tiprel_dict[x]).astype(np.int8)
    DF['indresi'] = DF['indresi'].map(lambda x: indresi_dict[x]).astype(np.int8)
    DF['indext'] = DF['indext'].map(lambda x: indext_dict[x]).astype(np.int8)
    DF['conyuemp'] = DF['conyuemp'].map(lambda x: conyuemp_dict[x]).astype(np.int8)
    DF['canal_entrada'] = DF['canal_entrada'].map(lambda x: canal_dict[x]).astype(np.int16)
    DF['indfall'] = DF['indfall'].map(lambda x: indfall_dict[x]).astype(np.int8)
    DF['pais_residencia'] = DF['pais_residencia'].map(lambda x: pais_dict[x]).astype(np.int8)
    DF['tipodom'] = DF['tipodom'].astype(np.int8)
    DF['cod_prov'] = DF['cod_prov'].astype(np.int8)

    DF.drop('nomprov',axis=1,inplace=True)

    DF['ind_actividad_cliente'] = DF['ind_actividad_cliente'].astype(np.int8)
    DF['fecha_dato_month'] = DF['fecha_dato'].map(lambda x: int(x[5:7])).astype(np.int8)
    DF['fecha_dato_year'] = DF['fecha_dato'].map(lambda x: int(x[0:4]) - 2015).astype(np.int8)
    DF['month_int'] = (DF['fecha_dato_month'] + 12 * DF['fecha_dato_year']).astype(np.int8)
    DF.drop('fecha_dato',axis=1,inplace=True)
    DF['fecha_alta'] = DF['fecha_alta'].map(lambda x: '2020-01-01' if x == -1 else x)
    DF['fecha_alta_month'] = DF['fecha_alta'].map(lambda x: int(x[5:7])).astype(np.int16)
    DF['fecha_alta_year'] = DF['fecha_alta'].map(lambda x: int(x[0:4]) - 1995).astype(np.int16)
    DF['fecha_alta_day'] = DF['fecha_alta'].map(lambda x: int(x[8:10])).astype(np.int16)
    DF['fecha_alta_month_int'] = (DF['fecha_alta_month'] + 12 * DF['fecha_alta_year']).astype(np.int16)
    DF['fecha_alta_day_int'] = (DF['fecha_alta_day'] + 30 * DF['fecha_alta_month'] + 365 * DF['fecha_alta_year']).astype(np.int32)
    DF.drop('fecha_alta',axis=1,inplace=True)
    DF['ult_fec_cli_1t'] = DF['ult_fec_cli_1t'].map(lambda x: '2020-01-01' if x == -1 else x)
    DF['ult_fec_cli_1t_month'] = DF['ult_fec_cli_1t'].map(lambda x: int(x[5:7])).astype(np.int16)
    DF['ult_fec_cli_1t_year'] = DF['ult_fec_cli_1t'].map(lambda x: int(x[0:4]) - 2015).astype(np.int16)
    DF['ult_fec_cli_1t_day'] = DF['ult_fec_cli_1t'].map(lambda x: int(x[8:10])).astype(np.int16)
    DF['ult_fec_cli_1t_month_int'] = (DF['ult_fec_cli_1t_month'] + 12 * DF['ult_fec_cli_1t_year']).astype(np.int8)
    DF.drop('ult_fec_cli_1t',axis=1,inplace=True)

    DF['segmento'] = DF['segmento'].map(lambda x: segmento_dict[x]).astype(np.int8)

    for col in target_cols:
        if is_DF:
            DF[col] = DF[col].astype(np.int8)

    return DF

#reader = pd.read_csv('./train_ver2.csv', chunksize=100000, usecols = ['ind_actividad_cliente'], header=0)

reader = pd.read_csv('./train_ver2.csv', chunksize=100000, header=0)
train = pd.concat([preprocess(chunk) for chunk in reader])

#DF = pd.read_csv('./train_ver2.csv', header=0,nrows=5000000)
#DF = pd.read_csv('./train_ver2.csv',nrows=2000000)
#DF.replace(' NA', -1, inplace=True)
#DF.fillna(-1, inplace=True)
#DF['ncodpers'] = DF['ncodpers'].astype(np.int32)

reader_2 = pd.read_csv('./test_ver2.csv', chunksize=10000, header=0)
test = pd.concat([preprocess(chunk,is_DF=False) for chunk in reader_2])
toc=timeit.default_timer()
print('Load Time',toc - tic)
#%%
train.to_csv('train_hash.csv', index=False)
test.to_csv('test_hash.csv', index=False)
'''
raw = reader.loc[reader['fecha_dato'] == '2016-05-28',:]
raw.to_csv('raw.csv', index=False)
'''
toc=timeit.default_timer()
print('Total Time',toc - tic0)

'''
reader = pd.read_csv('./train_ver2.csv', chunksize=10000, header=0)
DF = pd.concat([resize_data(chunk) for chunk in reader])
reader = pd.read_csv('./test_ver2.csv', chunksize=10000, header=0)
test = pd.concat([resize_data(chunk,is_DF=False) for chunk in reader])

DF = pd.read_csv('./train_ver2.csv', dtype = dtype_dict, header=0)
toc=timeit.default_timer()
print('Load Time',toc - tic)
'''

#for col in target_cols:
#    DF[col] = DF[col].astype(np.int8)
#DF['ncodpers'] = DF['ncodpers'].astype(np.int32)

#def convert_strings_to_ints(input_df,col_name,output_col_name,do_sort=True):
#
##    input_df.sort_values(col_name,inplace=True)
#    labels, levels = pd.factorize(input_df[col_name],sort = do_sort)
#    input_df[output_col_name] = labels
#    input_df[output_col_name] = input_df[output_col_name].astype(int)
#    output_dict = dict(zip(input_df[col_name],input_df[output_col_name]))
#    return (output_dict,input_df)
#%%
#(canal_dict,train) = convert_strings_to_ints(train,'canal_entrada','canal_entrada_hash',do_sort=False)
#%%
#(pais_dict,train) = convert_strings_to_ints(train,'pais_residencia','pais_residencia_hash',do_sort=False)
#%%
#DF.drop('pais_residencia',axis=1,inplace=True)
##%%

#DF['pais_hash'] = DF['pais_hash'].astype(np.int8)

'''
DF['renta'] = DF['renta'].astype(np.float32)
may want to verify string vs ints at some point
DF['fecha_dato'] = pd.to_datetime(DF['fecha_dato'])
'''
#test_samp = test.sample(frac = 0.001, random_state = 111)
'''
tic=timeit.default_timer()
GROUPS = []
for k,g in DF.groupby(np.arange(len(DF))//10000):
     g['ncodpers'] = g['ncodpers'].astype(np.int32)
     g['ind_empleado'] = g['ind_empleado'].map(lambda x: emp_dict[x])
     GROUPS.append(g)
DF = pd.concat(GROUPS,ignore_index=True)
toc=timeit.default_timer()
print('Preprocessing Time',toc - tic)
train.to_csv('train_hash.csv', index=False)
test.to_csv('test_hash.csv', index=False)

toc=timeit.default_timer()
print('Total Time',toc - tic0)
'''
'''
target_cols = ['ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1',
               'ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1',
               'ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1',
               'ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1',
               'ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1',
               'ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1',
               'ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1',
               'ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']
#%%
dtype_dict = {'ind_ahor_fin_ult1':np.int8,'ind_aval_fin_ult1':np.int8,'ind_cco_fin_ult1':np.int8,
               'ind_cder_fin_ult1':np.int8,'ind_cno_fin_ult1':np.int8,'ind_ctju_fin_ult1':np.int8,
               'ind_ctma_fin_ult1':np.int8,'ind_ctop_fin_ult1':np.int8,'ind_ctpp_fin_ult1':np.int8,
               'ind_deco_fin_ult1':np.int8,'ind_deme_fin_ult1':np.int8,'ind_dela_fin_ult1':np.int8,
               'ind_ecue_fin_ult1':np.int8,'ind_fond_fin_ult1':np.int8,'ind_hip_fin_ult1':np.int8,
               'ind_plan_fin_ult1':np.int8,'ind_pres_fin_ult1':np.int8,'ind_reca_fin_ult1':np.int8,
               'ind_tjcr_fin_ult1':np.int8,'ind_valo_fin_ult1':np.int8,'ind_viv_fin_ult1':np.int8,
               'ind_nomina_ult1':np.int8,'ind_nom_pens_ult1':np.int8,'ind_recibo_ult1':np.int8}

def analysis(data):

    target_cols = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
                   'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
                   'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
                   'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
                   'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
                   'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
                   'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
                   'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']
    # %%
    dtype_dict = {'ind_ahor_fin_ult1': np.int8, 'ind_aval_fin_ult1': np.int8, 'ind_cco_fin_ult1': np.int8,
                  'ind_cder_fin_ult1': np.int8, 'ind_cno_fin_ult1': np.int8, 'ind_ctju_fin_ult1': np.int8,
                  'ind_ctma_fin_ult1': np.int8, 'ind_ctop_fin_ult1': np.int8, 'ind_ctpp_fin_ult1': np.int8,
                  'ind_deco_fin_ult1': np.int8, 'ind_deme_fin_ult1': np.int8, 'ind_dela_fin_ult1': np.int8,
                  'ind_ecue_fin_ult1': np.int8, 'ind_fond_fin_ult1': np.int8, 'ind_hip_fin_ult1': np.int8,
                  'ind_plan_fin_ult1': np.int8, 'ind_pres_fin_ult1': np.int8, 'ind_reca_fin_ult1': np.int8,
                  'ind_tjcr_fin_ult1': np.int8, 'ind_valo_fin_ult1': np.int8, 'ind_viv_fin_ult1': np.int8,
                  'ind_nomina_ult1': np.int8, 'ind_nom_pens_ult1': np.int8, 'ind_recibo_ult1': np.int8}
    data.replace(to_replace='unknown', value=np.NaN, inplace=True)
    missing_values_count = data.isnull().sum()
    info_cols = ['fecha_dato', 'ncodpers', 'ind_empleado', 'pais_residencia', 'sexo', 'age', 'fecha_alta',
                  'ind_nuevo', 'antiguedad', 'indrel', 'ult_fec_cli_1t', 'indrel_1mes', 'tiprel_1mes',
                  'indresi', 'indext', 'conyuemp', 'canal_entrada', 'indfall', 'tipodom', 'cod_prov',
                  'nomprov', 'ind_actividad_cliente', 'renta', 'segmento']
    values = []
    for k in info_cols:
        values.append(missing_values_count[k])
    y_pos = np.arange(len(info_cols))
    # 缺失的图
    plt.barh(y_pos ,values, align='center', alpha=0.4)
    plt.yticks(y_pos, info_cols)
    plt.xlabel('unknown')
    plt.xlim(0, 2000000)
    plt.title('The unknown in the features')
    plt.show()

    values = []
    for k in target_cols:
        values.append(missing_values_count[k])
    y_pos = np.arange(len(target_cols))
    # 缺失的图
    plt.barh(y_pos ,values, align='center', alpha=0.4)
    plt.yticks(y_pos, target_cols)
    plt.xlabel('unknown')
    plt.xlim(0, 20000)
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
    # data = pd.read_csv('train_ver2.csv', sep=',')
    # data2 = pd.read_csv('test_ver2.csv', sep=',')
    #raw = data.loc[data['fecha_dato'] == '2016-05-28',:]
    x1 = data.loc[:, data.columns == 'ind_ahor_fin_ult1']
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
'''