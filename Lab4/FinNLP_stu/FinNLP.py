#encoding=utf-8
from FinNLP_stu.data import get_data
from FinNLP_stu.data import text_preprocess
from FinNLP_stu.classification import classification2
import sys

if __name__ == '__main__':
	token = 'a6dae538a760f0b9e39432c1bff5e50a1c462a1a087e994dae18fa04'
	N = 80 # number for filtering classes with more than N records
	merged_df = get_data(token, N)
	processed_df = text_preprocess(merged_df)
	results2 = classification2(processed_df)
	#results = classification(processed_df)
	#print(results)

