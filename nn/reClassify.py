# -*- coding:utf-8 -*-
# author: LXY
# 20180606
import os
import time

import nn.train as train
from new_conf import My_config


class ClassifyHandler(object):
	"""
	the 
	"""
	def __init__(self, ini_path):
		"""
		ini_path: the path of config file, scene classify
		"""
		# self.ini_path = ini_path
		# self.conf = configparser.ConfigParser()
		self.conf = My_config(ini_path)
		# self.expMQ_path = self.conf.get("dataUpdate", "EXPMD_PATH") # 场景扩展问正排索引文件路径
		self.basic_name = self.conf.get("reClassify", "BASIC_NAME") #通用单轮分类模型名（应当为一个路径，不带数字，不需要更改）
		self.basic_num = int(str(self.conf.get("reClassify", "BASIC_NUM"))) #当前单轮分类模型的数字后缀，新训练模型后需要加一
		self.modular_name = self.conf.get("reClassify", "MODULAR_NAME") #通用场景分类模型名（应当为一个路径，不带数字，不需要更改）
		self.modular_num = int(str(self.conf.get("reClassify", "MODULAR_NUM"))) #当前场景分类模型的数字后缀，新训练模型后需要加一
		

	def updateModel(self, dirname, num):
		print("in updateModel")
		new_dir = dirname + str(num)
		print(new_dir)
		input_path = os.path.abspath(os.path.join(new_dir, "data"))
		print(input_path)
		startTime = time.time()
		new_model_name = os.path.abspath(os.path.join(new_dir, "model"))
		if not os.path.exists(input_path):
			print("The data is not ready for new classifier, and the new version shall be number " + str(num))
			return False
		elif os.path.exists(new_model_name):
			print("There already exist model " + new_model_name + ", there is no need to train or the num is wrong")
			return False
		# try:
		new_model = train.Train(input_path, new_model_name)
		endTime = time.time()
		print("It costs %d min to retrain new model." % ((endTime - startTime) / 60))
		return new_dir
		# except:
		# 	if os.path.exists(new_dir):
		# 		shutil.rmtree(new_dir)
		# 	print("Create new classifier model failed!")
		# 	return False


	def change_ini(self, change_list):
		for option, value in change_list:
			self.conf.set("reClassify", option, str(value))
		

if __name__ == "__main__":
	classifier = ClassifyHandler("./conf.txt")
	print(classifier.updateModel(classifier.modular_name, classifier.modular_num))
	# basic = classifier.updateModel(classifier.basic_name, classifier.basic_num)
	# if basic:
	# 	classifier.change_ini([["BASIC_NUM", classifier.basic_num + 1]])
	# modular = classifier.updateModel(classifier.modular_name, classifier.modular_num)
	# if modular:
	# 	classifier.change_ini([["MODULAR_NUM", classifier.modular_num + 1]])