import numpy as np
import torch
from torch.utils.data import Dataset


class DealDataset(Dataset):
	"""Download datasets, normalize data"""
	def __init__(self, data_dir):
		self.data_dir = data_dir
		data = open(self.data_dir, 'r')
		xLines = data.readlines()
		numberOfLines = len(xLines)
		train_dim = 4
		x_train = np.zeros((numberOfLines,train_dim))
		index = 0	
		for line in xLines:
			line = line.strip()
			# #line=line.strip("\n")
			line = line.split(' ')
			x_train[index, 0:4] = line[0:4]
			index += 1

		# Swap columns u and v
		x_train_temp = np.zeros((numberOfLines, 1))
		x_train_temp[:, 0] = x_train[:, 0]
		x_train[:, 0] = x_train[:, 1]
		x_train[:, 1] = x_train_temp[:, 0]

		numberOfLines = numberOfLines-10
		# print('numberOfLines:', numberOfLines)
		x_train_temp = np.zeros((numberOfLines, 4))
		for i in range(numberOfLines):
			x_train_temp[i, :] = x_train[i+10, :]
		x_train = np.zeros((numberOfLines, 4))
		x_train[:, :] = x_train_temp[:, :]
		x_train = torch.FloatTensor(x_train)

		# # Normalized data
		# print('x_train:',x_train.size())
		# # [-1，1]
		# data = torch.zeros(numberOfLines, train_dim)
		# for i in range(0, train_dim):
		# 	absmax_x_train = abs(x_train[:, i]).max()
		# 	normalize = x_train[:, i]/absmax_x_train
		# 	# print(normalize)
		# 	data[:, i] = normalize
		# 	# print('data[:,i]',data[:,i])
		# # print(data.size())
		# x_train = data

		x_train = x_train.T
		x_train = x_train.unsqueeze(0)
		train_size = x_train[0, 0, :].size()
		train_num = int(train_size[0]/10)

		data_temp = torch.empty(0, train_dim, 10)
		for i, data_i in enumerate(x_train.chunk(train_num, dim=2)):
			data_temp = torch.cat((data_temp, data_i), dim=0)  # data are connected through dimension dim=0
		
		x_train = data_temp
		self.x_train = x_train
		# print(x_train.size())
		self.train_num = train_num
		# print('x_train[0,:,:]',x_train[0,:,:])

	def __getitem__(self, index):
		return self.x_train[index, :, :]

	def __len__(self):
		return self.train_num





