import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from deal_data import DealDataset
from scipy.stats.distributions import chi2
from scipy.stats import f


# The dataset
data_dir = 'real_data/shuiji_new_1.txt'
test_num = 1500

test_size = test_num * 10
test_time = test_size * 0.02
test_dim = 4


def draw_original_data():
	deal_dataset = DealDataset(data_dir)
	test_loader = DataLoader(dataset=deal_dataset, batch_size=1, shuffle=False)
	# print('test_loader:',len(test_loader))

	x_temp = torch.empty(0, test_dim, 10)
	for step, batch_data in enumerate(test_loader):
		x_temp = torch.cat((x_temp, batch_data), dim=0)
	x = x_temp
	print(x.size())
	x = x[0:test_num, :, :]

	# Collect data
	t = torch.arange(0, test_time, 0.02)
	x_0 = x[:, 0, :10]		# Steering  Angle
	x_0 = x_0.flatten(0)
	x_0 = x_0.data.numpy()

	x_1 = x[:, 1, :10]		# Vehicle  Speed
	x_1 = x_1.flatten(0)
	x_1 = x_1.data.numpy()

	x_2 = x[:, 2, :10]		# Lateral  Acceleration
	x_2 = x_2.flatten(0)
	x_2 = x_2.data.numpy()

	x_3 = x[:, 3, :10]		# Yaw  Rate
	x_3 = x_3.flatten(0)
	x_3 = x_3.data.numpy()

	# # Draw the steering angle and the vehicle speed collected by USBCAN-I/II+
	fig = plt.figure(figsize=(10, 7))
	ax1 = fig.add_subplot(2, 1, 1)
	ax1.plot(t, x_0, 'b', linewidth=0.9)
	plt.xlim(0, test_time)
	plt.ylim(-1, 1)
	plt.tick_params(labelsize=15)
	plt.xlabel(u"Time (s) \n (a)", fontproperties='Times New Roman', fontsize=20)
	plt.ylabel(u'Steering  Angle (rad)', fontproperties='Times New Roman', fontsize=20)
	plt.grid()

	ax1 = fig.add_subplot(2, 1, 2)
	ax1.plot(t, x_1, 'b', linewidth=0.5)
	plt.xlim(0, test_time)
	plt.ylim(0, 2)
	plt.tick_params(labelsize=15)
	plt.xlabel(u"Time (s) \n (b)", fontproperties='Times New Roman', fontsize=20)
	plt.ylabel(u'Vehicle  Speed (m/s)', fontproperties='Times New Roman', fontsize=20)
	plt.grid()

	fig.tight_layout()
	plt.subplots_adjust(left=0.2, bottom=0.15, right=0.8, top=0.9, hspace=0.5, wspace=0.2)
	plt.savefig('images/Steering_angle_and_vehicle_speed.jpg', dpi=600)

	# Draw the lateral acceleration and the yaw rate collected by EPSON-G320
	fig = plt.figure(figsize=(10, 7))
	ax1 = fig.add_subplot(2, 1, 1)
	ax1.plot(t, x_2, 'b', linewidth=0.9)
	plt.xlim(0, test_time)
	plt.ylim(-2, 2)
	plt.tick_params(labelsize=12)
	plt.xlabel(u"Time (s) \n (a)", fontproperties='Times New Roman', fontsize=20)
	plt.ylabel(u'Lateral  Acceleration (m/s^2)', fontproperties='Times New Roman', fontsize=20)
	plt.grid()

	ax1 = fig.add_subplot(2, 1, 2)
	ax1.plot(t, x_3, 'b', linewidth=0.9)
	plt.xlim(0, test_time)
	plt.ylim(-0.5, 0.5)
	plt.tick_params(labelsize=12)
	plt.xlabel(u"Time (s) \n (b)", fontproperties='Times New Roman', fontsize=20)
	plt.ylabel(u'Yaw  Rate (rad/s)', fontproperties='Times New Roman', fontsize=20)
	plt.grid()

	fig.tight_layout()
	plt.subplots_adjust(left=0.2, bottom=0.15, right=0.8, top=0.9, hspace=0.5)
	plt.savefig('images/Lateral_acceleration_and_yaw_rate.jpg', dpi=600)

	plt.show()


if __name__ == '__main__':
	draw_original_data()
