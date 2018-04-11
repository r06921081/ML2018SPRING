import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torchvision

# Download MNIST from 
train_dataset = datasets.MNIST(root="./data",
							   train=True,
							   transform=transforms.ToTensor(),
							   download=True)

test_dataset = datasets.MNIST(root="./data",
							  train=False,
							  transform=transforms.ToTensor(),
							  download=True)

print("train_dataset train_data :", train_dataset.train_data.size())
print("train_dataset train_labels:", train_dataset.train_labels.size())
print("test_dataset test_data :", test_dataset.test_data.size())
print("test_dataset test_labels :", test_dataset.test_labels.size())

batch_size = 100
n_iters = 3000
num_epochs = n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
	                                       batch_size=batch_size,
	                                       shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
	                                      batch_size=batch_size,
	                                      shuffle=True)
test_data = torchvision.datasets.MNIST(root='./data', train=False)
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1)).type(torch.FloatTensor)[:2000].cuda()/255.   # Tensor on GPU
test_y = test_data.test_labels[:2000].cuda()
class CNNModel(nn.Module):
	def __init__(self):
		super(CNNModel, self).__init__()
		# conv 1                      # (1,28,28)
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
		self.relu1 = nn.ReLU()        # (16,28,28)  28+4-5+1
		# max 1
		self.maxpool1 = nn.MaxPool2d(kernel_size=2)
                                      # (16,14,14)
		# conv 2
		self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
		self.relu2 = nn.ReLU()        # (32,14,14) 14+4-5+1
		# max 2
		self.maxpool2 = nn.MaxPool2d(kernel_size=2)
		                              # (32, 7, 7)
		# fully connected layer
		self.fc1 = nn.Linear(32 * 7 * 7, 10)

	def forward(self, x):
		# conv 1
		x = self.conv1(x)
		x = self.relu1(x)
		x = self.maxpool1(x)
		# conv 2
		x = self.conv2(x)
		x = self.relu2(x)
		x = self.maxpool2(x)

		# flatten
		x = x.view(x.size(0), -1)
		out = self.fc1(x)

		return out

model = CNNModel()
model.cuda()
loss_func = nn.CrossEntropyLoss()
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

iters = 0
for epoch in range(num_epochs):
	for i, (images, labels) in enumerate(train_loader):
		print(type(train_loader))
		exit()
		# convert tensor to Variable
		images = Variable(images).cuda()
		labels = Variable(labels).cuda()

		# clear gradients w.r.t parameters
		optimizer.zero_grad()

		# forward pass
		outputs = model(images)

		# calculate loss
		loss = loss_func(outputs, labels)

		# get gradient w.r.t parameters
		loss.backward()

		# update parameters
		optimizer.step()

		iters += 1

		if iters % 500 == 0:
			test_output = model(test_x)
			# !!!!!!!! Change in here !!!!!!!!! #
			pred_y = torch.max(test_output, 1)[1].cuda().data.squeeze()  # move the computation in GPU

			accuracy = torch.sum(pred_y == test_y) / test_y.size(0)
			print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0], '| test accuracy: %.2f' % accuracy)



# get filters from parameters:
print(list(model.parameters())[0].size())    # (16, 1, 5, 5)
model.cpu()
fig = plt.figure()
axes = []
for i in range(4):
	for j in range(4):
		ax = plt.subplot2grid((4,4), (i,j), colspan=1, rowspan=1)
		axes.append(ax)

for i in range(16):
	im = axes[i].imshow(list(model.parameters())[0].data.numpy()[i,0,:,:], cmap="hot")
plt.colorbar(im)
plt.show()
