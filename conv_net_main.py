import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import time
import os
import copy
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.datasets import CocoDetection
from tqdm import tqdm
from glove_setup import glove

class CifarDataset(torch.utils.data.Dataset):
	
	def __init__(self, root_dir, transform):
		"""Initializes a dataset containing images and labels."""
		super().__init__()
		self.transform = transform
		self.root_dir = root_dir
		classes = os.listdir(root_dir)
		self.img_names = [os.path.join(os.path.join(root_dir, classes[i]), img_name) for i in range(len(classes)) for img_name in os.listdir(os.path.join(root_dir, classes[i]))]
		self.label_names = [i for i in range(len(classes)) for image in os.listdir(os.path.join(root_dir, classes[i]))]
		print(self.label_names)
		#print(self.img_names)
		#raise NotImplementedError

	def __len__(self):
		"""Returns the size of the dataset."""
		return len(self.img_names) 

	def __getitem__(self, index):
		img = (Image.open(self.img_names[index]))
		img = self.transform(Image.open(self.img_names[index]))
		label = self.label_names[index]

		return img, label

def train(model, device, dataloader, hyperparameters, name):
	model = model.to(device)
	optimizer = hyperparameters['optimizer']
	scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
	total_epochs = tqdm(range(hyperparameters['n_epochs']))
	best_acc = 0.0
	all_batchs_loss = 0
	all_batchs_corrects = 0

	for epochs in total_epochs:
		for phase in ['train', 'val']:
			if phase == 'train':
				model.train()
			else:
				model.eval() 	

			for images, targets in data_loader[phase]:
				images = images.to(device)
				targets = targets.to(device)
				optimizer.zero_grad()

				with torch.set_grad_enabled(phase == 'train'):
					outputs = model(images)
					if phase == 'train':					
						losses.backward()
						optimizer.step()

				all_batchs_loss += loss.item() * inputs.size(0)
				all_batchs_corrects += torch.sum(preds == labels.data)

				if phase == 'train':
					scheduler.step()

				epoch_loss = all_batchs_loss / dataset_sizes[phase]
				epoch_acc = all_batchs_corrects.double() / dataset_sizes[phase]

				print(epoch_acc)

				if phase == 'val' and epoch_acc > best_acc:
					best_acc = epoch_acc
					best_model_wts = copy.deepcopy(model.state_dict())
					#permission issue fix when trying to write to an existing file
					if os.path.exists("model_weight_" + str(hyperparameters['lr']) + "_" + str(hyperparameters['n_epochs']) + ".pth"):
						os.remove("model_weight_" + str(hyperparameters['lr']) + "_" + str(hyperparameters['n_epochs']) + ".pth")
					torch.save(best_model_wts , "model_weight_" + str(hyperparameters['lr']) + "_" + str(hyperparameters['n_epochs']) + ".pth")

	print(best_acc)

def test(dataloader, dataset_sizes, num_classes, device):
	model = models.mobilenet_v2()
	model.fc = nn.Linear(model.classifier[1].in_features, num_classes)
	model = model.to(device)
	#model.load_state_dict(torch.load('best_model_weight.pth1lr3'))
	all_batchs_corrects = 0
	model.eval()
	phase = 'test'


	for inputs, labels in dataloader[phase]:
		inputs = inputs.to(device)
		labels = labels.to(device)	
		outputs = model(inputs)
		_, preds = torch.max(outputs, 1)
		all_batchs_corrects += torch.sum(preds == labels.data)
		#####################################################ADDED CODE##################################################################
		#concatenate CUDA tensors -> numpy arrays for confusion matrix generataion for the true and predicted values
		if iters == 0:
			pred_conf = preds.cpu().detach().numpy()
			true = labels.data.cpu().detach().numpy()
		else:
			pred_conf = np.append(pred_conf, preds.cpu().detach().numpy())
			true = np.append(true, labels.data.cpu().detach().numpy())

		iters += 1
		#####################################################ADDED CODE##################################################################

	epoch_acc = all_batchs_corrects.double() / dataset_sizes[phase]

	return epoch_acc

def main():
	parser = argparse.ArgumentParser(description="")
	parser.add_argument('--lr', type=float)
	parser.add_argument('--epochs', type=int)
	args = parser.parse_args()

	#transform for normal training
	transform = transforms.Compose([
		transforms.Grayscale(num_output_channels=1),
		transforms.ToTensor()
	])

	#transform for transfer learning
	"""transform = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor()
	])"""

	#create train and test dataloaders
	TRAIN_DIRECTORY_PATH = "cifar10_train"
	train_dataset = CifarDataset(TRAIN_DIRECTORY_PATH, transform)

	#use a split of 80 20 for train and test data
	train_size = int(0.8*len(train_dataset))
	test_size = len(train_dataset) - train_size
	train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, (train_size, test_size)) 
	train_dataloader = torch.utils.data.DataLoader(train_dataset,
											batch_size=batch_size,
											shuffle=True)
	test_dataloader = torch.utils.data.DataLoader(test_dataset,
										batch_size=batch_size,
										shuffle=True) 

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = models.mobilenet_v2(pretrained=True)
	embedding_dim = glove['the'].shape[1]
	model.fc = nn.Embedding(model.classifier[1].in_features, embedding_dim)
	model = nn.Sequential(model, nn.Linear(embedding_dim, num_classes))
	hyperparameters = {'learning_rate': args.lr, 'optimizer': optimizer, 'n_epochs': args.epochs}




