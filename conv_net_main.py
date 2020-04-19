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
from tqdm import tqdm
from absl import app, flags
from skip_gram_main import Word2Vec
from mobilenet_v2 import mobilenet_v2

FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate.')
flags.DEFINE_float('weight_decay', 0, 'Weight decay (L2 regularization).')
flags.DEFINE_integer('batch_size', 100, 'Number of examples per batch.')
flags.DEFINE_integer('epochs', 20, 'Number of epochs for training.')
flags.DEFINE_string('experiment_name', 'exp', 'Defines experiment name.')

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class Embedding(nn.Module):
    def __init__(self, embedding_dim):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(1280, embedding_dim)
        
    def forward(self, x):
    	x = self.embedding(x)
        return x

class Classify(nn.Module):
    def __init__(self, num_classes):
        super(Classify, self).__init__()
        self.classifier= nn.Linear(1280, num_classes)
        
    def forward(self, x):
    	x = self.classifier(x)
        return x

class CifarDataset(torch.utils.data.Dataset):
	
	def __init__(self, root_dir, transform):
		"""Initializes a dataset containing images and labels."""
		super().__init__()
		self.transform = transform
		self.root_dir = root_dir
		classes = os.listdir(root_dir)
		self.img_names = [os.path.join(os.path.join(root_dir, classes[i]), img_name) for i in range(len(classes)) for img_name in os.listdir(os.path.join(root_dir, classes[i]))]
		self.label_names = [i for i in range(len(classes)) for image in os.listdir(os.path.join(root_dir, classes[i]))]
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

def train(model, layer1, layer2, device, dataloader):
	model = model.to(device)
	optimizer = torch.optim.Adam(model.parameters(), 
								lr=FLAGS.learning_rate, 
								weight_decay=FLAGS.weight_decay)
	scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
	total_epochs = tqdm(range(FLAGS.epochs))
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
					output = model(images) 
					embedding_output = layer1(output)
					class_output = layer2(output)
	
					#loss = 

					if phase == 'train':					
						loss.backward()
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
					if os.path.exists("model_weight_" + str(FLAGS.learning_rate) + "_" + str(FLAGS.epochs) + ".pth"):
						os.remove("model_weight_" + str(FLAGS.learning_rate) + "_" + str(FLAGS.epochs) + ".pth")
					torch.save(best_model_wts , "model_weight_" + str(FLAGS.learning_rate) + "_" + str(FLAGS.epochs) + ".pth")

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
	dataset = CifarDataset(TRAIN_DIRECTORY_PATH, transform)

	#use a split of 60 20 20 for train val and test splits 
	train_size = int(0.6*len(dataset))
	test_size = int((len(dataset) - train_size)/2)
	val_size = test_size

	train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, (train_size, val_size, test_size)) 
	dataloader = {}
	dataloader['train'] = torch.utils.data.DataLoader(train_dataset,
											batch_size=100,
											shuffle=True)
	dataloader['val'] = torch.utils.data.DataLoader(val_dataset,
										batch_size=100,
										shuffle=True) 
	dataloader['test'] = torch.utils.data.DataLoader(test_dataset,
										batch_size=100,
										shuffle=True) 

	num_classes = 10

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = models.mobilenet_v2(pretrained=True)
	model.classifier = Identity()

	embedding_dim = Word2Vec("word2vec_model.txt").get_embedding_dims()
	layer1 = Embedding(embedding_dim)
	layer2 = Classify(num_classes)


	
	#model = mobilenet_v2(embedding_dim=embedding_dim, pretrained=True, progress=True)	 
	#model = nn.Sequential(model, nn.Linear(embedding_dim, num_classes))
	#print(model)

	if FLAGS.task_type == 'training':
		train(model, layer1, layer2, device, dataloader)
	elif FLAGS.task_type == 'testing':
		test(model, layer1, layer2, device, dataloader)

if __name__ == "__main__":
	main()