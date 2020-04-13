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
from tqdm import tqdms
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
