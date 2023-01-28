import argparse
import torch 
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json


def label_mapping():    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        
    return cat_to_name



def get_input_args():

    parser = argparse.ArgumentParser()
        
    parser.add_argument('dir', type = str,help = 'data_directory')
    parser.add_argument('--save_dir', type = str,default = None,help = 'save create data_directory checkpoint')
    parser.add_argument("--arch",default='densenet',help = 'sets any pretrained model from train.py')
    parser.add_argument('--learning_rate', type=float, default=0.003, help='learning rate for model')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--gpu', action='store_true',default=False,help='activate the gpu')
    parser.add_argument('--hidden_units', nargs=3, type=int, default=[512,256,128], help='list of hidden layers')

    return parser.parse_args()


def Load_datasets(data_dir):
#     data_dir = 'flowers'
    data_dir = data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomVerticalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=valid_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloaders = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    validloaders = torch.utils.data.DataLoader(valid_datasets, batch_size=32)
    testloaders = torch.utils.data.DataLoader(valid_datasets, batch_size=32)

    return trainloaders,validloaders,testloaders,train_datasets.class_to_idx


def model_selection(input_model):
        
    
    resnet18 = models.resnet18(pretrained=True)
    alexnet = models.alexnet(pretrained=True)
    vgg16 = models.vgg16(pretrained=True)
    densenet121 = models.densenet121(pretrained=True)
    
    model_list = {'resnet': resnet18, 'alexnet': alexnet, 'vgg': vgg16,'densenet':densenet121}

    return model_list[input_model]   
    
    
def criterion():
    return nn.NLLLoss()

def optimizer(model,learn_rate):
    return optim.Adam(model.classifier.parameters(), lr=learn_rate)
