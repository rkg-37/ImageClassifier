import torch
from torchvision import datasets, transforms, models
from model_class import MyNetwork
import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from utility_module import label_mapping
import seaborn as sb

def get_input_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('path', type = str,help = 'data_directory of input')
    parser.add_argument('checkpoint', type = str,help = 'checkpoint path')
    parser.add_argument('--top_k', type=int, default=3, help='number of epochs to train')
    parser.add_argument('--category_names', type=str,help='category_name')
    parser.add_argument('--gpu', action='store_true',default=False,help='activate the gpu')

    return parser.parse_args()


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    model = checkpoint['model']
    if(model == 'densenet'):
        model = models.densenet121(pretrained=True)
    elif(model == 'resnet'):
        model = models.resnet18(pretrained=True)
    elif(model == 'alexnet'):
        model = models.alexnet(pretrained=True)
    else:
        model = models.vgg16(pretrained=True)

    model.classifier = MyNetwork(checkpoint['input_size'],
                          checkpoint['output_size'],
                          checkpoint['hidden_layers'])
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def process_image(image):
    # TODO: Process a PIL image for use in a PyTorch model
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img = Image.open(image)
    img = img.resize((256, 256)).crop((16, 16, 240, 240))
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    img = np.array(img)
    img = img / 255
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)
    
    return img
    
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    print(type(image))
#     image = image.numpy().transpose((1, 2, 0))
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk=5):
    # TODO: Implement the code to predict the class from an image file
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    img = process_image(image_path)
    img = torch.from_numpy(img).type(torch.FloatTensor)
    img = img.unsqueeze(0)   # fit to used as input
    model.to('cpu')
    img = img.to('cpu')
    ps = torch.exp(model.forward(img))  
    top_ps, top_labs = ps.topk(topk)
    
    top_ps = top_ps.detach().numpy().tolist()[0]     # make list
    top_labs = top_labs.detach().numpy().tolist()[0]
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [idx_to_class[lab] for lab in top_labs]
    
    return top_ps, top_classes
    
# TODO: Display an image along with the top 5 classes
def predict_show(image_path, model):
    cat_to_name =label_mapping()
    plt.figure(figsize = (4, 8))
    ax = plt.subplot(2,1,1)
    title = cat_to_name[image_path.split('/')[2]]
    img = process_image(image_path)
    imshow(img, ax, title)
    
    ps, classes = predict(image_path, model)
    flower_name = [cat_to_name[cla] for cla in classes]
    plt.subplot(2,1,2)
    sb.barplot(x=ps, y=flower_name, color=sb.color_palette()[1])
    plt.show()  