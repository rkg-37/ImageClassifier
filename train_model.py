from workspace_utils import active_session
import torch

def validation(model, validloaders, criterion,device='cpu'):
    test_loss = 0
    accuracy = 0
    model.to(device)
    with active_session():
        for images, labels in validloaders:
            images, labels = images.to(device), labels.to(device)
            outputs = model.forward(images)
            test_loss += criterion(outputs, labels).item()    
            equality = (labels.data == outputs.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()
        return test_loss, accuracy


def train_model(model, trainloaders, epochs,validloaders, print_every, criterion, optimizer, device = 'cpu'):
    epochs = epochs
    print_every = print_every
    steps = 0

    # change to cuda
    model.to(device)
    with active_session():
        for e in range(epochs):
            model.train()
            running_loss = 0
            for ii, (inputs, labels) in enumerate(trainloaders):
                steps += 1
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                if steps % print_every == 0:
                    model.eval()
                    test_loss, accuracy = validation(model, validloaders, criterion,device)
                    print("Epoch: {}/{}... ".format(e+1, epochs),
                          "Train Loss: {:.3f}".format(running_loss/print_every),
                         "Valid Loss: {:.3f}".format(test_loss/len(validloaders)),
                         "Valid Accuracy :{:.3f}%".format(100*accuracy/len(validloaders)))

                    running_loss = 0
                    model.train()
                    
                    
def test_model(model, testloaders,device='cpu'):
    model.eval()
    model.to(device)
    accuracy = 0
    total_acc = 0
    with torch.no_grad():
        with active_session():
            for ii, (images, labels) in enumerate(testloaders):
                images, labels = images.to(device), labels.to(device)

                outputs = model.forward(images)
                _, predicted = outputs.max(dim=1)
                equality = labels.data == predicted

                if ii == 0:
                    print(predicted)      # idx of predicted class
                    print(torch.exp(_))   # probability of prediction
                    print(equality)       
                accuracy = equality.type(torch.FloatTensor).mean()
                total_acc += accuracy
                print(accuracy)
    print("Prediction accuracy in test set is {:.3f}%".format(100*total_acc/len(testloaders)))
    
    
    
    

def checkpoint(model,class_to_idx,optimizer,criterion,epoch,arch):

    model.class_to_idx = class_to_idx
    checkpoint = {'features': model.features,
                'input_size': 1024, 
                'output_size': 102,
                'hidden_layers': [each.out_features for each in model.classifier.hidden_layers],
                'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx,
                'epoch':epoch,
                'optimizer':optimizer,
                'criterion':criterion,
                'model': arch
                }
    torch.save(checkpoint, 'checkpoint.pth')
    print("checkpoint.pth file created ")
    
