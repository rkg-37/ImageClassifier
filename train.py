from utility_module import get_input_args , Load_datasets , label_mapping , model_selection , criterion , optimizer
from model_class import MyNetwork
from workspace_utils import active_session
from train_model  import validation,train_model,test_model,checkpoint

args = get_input_args()

trainloaders ,validloaders,testloaders , class_to_idx = Load_datasets(args.dir)

cat_to_name = label_mapping()

my_classifer = MyNetwork(1024,102,args.hidden_units,drop_p=0.2)

model = model_selection(args.arch)

for param in model.parameters():
    param.requires_grad = False
model.classifier = my_classifer


criterion = criterion()
optimizer = optimizer(model,args.learning_rate)


print("\n starting training process..........................................")
with active_session():
    # do long-running work here
    if args.gpu == True:
        train_model(model, trainloaders,10,validloaders,40, criterion, optimizer, 'cuda')
    else:
        train_model(model, trainloaders,10,validloaders,40, criterion, optimizer, 'cpu')

print("\n Ending training process..........................................")

print("testing the model ....................................................")
if args.gpu == True:
    test_model(model, testloaders,device='cuda')
else:
    test_model(model, testloaders,device='cpu')
 
print("\n Ending testing process..........................................")


checkpoint(model,class_to_idx,optimizer,criterion,args.epochs,args.arch)

