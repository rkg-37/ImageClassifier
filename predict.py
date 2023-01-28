from predict_utility import load_checkpoint,get_input_args,predict,process_image
from utility_module import label_mapping
import warnings
warnings.filterwarnings("ignore")
from prettytable import PrettyTable
x = PrettyTable()

args = get_input_args()

model = load_checkpoint(args.checkpoint)

top_ps,top_class = predict(image_path=args.path,model=model,topk=args.top_k)

print("\nprediction to the given image of the flower\n")

flower_to_name = label_mapping()

prob = [round(p,5) for p in top_ps]
top_class_name = [flower_to_name[c] for c in top_class]

x.add_column("flower name",top_class_name)
x.add_column("prediction probability", prob)
print(x)
