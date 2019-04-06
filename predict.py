import torch
import numpy as np
import utils
import model
import argparse
import json

parser = argparse.ArgumentParser()

parser.add_argument('image_path',
                    help='Image directory path')
parser.add_argument('checkpoint',
                    help='Checkpoint of the model')
parser.add_argument('--top_k', action='store',
                    dest='topk',
                    default=5,
                    help='Top k prediction probabilities')
parser.add_argument('--category_names', action='store',
                    dest='category_names',
                    default=None,
                    help='Json association between category and names')
parser.add_argument('--gpu', action='store_true',
                    default=False,
                    dest='gpu',
                    help='Set training to gpu')

results = parser.parse_args()

model = model.load_checkpoint("model_checkpoint.pth")

device = ("cuda" if results.gpu else "cpu")
model.to(device)
    
image = utils.process_image(results.image_path).to(device)
# Adding dimension to image (first dimension)
np_image = image.unsqueeze_(0)

model.eval()
with torch.no_grad():
    logps = model.forward(np_image)

ps = torch.exp(logps)
top_k, top_classes_idx = ps.topk(int(results.topk), dim=1)
top_k, top_classes_idx = np.array(top_k.to('cpu')[0]), np.array(top_classes_idx.to('cpu')[0])

# Inverting dictionary
idx_to_class = {x: y for y, x in model.class_to_idx.items()}

top_classes = []
for index in top_classes_idx:
    top_classes.append(idx_to_class[index])
    
if results.category_names != None:
    with open(results.category_names, 'r') as f:
        cat_to_name = json.load(f)
        top_class_names = [cat_to_name[top_class] for top_class in list(top_classes)]
        print(f'Top {results.topk} probabilities: {list(top_k)}')
        print(f'Top {results.topk} classes: {top_class_names}')
else:
    print(f'Top {results.topk} probabilities: {list(top_k)}')
    print(f'Top {results.topk} classes: {list(top_classes)}')