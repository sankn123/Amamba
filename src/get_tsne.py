import numpy as np
import torch
import timm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from torch.utils.data import dataset, DataLoader 
import torchvision.transforms as transforms 
import json
import seaborn as sns
import sys
sys.path.append("path to Vim-main/vim")
import models_mamba
sys.path.append("path to utils")
from custom_dataloaders import *
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import torch.nn as nn

# dataset = "DCASE19"
dataset = "vocalsound"
# dataset = "ESC10"
# dataset = "FSC22"

# model_name = "vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2"
# model_name = "vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2"


match dataset:
    case "FSC22":
        from paths.FSC22_pathname import *
        Num_Classes = 27
        

    case "DCASE19":
        from paths.DCASE19_pathname import *
        Num_Classes = 10       
        

    case "ESC10":
        from paths.ESC10_pathname import *
        Num_Classes = 10
        
    case "vocalsound":
        from paths.vocalsound_pathname import *
        Num_Classes = 6
        

if "vim_tiny" in model_name:
    model_weights = f'log_files/{dataset}/tiny_model.pt'
    model = timm.create_model(model_name,num_classes = Num_Classes)
    
    checkpoint = torch.load(model_weights, map_location='cpu')
    model.load_state_dict(checkpoint, strict=False)
    
elif "vim_small" in model_name:
    model_weights = f'log_files/{dataset}/model.pt'
    model = timm.create_model(model_name,num_classes = Num_Classes)
    
    checkpoint = torch.load(model_weights, map_location='cpu')
    model.load_state_dict(checkpoint, strict=False)

elif "mixer" in model_name:
    model_weights = f'log_files/{dataset}/mixer_b16_224_model.pt'
    
    pt_model = timm.create_model(model_name, num_classes = Num_Classes)
    
    class custom_model(nn.Module):
        
        def __init__(self, model, num_classes):
            super().__init__()
            self.pretrained_layers = nn.Sequential(*list(model.children())[:-1])
            out = model.head.in_features
            self.fc = nn.Linear(out, Num_Classes)
            # self.sm = nn.Softmax()
            
        def forward(self,x):
            x = self.pretrained_layers(x)
            # x = self.fc(x)
            return x
    
    model = custom_model(model = pt_model, num_classes = Num_Classes)
    
    checkpoint = torch.load(model_weights, map_location='cpu')
    model.load_state_dict(checkpoint, strict=False)


    
        
        
             


model = model.to("cuda")
model.eval()



# testing
data_transformations = transforms.Compose([ # Training Transform 
    transforms.ToTensor()])


train_dataset=Train_Data(Train_Label_Path, Train_Data_Path, H5py_Path, transform=data_transformations) 
train_loader=DataLoader(dataset=train_dataset,batch_size=1,shuffle=True, drop_last = True)

loader_iter = iter(train_loader)
y_pred = []
print(len(train_loader))

labels = []
for _ in range(len(train_loader)):
    data, label = next(loader_iter) 
    labels.append(label)
    
    data = data.to("cuda")
    label = label.to("cuda")
    with torch.no_grad():
        if "vim" in model_name:
            feats = model(data, return_features = True)
        
        elif "mixer" in model_name:
            feats = model(data)
            # print(feats.shape)
            feats = feats.mean(dim=1)
            
    feats = np.array(feats.cpu())
    
    y_pred.append(feats)
        

y_pred = np.array(y_pred)
labels = np.array(labels)

labels = np.squeeze(labels,axis=1)
y_pred = np.squeeze(y_pred, axis=1)

print(y_pred.shape, labels.shape)
# y_pred = y_pred.ravel()

n_components = 2
tsne = TSNE(n_components)
tsne_result = tsne.fit_transform(y_pred)

# (1000, 2)
# Two dimensions for each of our images
 
# Plot the result of our TSNE with the label color coded
# A lot of the stuff here is about making the plot look pretty and not TSNE

# Create a DataFrame to store the t-SNE result

tsne_result_df = pd.DataFrame({'Feature-1': tsne_result[:,0], 'Feature-2': tsne_result[:,1], 'label': labels})
fig, ax = plt.subplots(1)
sns.scatterplot(x='Feature-1', y='Feature-2', hue='label', data=tsne_result_df, ax=ax,s=120,  palette='Set2')
lim = (tsne_result.min()-5, tsne_result.max()+5)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_aspect('equal')
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
# ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=10, borderaxespad=0.)
plt.tight_layout()

if "vim_small" in model_name:
    plt.savefig(f'log_files/{dataset}/{dataset}_tsne.pdf', bbox_inches='tight')
elif "vim_tiny" in model_name:
    plt.savefig(f'log_files/{dataset}/tiny_{dataset}_tsne.pdf', bbox_inches='tight')
elif "mixer" in model_name:
    plt.savefig(f'log_files/{dataset}/{model_name}_{dataset}_tsne.pdf', bbox_inches='tight')
