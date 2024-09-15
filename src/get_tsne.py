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
sys.path.append("/mnt/r/sankn-2/Amamba/Vim-main/vim")
import models_mamba
sys.path.append("/mnt/r/sankn-2/Adaptive_KD/training_scripts/utils")
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
model_name = "vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2"

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

    model_weights = f'/mnt/r/sankn-2/Amamba/log_files/{dataset}/tiny_model.pt'
    
elif "vim_small" in model_name:
    model_weights = f'/mnt/r/sankn-2/Amamba/log_files/{dataset}/model.pt'



    
        
        
        
        
model = timm.create_model(model_name,num_classes = Num_Classes)
checkpoint = torch.load(model_weights, map_location='cpu')

# checkpoint_model = checkpoint['model']
checkpoint_model = checkpoint
state_dict = model.state_dict()
# for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
#     if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
#         print(f"Removing key {k} from pretrained checkpoint")
#         del checkpoint_model[k]

# # interpolate position embedding
# pos_embed_checkpoint = checkpoint_model['pos_embed']
# embedding_size = pos_embed_checkpoint.shape[-1]
# num_patches = model.patch_embed.num_patches
# num_extra_tokens = model.pos_embed.shape[-2] - num_patches
# # height (== width) for the checkpoint position embedding
# orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
# # height (== width) for the new position embedding
# new_size = int(num_patches ** 0.5)
# # class_token and dist_token are kept unchanged
# extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
# # only the position tokens are interpolated
# pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
# pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
# pos_tokens = torch.nn.functional.interpolate(
#     pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
# pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
# new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
# checkpoint_model['pos_embed'] = new_pos_embed

model.load_state_dict(checkpoint_model, strict=False)

model = model.to("cuda")
model.eval()



# testing
data_transformations = transforms.Compose([ # Training Transform 
    transforms.Resize([224,224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])])

# Test_Dataset=Test_Data(Test_Label_Path, Test_Data_Path, H5py_Path,transform=data_transformations)
# test_loader=DataLoader(dataset=Test_Dataset,batch_size=1,shuffle=True,drop_last=True) # Create Test Dataloader 

train_dataset=Train_Data(Train_Label_Path, Train_Data_Path, H5py_Path, transform=data_transformations) 
train_loader=DataLoader(dataset=train_dataset,batch_size=1,shuffle=True, drop_last = True)

loader_iter = iter(train_loader)
y_pred = []
print(len(train_loader))

labels = []
for _ in range(len(train_loader)):
    data, label = next(loader_iter) 
    labels.append(label)
    
    data=torch.cat([data,data,data],dim=1)
    data = data.to("cuda")
    label = label.to("cuda")
    with torch.no_grad():
        feats = model(data, return_features = True)
    feats = np.array(feats.cpu())
    
    y_pred.append(feats)
        

y_pred = np.array(y_pred)
labels = np.array(labels)

labels = np.squeeze(labels,axis=1)
y_pred = np.squeeze(y_pred, axis=1)
y_pred = y_pred[:,:-1]
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

tsne_result_df = pd.DataFrame({'Features 1': tsne_result[:,0], 'Features 2': tsne_result[:,1], 'label': labels})
fig, ax = plt.subplots(1)
sns.scatterplot(x='Features 1', y='Features 2', hue='label', data=tsne_result_df, ax=ax,s=120,  palette='Set2')
lim = (tsne_result.min()-5, tsne_result.max()+5)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_aspect('equal')
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

if "vim_small" in model_name:
    plt.savefig(f'/mnt/r/sankn-2/Amamba/log_files/{dataset}/{dataset}_tsne.pdf', bbox_inches='tight')
elif "vim_tiny" in model_name:
    plt.savefig(f'/mnt/r/sankn-2/Amamba/log_files/{dataset}/tiny_{dataset}_tsne.pdf', bbox_inches='tight')

