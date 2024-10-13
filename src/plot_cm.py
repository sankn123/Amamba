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
from torchmetrics import Accuracy


# dataset = "DCASE19"
# dataset = "EATD"
dataset = "ESC10"
# dataset = "FSC22"
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
        
    case "EATD":
        from paths.EATD_pathname import *
        Num_Classes = 2
        

log_file = f'log_files/{dataset}/log.txt'
model_weights = f'log_files/{dataset}/best_checkpoint.pth'
train_loss = []
test_loss = []
test_acc1 = []
epochs = []

with open(log_file, 'r') as f:
    for line in f:
        data = json.loads(line)
        train_loss.append(data['train_loss'])
        test_loss.append(data['test_loss'])
        test_acc1.append(data['test_acc1'])
        epochs.append(data['epoch'])


# Loading model
model = timm.create_model("vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2",num_classes = Num_Classes)
checkpoint = torch.load(model_weights, map_location='cpu')

checkpoint_model = checkpoint['model']
# checkpoint_model = checkpoint
state_dict = model.state_dict()
for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
        print(f"Removing key {k} from pretrained checkpoint")
        del checkpoint_model[k]

# interpolate position embedding
pos_embed_checkpoint = checkpoint_model['pos_embed']
embedding_size = pos_embed_checkpoint.shape[-1]
num_patches = model.patch_embed.num_patches
num_extra_tokens = model.pos_embed.shape[-2] - num_patches
# height (== width) for the checkpoint position embedding
orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
# height (== width) for the new position embedding
new_size = int(num_patches ** 0.5)
# class_token and dist_token are kept unchanged
extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
# only the position tokens are interpolated
pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
pos_tokens = torch.nn.functional.interpolate(
    pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
checkpoint_model['pos_embed'] = new_pos_embed

model.load_state_dict(checkpoint_model, strict=False)

model = model.to("cuda")
# testing
data_transformations = transforms.Compose([ # Training Transform 
        transforms.ToTensor()])

Test_Dataset=Test_Data(Test_Label_Path, Test_Data_Path, H5py_Path,transform=data_transformations)
test_loader=DataLoader(dataset=Test_Dataset,batch_size=8,shuffle=True,drop_last=True) # Create Test Dataloader 

y_true = []
y_pred = []


accuracy = Accuracy(task="multiclass", num_classes=Num_Classes).to("cuda")
acc = 0
for data,label in test_loader:
  
    data = data.to("cuda")
    label = label.to("cuda")
    
    preds = model(data)
    acc+=accuracy(preds,label)
    
    preds = preds.detach().cpu()
    preds=np.argmax(preds,axis=1)
    
    label = label.detach().cpu()
    
    label = np.array(label)
    preds = np.array(preds)
    
    y_true.append(label)
    
    y_pred.append(preds)
    # print(y_pred)
    # break

print(f"acc:{acc*100/len(test_loader)}")
y_true = np.array(y_true)
y_pred = np.array(y_pred)

y_true = y_true.ravel()
y_pred = y_pred.ravel()

cm = confusion_matrix(y_true, y_pred)
cmd = ConfusionMatrixDisplay(cm, display_labels=[0, 1])  # Adjust labels accordingly
cmd.plot(cmap=plt.cm.Blues)
plt.title(f"{dataset} Confusion Matrix")
plt.savefig(f'log_files/{dataset}/CF_{dataset}.pdf')


print("Classification Report:")
print(classification_report(y_true, y_pred))


fig, ax1 = plt.subplots()

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.plot(epochs, train_loss, label='Train Loss', color='tab:red')
ax1.plot(epochs, test_loss, label='Test Loss', color='tab:blue')
ax1.legend(loc='upper right')

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('Accuracy')
ax2.plot(epochs, test_acc1, label='Test Accuracy', color='tab:green')
ax2.legend(loc='upper left')

plt.title("Learning Curve")
plt.savefig(f'log_files/{dataset}/{dataset}_learning_curve.pdf') 
