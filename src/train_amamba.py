import numpy as np 
import torch 
import timm
import sys
sys.path.append("/mnt/r/sankn-2/Amamba/Vim-main/vim")
import models_mamba
sys.path.append("/mnt/r/sankn-2/Adaptive_KD/training_scripts/utils")

from custom_dataloaders import *
from pytorchtools import EarlyStopping

from sklearn.metrics import confusion_matrix
import random
import torch.optim as optim 
from torch.utils.data import dataset, DataLoader 
import torchvision.transforms as transforms 
import pandas as pd 
import torch.nn as nn 
import torch.nn.functional as F
import time # import time 
import os # Import OS
import warnings
warnings.filterwarnings("ignore")
import sklearn

import os 
from sklearn.metrics import precision_recall_fscore_support
import logging
import matplotlib.pyplot as plt
import seaborn as sn
from matplotlib.ticker import MaxNLocator

# dataset = "FSC22"
# dataset = "DCASE19"
# dataset = "ESC10"
dataset = "vocalsound"

if_pretrained = True
plot_cm = True



# model_name = "vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2"
model_name = "vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2"
# model_name = "mixer_b16_224"

match dataset:
    case "FSC22":
        from paths.FSC22_pathname import *
        Num_Classes = 27
        r = 0.9 # train/valid split
        learning_rate=1e-5
        batch_size=32
        classes = ("Fire", "Rain", "Thunderstorm", "WaterDrop", "Wind", "Silence", "TreeFalling", "Helicopter", "Engine", 
                   "Axe", "Chainsaw", "Generator", "Handsaw", "Firework", "Gunshot", "WoodChop", "Whistling", "Speaking", 
                   "Footsteps", "Clapping", "Insect", "Frog", "Chirping", "WingFlaping", "Lion", "WolfHowl", "Squirrel")
        

    case "DCASE19":
        from paths.DCASE19_pathname import *
        Num_Classes = 10       
        r = 0.9 # train/valid split
        learning_rate=2e-5
        batch_size=16
        classes = ('airport', 'shopping_mall', 'metro_station', 'street_pedestrian', 'public_square', 
           'street_traffic', 'tram', 'bus', 'metro', 'park')


    case "ESC10":
        from paths.ESC10_pathname import *
        Num_Classes = 10
        r = 0.9 # train/valid split
        learning_rate=2e-5
        batch_size=16
        classes = ('Dogbark', 'Rain', 'Seawaves', 'Babycry', 'Clocktick',
           'sneeze', 'Helicopter', 'Chainsaw', 'Rooster', 'Firecrackling')
        
        
    case "vocalsound":
        from paths.vocalsound_pathname import *
        Num_Classes = 6
        r = 0.9 # train/valid split
        learning_rate=2e-5
        batch_size=16
        classes = ('Laughter', 'Sigh', 'Cough', 'ThroatClearing', 'Sneeze', 'Snif')
    
        
    case _:
        print(f"{dataset} Not implemented")
        sys.exit()


log_path = os.path.join(f'/mnt/r/sankn-2/Amamba/log_files/{dataset}')
os.makedirs(log_path,exist_ok = True)

MODEL_SAVE_PATH = os.path.join(log_path, f"{model_name}_model.pt")
logfile = os.path.join(log_path, f"{model_name}_output.log")




logging.basicConfig(
level=logging.INFO,
format="%(asctime)s %(message)s",
handlers=[
        logging.FileHandler(logfile, "w"),
        logging.StreamHandler()
])

SEED = 1234 # Initialize seed 
EPOCHS=1000

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda') # Define device type 
# warnings.filterwarnings("ignore")
data_transformations = transforms.Compose([ # Training Transform 
    transforms.Resize([224,224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])])

train_dataset=Train_Data(Train_Label_Path, Train_Data_Path, H5py_Path, transform=data_transformations) 

train_size = int(r * len(train_dataset)) 
valid_size = len(train_dataset) - train_size 
Train_Dataset,Valid_Dataset = torch.utils.data.random_split(train_dataset, [train_size, valid_size]) 

Test_Dataset=Test_Data(Test_Label_Path, Test_Data_Path, H5py_Path,transform=data_transformations)

train_loader=DataLoader(dataset=Train_Dataset,batch_size=batch_size,shuffle=True) # Create Training Dataloader 
valid_loader=DataLoader(dataset=Valid_Dataset,batch_size=batch_size,shuffle=False)# Create Validation Dataloader 
test_loader=DataLoader(dataset=Test_Dataset,batch_size=batch_size,shuffle=False) # Create Test Dataloader 

if "mixer" in model_name:
    pt_model = timm.create_model(model_name,num_classes = Num_Classes, pretrained = if_pretrained)
    
    class custom_model(nn.Module):
        def __init___(self, model, Num_Classes):
            super().__init__()
            self.pretrained_layers = nn.Sequential(*list(model.children()[:-1]))
            out = model.head.in_features
            self.fc = nn.Linear(out, Num_Classes)
            self.sm = nn.Softmax()
            
        def forward(self,x):
            x = self.pretrained_layers(x)
            x = self.fc(x)
            return self.sm(x)
    
    model = custom_model(pt_model,Num_Classes)
    
else:
    model = timm.create_model(model_name,num_classes = Num_Classes)

    if if_pretrained:
       
        if "vim_small" in model_name:
            checkpoint = torch.load("/mnt/r/sankn-iit/pretrained_model_weights/vim_s_midclstok_ft_81p6acc.pth", map_location='cpu')
        elif "vim_tiny" in model_name:
            checkpoint = torch.load("/mnt/r/sankn-iit/pretrained_model_weights/vim_t_midclstok_ft_78p3acc.pth", map_location='cpu')
            

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




model=model.to(device)
optimizer = optim.Adam(model.parameters(),lr=learning_rate)
criterion = nn.CrossEntropyLoss() # Define Loss Function 

def calculate_accuracy(fx, y): # caluate accuracy 
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = correct.float()/preds.shape[0]
    return acc

def train(model,device,iterator, optimizer, criterion): 
    # early_stopping = EarlyStopping(patience=7, verbose=True)
    
    epoch_loss = 0
    epoch_acc = 0
    count=0
    model.train() # call model object for training 
    for i,(x,y) in enumerate(iterator):
        print(i)
        x=x.float()
        x=torch.cat([x,x,x],dim=1)
        
        x=x.to(device)
        y=y.to(device)# Transfer label  to device
        optimizer.zero_grad() # Initialize gredients as zeros 
        count=count+1
        #print(x.shape)
        Predicted_Train_Label=model(x)
        # print(Predicted_Train_Label.shape)
        # Predicted_Train_Label=x = Predicted_Train_Label.mean(dim=1)

        # print(Predicted_Train_Label)
        loss = criterion(Predicted_Train_Label, y) # training loss
        acc = calculate_accuracy(Predicted_Train_Label, y) # training accuracy
        loss.backward() # backpropogation 
        optimizer.step() # optimize the model weights using an optimizer 
        epoch_loss += loss.item() # sum of training loss
        epoch_acc += acc.item() # sum of training accuracy  
  
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
def evaluate(model,device,iterator, criterion, final_test = False): # Evaluate Validation accuracy 
    #print("Validation Starts")
    epoch_loss = 0
    epoch_acc = 0
    count=0
    precision=0
    recall=0
    fscore=0
    model.eval() # call model object for evaluation 
    if final_test:
        y_pred = []
        y_true = []
        
    
    with torch.no_grad(): # Without computation of gredient 
        for (x, y) in iterator:
            x=x.float()
            x=torch.cat([x,x,x],dim=1)
            #x=ImageToPatches(x,16)
            x=x.to(device) # Transfer data to device 
            y=y.to(device) # Transfer label  to device 
            count=count+1
            Predicted_Label = model(x) # Predict claa label 
            # Predicted_Label=Predicted_Label.mean(dim=1)
            loss = criterion(Predicted_Label, y) # Compute Loss 
            acc = calculate_accuracy(Predicted_Label, y) # compute Accuracy
            Predicted_Label_2=Predicted_Label.detach().cpu().numpy()
            Predicted_Label_2=np.argmax(Predicted_Label_2,axis=1)
            y_2=y.detach().cpu().numpy()
            precision1, recall1, fscore1, sup = sklearn.metrics.precision_recall_fscore_support(y_2, Predicted_Label_2, average='weighted')

            epoch_loss += loss.item() # Compute Sum of  Loss 
            epoch_acc += acc.item() # Compute  Sum of Accuracy
            precision=precision + precision1
            recall=recall+recall1
            fscore=fscore+fscore1
            
            if final_test:
                output = (torch.max(torch.exp(Predicted_Label), 1)[1]).data.cpu().numpy()
                y_pred.extend(output)
                
                labels = y.data.cpu().numpy()
                y_true.extend(labels)
            
         
                 

    if final_test:
        return epoch_loss / len(iterator), epoch_acc / len(iterator) , precision/len(iterator), recall/len(iterator), fscore/len(iterator) , y_pred, y_true
        
          
    return epoch_loss / len(iterator), epoch_acc / len(iterator) , precision/len(iterator), recall/len(iterator), fscore/len(iterator)
 
best_valid_loss = float('inf')

logging.info("Training ...") 
total_time=0
logging.info("---------------------------------------------------------------------------------------------------------------------")   
early_stopping = EarlyStopping(patience=7, verbose=True) # early Stopping Criteria


vals = {}
vals["train_loss"] = []
vals["train_acc"] = []
vals["val_loss"] = []
vals["val_acc"] = []

for epoch in range(EPOCHS):
    start_time=time.time() # Compute Start Time 
    train_loss, train_acc = train(model,device,train_loader,optimizer, criterion) # Call Training Process 
    train_loss=round(train_loss,2) # Round training loss 
    train_acc=round(train_acc,2) # Round training accuracy 
    valid_loss, valid_acc,_,_,_ = evaluate(model,device,valid_loader,criterion) # Call Validation Process 
    valid_loss=round(valid_loss,2) # Round validation loss
    valid_acc=round(valid_acc,2) # Round accuracy 
    end_time=(time.time()-start_time) # Compute End time 
    end_time=round(end_time,2)  # Round End Time 
    total_time=total_time+end_time
    logging.info(f" | Epoch={epoch} | Training Accuracy={train_acc*100} | Validation Accuracy= {valid_acc*100} | Training Loss= {train_loss} | Validation_Loss= {valid_loss} Time Taken(Seconds)={end_time}|")
    logging.info("---------------------------------------------------------------------------------------------------------------------")
    
    vals["train_acc"].append(train_acc*100)
    vals["train_loss"].append(train_loss)
    vals["val_loss"].append(valid_loss)
    vals["val_acc"].append(valid_acc*100)

  
    
    
    early_stopping(valid_loss,model,MODEL_SAVE_PATH) # call Early Stopping to Prevent Overfitting 
    if early_stopping.early_stop:
        logging.info("Early stopping")
        logging.info(f'Total Time: {total_time} sec')
        break
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))

model.load_state_dict(torch.load(MODEL_SAVE_PATH)) # load the trained model 
test_loss, test_acc,p,r,f1, y_pred, y_true = evaluate(model, device, test_loader, criterion,final_test = True) # Compute Test Accuracy on Unseen Signals 
test_loss=round(test_loss,2)# Round test loss
test_acc=round(test_acc,2) # Round test accuracy

p=round(p,3) 
r=round(r,3) 
f1=round(f1,3)

logging.info(f"|Test Loss= {test_loss} Test Accuracy= {test_acc*100}") # print test accuracy 
logging.info(f"P: {p}, R: {r}, F1: {f1}")  


if plot_cm:
  
    
    plt.plot(vals["train_loss"], label="Train Loss", color="blue", marker="o")
    plt.plot(vals["val_loss"], label="Validation Loss", color="orange", marker="o")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=10))
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    plt.savefig(f'/mnt/r/sankn-2/Amamba/log_files/{dataset}/{model_name}_{dataset}_loss_curve.pdf')

    plt.clf()
    plt.plot(vals["train_acc"], label="Train Accuracy", color="green", marker="o")
    plt.plot(vals["val_acc"], label="Validation Accuracy", color="red", marker="o")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=10))
    
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    
    plt.savefig(f'/mnt/r/sankn-2/Amamba/log_files/{dataset}/{model_name}_{dataset}_acc_curve.pdf')

  
    plt.clf()
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                        columns = [i for i in classes])
    # plt.figure(figsize=(Num_Classes * 0.7, Num_Classes * 0.5))
    plt.figure(figsize=(14,7))
    heatmap = sn.heatmap(df_cm, annot=True, cmap = "crest", annot_kws={"size": 14})
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=90, fontsize=14)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, fontsize=14)
    plt.savefig(f'/mnt/r/sankn-2/Amamba/log_files/{dataset}/{model_name}_{dataset}_cm.pdf',bbox_inches='tight')
    
    if "FSC" in dataset:
        plt.clf()
        cf_matrix = confusion_matrix(y_true, y_pred)
        df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                            columns = [i for i in classes])
        plt.figure(figsize=(Num_Classes * 0.8, Num_Classes * 0.6))
        heatmap = sn.heatmap(df_cm, annot=True, cmap = "crest", annot_kws={"size": 14}, fmt=".2f")
        heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=90, fontsize=14)
        heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=14)
        plt.savefig(f'/mnt/r/sankn-2/Amamba/log_files/{dataset}/{model_name}_{dataset}_cm_2.pdf',bbox_inches='tight')
   
    

