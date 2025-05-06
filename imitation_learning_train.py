import numpy as np
import torch
import h5py
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from robomimic.utils.dataset import SequenceDataset
import robosuite

obs_keys=[                     # observations we want to appear in batches
            "robot0_eef_pos", 
            "robot0_eef_quat", 
            "robot0_gripper_qpos", 
            "object-state",
        ]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_data_loader(path,mode,batchsize=100):
    observation_keys = obs_keys.copy()
    if "object-state" in obs_keys:
         observation_keys[obs_keys.index('object-state')] = "object"
    dataset = SequenceDataset(
        hdf5_path= path,
        obs_keys= observation_keys,
        dataset_keys=(                  # can optionally specify more keys here if they should appear in batches
            "actions", 
            "rewards", 
            "dones",
        ),
        load_next_obs=True,
        frame_stack=1,
        seq_length=1,                  # length-10 temporal sequences
        pad_frame_stack=True,
        pad_seq_length=True,            # pad last obs per trajectory to ensure all sequences are sampled
        get_pad_mask=False,
        goal_mode=None,
        hdf5_cache_mode="all",          # cache dataset in memory to avoid repeated file i/o
        hdf5_use_swmr=True,
        hdf5_normalize_obs=False,
        filter_by_attribute=mode,       # can optionally provide a filter key here
    )

    data_loader = DataLoader(
        dataset=dataset,
        sampler=None,       # no custom sampling logic (uniform sampling)
        batch_size=batchsize,     # batches of size 100
        shuffle=True,
        num_workers=0,
        drop_last=True      # don't provide last batch in dataset pass if it's less than 100 in size
    )
    return data_loader

class BCModel(nn.Module):
    def __init__(self,input_size,outpit_size):
        super(BCModel,self).__init__()
        self.device = device
        self.input_size = input_size
        self.output_size = outpit_size
        self.layers = nn.Sequential(
            nn.Linear(input_size,32,True,self.device),
            nn.Tanh(),
            nn.Linear(32,64,True,self.device),
            nn.Tanh(),
            nn.Linear(64,outpit_size,True,self.device)
        )
        
    def forward(self,x):
        return self.layers(x)
    
def process_inputs(inputs):
    return torch.concatenate([inputs[k] for k in inputs.keys()],2).to(device,dtype = torch.float32)
    
train_data_loader = get_data_loader("low_dim.hdf5","50_percent_train")
validation_data_loader = get_data_loader("low_dim.hdf5","50_percent_valid",batchsize=1)

model = BCModel(19,7)
criterion = nn.MSELoss()
optimizer = optim.Adam(params=model.parameters(),lr=0.0001)
num_epochs = 800

file = open("bc_losses.txt","a")
for epoch in range(1,num_epochs+1):
    running_loss = 0.0
    for i, data in enumerate(train_data_loader, 0):
        inputs, labels = data["obs"],data["actions"].to(device)
        inputs = process_inputs(inputs)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    
    if epoch%50 == 0:
        with torch.no_grad():
            average_loss = 0
            for j,data in enumerate(validation_data_loader,1):
                test_inputs, test_labels = data["obs"],data["actions"].to(device)
                test_inputs = process_inputs(test_inputs)
                test_outputs = model(test_inputs)
                loss = criterion(test_outputs, test_labels)
                average_loss += loss.item()
            print(f'Epoch {epoch}, Training Loss: {running_loss / len(train_data_loader)}')
            print(f'Validation loss epoch {epoch}: {average_loss/j}')
            file.write(f"Epoch {epoch}, Training Loss: {running_loss / len(train_data_loader)} Validation loss {epoch}: {average_loss/j} \n")
    if epoch%100 == 0:
        torch.save(model.state_dict(), f"models/model_{epoch}.pt")

file.close()         