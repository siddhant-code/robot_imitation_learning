import numpy as np
import torch
import h5py
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from robomimic.utils.dataset import SequenceDataset
import robosuite
from robosuite.wrappers import GymWrapper
import os

obs_keys=[                     # observations we want to appear in batches
            "robot0_eef_pos", 
            "robot0_eef_quat", 
            "robot0_gripper_qpos",
            "robot0_joint_pos",
            "robot0_joint_vel",
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

def process_inputs(inputs): 
    return torch.concatenate([inputs[k] for k in inputs.keys()],2).to(device,dtype = torch.float32)

# Define the Generator (Policy Network)
class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size,96,True,device=device),
            nn.Tanh(),
            nn.Linear(96,128,True,device=device),
            nn.Tanh(),
            nn.Linear(128,output_size,device=device)
        )
        
    def forward(self,x):
        return self.layers(x)

# Define the Discriminator 
class Discriminator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim + output_dim, 64,device=device)
        self.fc2 = nn.Linear(64, 64,device=device)
        self.fc3 = nn.Linear(64, 1,device=device)  # Output a scalar value for binary classification
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)  # Combine state and action
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Output between 0 and 1
        return x

# Training function for GAIL using DataLoader
def train_gail(generator, discriminator, dataloader, num_epochs=100, lr=1e-3):
    # Optimizers for both networks
    gen_optimizer = optim.Adam(generator.parameters(), lr=1e-3)
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=1e-4)
    
    criterion = nn.BCELoss()  # Binary Cross Entropy for discriminator
    file = open("gail_losses.txt","a")
    for epoch in range(1,num_epochs+1):
        generator.train()
        discriminator.train()

        total_disc_loss = 0
        total_gen_loss = 0
        for i, data in enumerate(dataloader, 0):
            inputs, batch_expert_actions = data["obs"],data["actions"].to(device)
            batch_states = process_inputs(inputs)
        
            # Train Discriminator
            fake_actions = generator(batch_states)
            
            # Labels: 1 for expert, 0 for fake actions from generator
            real_labels = torch.ones(batch_states.size(0), 1,device=device)
            fake_labels = torch.zeros(batch_states.size(0), 1,device=device)
            
            labels = torch.cat([real_labels, fake_labels], dim=-1)
            
            # Discriminator on expert actions (real)
            disc_real = discriminator(batch_states, batch_expert_actions).squeeze(1)
            #disc_loss_real = criterion(disc_real, real_labels)
            
            # Discriminator on generated actions (fake)
            disc_fake = discriminator(batch_states, fake_actions.detach()).squeeze(1)  # detach to avoid backprop in the generator
            #disc_loss_fake = criterion(disc_fake, fake_labels)
            
            # Total discriminator loss
            preds = torch.cat([disc_real, disc_fake], dim=-1)
            #disc_loss = disc_loss_real + disc_loss_fake
            disc_loss = criterion(preds,labels)
            # Backpropagation for discriminator
            disc_optimizer.zero_grad()
            disc_loss.backward()
            disc_optimizer.step()
            
            total_disc_loss += disc_loss.item()

            # Train Generator
            # Generator tries to fool the discriminator
            disc_fake = discriminator(batch_states, fake_actions).squeeze(1)     
            gen_loss = criterion(disc_fake, real_labels)
            
            # Backpropagation for generator
            gen_optimizer.zero_grad()
            gen_loss.backward()
            gen_optimizer.step()

            total_gen_loss += gen_loss.item()

        # Print average loss for every 10th epoch
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{num_epochs}, Disc Loss: {total_disc_loss/len(dataloader)}, Gen Loss: {total_gen_loss/len(dataloader)}')
            file.write(f'Epoch {epoch}/{num_epochs}, Disc Loss: {round(total_disc_loss/len(dataloader),2)}, Gen Loss: {round(total_gen_loss/len(dataloader),2)} \n')
            
        if epoch % 2000 == 0:
            torch.save(generator.state_dict(), f"models/model_gail_{epoch}.pt")
            
    file.close()


# Define input and output dimensions
input_dim = 33  # States 
output_dim = 7  # Actions 

train_data_loader = get_data_loader("low_dim.hdf5","50_percent_train")

# Create the generator and discriminator
generator = Generator(input_dim, output_dim)
discriminator = Discriminator(input_dim, output_dim)

# Train GAIL with DataLoader
train_gail(generator, discriminator, train_data_loader, num_epochs=100000, lr=1e-3)
