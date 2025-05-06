import robosuite
import h5py
import torch
import torch.nn as nn
import numpy as np

env = robosuite.make(
        #**env_info,
        env_name="Lift",
        robots="Panda",
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=False,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )

obs_keys=[                     # observations we want to appear in batches
            "robot0_eef_pos", 
            "robot0_eef_quat", 
            "robot0_gripper_qpos",
            "robot0_joint_pos",
            "robot0_joint_vel",
            "object-state",            
        ]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

f = h5py.File("low_dim.hdf5", "r")
demos = set(k.decode('UTF-8') for k in f["mask"]["50_percent_train"])
all_demos = set(f"demo_{idx}" for idx in range(len(f["data"])))
validation_demos = list(all_demos - demos)

demos = validation_demos
demo_index = 0
model_xml = f["data/{}".format(demos[demo_index])].attrs["model_file"]
env.reset()
xml = env.edit_model_xml(model_xml)
env.reset_from_xml_string(xml)
env.sim.reset()
env.viewer.set_camera(0)
observation = env.reset()

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


done = False 
total_reward = 0
model = Generator(33,7)
device = "cuda" if torch.cuda.is_available() else "cpu"
model_state_dict = torch.load("models/model_gail_100000.pt",map_location=torch.device(device))
model.load_state_dict(model_state_dict)


success = 0
total_demos = 0
while True:
    action=model(torch.tensor(np.hstack([observation[key] for key in obs_keys]),device=device,dtype=torch.float32))
    observation, reward, done, info  = env.step(action.tolist())   
    total_reward+=reward   
    if done or int(total_reward) >= 15:  
        if int(total_reward) >= 15:
            success+=1       
        demo_index+=1
        if demo_index == len(demos):
            break
        total_reward = 0
        model_xml = f["data/{}".format(demos[demo_index])].attrs["model_file"]
        env.reset()
        xml = env.edit_model_xml(model_xml)
        env.reset_from_xml_string(xml)
        env.sim.reset()
        env.viewer.set_camera(0)
        observation = env.reset()
        total_demos+=1
        print(f"Total demos run {total_demos} , Successful run : {success} Success rate : {success/total_demos * 100}")