from robot_env import RobotEnv
from controllers.oculus_controller import VRPolicy
from data_collector import DataCollecter
from user_interface.gui import RobotGUI
import numpy as np
import torch

policy = torch.load('/Users/sasha/Desktop/robot_training/run1/id0/models/2.pt')

policy.eval()

# Make the robot env
env = RobotEnv('172.16.0.1')
controller = VRPolicy()

# Make the data collector
data_collector = DataCollecter(env=env, controller=controller, policy=policy)

# Make the GUI
user_interface = RobotGUI(robot=data_collector)

