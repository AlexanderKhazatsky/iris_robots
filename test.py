from robot_env import RobotEnv
from controllers.oculus_controller import VRPolicy
import numpy as np
import time
from absl import logging
logging.set_verbosity(logging.WARNING)

# env = RobotEnv(ip_address='127.0.0.1')
env = RobotEnv()
controller = VRPolicy()

STEP_ENV = True
 
env.reset()
controller.reset_state()


# (Pdb) obs['images'][0]['array'].shape
# (480, 640, 3)
# (Pdb) obs['images'][1]['array'].shape
# (480, 640, 3)
# (Pdb) import sys
# (Pdb) sys.getsizeof(obs)

import pdb; pdb.set_trace()

max_steps = 10000
for i in range(max_steps):
	obs = env.get_state()
	a = controller.get_action(obs)
	if STEP_ENV: env.step(a)
	else: time.sleep(0.2)
	#print(np.round(a, 3))