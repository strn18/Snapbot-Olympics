import sys,mujoco
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('../package/helper/')
sys.path.append('../package/mujoco_usage/')
sys.path.append('../package/gym/')
sys.path.append('../package/rl/')
from mujoco_parser import *
from slider import *
from utility import *
from snapbot_env import *
from sac import *
np.set_printoptions(precision=2,suppress=True,linewidth=100)
plt.rc('xtick',labelsize=6); plt.rc('ytick',labelsize=6)
# %config InlineBackend.figure_format = 'retina'
# %matplotlib inline

xml_path = '../asset/snapbot/scene_snapbot.xml'
env = MuJoCoParserClass(name='Snapbot',rel_xml_path=xml_path,verbose=False)
gym = SnapbotGymClass(
    env = env,
    HZ  = 50,
    history_total_sec = 0.2,
    history_intv_sec  = 0.1,
    VERBOSE = False,
)

# Load 
use_old = False
epi_idx = 1000
print ("epi_idx:[%d]"%(epi_idx))

# Configuration
max_epi_sec  = 5.0 # maximum episode length in second
max_torque = 5.0

max_epi_tick = int(max_epi_sec*gym.HZ) # maximum episode length in tick
# Actor
device     = 'cpu' # cpu / mps / cuda
init_alpha = 0.2
lr_actor   = 0.0003
lr_alpha   = 0.0000
actor = ActorClass(
    obs_dim    = gym.o_dim,
    h_dims     = [256,256],
    out_dim    = gym.a_dim,
    max_out    = max_torque,
    init_alpha = init_alpha,
    lr_actor   = lr_actor,
    lr_alpha   = lr_alpha,
    device     = device,
).to(device)
# Load pth
if use_old:
    pth_path = './old_result/weights/sac_%s/episode_%d.pth'%(gym.name.lower(),epi_idx)
else:
    pth_path = './result/weights/sac_%s/episode_%d.pth'%(gym.name.lower(),epi_idx)
# pth_path = './task_1_high_jump/weights/episode_%d.pth'%(epi_idx)
actor.load_state_dict(torch.load(pth_path,map_location=device))
# Run
gym.init_viewer()
s = gym.reset()
gym.viewer_pause() # pause
print ("   Viewer paused. Press [space] to resume.")
reward_total = 0.0
# flag = False
for tick in range(max_epi_tick):
    a,_ = actor(np2torch(s,device=device),SAMPLE_ACTION=False)
    s_prime,reward,done, info = gym.step(torch2np(a),max_time=max_epi_sec)
    gym.render(
        TRACK_TORSO      = True,
        PLOT_WORLD_COORD = True,
        PLOT_TORSO_COORD = True,
        PLOT_SENSOR      = True,
        PLOT_CONTACT     = True,
        PLOT_TIME        = True,
    )
    reward_total += reward
    s = s_prime
    if not gym.is_viewer_alive(): break

gym.close_viewer()
y_diff = gym.env.get_p_body('torso')[1]
print ("  [Eval] reward:[%.3f] y_diff:[%.3f] epi_len:[%d/%d]"%
       (reward_total,y_diff,tick,max_epi_tick))