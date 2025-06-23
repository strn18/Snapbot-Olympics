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
print ("Ready.")

xml_path = '../asset/snapbot/scene_snapbot.xml'
env = MuJoCoParserClass(name='Snapbot',rel_xml_path=xml_path,verbose=True)
gym = SnapbotGymClass(
    env = env,
    HZ  = 50,
    history_total_sec = 0.2,
    history_intv_sec  = 0.1,
    VERBOSE =True,
)
print ("Ready.")

max_epi_sec       = 5.0 # maximum episode length in second (IMPORTANT)
max_torque        = 2.0
n_episode         = 1000 # number of total episodes (rollouts)

max_epi_tick      = int(max_epi_sec*gym.HZ) # maximum episode length in tick
n_warmup_epi      = 10 # number of warm-up episodes
buffer_limit      = 50000 # 50000
buffer_warmup     = buffer_limit // 5
init_alpha        = 0.2
# Update
lr_actor          = 0.0005 # 0.0002
lr_alpha          = 0.0001 # 0.0003
lr_critic         = 0.0001
n_update_per_tick = 1 # number of updates per tick
batch_size        = 256
gamma             = 0.99
tau               = 0.005
# Debug
print_every       = 50
eval_every        = 50
save_every        = 50
RENDER_EVAL       = False # False
print ("n_episode:[%d], max_epi_sec:[%.2f], max_epi_tick:[%d]"%
       (n_episode,max_epi_sec,max_epi_tick))
print ("n_warmup_epi:[%d], buffer_limit:[%.d], buffer_warmup:[%d]"%
       (n_warmup_epi,buffer_limit,buffer_warmup))

device = 'cpu' # cpu / mps / cuda
replay_buffer = ReplayBufferClass(buffer_limit, device=device)
actor_arg = {'obs_dim':gym.o_dim,'h_dims':[256,256],'out_dim':gym.a_dim,
             'max_out':max_torque,'init_alpha':init_alpha,'lr_actor':lr_actor,
             'lr_alpha':lr_alpha,'device':device}
critic_arg = {'obs_dim':gym.o_dim,'a_dim':gym.a_dim,'h_dims':[256,256],'out_dim':1,
              'lr_critic':lr_critic,'device':device}
actor           = ActorClass(**actor_arg).to(device)
critic_one      = CriticClass(**critic_arg).to(device)
critic_two      = CriticClass(**critic_arg).to(device)
critic_one_trgt = CriticClass(**critic_arg).to(device)
critic_two_trgt = CriticClass(**critic_arg).to(device)
print ("Ready.")

# Modify floor friction priority
env.model.geom('floor').priority = 1 # 0=>1
print ("Floor priority:%s"%(env.model.geom('floor').priority))
gym.env.ctrl_ranges[:,0] = -max_torque
gym.env.ctrl_ranges[:,1] = +max_torque
print ("gym.env.ctrl_ranges:\n",gym.env.ctrl_ranges)

REMOVE_PREV_FILES = False # remove previous files

# Loop
np.random.seed(seed=0) # fix seed
print ("Start training.")
for epi_idx in range(n_episode+1): # for each episode
    zero_to_one = epi_idx/n_episode
    one_to_zero = 1-zero_to_one
    # Reset gym
    s = gym.reset()
    # Loop
    USE_RANDOM_POLICY = (np.random.rand()<(0.1*one_to_zero)) or (epi_idx < n_warmup_epi)
    reward_total,reward_forward = 0.0,0.0
    for tick in range(max_epi_tick): # for each tick in an episode
        if USE_RANDOM_POLICY:
            a_np = gym.sample_action()
        else:
            a,log_prob = actor(np2torch(s,device=device))
            a_np = torch2np(a)
        # Step
        s_prime,reward,done,info = gym.step(a_np,max_time=max_epi_sec)
        replay_buffer.put((s,a_np,reward,s_prime,done))
        reward_total += reward 
        # reward_forward += info['r_forward']
        s = s_prime
        if done is True: break # terminate condition
        
        # Replay buffer
        if replay_buffer.size() > buffer_warmup:
             for _ in range(n_update_per_tick): 
                mini_batch = replay_buffer.sample(batch_size)
                # Update critics
                td_target = get_target(
                    actor,
                    critic_one_trgt,
                    critic_two_trgt,
                    gamma      = gamma,
                    mini_batch = mini_batch,
                    device     = device,
                )
                critic_one.train(td_target,mini_batch)
                critic_two.train(td_target,mini_batch)
                # Update actor
                actor.train(
                    critic_one,
                    critic_two,
                    target_entropy = -gym.a_dim,
                    mini_batch     = mini_batch,
                )
                # Soft update of critics
                critic_one.soft_update(tau=tau,net_target=critic_one_trgt)
                critic_two.soft_update(tau=tau,net_target=critic_two_trgt)

    # Compute x_diff
    x_diff = gym.env.get_p_body('torso')[0]
    
    # Print
    if (epi_idx%print_every)==0:
        epi_tick = tick
        print ("[%d/%d][%.1f%%]"%(epi_idx,n_episode,100.0*(epi_idx/n_episode)))
        print ("  reward:[%.1f] x_diff:[%.3f] epi_len:[%d/%d] buffer_size:[%d] alpha:[%.2f]"%
               (reward_total,x_diff,epi_tick,max_epi_tick,
                replay_buffer.size(),actor.log_alpha.exp()))
    
    # Evaluation
    if (epi_idx%eval_every)==0:
        if RENDER_EVAL: gym.init_viewer()
        s = gym.reset()
        reward_total = 0.0
        for tick in range(max_epi_tick):
            a,_ = actor(np2torch(s,device=device),SAMPLE_ACTION=False)
            s_prime,reward,done,info = gym.step(torch2np(a),max_time=max_epi_sec)
            reward_total += reward
            if RENDER_EVAL and ((tick%5) == 0):
                gym.render(
                    TRACK_TORSO      = True,
                    PLOT_WORLD_COORD = True,
                    PLOT_TORSO_COORD = True,
                    PLOT_SENSOR      = True,
                    PLOT_CONTACT     = True,
                    PLOT_TIME        = True,
                )
            s = s_prime
            if RENDER_EVAL:
                if not gym.is_viewer_alive(): break
        if RENDER_EVAL: gym.close_viewer()
        x_diff = gym.env.get_p_body('torso')[0]
        print ("  [Eval] reward:[%.3f] x_diff:[%.3f] epi_len:[%d/%d]"%
               (reward_total,x_diff,tick,max_epi_tick))

    # Save network
    if (epi_idx%save_every)==0:
        pth_path = './result/weights/sac_%s/episode_%d.pth'%(gym.name.lower(),epi_idx)
        dir_path = os.path.dirname(pth_path)
        if not os.path.exists(dir_path): os.makedirs(dir_path)
        if (epi_idx == 0) and REMOVE_PREV_FILES: # remove all existing files
            files = os.listdir(path=dir_path)
            print ("  [Save] Remove existing [%d] pth files."%(len(files)))
            for file in files: os.remove(os.path.join(dir_path,file))
        torch.save(actor.state_dict(),pth_path)
        print ("  [Save] [%s] saved."%(pth_path))

print ("Done.")