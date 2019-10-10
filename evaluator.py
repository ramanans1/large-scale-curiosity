from wrappers import ProcessFrame84, make_dm_suite
from baselines.common.atari_wrappers import FrameStack
from baselines.common.distributions import make_pdtype
from baselines import logger
import numpy as np
from cnn_policy import CnnPolicy
import cv2
import os
from tensorflow.python.tools import inspect_checkpoint as chkp
import gym

class Evaluator(object):
    def __init__(self, env_name, num_episodes, exp_name, policy):
        self.exp_name = exp_name
        self.env = make_dm_suite(task=env_name)
        #self.env = ProcessFrame84(self.env, crop=False)
        #self.env = FrameStack(self.env, 4)
        self.num_episodes = 1
        self.policy = policy

    def eval_model(self, ep_num):
        for i in range(self.num_episodes):
            ep_images = []
            ob = self.env.reset()
            ob = np.array(ob)
            eprews = []
            if i == 0:
                ep_images.append(self.env.last_observation)
            for step in range(900):
                action, vpred, nlp = self.policy.get_ac_value_nlp_eval(ob)
                ob, rew, done, info = self.env.step(action[0])
                if i == 0:
                    ep_images.append(self.env.last_observation)
                if rew is None:
                    eprews.append(0)
                else:
                    eprews.append(rew)
                if done:
                    print("Episode finished after {} timesteps".format(step+1))
                    print("Episode Reward is {}".format(sum(eprews)))
                    break
            #dirname = os.path.abspath(os.path.dirname(__file__))
            dirname = logger.get_dir()
            image_folder = 'images'
            image_path = os.path.join(dirname,image_folder)
            os.makedirs(image_path, exist_ok=True)  # succeeds even if directory exists.
            vid_file = os.path.join(image_path, self.exp_name +"_{}_{}_".format(ep_num, i) + ".avi")
            out = cv2.VideoWriter(vid_file,cv2.VideoWriter_fourcc(*'DIVX'), 15, (84,84))
            for j in range(len(ep_images)):
                image_file = os.path.join(dirname, image_folder, self.exp_name +"_{}_{}_{}_".format(ep_num, i, j) + ".png")
                print('--EPLEN--',len(ep_images))
                #assert 1==2
                #print(ep_images[j])
                #assert 1==2
                out.write(ep_images[j])
                #cv2.imwrite(image_file, ep_images[j])
            out.release()
            print("Episode {} cumulative reward: {}".format(i, sum(eprews)))
