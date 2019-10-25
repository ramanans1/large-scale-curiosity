#!/usr/bin/env python
import os
import yaml

os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

try:
    from OpenGL import GLU
except:
    print("no OpenGL.GLU")
import functools
import os.path as osp
from functools import partial

import gym
import tensorflow as tf
from baselines import logger
from baselines.bench import Monitor
from baselines.common.atari_wrappers import NoopResetEnv, FrameStack
from mpi4py import MPI

from auxiliary_tasks import FeatureExtractor, InverseDynamics, VAE, JustPixels
from cnn_policy import CnnPolicy
from cppo_agent import PpoOptimizer
from dynamics import Dynamics, UNet
from utils import random_agent_ob_mean_std
from wrappers import MontezumaInfoWrapper, make_dm_suite, make_mario_env, make_robo_pong, make_robo_hockey, \
    make_multi_pong, AddRandomStateToInfo, MaxAndSkipEnv, ProcessFrame84, ExtraTimeLimit, TempMonitor

import datetime
import multiprocessing

def start_experiment(**args):
    log, tf_sess, ldir = get_experiment_environment(**args)
    make_env = partial(make_env_all_params, add_monitor=True, args=args, logdir=ldir)
    trainer = Trainer(make_env=make_env,
                      num_timesteps=args['num_timesteps'], hps=args,
                      envs_per_process=args['envs_per_process'],
                      num_dyna=args['num_dynamics'],
                      var_output=args['var_output'])

    with log, tf_sess:
        logdir = logger.get_dir()
        print("results will be saved to ", logdir)
        with open(os.path.join(logdir,'run_config.yaml'),'w') as outfile:
            yaml.dump(args,outfile,default_flow_style=False)
        trainer.train()


class Trainer(object):
    def __init__(self, make_env, hps, num_timesteps, envs_per_process, num_dyna, var_output):
        self.make_env = make_env
        self.hps = hps
        self.envs_per_process = envs_per_process
        self.num_timesteps = num_timesteps
        self._set_env_vars()

        self.policy = CnnPolicy(
            scope='pol',
            ob_space=self.ob_space,
            ac_space=self.ac_space,
            hidsize=512,
            feat_dim=512,
            ob_mean=self.ob_mean,
            ob_std=self.ob_std,
            layernormalize=False,
            nl=tf.nn.leaky_relu)

        self.feature_extractor = {"none": FeatureExtractor,
                                  "idf": InverseDynamics,
                                  "vaesph": partial(VAE, spherical_obs=True),
                                  "vaenonsph": partial(VAE, spherical_obs=False),
                                  "pix2pix": JustPixels}[hps['feat_learning']]
        self.feature_extractor = self.feature_extractor(policy=self.policy,
                                                        features_shared_with_policy=False,
                                                        feat_dim=512,
                                                        layernormalize=hps['layernorm'])

        self.dynamics_class = Dynamics if hps['feat_learning'] != 'pix2pix' else UNet

        #create dynamics list
        self.dynamics_list = []
        for i in range(num_dyna):
            self.dynamics_list.append(self.dynamics_class(auxiliary_task=self.feature_extractor,
                                          predict_from_pixels=hps['dyn_from_pixels'],
                                          feat_dim=512, env_kind=hps['env_kind'],
                                          scope='dynamics_{}'.format(i),
                                          var_output=var_output)
                                      )

        self.agent = PpoOptimizer(
            scope='ppo',
            ob_space=self.ob_space,
            ac_space=self.ac_space,
            stochpol=self.policy,
            use_news=hps['use_news'],
            gamma=hps['gamma'],
            lam=hps["lambda"],
            nepochs=hps['nepochs'],
            nminibatches=hps['nminibatches'],
            lr=hps['lr'],
            cliprange=0.1,
            nsteps_per_seg=hps['nsteps_per_seg'],
            nsegs_per_env=hps['nsegs_per_env'],
            ent_coef=hps['ent_coeff'],
            normrew=hps['norm_rew'],
            normadv=hps['norm_adv'],
            ext_coeff=hps['ext_coeff'],
            int_coeff=hps['int_coeff'],
            dynamics_list=self.dynamics_list,
            exp_name = hps['exp_name'],
            env_name = hps['env']
        )

        self.agent.to_report['aux'] = tf.reduce_mean(self.feature_extractor.loss)
        self.agent.total_loss += self.agent.to_report['aux']
        
        self.agent.to_report['dyn_loss'] = tf.reduce_mean(self.dynamics_list[0].partial_loss)
        for i in range(1,num_dyna):
            self.agent.to_report['dyn_loss'] += tf.reduce_mean(self.dynamics_list[i].partial_loss)

        self.agent.total_loss += self.agent.to_report['dyn_loss']
        self.agent.to_report['feat_var'] = tf.reduce_mean(tf.nn.moments(self.feature_extractor.features, [0, 1])[1])

    def _set_env_vars(self):
        env = self.make_env(0, add_monitor=False)
        assert self.hps['env_kind'] == "dm_suite" #Current code only runs with dm_suite, because of added action space min and max for clipping
        self.ac_space_min, self.ac_space_max = env.ac_space_min , env.ac_space_max
        self.ob_space, self.ac_space = env.observation_space, env.action_space
        self.ob_mean, self.ob_std = random_agent_ob_mean_std(env)
        del env
        self.envs = [functools.partial(self.make_env, i) for i in range(self.envs_per_process)]

    def train(self):
        self.agent.start_interaction(self.envs, nlump=self.hps['nlumps'], dynamics_list=self.dynamics_list)
        if self.hps['ckptpath'] is not None:
            self.agent.restore_model(logdir = self.hps['ckptpath'], exp_name = self.hps['exp_name'])
        while True:
            info = self.agent.step()
            #print('PRINTINFO',info)
            #assert 1==2
            if info['update']:
                logger.logkvs(info['update'])
                logger.dumpkvs()
                if info['update']['n_updates'] % 30 == 0:
                    self.agent.save_model(logdir = logger.get_dir(), exp_name = self.hps['exp_name'], global_step = info['update']['n_updates'])
            if self.agent.rollout.stats['tcount'] > self.num_timesteps:
                break

        self.agent.stop_interaction()


def make_env_all_params(rank, add_monitor, args, logdir):
    if args["env_kind"] == 'atari':
        env = gym.make(args['env'])
        assert 'NoFrameskip' in env.spec.id
        env = NoopResetEnv(env, noop_max=args['noop_max'])
        env = MaxAndSkipEnv(env, skip=4)
        env = ProcessFrame84(env, crop=False)
        env = FrameStack(env, 4)
        env = ExtraTimeLimit(env, args['max_episode_steps'])
        if 'Montezuma' in args['env']:
            env = MontezumaInfoWrapper(env)
        env = AddRandomStateToInfo(env)
    elif args["env_kind"] == 'mario':
        env = make_mario_env()
    elif args["env_kind"] == "retro_multi":
        env = make_multi_pong()
    elif args["env_kind"] == 'robopong':
        if args["env"] == "pong":
            env = make_robo_pong()
        elif args["env"] == "hockey":
            env = make_robo_hockey()
    elif args["env_kind"] == "dm_suite":
        env = make_dm_suite(task=args["env"],logdir=logdir)

    if add_monitor:
        env = TempMonitor(env)

    return env


def get_experiment_environment(**args):
    from utils import setup_mpi_gpus, setup_tensorflow_session
    from baselines.common import set_global_seeds
    from gym.utils.seeding import hash_seed
    process_seed = args["seed"] + 1000 * MPI.COMM_WORLD.Get_rank()
    process_seed = hash_seed(process_seed, max_bytes=4)
    set_global_seeds(process_seed)
    setup_mpi_gpus()
    logdir = './'+args["logdir"]+'/' + datetime.datetime.now().strftime(args["expID"] + "-openai-%Y-%m-%d-%H-%M-%S-%f")
    logger_context = logger.scoped_configure(dir=logdir,
                                             format_strs=['stdout', 'log',
                                                          'csv', 'tensorboard']
                                             if MPI.COMM_WORLD.Get_rank() == 0 else ['log'])
    tf_context = setup_tensorflow_session()
    return logger_context, tf_context, logdir


def add_environments_params(parser):
    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4',
                        type=str)
    parser.add_argument('--max-episode-steps', help='maximum number of timesteps for episode', default=4500, type=int)
    parser.add_argument('--env_kind', type=str, default="atari")
    parser.add_argument('--noop_max', type=int, default=30)


def add_optimization_params(parser):
    parser.add_argument('--lambda', type=float, default=0.95)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--nminibatches', type=int, default=8)
    parser.add_argument('--norm_adv', type=int, default=1)
    parser.add_argument('--norm_rew', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--ent_coeff', type=float, default=0.001)
    parser.add_argument('--nepochs', type=int, default=3)
    parser.add_argument('--num_timesteps', type=int, default=int(1e8))


def add_rollout_params(parser):
    parser.add_argument('--nsteps_per_seg', type=int, default=128)
    parser.add_argument('--nsegs_per_env', type=int, default=1)
    parser.add_argument('--envs_per_process', type=int, default=128)
    parser.add_argument('--nlumps', type=int, default=1)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_environments_params(parser)
    add_optimization_params(parser)
    add_rollout_params(parser)
    multiprocessing.set_start_method('spawn')
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--dyn_from_pixels', type=int, default=0)
    parser.add_argument('--use_news', type=int, default=0)
    parser.add_argument('--ext_coeff', type=float, default=0.)
    parser.add_argument('--int_coeff', type=float, default=1.)
    parser.add_argument('--layernorm', type=int, default=0)
    parser.add_argument('--feat_learning', type=str, default="none",
                        choices=["none", "idf", "vaesph", "vaenonsph", "pix2pix"])
    parser.add_argument('--expID',type=str,default='0001')
    parser.add_argument('--logdir',type=str,default='logs')
    parser.add_argument('--ckptpath',type=str,default=None)
    parser.add_argument('--num_dynamics', type=int, default=5)
    parser.add_argument('--var_output', action='store_true', default=True)

    args = parser.parse_args()

    start_experiment(**args.__dict__)
