import gym
from gym.envs.registration import register
#import dm_suite_envs as dm_suite
#from .dm_suite_envs import DiscretizeActionWrapper, MultiDiscreteToUsual, ActionRepeat, PixelObservations, MaximumDuration, ConvertTo32Bit, ConcatObservation
#from .joint_pong import DiscretizeActionWrapper, MultiDiscreteToUsual

register(
    id='RoboschoolPong-v2',
    entry_point='.joint_pong:RoboschoolPongJoint',
    max_episode_steps=10000,
    tags={"pg_complexity": 20 * 1000000},
)

# register(
#     id='RoboschoolHockey-v1',
#     entry_point='.joint_hockey:RoboschoolHockeyJoint',
#     max_episode_steps=1000,
#     tags={"pg_complexity": 20 * 1000000},
#)
register(
    id='CheetahRun-v1',
    entry_point='roboenvs.dm_suite_envs:CheetahRun',
    max_episode_steps = 1000,
    tags={"pg_complexity": 20 * 1000000},
)

register(
    id='WalkerWalk-v1',
    entry_point='roboenvs.dm_suite_envs:WalkerWalk',
    max_episode_steps = 1000,
    tags={"pg_complexity": 20 * 1000000},
)

def make_robopong():
    return gym.make("RoboschoolPong-v2")


def make_robohockey():
    return gym.make("RoboschoolHockey-v1")

def make_cheetah_run():
    env_param = {}
    env_param["action_repeat"] = 4
    env_param["max_length"] = 1000 // env_param["action_repeat"]
    #env_param["max_length"] = 250
    env_param["state_components"] = ['reward', 'position', 'velocity']
    env = gym.make("CheetahRun-v1")
    return env, env_param

def make_walker_walk():
    env_param = {}
    env_param["action_repeat"] = 2
    env_param["max_length"] = 1000 // env_param["action_repeat"]
    #env_param["max_length"] = 500
    env_param["state_components"] = ['reward', 'height', 'orientations', 'velocity']
    env = gym.make("WalkerWalk-v1")
    return env,env_param
