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
    max_episode_steps = 1000//4,
    tags={"pg_complexity": 20 * 1000000},
)

def make_robopong():
    return gym.make("RoboschoolPong-v2")


def make_robohockey():
    return gym.make("RoboschoolHockey-v1")

def make_cheetah():

    env = gym.make("CheetahRun-v1")
    # from dm_control import suite
    # import matplotlib.pyplot as plt
    # import numpy as np
    #
    # max_frame = 90
    #
    # width = 480
    # height = 480
    # video = np.zeros((90, height, width, 3), dtype=np.uint8)
    #
    # # Load one task:
    # #env = suite.load(domain_name="cartpole", task_name="swingup")
    #
    # # Step through an episode and print out reward, discount and observation.
    # # action_spec = env.action_spec()
    # #time_step = env.reset()
    # #while not time_step.last():
    # env.reset()
    # for i in range(max_frame):
    # # action = np.random.uniform(action_spec.minimum,
    # #                          action_spec.maximum,
    # #                          size=action_spec.shape)
    # # time_step = env.step(action)
    #     env.step(env.action_space.sample())
    #     #env.physics.render(480,480,camera_id=1)
    #     img = plt.imshow(env.render())
    #     plt.pause(0.01)
    #     plt.draw()
    #     #video[i] = np.hstack([env.physics.render(height, width, camera_id=0),
    #     #                  env.physics.render(height, width, camera_id=1)])
    # #print(time_step.reward, time_step.discount, time_step.observation)
    # # for i in range(max_frame):
    # #     img = plt.imshow(video[i])
    # #     plt.pause(0.01)  # Need min display time > 0.0.
    # #     plt.draw()
    # assert 1==2
    return env
