import sys
from rlai.policy_gradient.policies.continuous_action import ContinuousActionNormalDistributionPolicy
from rlai.policy_gradient import ParameterizedMdpAgent
from rlai.core.environments.gymnasium import Gym
from rlai.core.environments.gymnasium import ContinuousLunarLanderFeatureExtractor
from numpy.random import RandomState
from rlai.policy_gradient.monte_carlo.reinforce import improve

# Initialize environment
environment = Gym(
    gym_id='LunarLanderContinuous-v3',
    render_every_nth_episode=1,
    T=1000,
    random_state=RandomState(1234)
)

# Initialize feature extractor
feature_extractor = ContinuousLunarLanderFeatureExtractor(
    scale_features=True
)

# Initialize policy
policy = ContinuousActionNormalDistributionPolicy(
    environment=environment,
    feature_extractor=feature_extractor,
    plot_policy=True
)

# Initialize agent
agent = ParameterizedMdpAgent(
    name='LunarLander-PolicyGradient',
    gamma=0.99,
    pi=policy,
    v_S=None,
    random_state=RandomState(1234)
)

# Train agent
improve(
    agent=agent,
    environment=environment,
    num_episodes=100,
    update_upon_every_visit=True,
    alpha=0.001,
    thread_manager=None,
    plot_state_value=True
)

# Save agent
import pickle
with open('trained_agents/lunarlander/policy_gradient/lunarlander_agent_new.pickle', 'wb') as f:
    pickle.dump(agent, f) 