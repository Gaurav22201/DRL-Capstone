import sys
from rlai.gpi.temporal_difference.evaluation import Mode, evaluate_q_pi
from rlai.gpi.state_action_value import ActionValueMdpAgent
from rlai.core.environments.gymnasium import Gym
from rlai.gpi.state_action_value.tabular import TabularStateActionValueEstimator
from numpy.random import RandomState

# Initialize environment
environment = Gym(
    gym_id='CartPole-v1',
    render_every_nth_episode=1,
    T=1000,
    random_state=RandomState(1234)
)

# Initialize agent
agent = ActionValueMdpAgent(
    name='CartPole-Q-Learning',
    gamma=0.99,
    q_S_A=TabularStateActionValueEstimator(
        environment=environment,
        epsilon=0.1,
        continuous_state_discretization_resolution=0.1
    ),
    random_state=RandomState(1234)
)

# Train agent
evaluate_q_pi(
    agent=agent,
    environment=environment,
    num_episodes=100,
    num_updates_per_improvement=None,
    alpha=0.1,
    mode=Mode.Q_LEARNING,
    n_steps=1,
    planning_environment=None
)

# Save agent
import pickle
with open('trained_agents/cartpole/tabular/cartpole_agent_new.pickle', 'wb') as f:
    pickle.dump(agent, f) 