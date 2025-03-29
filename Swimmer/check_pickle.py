import pickle

with open('trained_agents/cartpole/tabular/cartpole_agent.pickle', 'rb') as f:
    agent = pickle.load(f)
    print(f"Agent class: {agent.__class__}")
    print(f"Agent module: {agent.__class__.__module__}") 