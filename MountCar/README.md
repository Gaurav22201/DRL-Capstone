# Deep Q-Learning (DQN) for Atari Games

This implementation provides a Deep Q-Learning (DQN) agent that can learn to play Atari games. The implementation includes experience replay and a target network for stable training.

## Features

- Deep Q-Network (DQN) implementation using PyTorch
- Experience replay buffer for improved training stability
- Target network to reduce overestimation
- Atari environment preprocessing (grayscale, resizing, frame skipping)
- Epsilon-greedy exploration strategy

## Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
```

## Usage

To train the DQN agent on the Pong environment:

```bash
python3 dqn_atari.py
```

## Implementation Details

The implementation includes several key components:

1. `AtariPreprocessing`: Handles frame preprocessing (grayscale conversion, resizing, frame skipping)
2. `DQN`: The Q-network architecture with convolutional and fully connected layers
3. `ReplayBuffer`: Stores and samples transitions for experience replay
4. `DQNAgent`: Manages the training process, including action selection and network updates

## Parameters

The main training parameters can be modified in the `main()` function:

- `episodes`: Number of episodes to train (default: 1000)
- `target_update`: Frequency of target network updates (default: 1000 frames)
- `print_every`: Frequency of printing training progress (default: 1 episode)

Agent parameters can be modified when initializing the `DQNAgent`:

- `learning_rate`: Learning rate for the optimizer (default: 1e-4)
- `gamma`: Discount factor (default: 0.99)
- `buffer_size`: Size of the replay buffer (default: 100000)
- `batch_size`: Number of samples per training batch (default: 32)
- `epsilon_start`: Initial exploration rate (default: 1.0)
- `epsilon_final`: Final exploration rate (default: 0.01)
- `epsilon_decay`: Number of frames over which to decay epsilon (default: 10000) 