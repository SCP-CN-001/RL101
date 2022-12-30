Language: [中文](./README.md) | English

# RL101

## About

This repository is a beginner's tutorial about Reinforcement Learning (RL). We expect the readers to have some background in *Probability and Statistics* and *Machine Learning*. Specifically, suppose the readers know the expectation, Bayesian theory, Markov Chain, neural networks, and activation functions. In that case, it will be easier to understand the tutorial's content. It is okay that you know nothing about all the above terms. This tutorial is trying to tell a story that everybody can understand.

In this tutorial, we will start with the basic concepts, then go over the development history of RL. The relationship between RL and optimization methods will be revealed. The Bellman equation will be explained in detail. We will also show the mechanism and implementation of the milestone RL algorithms like DQN, DDPG, SAC, and PPO and their deviations. Finally, we will discuss the specialty of multi-agent RL.

### Motivation

As a researcher working on decision-making in autonomous driving, I have high expectations of RL. The data-driven strategy-based (RL, IL) methods are more flexible. They may handle the corner cases better than the rule-based methods relying on hand-articulated rules. As long as we find the proper algorithm, the data-driven methods will have strong robustness and apply to diversified scenarios. The core of the L3/L4 level autonomous driving belongs to data-driven methods (in my belief).

Although I am very optimistic about the future of data-driven driving decision-making, it is impossible to solve all the problems in this field by myself. I need a community and cooperators to work together. Providing an easy-to-start tutorial with elegant explanations and interesting samples is a good idea to make this field attractive to my potential colleagues. On the other side, I have significantly benefited from the open-source and shared knowledge concept. I have seen insufficiency in current RL tutorials and introductions (I will explain this in *Features* section), so it seems like my chance to contribute to the community. 

This tutorial follows [GPL-3](https://choosealicense.com/licenses/gpl-3.0/#) license. Everybody is welcome to raise a pull request to this repository.

### Features

## Quick Start

### Folder Structure

The folder structure of this repository is as follows:

```shell
.
│   # Record the training process of some vanilla RL algorithms
├── demo_rl
│   │   # Records of the average rewards
│   ├── logs
│   │   # Records of the trained parameters
│   └── models
│   # Implementation of some vanilla RL algorithms based on the original papers. 
│   # All the algorithms are tested executable.
├── rllib
│   ├── algorithms
│   └── utils
│   # A series of Jupyter notebooks that introduce the concepts, mechanism, 
│   # and details of the classic methods with code examples.
└── tutorials
```

### Environment Configuration

## Reference Materials

### Books

- Sutton, Richard S., and Andrew G. Barto. *Reinforcement learning: An introduction*. MIT press, 2018.

### Papers

### Blogs and Websites
