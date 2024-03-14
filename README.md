<h1> MADDPG (Multi-Agent Deep Deterministic Policy Gradient) Algorithm </h1>
<h2>Introduction</h2>

MADDPG is a reinforcement learning algorithm designed for cooperative multi-agent environments.
    
<h2>Key Features</h2>
    <ul>
        <li>Decentralized execution with centralized training</li>
        <li>Utilizes actor-critic architecture</li>
        <li>Supports continuous action spaces</li>
        <li>Uses off-policy learning with replay buffer</li>
    </ul>

<h2>Usage</h2>
To use MADDPG:
    <ol>
        <li>Initialize the agent neural networks</li>
        <li>Interact with the environment and collect experiences</li>
        <li>Update the actor and critic networks using the collected experiences</li>
        <li>Repeat steps 2-3 until convergence</li>
    </ol>

<h2>References</h2>
    <ul>
        <li><a href="https://arxiv.org/abs/1706.02275">MADDPG Paper</a></li>
        <li><a href="https://github.com/openai/maddpg">Official MADDPG GitHub Repository</a></li>
    </ul>


