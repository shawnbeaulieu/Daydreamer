# Daydreamer: 

### Under (heavy) construction

Dependencies: NumPy, TensorFlow, Keras

For use in Python 3.6. An extension of [World Models](https://arxiv.org/pdf/1803.10122.pdf) (Ha, Schmidhuber 2018) wherein
agents learn to generate mappings (or more poetically, create analogies) between environments to learn the general
structure underlying similar tasks. Using the OpenAI Gym, the Atari games Montezuma's Revenge and Frostbite are alternately
simulated for E iterations, with the frames seen at a given time step in the previous simulation acting as the target for the agents' 
generative model at the same timestep in the current simulation. Where no prior "memory" exists, the generative model produces
its best guess for what the corresponding frame "should" be.

In contrast with the original World Models paper, reinforcement learing (in this case, a skeletal version of Q-Learning) is used
instead of CMA-ES for optimizing the controller (Keras). This allows for a straightforward implementation of "daydreaming", as opposed to
one in which some population-level feature dictates the mapping from one environment to another.

An untrained Mixture Density RNN (TensorFlow) is used as a reservoir to supply the generative model with temporal information. 
The variational autoencoder (TensorFlow) is, in theory, equipped to handle convolutional encodings, but is currently built for 
simple feedforward network architectures.

At this rudimentary stage, the system is being forced to "dream" in one environment while navigating another. Future versions will 
seek to generate unforced mappings between superficially disparate situations. This process is analogous (!) to how we unconsciously 
extract lessons from literature or film by mapping what we see or read onto aspects of our own life. See [Hofstadter](https://www.amazon.com/Am-Strange-Loop-Douglas-Hofstadter/dp/0465030793) for a more philosophical treatment.


Much more yet to come!

