#!/usr/bin/env python3.6
# Daydreamer.py
# Author: Shawn Beaulieu
# August 6th, 2018

import os
import gym
import keras
import random
import atari_py

import numpy as np
import matplotlib.pyplot as plt

from Reservoir import MDNRNN
from VariationalBayes import VAE
from skimage.color import rgb2gray
from skimage.transform import resize


def Get_Preprocessed_Frame(observation):

    """
    0) Atari frames: 210 x 160
    1) Get image grayscale
    2) Rescale image 110 x 84
    3) Crop center 84 x 84 (you can crop top/bottom according to the game)
    """
    return(resize(rgb2gray(observation), (110, 84))[13:110-13, :])

def Show_Dream(image):

    plt.imshow(image.reshape(84,84), cmap='Greys')
    plt.show()

class Daydreamer():

    P = {

        'epochs': 20,
        'environments': {0:'MontezumaRevenge-v0', 1:'Frostbite-v0'},
        'latent_dim': 32,
        'gamma': 0.9,
        'epsilon': 1.0,
        'action_space': 18,
        'vae_params': {
                     
            'blueprint': [84*84, 200, 100, 32], #105*80 after flattening
            'convolutions': 0,
            "batch_size": 4000,
            "regularizer": 1E-6,
            "learning_rate": 3E-4,
            "dropout": True,
            "dropout_rate": 0.50,
            "num_classes": 0
        }

    }

    def __init__(self, parameters=dict()):
        
        # Create self."x" variables for each x in P Daydreamer.P, which
        # is updated to contain newly passed variables from parameters dictionary
        self.__dict__.update(Daydreamer.P, **parameters)
        self.world_model = VAE(self.vae_params, meta_graph=None, new_graph=True)
        self.reservoir = MDNRNN()
        self.controller = self.Build_Controller()
        self.memory_bank = []

        self.Play()

    def Build_Controller(self):
       
        """
        For simplicity, use Keras to build a neural network with a single hidden layer.

        """
        controller = keras.models.Sequential()
        controller.add(keras.layers.Dense(units=100, activation='relu', input_shape=(288,)))
        controller.add(keras.layers.Dense(units=18))

        rms = keras.optimizers.RMSprop()
        controller.compile(loss='mean_squared_error', optimizer=rms, metrics=['accuracy'])
        
        return(controller)

    def Play(self):

        for e in range(self.epochs):

            if e % 2 == 0:
                game = self.environments[0]
            else:
                game = self.environments[1]

            print("Playing {0}".format(game))

            env = gym.make(game)
            env.reset()

            running_score = 0

            self.m = 0
            self.action = np.random.choice(range(self.action_space)) # random initial action
            hidden_state = None
            

            """
            DQN frame-skipping: states (4, 84, 84) fed to the controller consist
            of only every fourth state. A sliding window defines overlapping states:
            e.g. S1=(x,y,z,a), S2=(y,z,a,b), S3=(z,a,b,c)... 

            """

            phi_counter = 0 # for tracking frames encountered

            for t in range(10):
  
                """
                'env.step' returns four values:
                (1) observation (e.g. pixel data, sensor data...)
                (2) reward
                (3) done (Bool: yes=terminate, no=running)
                (4) info, for diagnostics

                """

                action_vector = np.zeros([1,self.action_space])
                action_vector[0, self.action] = 1
   
                if self.render:
                    env.render()

                """=========================== ACTION/PERCEPTION ==========================="""

                # Take an action and received feedback from the environment. Preprocess observation
                self.observation, self.reward, self.done, self.info = env.step(self.action)
                self.observation = Get_Preprocessed_Frame(self.observation)
                running_score += self.reward


                present = np.reshape(self.observation, [1,-1])

                if e > 0:

                    self.Dream(present)

                else:

                    self.Perceive(present)


                """=========================== COGITATION ==========================="""

                # Using the untrained reservoir, "predict" the next state and extract hidden_state
                inputs = np.hstack([self.Z, action_vector]).reshape(1,1, self.action_space + self.latent_dim)
                _, hidden_state = self.reservoir.Predict_Next_Frame(inputs, hidden_state)

                #hidden_state = hidden_state.reshape(1,1,50)
                self.Update(action_vector, hidden_state)
  
                """=========================== ACTION SELECTION ==========================="""

                # Epsilon-greed action selection.
                if random.random() < self.epsilon:
                    self.action = np.random.choice(range(self.action_space))

                else:
                    self.action = self.Generate_Behavior(hidden_state, action_vector)
 
                # Slowly introduce learning by intially taking random actions
                # As epsilon decays, actions will primarily result from the controller:
                if self.epsilon > 0.1:
                    self.epsilon -= 1/self.epochs
 
                if self.done:
                    print("Game has finished")
                    break

    def Dream(self, present):


        if self.m < len(self.memory_bank):

            past = np.reshape(self.memory_bank.pop(self.m), [1,-1])
            self.memory_bank = [self.observation] + self.memory_bank

            # Compute cost of the VAE and obtain the compressed representation of
            # the observation received above.
            opt, cost, self.Z, self.dream = self.world_model.Fit(present, past)
            Show_Dream(self.dream)
            self.m += 1

        else:

            """ 
                Once m exceeds length of memory bank, we want to stop triggering the 
                above conditional and instead trigger THIS conditional:

                If we lack corresponding memories from the last playthrough, then reconstruct
                an imagined dream. Latent vector will still contain relevant information about input.

            """
            
            self.m = 1E10
            self.Perceive(present)
            #Show_Dream(self.dream)                  

    def Perceive(self, present):

        self.memory_bank = [self.observation] + self.memory_bank # add new experience to memory
        self.Z, self.dream = self.world_model.Reconstruct(present)

    def Update(self, action_vector, hidden_state):
    
        # Vectorize received reward, where index is the action taken
        reward_vector = np.zeros_like(action_vector)
        reward_vector[0, self.action] = self.reward

        memory = hidden_state[1]
        inputs = np.hstack([self.Z, memory])

        # If our action didn't result in termination, imagine what the next move
        # could yield.
        if not self.done:

            # LSTM tuple contains cell state and memory state:
            imagined_reward = np.max(self.controller.predict(inputs, batch_size=1))
            reward_vector[0, self.action] += self.gamma*imagined_reward 

        # Update the controller on the received action vector
        if not np.sum(reward_vector) == 0:
            self.controller.fit(inputs, reward_vector, batch_size=1, epochs=2)
        

    def Generate_Behavior(self, hidden_state, action_vector):

        # LSTM tuple contains cell state and memory state:
        memory = hidden_state[1]
        inputs = np.hstack([self.Z, memory])
        predicted_reward = self.controller.predict(inputs, batch_size=1)
        action = np.argmax(predicted_reward)
       
        return(action)
