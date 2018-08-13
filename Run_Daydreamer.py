#!/usr/bin/env python3.6
# Daydreamer.py
# Author: Shawn Beaulieu
# August 7th, 2018


from Daydreamer import Daydreamer

def main():


    params = {

        'render': False,
        'phi_length': 4,
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

    Daydreamer(params)


if __name__ == '__main__':
    main()
