# Learn To Race

Neural network learns how to drive a car on a track using Evolutionary Algorithms and Reinforcement Learning
Simple 2D simulation with **pyglet** & **numpy**.

This fork was utilized to write the article [Learn to Race: solving with Evolutionary Algorithm and Reinforcement Learning](/Learn%20to%20Race%20solving%20with%20Evolutionary%20Algorithms%20and%20Reinforcement%20Learning.pdf)

## ▶️️ HOW TO RUN?

### Install packages

    pip install -r requirements.txt

### Run main file

Should work with `Python 3.0` and higher.

For example:

    py -3.10 .\__main__.py

Or use a virtual environment.

### Config (Optional)

**evolutionary_settings.json**

    {
        "width": 1280
        "height": 720
        "friction": 0.1
        "render_timestep": 0.025 // time between frames in seconds - 0.025s = 40 FPS
        "timeout_seconds": 30 // maximum time for each gen
        "population": 40 // number of cars
        "mutation_rate": 0.6 // mutation rate after gen
    }

**reinforcement_settings.json**

    	{
    		"width": 1280,
    		"height": 720,
    		"friction": 0.1,
    		"render_timestep": 0.025,
    		"timeout_seconds": 30,
    		"population": 1,
    		"mutation_rate": 0.0,
    		"gamma": 0.99,
    		"batch_size": 256
    	}

**default_nn_config.json** - Default car config for new saves.

    {
        "name": "test",
        "acceleration": 1,
        "friction": 0.95,
        "max_speed": 30,
        "rotation_speed": 4,
        "shape": [6, 4, 3, 2], // neural network shape - do not change first and last layer
        "max_score": 0,
        "gen_count": 0
    }

**ddpg_config.json**

    	{
    		"name": "DDPG",
    		"acceleration": 1,
    		"friction": 0.95,
    		"max_speed": 30,
    		"rotation_speed": 4,
    		"shape": [6, 32, 32, 2],
    		"max_score": 0,
    		"gen_count": 0
    	}

**td3_config.json**

    	{
    		"name": "TD3",
    		"acceleration": 1,
    		"friction": 0.95,
    		"max_speed": 30,
    		"rotation_speed": 4,
    		"shape": [6, 32, 32, 2],
    		"max_score": 0,
    		"gen_count": 0
    	}

## 🏎️ Environment and Trach Generation

| ![image](https://user-images.githubusercontent.com/46631861/161503165-7a99e1e1-d726-4797-8167-4bb582fa3457.png) | ![track-generation](https://user-images.githubusercontent.com/46631861/161503022-bf0ca0d1-f678-48ce-b570-5bcaaa47b6f3.gif) |
| --------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |

## Evolutionary Algorithm

Write "e"/"evolutionary" when running the app to choose the evolutionary algorithm to train the MLPs

## Reinforcement Learning

Write "r"/"reinforcement" when running the app to choose the reinforcement algorithm to train the MLPs

### DDPG and TD3

Write down "d"/"ddpg" or "t"/"td3" to choose which algorithm you want to utilize to train.

## ⚖️ LICENSE

Shield: [![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg
[car-sensors]: https://raw.githubusercontent.com/JPVercosa/LearnToRace/master/.github/images/car_sensors.png
