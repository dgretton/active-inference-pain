
## Development Status
⚠️ **Note: This project is currently under active development** ⚠️
This is a work in progress, and many features are still being implemented or refined. The current version should be considered a proof of concept rather than a final implementation. We expect significant changes and improvements in the coming weeks.


# Active Inference Models of Pain Learning

## Overview
Our aim is to use active inference to provide a computational account of how pain arises and may become chronic. This project implements a simple demonstration of this phenomenon through a simulated "worm" agent that learns to avoid noxious stimuli.

## Theoretical Background
For an introduction to the Active Inference Framework (AIF), we suggest [this tutorial](https://www.sciencedirect.com/science/article/pii/S0022249621000973) and also [this paper](https://www.nature.com/articles/s41598-021-89047-0.pdf) applying AIF to modeling therapeutic change.

## Current Implementation
Our model implements a simple organism (visualized as a segmented worm) that must learn about and navigate potentially harmful regions in its environment. The simulation includes:

### Environment Features
- Weird smell region: A zone that provides predictive information about approaching danger
- Nociceptive region: A zone that represents actual tissue damage/pain
- Continuous physical movement with realistic segment physics

### Agent Architecture
The agent implements active inference with:
- Joint observation modality combining:
  - Nociception (pain) signals
  - Weird smell signals
- Two possible hidden states:
  - Safe
  - Harmful
- Two possible actions:
  - Stay
  - Retreat
- Learning dynamics implemented through:
  - Updated A-matrix (likelihood mapping) for weird smell-danger associations
  - Preference learning for pain avoidance

### Key Features
1. Real-time visualization of:
   - Agent position and movement
   - Current beliefs about safety
   - Warning and nociception status
   - Selected actions
2. Configurable parameters for:
   - Learning rates
   - Movement speeds
   - Environment dimensions
   - Weird smell/nociception zone properties

## Usage
To run the simulation, after installing the relevant dependencies:
```bash
cd worm
python3 worm_simulation.py
```

For parallel simulation experiments:
```bash
cd worm
python3 worm_simulation.py --parallel
```


## License
MIT License

Copyright (c) 2024 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
# Learning Worm Simulation

A visual simulation that demonstrates how artificial agents can learn from experience, similar to how animals learn to avoid danger through association.

## What This Shows

Imagine a simple worm-like creature moving through an environment. At first, it doesn't know what's safe or dangerous. Through trial and error, it learns that certain smells predict painful experiences, just like how a real animal might learn to avoid areas that smell like predators.

### The Learning Process

**What the worm experiences:**
- A "weird smell" zone (shown in orange)
- A "nociception" (pain) zone (shown in red) 
- The smell zone comes before the pain zone

**How learning happens:**
1. **Early episodes**: The worm explores randomly, sometimes encountering smell followed by pain
2. **Gradual learning**: After repeated experiences, the worm starts to associate the smell with upcoming pain
3. **Smart behavior**: Eventually, the worm learns to retreat when it detects the smell, avoiding pain entirely

This mimics how real animals learn - a mouse that gets shocked after hearing a tone will eventually run away just from hearing the tone, even before any shock occurs.

## Running the Simulation

### Requirements
First install the needed software:
```bash
pip install numpy pygame pymdp
```

### Start the Simulation
```bash
cd worm
python worm_simulation.py
```

## What You'll See

### The Main Window
- **Red circles**: The worm (head and body segments)
- **Orange rectangle**: "Weird smell" warning zone
- **Red rectangle**: "Nociception" (pain) zone
- **Status text**: Shows current sensations and actions

### The Analysis Panel (Right Side)
- **A Matrices**: Shows what the worm has learned about when smells and pain occur
- **State Beliefs**: Bar chart showing what the worm thinks about its current situation (Safe/Warning/Harmful)
- **Observation Counts**: Statistics on what the worm has experienced

### Console Output
The program prints learning progress after each episode:
```
=== Episode 1 ===
Learning Progress:
  Smell aversion: -0.000    (starts at zero, becomes negative as worm learns to dislike smell)
  A(smell|warning): 0.330   (starts around 0.33, increases as worm learns smell predicts danger)
```

## What to Watch For

### Early Episodes (1-5)
- Worm moves somewhat randomly
- Sometimes reaches the pain zone
- "Smell aversion" stays near 0.000
- Worm doesn't show consistent avoidance behavior

### Learning Phase (5-20 episodes)
- "Smell aversion" becomes increasingly negative (worm dislikes smell more)
- "A(smell|warning)" increases (worm learns smell predicts danger)
- You may see the worm start to retreat when entering the orange smell zone

### Learned Behavior (20+ episodes)
- Worm consistently retreats when detecting smell
- Rarely reaches the red pain zone
- Strong negative "smell aversion" values
- High "A(smell|warning)" values (0.8 or higher)

## The Science Behind It

This simulation demonstrates **classical conditioning** - the same learning process discovered by Ivan Pavlov with his famous dogs. Just as Pavlov's dogs learned to salivate when hearing a bell (because it predicted food), our artificial worm learns to retreat when smelling the odor (because it predicts pain).

The worm uses a computational framework called "active inference" - a theory about how brains might work by constantly predicting what will happen next and updating those predictions based on experience.

## Troubleshooting

**Worm not learning?** 
- Let it run longer (20+ episodes)
- The worm needs to experience the smell→pain sequence multiple times

**Worm always retreating from the start?**
- This means the learning rate might be too high - the worm learned too quickly from just a few experiences

**No change in numbers?**
- Check that the worm is actually reaching the colored zones
- The learning might be very gradual - look for small changes over many episodes

This simulation helps us understand how simple learning rules can lead to intelligent-seeming behavior, and provides insights into both artificial intelligence and animal cognition.
