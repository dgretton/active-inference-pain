
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
- Warning region: A zone that provides predictive information about approaching danger
- Nociceptive region: A zone that represents actual tissue damage/pain
- Continuous physical movement with realistic segment physics

### Agent Architecture
The agent implements active inference with:
- Two observation modalities:
  - Nociception (pain) signals
  - Warning signals
- Two possible states:
  - Safe
  - Harmful
- Two possible actions:
  - Stay
  - Retreat
- Learning dynamics implemented through:
  - Updated A-matrix (likelihood mapping) for warning-danger associations
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
   - Warning/nociception zone properties

## Usage
To run the simulation, after installing the relevant dependancies:
```python3 worm_simulation.py```


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
