# 🧠 Decode My Brain

An interactive neuroscience visualization app for exploring neural coding and decoding concepts.

## Overview

This app simulates a population of direction-tuned neurons and demonstrates:
- **Neural Tuning Curves**: How neurons respond to different movement directions
- **Population Coding**: How direction is represented by the combined activity of many neurons
- **Decoding**: How we can infer movement direction from neural activity
- **Game Mode**: Test your ability to decode direction from spike patterns against a model decoder!

Perfect for neuroscience educators, students, or anyone curious about how the brain encodes information.

## Quick Start

### Installation

```bash
# Clone or download this repository
cd NeuroVisualizationApp

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the App

```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`.

## Features

### 📊 Tab 1: Simulation Controls
- Configure the neural population (number of neurons, baseline rate, modulation depth)
- Set trial parameters (duration, noise level, number of trials)
- Run simulations and view summary statistics

### 👁️ Tab 2: Visualize Neural Code
- **Tuning Curves**: See how each neuron responds to different directions
- **Spike Heatmaps**: View spike counts across the population
- **Population Activity**: Bar and polar plots of neural activity
- **Time-Binned Rasters**: Explore activity across time

### 🎮 Tab 3: Decode My Brain (Game Mode)
- Challenge yourself to guess the hidden movement direction from neural activity
- Compete against model decoders (Population Vector or Maximum Likelihood)
- Track your performance with a running scoreboard
- See detailed results with polar comparison plots

### 🔬 Tab 4: Analysis
- **Noise vs Performance**: Explore how decoder accuracy changes with neural variability
- **Normal vs Lesioned**: Compare populations with reduced tuning strength
- **Export Data**: Download simulation data in NPZ or CSV format

## The Neural Model

Each neuron has a **preferred direction** and responds according to a cosine tuning curve:

```
λ(θ) = r₀ + k · cos(θ − μ)
```

Where:
- `λ(θ)` = firing rate for direction θ
- `r₀` = baseline firing rate (Hz)
- `k` = modulation depth (Hz)
- `μ` = preferred direction of the neuron

Spike counts are generated from a Poisson distribution with rate `λ(θ) × T`, where T is the trial duration.

## Decoders

### Population Vector Decoder
Estimates direction by computing the weighted vector sum of preferred directions:

```
θ̂ = atan2(Σ nᵢ sin(μᵢ), Σ nᵢ cos(μᵢ))
```

Where `nᵢ` is the spike count and `μᵢ` is the preferred direction of neuron i.

### Maximum Likelihood Decoder
Computes the log-likelihood over a grid of candidate directions assuming independent Poisson neurons:

```
θ̂ = argmax Σᵢ [nᵢ log λᵢ(θ) - λᵢ(θ)]
```

## File Structure

```
NeuroVisualizationApp/
├── app.py              # Streamlit entry point with tabbed UI
├── simulation.py       # Neural population & spike generation
├── decoders.py         # Population vector & ML decoders
├── visualization.py    # Plotly plotting functions
├── utils.py            # Angle math & export helpers
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Requirements

- Python 3.8+
- Streamlit 1.28+
- NumPy 1.24+
- SciPy 1.10+
- Plotly 5.18+
- Pandas 2.0+

## Tips for Educators

1. **Start Simple**: Begin with a small population (20-30 neurons) to see clear patterns
2. **Explore Tuning**: Use the visualization tab to explain how neurons encode direction
3. **Add Noise**: Increase variance scale to show how noise affects decoding
4. **Lesion Effects**: Use the analysis tab to demonstrate population coding redundancy
5. **Game Mode**: Let students compete to understand the decoding challenge

## License

MIT License - feel free to use, modify, and share!

---

*Built with ❤️ for neuroscience education*

