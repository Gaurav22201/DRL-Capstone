#!/bin/bash

# Script to visualize the final training statistics

# Create plots directory if it doesn't exist
mkdir -p plots

# Run the visualization script
python3 visualize_stats.py stats/final_stats.json --output-dir plots

echo "Visualization complete! Results saved in plots/ directory." 