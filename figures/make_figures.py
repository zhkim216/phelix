import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt
import os

# Sample data (replace these lists with your actual data)
data_dict = {
    0 : {
    'Sidechain MPNN-GVP': [46.7, 47.6, 48.3, 49.0, 49.8, 50.5, 51.2, 51.8, 52.4, 52.7], 
    'Sidechain MPNN': [46.2, 47.0, 47.7, 48.5, 49.2, 49.9, 50.5, 51.2, 51.8, 52.1],
    'Baseline MPNN': [46.1, 46.4, 46.8, 47.2, 47.5, 47.8, 48.1, 48.4, 48.7, 48.8],
    },
    0.3 : {
    'Sidechain MPNN-GVP': [35.5,36.6, 37.6, 38.6, 39.5, 40.5, 41.3, 42.1, 42.9, 43.4], 
    'Sidechain MPNN': [33.8,34.9,35.9, 36.8, 37.7, 38.6, 39.4, 40.3, 41.1, 41.5],
    'Baseline MPNN': [32.5,33, 33.5, 34, 34.4, 34.7, 35.2, 35.6, 35.9, 36.0],
    }
}

# Define x-axis values: % Partial Sequence
x_values = np.linspace(0, 0.9, 10)

# Colors and styles for the lines
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
linestyles = ['-', '--', '-.']

# Generate a separate plot for each noise level
for noise_level, models_data in data_dict.items():
    plt.figure(figsize=(8, 6))
    
    # Plot each model's data with specified style
    for (label, y_values), color, linestyle in zip(models_data.items(), colors, linestyles):
        plt.plot(x_values, y_values, label=label, color=color, linestyle=linestyle, linewidth=2, marker='o')
    
    # Labels and title
    plt.xlabel('% Partial Sequence', fontsize=14)
    plt.ylabel('Sequence Recovery', fontsize=14)
    plt.title(f'Sequence recovery at varying partial sequence conditioning\n(Noise Level: {noise_level})', fontsize=16)
    
    # Dynamic y-ticks based on min and max values in the data
    min_val = min(min(values) for values in models_data.values()) - 5
    max_val = max(max(values) for values in models_data.values()) + 5

    plt.ylim(min_val, max_val)
    plt.xticks(x_values)

    # Add legend and grid
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, alpha = 0.5)

    # Calculate the fold change in accuracy between the highest and lowest values at x=0.9
    y_values_at_09 = [model_data[9] for model_data in models_data.values()]
    max_value = max(y_values_at_09)
    min_value = min(y_values_at_09)
    fold_change = max_value / min_value

    # Add vertical red line at x = 0.9, only between the min and max y-values, with arrows at both ends
    plt.vlines(0.9, min_value, max_value, color='red', linestyle='-', linewidth=2, zorder=5)
    plt.annotate('', xy=(0.9, max_value), xytext=(0.9, min_value), 
                 arrowprops=dict(facecolor='red', edgecolor='red', shrink=0.05, width=1.5, headwidth=8, zorder=5))

    # Annotate the fold change on the left of the line in red
    plt.text(0.815, (max_value + min_value) / 2, f'{fold_change:.2f}x', fontweight='bold', color='red', fontsize=12, verticalalignment='center', zorder=5)
    
    # Save the plot to the specified path
    output_path = os.path.join('out', f'seq_recovery_noise_{noise_level}.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    # Show the plot
    plt.show()
