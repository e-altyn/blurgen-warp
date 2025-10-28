
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def plot(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('save_path', type=str, default='assets/loss_terms_plot.png')
    args = parser.parse_args()
    
    # Create assets folder if it doesn't exist
    os.makedirs('assets', exist_ok=True)

    # Read the file
    with open(args.path, 'r') as f:
        content = f.read()
        
    # Extract loss terms data
    # Pattern: "Loss terms: <value1> | <value2> | <value3> | <value4>"
    loss_terms_pattern = r'\[Val (\d+)/\d+\][^\n]*\nLoss terms: ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+)'
    matches = re.findall(loss_terms_pattern, content)

    # Convert to structured data
    data = []
    for match in matches:
        epoch = int(match[0])
        term1 = float(match[1])
        term2 = float(match[2])
        term3 = float(match[3])
        term4 = float(match[4])
        data.append([epoch, term1, term2, term3, term4])

    # Create DataFrame
    df = pd.DataFrame(data, columns=['Epoch', 'Loss_Term_1', 'Loss_Term_2', 'Loss_Term_3', 'Loss_Term_4'])

    # Create the plot
    plt.figure(figsize=(12, 7))

    plt.plot(df['Epoch'], df['Loss_Term_1'], label='Loss Term 1', linewidth=2, marker='o', markersize=3, markevery=20)
    plt.plot(df['Epoch'], df['Loss_Term_2'], label='Loss Term 2', linewidth=2, marker='s', markersize=3, markevery=20)
    plt.plot(df['Epoch'], df['Loss_Term_3'], label='Loss Term 3', linewidth=2, marker='^', markersize=3, markevery=20)
    plt.plot(df['Epoch'], df['Loss_Term_4'], label='Loss Term 4', linewidth=2, marker='d', markersize=3, markevery=20)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.title('Training Loss Terms Progression Across Epochs', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    # Save the plot
    output_path = 'assets/loss_terms_plot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}\n")

    plt.close()
