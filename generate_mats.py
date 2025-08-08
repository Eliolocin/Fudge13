#!/usr/bin/env python3
"""
Academic Materials Generator for LLM Translation Judge Analysis
Generates graphs and tables comparing Prompt-engineered vs Agentic approaches
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

# =============================================================================
# CONFIGURATION CONSTANTS - Adjust these to customize output
# =============================================================================

# Font settings
FONT_SIZE = 18  # Default font size (academic papers typically use 10-12)
FONT_STYLE = 'Times New Roman'  # Academic standard font
SHOW_TITLE = True  # Whether to show titles on charts

# Color scheme
PROMPT_COLOR = '#FF8C00'  # Orange for Prompt-engineered
AGENTIC_COLOR = '#20B2AA'  # Teal for Agentic

# Output settings
OUTPUT_DIR = 'materials'  # Directory for generated materials
DPI = 300  # High-quality output for publication

# Data paths
PROMPT_ANALYSIS_PATH = 'analysis_results/prompt_analysis/llm_judge_analysis_20250805_192856.json'
AGENTIC_ANALYSIS_PATH = 'analysis_results/agentic_analysis/llm_judge_analysis_20250805_101404.json'

# =============================================================================
# MATPLOTLIB CONFIGURATION
# =============================================================================

def configure_matplotlib():
    """
    Configure matplotlib for academic publication quality
    """
    # 1. Set font properties
    plt.rcParams.update({
        'font.family': ['serif'],
        'font.serif': [FONT_STYLE],
        'font.size': FONT_SIZE,
        'axes.labelsize': FONT_SIZE,
        'axes.titlesize': FONT_SIZE + 1,
        'xtick.labelsize': FONT_SIZE - 1,
        'ytick.labelsize': FONT_SIZE - 1,
        'legend.fontsize': FONT_SIZE - 1,
        'figure.titlesize': FONT_SIZE + 2,
        
        # 2. Set high DPI for crisp output
        'figure.dpi': DPI,
        'savefig.dpi': DPI,
        'savefig.format': 'pdf',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        
        # 3. Clean academic styling
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.axisbelow': True,
        
        # 4. Color and styling
        'axes.edgecolor': 'black',
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
    })

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_analysis_data(prompt_path, agentic_path):
    """
    Load analysis results from JSON files
    
    Args:
        prompt_path (str): Path to prompt-engineered analysis results
        agentic_path (str): Path to agentic analysis results
    
    Returns:
        tuple: (prompt_data, agentic_data) dictionaries
    """
    try:
        # 1. Load prompt-engineered data
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_data = json.load(f)
        
        # 2. Load agentic data
        with open(agentic_path, 'r', encoding='utf-8') as f:
            agentic_data = json.load(f)
        
        print(f"[OK] Loaded prompt analysis: {prompt_data['analysis_metadata']['total_evaluations_analyzed']} evaluations")
        print(f"[OK] Loaded agentic analysis: {agentic_data['analysis_metadata']['total_evaluations_analyzed']} evaluations")
        
        return prompt_data, agentic_data
        
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Analysis file not found: {e}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")

def extract_performance_metrics(prompt_data, agentic_data):
    """
    Extract key performance metrics for comparison
    
    Args:
        prompt_data (dict): Prompt-engineered analysis results
        agentic_data (dict): Agentic analysis results
    
    Returns:
        pd.DataFrame: Performance metrics comparison
    """
    # 1. Extract metrics from both datasets
    prompt_metrics = prompt_data['classification_metrics']['metrics']
    agentic_metrics = agentic_data['classification_metrics']['metrics']
    
    # 2. Create comparison dataframe
    metrics_df = pd.DataFrame({
        'Prompt-engineered': [
            prompt_metrics['accuracy'],
            prompt_metrics['precision'], 
            prompt_metrics['recall'],
            prompt_metrics['f1_score']
        ],
        'Agentic': [
            agentic_metrics['accuracy'],
            agentic_metrics['precision'],
            agentic_metrics['recall'], 
            agentic_metrics['f1_score']
        ]
    }, index=['Accuracy', 'Precision', 'Recall', 'F1-Score'])
    
    return metrics_df

def extract_correlation_data(prompt_data, agentic_data):
    """
    Extract correlation analysis data for scatter plots
    
    Args:
        prompt_data (dict): Prompt-engineered analysis results  
        agentic_data (dict): Agentic analysis results
    
    Returns:
        tuple: (prompt_pairs, agentic_pairs) DataFrames with LLM/Human score pairs
    """
    # 1. Extract paired data
    prompt_pairs = pd.DataFrame(prompt_data['correlation_analysis']['paired_data'])
    agentic_pairs = pd.DataFrame(agentic_data['correlation_analysis']['paired_data'])
    
    # 2. Add correlation stats
    prompt_corr = prompt_data['correlation_analysis']['correlation_analysis']['spearman_correlation']
    agentic_corr = agentic_data['correlation_analysis']['correlation_analysis']['spearman_correlation']
    
    return prompt_pairs, agentic_pairs, prompt_corr, agentic_corr

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_performance_comparison(metrics_df, output_path):
    """
    Create performance metrics comparison bar chart
    
    Args:
        metrics_df (pd.DataFrame): Performance metrics comparison
        output_path (str): Output file path
    """
    # 1. Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 2. Create grouped bar chart
    x = np.arange(len(metrics_df.index))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, metrics_df['Prompt-engineered'], width, 
                   label='Prompt-engineered', color=PROMPT_COLOR, alpha=0.8)
    bars2 = ax.bar(x + width/2, metrics_df['Agentic'], width,
                   label='Agentic', color=AGENTIC_COLOR, alpha=0.8)
    
    # 3. Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=FONT_SIZE-1)
    
    # 4. Customize chart
    ax.set_xlabel('Performance Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Performance Comparison: Prompt-engineered vs Agentic' if SHOW_TITLE else '')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df.index)
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    # 5. Save to PDF
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[OK] Generated: {output_path}")

def create_score_distributions(prompt_data, agentic_data, output_path):
    """
    Create score distribution analysis with box plots
    
    Args:
        prompt_data (dict): Prompt-engineered analysis results
        agentic_data (dict): Agentic analysis results  
        output_path (str): Output file path
    """
    # 1. Extract score data
    prompt_pairs = pd.DataFrame(prompt_data['correlation_analysis']['paired_data'])
    agentic_pairs = pd.DataFrame(agentic_data['correlation_analysis']['paired_data'])
    
    # 2. Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # 3. LLM Score Distributions
    ax1.boxplot([prompt_pairs['llm_score'], agentic_pairs['llm_score']], 
                labels=['Prompt-engineered', 'Agentic'],
                patch_artist=True,
                boxprops=dict(facecolor=PROMPT_COLOR, alpha=0.7),
                medianprops=dict(color='black', linewidth=2))
    
    # Update colors for second box
    boxes = ax1.findobj(patches.PathPatch)
    if len(boxes) >= 2:
        boxes[1].set_facecolor(AGENTIC_COLOR)
    
    ax1.set_title('LLM Score Distributions' if SHOW_TITLE else '')
    ax1.set_ylabel('LLM Score')
    ax1.set_ylim(0.5, 5.5)
    
    # 4. Human Score Distributions  
    ax2.boxplot([prompt_pairs['human_score'], agentic_pairs['human_score']],
                labels=['Prompt-engineered', 'Agentic'],
                patch_artist=True,
                boxprops=dict(facecolor=PROMPT_COLOR, alpha=0.7),
                medianprops=dict(color='black', linewidth=2))
    
    # Update colors for second box
    boxes2 = ax2.findobj(patches.PathPatch)
    if len(boxes2) >= 2:
        boxes2[1].set_facecolor(AGENTIC_COLOR)
    
    ax2.set_title('Human Score Distributions' if SHOW_TITLE else '')
    ax2.set_ylabel('Human Score')
    ax2.set_ylim(0.5, 5.5)
    
    # 5. Save to PDF
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[OK] Generated: {output_path}")

def create_correlation_analysis(prompt_pairs, agentic_pairs, prompt_corr, agentic_corr, output_path):
    """
    Create correlation analysis scatter plots
    
    Args:
        prompt_pairs (pd.DataFrame): Prompt-engineered score pairs
        agentic_pairs (pd.DataFrame): Agentic score pairs
        prompt_corr (float): Prompt-engineered correlation coefficient
        agentic_corr (float): Agentic correlation coefficient
        output_path (str): Output file path
    """
    # 1. Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 2. Prompt-engineered scatter plot
    ax1.scatter(prompt_pairs['human_score'], prompt_pairs['llm_score'], 
               color=PROMPT_COLOR, alpha=0.6, s=30)
    ax1.plot([1, 5], [1, 5], 'k--', alpha=0.5, linewidth=1)  # Perfect correlation line
    ax1.set_xlabel('Human Score')
    ax1.set_ylabel('LLM Score') 
    ax1.set_title(f'Prompt-engineered (r={prompt_corr:.3f})' if SHOW_TITLE else f'r={prompt_corr:.3f}')
    ax1.set_xlim(0.5, 5.5)
    ax1.set_ylim(0.5, 5.5)
    ax1.grid(True, alpha=0.3)
    
    # 3. Agentic scatter plot
    ax2.scatter(agentic_pairs['human_score'], agentic_pairs['llm_score'],
               color=AGENTIC_COLOR, alpha=0.6, s=30)
    ax2.plot([1, 5], [1, 5], 'k--', alpha=0.5, linewidth=1)  # Perfect correlation line
    ax2.set_xlabel('Human Score')
    ax2.set_ylabel('LLM Score')
    ax2.set_title(f'Agentic (r={agentic_corr:.3f})' if SHOW_TITLE else f'r={agentic_corr:.3f}')
    ax2.set_xlim(0.5, 5.5)
    ax2.set_ylim(0.5, 5.5)
    ax2.grid(True, alpha=0.3)
    
    # 4. Save to PDF
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[OK] Generated: {output_path}")

def create_consistency_analysis(prompt_data, agentic_data, output_path):
    """
    Create consistency evaluation bar chart
    
    Args:
        prompt_data (dict): Prompt-engineered analysis results
        agentic_data (dict): Agentic analysis results
        output_path (str): Output file path
    """
    # 1. Extract consistency metrics
    prompt_consistency = prompt_data['variation_analysis']['variation_summary']['overall_pass_consistency']
    agentic_consistency = agentic_data['variation_analysis']['variation_summary']['overall_pass_consistency']
    
    prompt_std = prompt_data['variation_analysis']['variation_summary']['overall_score_std']
    agentic_std = agentic_data['variation_analysis']['variation_summary']['overall_score_std']
    
    # 2. Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 3. Pass Consistency comparison
    methods = ['Prompt-engineered', 'Agentic']
    consistency_values = [prompt_consistency, agentic_consistency]
    colors = [PROMPT_COLOR, AGENTIC_COLOR]
    
    bars1 = ax1.bar(methods, consistency_values, color=colors, alpha=0.8)
    ax1.set_ylabel('Pass Consistency')
    ax1.set_title('Pass Consistency Comparison' if SHOW_TITLE else '')
    ax1.set_ylim(0, 1.1)
    
    # Add value labels
    for bar, value in zip(bars1, consistency_values):
        ax1.annotate(f'{value:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=FONT_SIZE-1)
    
    # 4. Score Variability (lower is better)
    std_values = [prompt_std, agentic_std]
    bars2 = ax2.bar(methods, std_values, color=colors, alpha=0.8)
    ax2.set_ylabel('Score Standard Deviation')
    ax2.set_title('Score Variability (Lower is Better)' if SHOW_TITLE else '')
    
    # Add value labels
    for bar, value in zip(bars2, std_values):
        ax2.annotate(f'{value:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 3), textcoords="offset points", 
                    ha='center', va='bottom', fontsize=FONT_SIZE-1)
    
    # 5. Save to PDF
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[OK] Generated: {output_path}")

def create_confusion_matrices(prompt_data, agentic_data, output_path):
    """
    Create confusion matrix comparison heatmaps
    
    Args:
        prompt_data (dict): Prompt-engineered analysis results
        agentic_data (dict): Agentic analysis results
        output_path (str): Output file path
    """
    # 1. Extract confusion matrix data
    prompt_cm = prompt_data['classification_metrics']['confusion_matrix']
    agentic_cm = agentic_data['classification_metrics']['confusion_matrix']
    
    # 2. Convert to matrix format
    prompt_matrix = np.array([[prompt_cm['true_negative'], prompt_cm['false_positive']],
                             [prompt_cm['false_negative'], prompt_cm['true_positive']]])
    
    agentic_matrix = np.array([[agentic_cm['true_negative'], agentic_cm['false_positive']],
                              [agentic_cm['false_negative'], agentic_cm['true_positive']]])
    
    # 3. Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    labels = ['Predicted\nPoor', 'Predicted\nGood']
    tick_labels = ['Actual\nPoor', 'Actual\nGood']
    
    # 4. Prompt-engineered confusion matrix
    im1 = ax1.imshow(prompt_matrix, cmap='Oranges', alpha=0.8)
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(labels)
    ax1.set_yticklabels(tick_labels)
    ax1.set_title('Prompt-engineered' if SHOW_TITLE else '')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax1.text(j, i, prompt_matrix[i, j], ha="center", va="center", 
                           color="black", fontsize=FONT_SIZE+2, weight='bold')
    
    # 5. Agentic confusion matrix  
    im2 = ax2.imshow(agentic_matrix, cmap='GnBu', alpha=0.8)
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(labels)
    ax2.set_yticklabels(tick_labels)
    ax2.set_title('Agentic' if SHOW_TITLE else '')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax2.text(j, i, agentic_matrix[i, j], ha="center", va="center",
                           color="black", fontsize=FONT_SIZE+2, weight='bold')
    
    # 6. Save to PDF
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[OK] Generated: {output_path}")

# =============================================================================
# LATEX TABLE GENERATION
# =============================================================================

def generate_latex_tables(prompt_data, agentic_data, output_path):
    """
    Generate LaTeX tables for academic publication
    
    Args:
        prompt_data (dict): Prompt-engineered analysis results
        agentic_data (dict): Agentic analysis results
        output_path (str): Output file path for .txt file
    """
    latex_content = []
    
    # 1. Performance Metrics Table
    latex_content.append("% Performance Metrics Comparison Table")
    latex_content.append("\\begin{table}[h]")
    latex_content.append("\\centering")
    latex_content.append("\\caption{Performance Metrics Comparison: Prompt-engineered vs Agentic LLM Judge}")
    latex_content.append("\\label{tab:performance_metrics}")
    latex_content.append("\\begin{tabular}{lcc}")
    latex_content.append("\\hline")
    latex_content.append("\\textbf{Metric} & \\textbf{Prompt-engineered} & \\textbf{Agentic} \\\\")
    latex_content.append("\\hline")
    
    # Extract metrics
    p_metrics = prompt_data['classification_metrics']['metrics']
    a_metrics = agentic_data['classification_metrics']['metrics']
    
    latex_content.append(f"Accuracy & {p_metrics['accuracy']:.3f} & {a_metrics['accuracy']:.3f} \\\\")
    latex_content.append(f"Precision & {p_metrics['precision']:.3f} & {a_metrics['precision']:.3f} \\\\")
    latex_content.append(f"Recall & {p_metrics['recall']:.3f} & {a_metrics['recall']:.3f} \\\\")
    latex_content.append(f"F1-Score & {p_metrics['f1_score']:.3f} & {a_metrics['f1_score']:.3f} \\\\")
    latex_content.append("\\hline")
    latex_content.append("\\end{tabular}")
    latex_content.append("\\end{table}")
    latex_content.append("")
    
    # 2. Descriptive Statistics Table
    latex_content.append("% Descriptive Statistics Table")
    latex_content.append("\\begin{table}[h]")
    latex_content.append("\\centering")
    latex_content.append("\\caption{Descriptive Statistics: LLM vs Human Scores}")
    latex_content.append("\\label{tab:descriptive_stats}")
    latex_content.append("\\begin{tabular}{lcccc}")
    latex_content.append("\\hline")
    latex_content.append("\\textbf{Method} & \\textbf{Scorer} & \\textbf{Mean} & \\textbf{Std} & \\textbf{Range} \\\\")
    latex_content.append("\\hline")
    
    # Extract descriptive stats
    p_llm = prompt_data['correlation_analysis']['descriptive_statistics']['llm_scores']
    p_human = prompt_data['correlation_analysis']['descriptive_statistics']['human_scores']
    a_llm = agentic_data['correlation_analysis']['descriptive_statistics']['llm_scores']
    a_human = agentic_data['correlation_analysis']['descriptive_statistics']['human_scores']
    
    latex_content.append(f"Prompt-engineered & LLM & {p_llm['mean']:.2f} & {p_llm['std']:.2f} & {p_llm['min']:.0f}-{p_llm['max']:.0f} \\\\")
    latex_content.append(f" & Human & {p_human['mean']:.2f} & {p_human['std']:.2f} & {p_human['min']:.0f}-{p_human['max']:.0f} \\\\")
    latex_content.append(f"Agentic & LLM & {a_llm['mean']:.2f} & {a_llm['std']:.2f} & {a_llm['min']:.0f}-{a_llm['max']:.0f} \\\\")
    latex_content.append(f" & Human & {a_human['mean']:.2f} & {a_human['std']:.2f} & {a_human['min']:.0f}-{a_human['max']:.0f} \\\\")
    latex_content.append("\\hline")
    latex_content.append("\\end{tabular}")
    latex_content.append("\\end{table}")
    latex_content.append("")
    
    # 3. Consistency Analysis Table
    latex_content.append("% Consistency Analysis Table")
    latex_content.append("\\begin{table}[h]")
    latex_content.append("\\centering")
    latex_content.append("\\caption{Consistency Analysis: Pass Consistency and Score Variability}")
    latex_content.append("\\label{tab:consistency}")
    latex_content.append("\\begin{tabular}{lcccc}")
    latex_content.append("\\hline")
    latex_content.append("\\textbf{Method} & \\textbf{Pass Consistency} & \\textbf{Score Std} & \\textbf{Correlation (r)} & \\textbf{p-value} \\\\")
    latex_content.append("\\hline")
    
    # Extract consistency data
    p_pass = prompt_data['variation_analysis']['variation_summary']['overall_pass_consistency']
    p_std = prompt_data['variation_analysis']['variation_summary']['overall_score_std']
    p_corr = prompt_data['correlation_analysis']['correlation_analysis']['spearman_correlation']
    p_pval = prompt_data['correlation_analysis']['correlation_analysis']['p_value']
    
    a_pass = agentic_data['variation_analysis']['variation_summary']['overall_pass_consistency']
    a_std = agentic_data['variation_analysis']['variation_summary']['overall_score_std']
    a_corr = agentic_data['correlation_analysis']['correlation_analysis']['spearman_correlation']
    a_pval = agentic_data['correlation_analysis']['correlation_analysis']['p_value']
    
    latex_content.append(f"Prompt-engineered & {p_pass:.3f} & {p_std:.3f} & {p_corr:.3f} & {p_pval:.3f} \\\\")
    latex_content.append(f"Agentic & {a_pass:.3f} & {a_std:.3f} & {a_corr:.3f} & {a_pval:.3f} \\\\")
    latex_content.append("\\hline")
    latex_content.append("\\end{tabular}")
    latex_content.append("\\end{table}")
    latex_content.append("")
    
    # 4. Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(latex_content))
    
    print(f"[OK] Generated LaTeX tables: {output_path}")

# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================

def main():
    """
    Main function to generate all materials
    """
    # 1. Configure matplotlib for academic output
    configure_matplotlib()
    
    # 2. Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[OK] Created output directory: {OUTPUT_DIR}")
    
    # 3. Load analysis data
    print("\n[INFO] Loading analysis data...")
    prompt_data, agentic_data = load_analysis_data(PROMPT_ANALYSIS_PATH, AGENTIC_ANALYSIS_PATH)
    
    # 4. Extract comparison data
    metrics_df = extract_performance_metrics(prompt_data, agentic_data)
    prompt_pairs, agentic_pairs, prompt_corr, agentic_corr = extract_correlation_data(prompt_data, agentic_data)
    
    # 5. Generate visualizations
    print("\n[INFO] Generating visualizations...")
    
    # Performance comparison
    create_performance_comparison(
        metrics_df, 
        os.path.join(OUTPUT_DIR, 'performance_comparison.pdf')
    )
    
    # Score distributions
    create_score_distributions(
        prompt_data, agentic_data,
        os.path.join(OUTPUT_DIR, 'score_distributions.pdf')
    )
    
    # Correlation analysis
    create_correlation_analysis(
        prompt_pairs, agentic_pairs, prompt_corr, agentic_corr,
        os.path.join(OUTPUT_DIR, 'correlation_analysis.pdf')
    )
    
    # Consistency evaluation
    create_consistency_analysis(
        prompt_data, agentic_data,
        os.path.join(OUTPUT_DIR, 'consistency_analysis.pdf')
    )
    
    # Confusion matrices
    create_confusion_matrices(
        prompt_data, agentic_data,
        os.path.join(OUTPUT_DIR, 'confusion_matrices.pdf')
    )
    
    # 6. Generate LaTeX tables
    print("\n[INFO] Generating LaTeX tables...")
    generate_latex_tables(
        prompt_data, agentic_data,
        os.path.join(OUTPUT_DIR, 'analysis_tables.txt')
    )
    
    print(f"\n[SUCCESS] All materials generated successfully in '{OUTPUT_DIR}' directory!")
    print(f"   - 5 PDF visualizations")
    print(f"   - 1 LaTeX tables file")

if __name__ == "__main__":
    main()