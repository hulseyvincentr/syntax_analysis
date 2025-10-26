"""
Automatically generate spectrograms for each HDBSCAN label.
This script samples 5 spectrograms from each cluster without requiring manual selection.

Usage:
    python auto_hdbscan_spectrograms.py path/to/data.npz [--samples_per_label 5] [--segment_length 200] [--output_dir imgs/auto_hdbscan]
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
import os
import argparse
import glob
from mpl_toolkits.axes_grid1 import make_axes_locatable

class AutoHDBSCANSpectrograms:
    def __init__(self, file_path, samples_per_label=5, segment_length=200, output_dir="plots/auto_hdbscan"):
        """
        Initialize the automatic HDBSCAN spectrogram generator.
        
        Args:
            file_path: Path to NPZ file containing HDBSCAN data
            samples_per_label: Number of spectrograms to generate per label
            segment_length: Length of spectrogram segments in timepoints
            output_dir: Directory to save output images
        """
        self.file_path = file_path
        self.samples_per_label = samples_per_label
        self.segment_length = segment_length
        self.output_dir = output_dir
        
        # Load data
        self.data = np.load(file_path, allow_pickle=True)
        self.spec = self.data["s"]
        self.hdbscan_labels = self.data["hdbscan_labels"]
        self.embedding = self.data["embedding_outputs"]
        
        # Use the exact same color approach as the original script
        # Generate tab20 color palette for HDBSCAN labels (same as original)
        # Handle noise label (-1) separately with gray color like the original script
        unique_labels = np.unique(self.hdbscan_labels)
        unique_labels_no_noise = unique_labels[unique_labels != -1]
        tab20_cmap = plt.get_cmap("tab20")
        self.hdbscan_colors = [mcolors.to_hex(tab20_cmap(i)) for i in range(len(unique_labels_no_noise))]
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"Loaded data with {len(self.hdbscan_labels)} timepoints")
        print(f"Found {len(np.unique(self.hdbscan_labels))} HDBSCAN labels (including noise)")
    
    def find_contiguous_segments(self, label):
        """
        Find all contiguous segments for a given label.
        
        Args:
            label: HDBSCAN label to find segments for
            
        Returns:
            List of (start_idx, length) tuples for each segment
        """
        segments = []
        current_start = None
        current_length = 0
        
        for i, lab in enumerate(self.hdbscan_labels):
            if lab == label:
                if current_start is None:
                    current_start = i
                current_length += 1
            else:
                if current_start is not None:
                    segments.append((current_start, current_length))
                    current_start = None
                    current_length = 0
        
        # Handle case where label extends to the end
        if current_start is not None:
            segments.append((current_start, current_length))
        
        return segments
    
    def sample_segments(self, segments, n_samples):
        """
        Randomly sample n_samples from the available segments.
        
        Args:
            segments: List of (start_idx, length) tuples
            n_samples: Number of samples to return
            
        Returns:
            List of (start_idx, length) tuples for sampled segments
        """
        if len(segments) <= n_samples:
            return segments
        
        # Weight by segment length (longer segments get higher probability)
        weights = [seg[1] for seg in segments]
        total_weight = sum(weights)
        probs = [w / total_weight for w in weights]
        
        # Sample without replacement
        sampled_indices = np.random.choice(len(segments), size=n_samples, replace=False, p=probs)
        return [segments[i] for i in sampled_indices]
    
    def create_spectrogram_plot(self, spec_segment, label, segment_info, embedding_coords):
        """
        Create a single spectrogram plot with metadata.
        
        Args:
            spec_segment: Spectrogram data for the segment
            label: HDBSCAN label
            segment_info: (start_idx, length) tuple
            embedding_coords: UMAP coordinates for the segment
            
        Returns:
            matplotlib Figure object
        """
        start_idx, length = segment_info
        
        fig, (ax_spec, ax_phase, ax_colorbar) = plt.subplots(3, 1, figsize=(8, 6), 
                                                             gridspec_kw={'height_ratios': [3, 1, 0.5]})
        
        # Main spectrogram - show full frequency range with expanded y-axis
        spec_to_plot = spec_segment.T
        
        # For better frequency visualization, we can extend the y-axis range
        # This makes the frequency bins more spread out and easier to see
        im = ax_spec.imshow(spec_to_plot, aspect='auto', origin='lower', cmap='viridis', 
                           extent=[0, spec_to_plot.shape[1], 0, spec_to_plot.shape[0]])
        
        ax_spec.set_title(f'HDBSCAN Label {label} - Segment {start_idx}:{start_idx+length}', fontsize=12)
        ax_spec.set_ylabel('Frequency Bins (0-195)', fontsize=10)
        ax_spec.set_xticks([])
        
        # Add frequency axis ticks for better interpretation
        if spec_to_plot.shape[0] > 10:  # Only add ticks if we have enough frequency bins
            freq_ticks = np.linspace(0, spec_to_plot.shape[0]-1, min(8, spec_to_plot.shape[0]))  # Increased from 6 to 8 ticks
            ax_spec.set_yticks(freq_ticks)
            # Show actual bin numbers (0-195) instead of normalized values
            tick_labels = [f'{int(i)}' for i in freq_ticks]
            ax_spec.set_yticklabels(tick_labels)
        else:
            ax_spec.set_yticks([])
        
        # Add colorbar for spectrogram
        divider = make_axes_locatable(ax_spec)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        
        # Phase space plot (UMAP coordinates)
        if len(embedding_coords) > 0:
            # Normalize coordinates to [0,1] for consistent coloring
            x_coords = embedding_coords[:, 0]
            y_coords = embedding_coords[:, 1]
            
            if x_coords.max() != x_coords.min():
                x_norm = (x_coords - x_coords.min()) / (x_coords.max() - x_coords.min())
            else:
                x_norm = np.zeros_like(x_coords)
                
            if y_coords.max() != y_coords.min():
                y_norm = (y_coords - y_coords.min()) / (y_coords.max() - y_coords.min())
            else:
                y_norm = np.zeros_like(y_coords)
            
            # Create phase colors
            phase_colors = np.column_stack([x_norm, y_norm, np.zeros_like(x_norm)])
            ax_phase.scatter(x_norm, y_norm, c=phase_colors, s=20, alpha=0.7)
            ax_phase.set_title('Phase Space (UMAP)', fontsize=10)
            ax_phase.set_xlim(0, 1)
            ax_phase.set_ylim(0, 1)
            ax_phase.set_aspect('equal')
            ax_phase.set_xticks([])
            ax_phase.set_yticks([])
            

        
        # Color bar showing the label color
        if label == -1:
            # Use the same gray color as the original script for noise labels
            label_color = "#7f7f7f"
        else:
            label_color = self.hdbscan_colors[label % len(self.hdbscan_colors)]
        # Convert hex to RGB for display
        rgb_color = mcolors.to_rgb(label_color)
        color_display = np.array(rgb_color).reshape(1, 1, 3)
        ax_colorbar.imshow(color_display, aspect='auto')
        ax_colorbar.set_title(f'Label {label} Color', fontsize=10)
        ax_colorbar.set_xticks([])
        ax_colorbar.set_yticks([])
        
        plt.tight_layout()
        return fig
    
    def generate_all_spectrograms(self):
        """
        Generate spectrograms for all HDBSCAN labels.
        """
        # Get unique labels (including noise label -1)
        unique_labels = np.unique(self.hdbscan_labels)
        
        print(f"Generating spectrograms for {len(unique_labels)} labels...")
        
        for label in unique_labels:
            print(f"Processing label {label}...")
            
            # Find all segments for this label
            segments = self.find_contiguous_segments(label)
            
            if not segments:
                print(f"  No segments found for label {label}")
                continue
            
            print(f"  Found {len(segments)} segments, sampling {self.samples_per_label}")
            
            # Sample segments
            sampled_segments = self.sample_segments(segments, self.samples_per_label)
            
            # Generate spectrograms for each sampled segment
            for i, (start_idx, length) in enumerate(sampled_segments):
                # Ensure we don't go out of bounds
                if start_idx + self.segment_length > len(self.spec):
                    continue
                
                # Extract spectrogram segment
                spec_segment = self.spec[start_idx:start_idx + self.segment_length]
                
                # Get corresponding embedding coordinates
                embedding_coords = self.embedding[start_idx:start_idx + self.segment_length]
                
                # Create the plot
                fig = self.create_spectrogram_plot(spec_segment, label, (start_idx, length), embedding_coords)
                
                # Save the figure
                filename = f"label_{label}_sample_{i+1}_start_{start_idx}.png"
                filepath = os.path.join(self.output_dir, filename)
                fig.savefig(filepath, dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                print(f"    Saved {filename}")
        
        print(f"\nAll spectrograms saved to {self.output_dir}")
    
    def create_summary_plot(self):
        """
        Create a summary plot showing all labels and their sample counts.
        """
        unique_labels = np.unique(self.hdbscan_labels)
        unique_labels = unique_labels[unique_labels != -1]
        
        # Count segments per label
        label_counts = {}
        for label in unique_labels:
            segments = self.find_contiguous_segments(label)
            label_counts[label] = len(segments)
        
        # Create summary plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Number of segments per label
        labels = list(label_counts.keys())
        counts = list(label_counts.values())
        
        # Handle colors for bars, using gray for -1 label
        bar_colors = []
        for l in labels:
            if l == -1:
                bar_colors.append("#7f7f7f")  # Gray for noise label
            else:
                bar_colors.append(self.hdbscan_colors[l % len(self.hdbscan_colors)])
        bars1 = ax1.bar(labels, counts, color=bar_colors)
        ax1.set_xlabel('HDBSCAN Label')
        ax1.set_ylabel('Number of Segments')
        ax1.set_title('Segments per HDBSCAN Label')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, count in zip(bars1, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{count}', ha='center', va='bottom')
        
        # Plot 2: UMAP embedding colored by HDBSCAN labels (including noise)
        plot_embedding = self.embedding
        plot_labels = self.hdbscan_labels
        
        # Create color mapping
        unique_plot_labels = np.unique(plot_labels)
        colors = []
        for l in plot_labels:
            if l == -1:
                colors.append("#7f7f7f")  # Gray for noise label
            else:
                colors.append(self.hdbscan_colors[l % len(self.hdbscan_colors)])
        
        scatter = ax2.scatter(plot_embedding[:, 0], plot_embedding[:, 1], 
                             c=colors, s=10, alpha=0.6)
        ax2.set_xlabel('UMAP 1')
        ax2.set_ylabel('UMAP 2')
        ax2.set_title('UMAP Embedding (HDBSCAN Labels)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save summary plot
        summary_path = os.path.join(self.output_dir, "summary_plot.png")
        fig.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Summary plot saved to {summary_path}")
    
    def create_hdbscan_color_legend(self):
        """
        Create a separate legend figure showing all HDBSCAN label colors.
        """
        unique_labels = np.unique(self.hdbscan_labels)
        
        # Create legend figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create color patches for each label
        legend_elements = []
        for label in sorted(unique_labels):
            if label == -1:
                color = "#7f7f7f"  # Gray for noise label
            else:
                color = self.hdbscan_colors[label % len(self.hdbscan_colors)]
            # Create a rectangle patch with the label color
            from matplotlib.patches import Rectangle
            patch = Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='black', linewidth=1)
            legend_elements.append((patch, f'Label {label}'))
        
        # Create legend
        ax.legend([elem[0] for elem in legend_elements], 
                 [elem[1] for elem in legend_elements], 
                 loc='center', fontsize=12, title='HDBSCAN Label Colors', 
                 title_fontsize=14)
        
        # Remove axes
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Add title
        plt.title('HDBSCAN Label Color Legend', fontsize=16, pad=20)
        
        # Save legend
        legend_path = os.path.join(self.output_dir, "hdbscan_color_legend.png")
        fig.savefig(legend_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"HDBSCAN color legend saved to {legend_path}")

def find_npz_files_recursively(parent_dir):
    """
    Recursively find all .npz files in a parent directory.
    
    Args:
        parent_dir: Path to parent directory to search
        
    Returns:
        List of paths to .npz files
    """
    npz_files = []
    for root, dirs, files in os.walk(parent_dir):
        for file in files:
            if file.endswith('.npz'):
                npz_files.append(os.path.join(root, file))
    return sorted(npz_files)

def process_single_file(file_path, samples_per_label, segment_length, output_dir):
    """
    Process a single NPZ file and generate all outputs.
    
    Args:
        file_path: Path to NPZ file
        samples_per_label: Number of spectrograms per label
        segment_length: Length of spectrogram segments
        output_dir: Base output directory
    """
    # Create a subdirectory for this file based on its name
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    file_output_dir = os.path.join(output_dir, file_name)
    
    print(f"\n{'='*60}")
    print(f"Processing: {file_path}")
    print(f"Output directory: {file_output_dir}")
    print(f"{'='*60}")
    
    try:
        # Create generator and run
        generator = AutoHDBSCANSpectrograms(
            file_path=file_path,
            samples_per_label=samples_per_label,
            segment_length=segment_length,
            output_dir=file_output_dir
        )
        
        # Generate all spectrograms
        generator.generate_all_spectrograms()
        
        # Create summary plot
        generator.create_summary_plot()
        
        # Create HDBSCAN color legend
        generator.create_hdbscan_color_legend()
        
        print(f"✓ Successfully processed {file_path}")
        
    except Exception as e:
        print(f"✗ Error processing {file_path}: {str(e)}")
        print(f"Continuing with next file...")

def main():
    parser = argparse.ArgumentParser(description="Automatically generate HDBSCAN spectrograms")
    parser.add_argument("path", type=str, help="Path to NPZ file OR parent directory containing .npz files")
    parser.add_argument("--samples_per_label", type=int, default=5, 
                       help="Number of spectrograms to generate per label (default: 5)")
    parser.add_argument("--segment_length", type=int, default=200,
                       help="Length of spectrogram segments in timepoints (default: 200)")
    parser.add_argument("--output_dir", type=str, default="plots/auto_hdbscan",
                       help="Output directory for generated images (default: plots/auto_hdbscan)")
    parser.add_argument("--recursive", action='store_true', 
                       help="Force recursive directory processing (auto-detected if path is directory)")
    
    args = parser.parse_args()
    
    # Determine if path is a file or directory
    if os.path.isfile(args.path):
        if args.path.endswith('.npz'):
            # Single NPZ file
            print(f"Processing single NPZ file: {args.path}")
            process_single_file(args.path, args.samples_per_label, args.segment_length, args.output_dir)
        else:
            print(f"Error: {args.path} is not a .npz file")
            return
    elif os.path.isdir(args.path):
        # Directory - find all .npz files recursively
        print(f"Searching for .npz files in directory: {args.path}")
        npz_files = find_npz_files_recursively(args.path)
        
        if not npz_files:
            print(f"No .npz files found in {args.path}")
            return
        
        print(f"Found {len(npz_files)} .npz files:")
        for i, file_path in enumerate(npz_files, 1):
            print(f"  {i:3d}. {file_path}")
        
        # Create base output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Process each file
        for file_path in npz_files:
            process_single_file(file_path, args.samples_per_label, args.segment_length, args.output_dir)
        
        print(f"\n{'='*60}")
        print(f"Completed processing {len(npz_files)} files")
        print(f"All outputs saved to: {args.output_dir}")
        print(f"{'='*60}")
        
    else:
        print(f"Error: {args.path} is not a valid file or directory")
        return
    
    print("Done!")

if __name__ == "__main__":
    main() 
    
    
"""
# --- Imports (needed in Spyder console) ---
from pathlib import Path
import numpy as np
import random
import importlib
import auto_hdbscan_spectrograms as ahs
importlib.reload(ahs)

# Reproducibility (optional)
np.random.seed(0)
random.seed(0)

# --- Single .npz file ---
npz_path = Path("/Volumes/my_own_ssd/2024_AreaX_lesions_NMA_and_sham/AreaXlesion_TweetyBERT_outputs/new_outputs/TweetyBERT_Pretrain_LLB_AreaX_FallSong_USA5336.npz")
outdir   = Path("/Volumes/my_own_ssd/2024_AreaX_lesions_NMA_and_sham/AreaXlesion_TweetyBERT_outputs/new_outputs")
outdir.mkdir(parents=True, exist_ok=True)

# Quick sanity checks
if not npz_path.is_file():
    raise FileNotFoundError(f"NPZ not found: {npz_path}")

with np.load(npz_path, allow_pickle=True) as npz:
    required = {"s", "hdbscan_labels", "embedding_outputs"}
    missing = required - set(npz.files)
    if missing:
        raise KeyError(f"Missing arrays in NPZ: {sorted(missing)}")
    print("Loaded shapes:",
          "s:", npz["s"].shape,
          "labels:", npz["hdbscan_labels"].shape,
          "embedding:", npz["embedding_outputs"].shape)

samples_per_label = 3
segment_length    = 200

file_out = outdir / npz_path.stem
file_out.mkdir(parents=True, exist_ok=True)

gen = ahs.AutoHDBSCANSpectrograms(
    file_path=str(npz_path),
    samples_per_label=samples_per_label,
    segment_length=segment_length,
    output_dir=str(file_out),
)

gen.generate_all_spectrograms()
gen.create_summary_plot()
gen.create_hdbscan_color_legend()

print("Done. Outputs in:", file_out)
for p in sorted(file_out.glob("*.png"))[:8]:
    print(" ", p.name)


# ==========================================
# OPTIONAL: Process an entire directory (class-only)
# ==========================================
# root = Path("/Users/mirandahulsey-vincent/Desktop/SfN_data/USA5323/npz_exports")  # <-- edit
# for npz in sorted(root.rglob("*.npz")):
#     print("\nProcessing:", npz)
#     file_out = outdir / npz.stem
#     file_out.mkdir(parents=True, exist_ok=True)
#     gen = ahs.AutoHDBSCANSpectrograms(
#         file_path=str(npz),
#         samples_per_label=samples_per_label,
#         segment_length=segment_length,
#         output_dir=str(file_out),
#     )
#     gen.generate_all_spectrograms()
#     gen.create_summary_plot()
#     gen.create_hdbscan_color_legend()
#     print("Saved to:", file_out)





"""