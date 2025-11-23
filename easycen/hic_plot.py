#!/usr/bin/env python3
"""
EasyCen Hi-C Plotting Module
Triangular Hi-C contact map visualization from .mcool files

Author: Yunyun Lv
Email: lvyunyun_sci@foxmail.com
Version: 1.0.0
License: MIT
"""

import argparse
import os
import sys
import logging
from pathlib import Path
import re

import cooler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Arial'
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# Try to import trackc, provide friendly error message
try:
    import trackc as tc
    TRACKC_AVAILABLE = True
except ImportError:
    TRACKC_AVAILABLE = False
    print("Warning: trackc package is not installed. Some features may be limited.")
    print("Please install it with: pip install trackc")

# ---------------------------
# Logging configuration
# ---------------------------
def setup_logging(verbose=False):
    """Set up logging level"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

# ---------------------------
# Colormap functions
# ---------------------------
def fruitpunch3_cmap():
    """Create custom colormap for Hi-C data"""
    return LinearSegmentedColormap.from_list(
        "fruitpunch3", [(0, "white"), (0.2, "r"), (1, "#0E3858")], N=100
    )

def get_cmap(name):
    """Get colormap, support built-in and custom colormaps"""
    cmap_dict = {
        "fruitpunch3": fruitpunch3_cmap(),
        "coolwarm": plt.get_cmap("coolwarm"),
        "viridis": plt.get_cmap("viridis"),
        "plasma": plt.get_cmap("plasma"),
        "inferno": plt.get_cmap("inferno"),
        "magma": plt.get_cmap("magma"),
        "RdYlBu_r": plt.get_cmap("RdYlBu_r"),
        "Spectral_r": plt.get_cmap("Spectral_r"),
    }
    
    if name in cmap_dict:
        return cmap_dict[name]
    else:
        try:
            return plt.get_cmap(name)
        except ValueError:
            logging.warning(f"Colormap '{name}' not found, using 'fruitpunch3' instead")
            return fruitpunch3_cmap()

# ---------------------------
# Parse regions
# ---------------------------
def parse_regions(regions_arg):
    """
    Support two formats:
    1) 'chr1:0-1000000,chr2:0-2000000'
    2) BED file: chrom  start  end [name ...]
    
    Returns:
        tuple: (regions_list, is_bed_file)
    """
    regions_arg = str(regions_arg).strip()
    
    # Check if it's a file
    if os.path.isfile(regions_arg):
        regions = []
        try:
            with open(regions_arg, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith("#") or line.startswith("track") or line.startswith("browser"):
                        continue
                    
                    parts = line.split()
                    if len(parts) < 3:
                        logging.warning(f"Skipping invalid BED line {line_num}: {line}")
                        continue
                    
                    chrom, start, end = parts[0], parts[1], parts[2]
                    
                    # Validate coordinates
                    try:
                        start_int = int(start)
                        end_int = int(end)
                        if start_int < 0 or end_int <= start_int:
                            logging.warning(f"Skipping invalid coordinates in line {line_num}: {chrom}:{start}-{end}")
                            continue
                    except ValueError:
                        logging.warning(f"Skipping line with non-numeric coordinates {line_num}: {line}")
                        continue
                    
                    # Generate region name
                    if len(parts) >= 4:
                        name = parts[3]
                        # Clean name of illegal filename characters
                        name = re.sub(r'[<>:"/\\|?*]', '_', name)
                    else:
                        name = f"{chrom}_{start}_{end}"
                    
                    region_str = f"{chrom}:{start}-{end}"
                    regions.append((region_str, name))
            
            if not regions:
                raise ValueError(f"No valid regions found in BED file: {regions_arg}")
                
            logging.info(f"Loaded {len(regions)} regions from BED file: {regions_arg}")
            return regions, True
            
        except IOError as e:
            raise ValueError(f"Cannot read BED file {regions_arg}: {e}")

    # String format regions
    regs = [r.strip() for r in regions_arg.split(",") if r.strip()]
    if not regs:
        raise ValueError("No regions provided in --regions argument")
    
    regions = []
    for i, r in enumerate(regs):
        # Validate region format
        if not re.match(r'^chr[\w]+:\d+-\d+$', r):
            logging.warning(f"Region format may be invalid: {r}")
        
        # Generate safe filename
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', r.replace(":", "_").replace("-", "_"))
        regions.append((r, safe_name))
    
    logging.info(f"Parsed {len(regions)} regions from string input")
    return regions, False

# ---------------------------
# Preprocess matrix data for color range
# ---------------------------
def preprocess_matrix(mat, vmin=None, vmax=None):
    """
    Preprocess matrix data, clip according to vmin/vmax
    
    Args:
        mat: Contact matrix object
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        
    Returns:
        Processed matrix object
    """
    data = mat.cmat.copy()
    
    if vmin is not None or vmax is not None:
        if vmin is None:
            vmin = np.nanmin(data)
        if vmax is None:
            vmax = np.nanmax(data)
        
        # Clip data to specified range
        data = np.clip(data, vmin, vmax)
    
    # Create a new similar object to store processed data
    class ProcessedMatrix:
        pass
    
    processed = ProcessedMatrix()
    processed.cmat = data
    return processed

# ---------------------------
# Plot one triangular Hi-C map
# ---------------------------
def plot_one_region(clr, region_list, outname, args):
    """
    Plot triangular Hi-C map for one region
    
    Args:
        clr: Cooler object
        region_list: List of regions, e.g., ["chr6:0-100000", "chr8:0-200000"]
        outname: Output file path
        args: Command line arguments
    """
    # Ensure output directory exists
    out_path = Path(outname)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get colormap
    cmap = get_cmap(args.cmap)

    try:
        if TRACKC_AVAILABLE:
            # Build tenon layout using trackc
            ten = tc.tenon(figsize=(args.fig_width, args.fig_height))

            # Parse bottom panel heights
            try:
                heights = [float(h.strip()) for h in args.bottom_heights.split(",")]
                if len(heights) < 2:
                    raise ValueError("Need at least 2 heights for bottom panels")
            except ValueError as e:
                logging.error(f"Invalid bottom_heights format: {e}")
                heights = [3.0, 0.4]  # Default values

            # Add bottom panels
            for h in heights:
                ten.add(pos="bottom", height=h)

            # Extract contact matrix
            logging.debug(f"Extracting contact matrix for regions: {region_list}")
            mat = tc.tl.extractContactRegions(clr=clr, row_regions=region_list)
            
            # Preprocess matrix data for color range
            if args.vmin is not None or args.vmax is not None:
                mat = preprocess_matrix(mat, args.vmin, args.vmax)

            # Plot triangular Hi-C map - using trackc supported parameters
            mapC_args = {
                'ax': ten.axs(0),
                'mat': mat.cmat,
                'map_type': "tri",
                'cmap': cmap,
            }
            
            # Check trackc version and select correct parameters
            try:
                # Try using lim parameter (newer trackc versions)
                if args.vmin is not None and args.vmax is not None:
                    mapC_args['lim'] = [args.vmin, args.vmax]
                elif args.vmin is not None:
                    mapC_args['lim'] = [args.vmin, np.nanmax(mat.cmat)]
                elif args.vmax is not None:
                    mapC_args['lim'] = [np.nanmin(mat.cmat), args.vmax]
            except TypeError:
                # If lim parameter not supported, fallback to other approaches
                logging.warning("'lim' parameter not supported, trying alternative approaches")
                # Preprocessed data already handles range limits, no extra parameters needed
            
            tc.pl.mapC(**mapC_args)

            # Draw scale track below
            tc.pl.multi_scale_track(
                ten.axs(1),
                regions=region_list,
                scale_adjust=args.scale_adjust,
                intervals=args.intervals,
                tick_rotation=args.tick_rotation,
                tick_fontsize=args.tick_fontsize,
            )

            # Save figure using trackc
            try:
                tc.savefig(str(out_path))
                logging.info(f"Saved figure using trackc: {out_path}")
            except Exception as e:
                logging.warning(f"trackc savefig failed, using matplotlib: {e}")
                plt.tight_layout()
                plt.savefig(str(out_path), dpi=300, bbox_inches='tight')
                plt.close()  # Close figure to free memory
                
        else:
            # Fallback to matplotlib-only implementation
            logging.warning("trackc not available, using matplotlib fallback")
            fig, ax = plt.subplots(1, 1, figsize=(args.fig_width, args.fig_height))
            
            # Extract contact matrix using cooler directly
            from cooler import Cooler
            import pandas as pd
            
            # Extract matrix using cooler
            matrices = []
            for region in region_list:
                mat = clr.matrix(balance=True).fetch(region)
                matrices.append(mat)
            
            # For triangular plot, we need to handle multiple regions
            if len(matrices) == 1:
                # Single region triangular plot
                mat = matrices[0]
                # Create triangular mask
                mask = np.triu(np.ones_like(mat, dtype=bool), k=1)
                mat_tri = np.ma.array(mat, mask=mask)
                
                im = ax.imshow(mat_tri, cmap=cmap, aspect='auto',
                              vmin=args.vmin, vmax=args.vmax)
                
                # Add colorbar
                plt.colorbar(im, ax=ax, label='Contact frequency')
                
            else:
                # Multiple regions - create combined visualization
                # This is a simplified version for fallback
                for i, mat in enumerate(matrices):
                    # Create triangular mask for each region
                    mask = np.triu(np.ones_like(mat, dtype=bool), k=1)
                    mat_tri = np.ma.array(mat, mask=mask)
                    
                    # Simple arrangement for multiple regions
                    # In a real implementation, you'd want a better layout
                    im = ax.imshow(mat_tri, cmap=cmap, aspect='auto',
                                  vmin=args.vmin, vmax=args.vmax)
                
                plt.colorbar(im, ax=ax, label='Contact frequency')
            
            ax.set_title(f"Hi-C Contact Map: {', '.join(region_list)}")
            ax.set_xlabel('Genomic Position')
            ax.set_ylabel('Genomic Position')
            
            plt.tight_layout()
            plt.savefig(str(out_path), dpi=300, bbox_inches='tight')
            plt.close()
            logging.info(f"Saved figure using matplotlib fallback: {out_path}")
            
    except Exception as e:
        logging.error(f"Failed to plot region {region_list}: {e}")
        raise

# ---------------------------
# Validate arguments
# ---------------------------
def validate_args(args):
    """Validate command line arguments"""
    # Check mcool file
    if not os.path.isfile(args.mcool):
        raise FileNotFoundError(f"mcool file not found: {args.mcool}")
    
    # Check if resolution is available
    cooler_ref = f"{args.mcool}::/resolutions/{args.resolution}"
    try:
        clr = cooler.Cooler(cooler_ref)
        logging.info(f"Loaded cooler with resolution {args.resolution}")
        # Test a small fetch to ensure it works
        test_region = clr.chromnames[0] + ":1-10000"
        _ = clr.matrix().fetch(test_region)
    except Exception as e:
        raise ValueError(f"Cannot access resolution {args.resolution} in {args.mcool}: {e}")
    
    # Validate output directory
    outdir = Path(args.outdir)
    try:
        outdir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise ValueError(f"Cannot create output directory {outdir}: {e}")
    
    # Validate figure dimensions
    if args.fig_width <= 0 or args.fig_height <= 0:
        raise ValueError("Figure dimensions must be positive")
    
    # Validate regions
    regions, _ = parse_regions(args.regions)
    if not regions:
        raise ValueError("No valid regions provided")
    
    return True

# ---------------------------
# Main plotting function
# ---------------------------
def plot_hic_triangular(mcool_file, resolution, regions, output_dir, 
                       single_plots=False, cmap="fruitpunch3", vmin=None, vmax=None,
                       fig_width=6.0, fig_height=4.0, bottom_heights="3,0.4",
                       scale_adjust="Mb", intervals=1, tick_rotation=0.0, 
                       tick_fontsize=10.0, verbose=False):
    """
    Main function to plot triangular Hi-C contact maps
    
    Args:
        mcool_file: Input .mcool file path
        resolution: Resolution for Hi-C data
        regions: Regions string or BED file path
        output_dir: Output directory for plots
        single_plots: Generate separate plot for each region
        cmap: Colormap for heatmap
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        fig_width: Figure width in inches
        fig_height: Figure height in inches
        bottom_heights: Heights for bottom panels (comma separated)
        scale_adjust: Scale adjustment for track
        intervals: Number of intervals for track
        tick_rotation: Rotation angle for ticks
        tick_fontsize: Font size for ticks
        verbose: Enable verbose logging
    """
    
    # Set up logging
    setup_logging(verbose)
    
    # Validate arguments
    args_dict = {
        'mcool': mcool_file,
        'resolution': resolution,
        'regions': regions,
        'outdir': output_dir,
        'single': single_plots,
        'cmap': cmap,
        'vmin': vmin,
        'vmax': vmax,
        'fig_width': fig_width,
        'fig_height': fig_height,
        'bottom_heights': bottom_heights,
        'scale_adjust': scale_adjust,
        'intervals': intervals,
        'tick_rotation': tick_rotation,
        'tick_fontsize': tick_fontsize,
        'verbose': verbose
    }
    
    # Create a simple namespace object to hold arguments
    class Args:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    args = Args(**args_dict)
    
    try:
        # Validate arguments
        validate_args(args)
        
        # Load cooler
        cooler_ref = f"{args.mcool}::/resolutions/{args.resolution}"
        clr = cooler.Cooler(cooler_ref)
        logging.info(f"Successfully loaded cooler: {cooler_ref}")
        logging.info(f"Available chromosomes: {clr.chromnames}")

        # Parse regions
        region_items, is_bed = parse_regions(args.regions)

        # Check trackc availability
        if not TRACKC_AVAILABLE:
            logging.warning("trackc is not available. Using matplotlib fallback mode.")
            logging.warning("Some features like multi-scale tracks may be limited.")

        # ---------------------------
        # 1) Multiple regions combined into one plot
        # ---------------------------
        if not args.single:
            region_list = [r[0] for r in region_items]
            outname = Path(args.outdir) / "combined_triangle_hic.pdf"
            logging.info(f"Generating combined plot with {len(region_list)} regions")
            plot_one_region(clr, region_list, str(outname), args)
            return

        # ---------------------------
        # 2) Individual plot for each region
        # ---------------------------
        logging.info(f"Generating individual plots for {len(region_items)} regions")
        success_count = 0
        for reg, name in region_items:
            try:
                outname = Path(args.outdir) / f"{name}.triangle_hic.pdf"
                plot_one_region(clr, [reg], str(outname), args)
                success_count += 1
            except Exception as e:
                logging.error(f"Failed to plot region {reg}: {e}")
                continue
        
        logging.info(f"Successfully generated {success_count}/{len(region_items)} plots")
        
    except Exception as e:
        logging.error(f"EasyCen Hi-C plotting failed: {e}")
        raise

# ---------------------------
# Command line interface
# ---------------------------
def main():
    parser = argparse.ArgumentParser(
        description="EasyCen Hi-C Plotting - Generate triangular Hi-C contact maps from .mcool files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--mcool", required=True, help="Input .mcool file path")
    parser.add_argument("--resolution", required=True, type=int, help="Resolution for Hi-C data")
    parser.add_argument("--regions", required=True, help="Regions string or BED file path")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--single", action="store_true", help="Generate separate plot for each region")
    parser.add_argument("--cmap", default="fruitpunch3", help="Colormap for heatmap")
    parser.add_argument("--vmin", type=float, default=None, help="Minimum value for colormap")
    parser.add_argument("--vmax", type=float, default=None, help="Maximum value for colormap")
    parser.add_argument("--fig_width", type=float, default=6.0, help="Figure width in inches")
    parser.add_argument("--fig_height", type=float, default=4.0, help="Figure height in inches")
    parser.add_argument("--bottom_heights", default="3,0.4", help="Heights for bottom panels (comma separated)")
    parser.add_argument("--scale_adjust", default="Mb", help="Scale adjustment for track")
    parser.add_argument("--intervals", type=int, default=1, help="Number of intervals for track")
    parser.add_argument("--tick_rotation", type=float, default=0.0, help="Rotation angle for ticks")
    parser.add_argument("--tick_fontsize", type=float, default=10.0, help="Font size for ticks")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Print EasyCen header
    print("=" * 60)
    print("EASYCEN Hi-C PLOTTING MODULE")
    print("=" * 60)
    print(f"Input file:    {args.mcool}")
    print(f"Resolution:    {args.resolution}")
    print(f"Output dir:    {args.outdir}")
    print(f"Trackc available: {TRACKC_AVAILABLE}")
    print("=" * 60)
    
    try:
        plot_hic_triangular(
            mcool_file=args.mcool,
            resolution=args.resolution,
            regions=args.regions,
            output_dir=args.outdir,
            single_plots=args.single,
            cmap=args.cmap,
            vmin=args.vmin,
            vmax=args.vmax,
            fig_width=args.fig_width,
            fig_height=args.fig_height,
            bottom_heights=args.bottom_heights,
            scale_adjust=args.scale_adjust,
            intervals=args.intervals,
            tick_rotation=args.tick_rotation,
            tick_fontsize=args.tick_fontsize,
            verbose=args.verbose
        )
        
        print(f"\nHi-C plotting complete. Results saved to: {args.outdir}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()