#!/usr/bin/env python3
"""
EasyCen Hi-C Plotting Module v1.0
Author: Yunyun Lv
Email: lvyunyun_sci@foxmail.com
"""

import argparse
import os
import sys
import logging
from pathlib import Path
import re
import math

import cooler
import numpy as np

try:
    import trackc as tc
    TRACKC_AVAILABLE = True
except ImportError:
    TRACKC_AVAILABLE = False
    print("Warning: trackc package is not installed. Some features may be limited.")
    print("Please install it with: pip install trackc")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Arial'
from matplotlib.colors import LinearSegmentedColormap

# Logging configuration
def setup_logging(verbose=False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

# Figure size calculation
def calculate_figure_size(region_list, is_combined=False):
    """Calculate figure size based on region lengths."""
    total_length = 0
    max_individual_length = 0
    individual_lengths = []
    
    for region in region_list:
        chrom, pos = region.split(':')
        start, end = map(int, pos.split('-'))
        length = end - start
        total_length += length
        individual_lengths.append(length)
        if length > max_individual_length:
            max_individual_length = length
    
    total_length_mb = total_length / 1e6
    max_length_mb = max_individual_length / 1e6
    num_regions = len(region_list)
    
    if is_combined:
        if num_regions == 1:
            width = max(5.0, min(max_length_mb * 0.8, 10.0))
        elif num_regions == 2:
            width = max(6.0, min(total_length_mb * 0.6, 12.0))
        elif num_regions == 3:
            width = max(7.0, min(total_length_mb * 0.5, 14.0))
        elif num_regions == 4:
            width = max(8.0, min(total_length_mb * 0.4, 16.0))
        else:
            width = max(9.0, min(total_length_mb * 0.3, 18.0))
        
        base_height = 3.0
        height = base_height + (num_regions - 1) * 0.5
        height = min(height, 8.0)
    else:
        base_size = 4.0
        if max_length_mb <= 4.0:
            width = base_size * (max_length_mb / 4.0) * 1.2
        else:
            width = base_size + (max_length_mb - 4.0) * 0.3
        height = max(3.0, width * 0.8)
    
    width = max(4.0, min(width, 20.0))
    height = max(2.0, min(height, 12.0))
    
    logging.info(f"Calculated figure size: {width:.1f} x {height:.1f} "
                f"for {num_regions} region(s) (max {max_length_mb:.1f}Mb)")
    return width, height

# Font size calculation
def calculate_font_sizes(fig_width, fig_height):
    """Calculate font sizes based on figure dimensions."""
    size_factor = math.sqrt(fig_width * fig_height)
    font_sizes = {
        'tick': max(6, min(10, int(8 * size_factor / 5))),
        'label': max(8, min(12, int(10 * size_factor / 5))),
        'title': max(10, min(14, int(12 * size_factor / 5))),
    }
    return font_sizes

# Custom colormap
def fruitpunch3_cmap():
    return LinearSegmentedColormap.from_list(
        "fruitpunch3", [(0, "white"), (0.2, "r"), (1, "#0E3858")], N=100
    )

def get_cmap(name):
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
    try:
        return plt.get_cmap(name)
    except ValueError:
        logging.warning(f"Colormap '{name}' not found, using 'fruitpunch3' instead")
        return fruitpunch3_cmap()

# Region parsing
def parse_regions(regions_arg, select_chroms=None):
    """Parse regions from string or BED file."""
    regions_arg = str(regions_arg).strip()
    if os.path.isfile(regions_arg):
        regions = []
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
                if select_chroms and chrom not in select_chroms:
                    continue
                try:
                    start_int = int(start)
                    end_int = int(end)
                    if start_int < 0 or end_int <= start_int:
                        logging.warning(f"Skipping invalid coordinates in line {line_num}: {chrom}:{start}-{end}")
                        continue
                except ValueError:
                    logging.warning(f"Skipping line with non-numeric coordinates {line_num}: {line}")
                    continue
                if len(parts) >= 4:
                    name = parts[3]
                    name = re.sub(r'[<>:"/\\|?*]', '_', name)
                else:
                    name = f"{chrom}_{start}_{end}"
                region_str = f"{chrom}:{start}-{end}"
                regions.append((region_str, name))
        if not regions:
            raise ValueError(f"No valid regions found in BED file: {regions_arg}")
        logging.info(f"Loaded {len(regions)} regions from BED file: {regions_arg}")
        return regions, True

    regs = [r.strip() for r in regions_arg.split(",") if r.strip()]
    if not regs:
        raise ValueError("No regions provided in --regions argument")
    regions = []
    for r in regs:
        if not re.match(r'^chr[\w]+:\d+-\d+$', r):
            logging.warning(f"Region format may be invalid: {r}")
        chrom = r.split(':')[0]
        if select_chroms and chrom not in select_chroms:
            continue
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', r.replace(":", "_").replace("-", "_"))
        regions.append((r, safe_name))
    logging.info(f"Parsed {len(regions)} regions from string input")
    return regions, False

# Generate regions for entire chromosomes
def generate_chromosome_regions(clr, select_chroms=None):
    chromsizes = dict(zip(clr.chromnames, clr.chromsizes))
    regions = []
    if select_chroms:
        for chrom in select_chroms:
            if chrom in chromsizes:
                region_str = f"{chrom}:0-{chromsizes[chrom]}"
                regions.append((region_str, chrom))
            else:
                logging.warning(f"Chromosome {chrom} not found in cooler file")
    else:
        for chrom in clr.chromnames:
            region_str = f"{chrom}:0-{chromsizes[chrom]}"
            regions.append((region_str, chrom))
    if not regions:
        raise ValueError("No valid chromosomes found")
    logging.info(f"Generated {len(regions)} chromosome regions")
    return regions

# Adjust region to chromosome boundaries
def adjust_region_to_bounds(region, chromsizes):
    chrom, pos = region.split(':')
    start, end = map(int, pos.split('-'))
    if chrom not in chromsizes:
        raise ValueError(f"Chromosome {chrom} not found in chromsizes.")
    chr_len = chromsizes[chrom]
    start_corrected = max(0, start)
    end_corrected = min(chr_len, end)
    if end_corrected <= start_corrected:
        raise ValueError(f"Region {region} is invalid after correction.")
    corrected_region = f"{chrom}:{start_corrected}-{end_corrected}"
    if corrected_region != region:
        print(f"[WARN] Region {region} out of bounds; adjusted to {corrected_region}")
    return corrected_region

# Matrix preprocessing
def preprocess_matrix(mat, vmin=None, vmax=None):
    data = mat.cmat.copy()
    if vmin is not None or vmax is not None:
        vmin = np.nanmin(data) if vmin is None else vmin
        vmax = np.nanmax(data) if vmax is None else vmax
        data = np.clip(data, vmin, vmax)
    class ProcessedMatrix:
        pass
    processed = ProcessedMatrix()
    processed.cmat = data
    return processed

# Plot one triangular Hi-C map
def plot_one_region(clr, region_list, outname, args):
    out_path = Path(outname)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmap = get_cmap(args.cmap)

    try:
        if TRACKC_AVAILABLE:
            if args.auto_size:
                fig_width, fig_height = calculate_figure_size(
                    region_list, 
                    is_combined=(len(region_list) > 1 and not args.single)
                )
            else:
                fig_width, fig_height = args.fig_width, args.fig_height
            
            font_sizes = calculate_font_sizes(fig_width, fig_height)
            effective_tick_fontsize = min(args.tick_fontsize, font_sizes['tick'])
            
            logging.info(f"Creating figure: {fig_width:.1f}x{fig_height:.1f}, tick font: {effective_tick_fontsize}")
            
            ten = tc.tenon(figsize=(fig_width, fig_height))
            try:
                heights = [float(h.strip()) for h in args.bottom_heights.split(",")]
                if len(heights) < 2:
                    raise ValueError
                total_specified = sum(heights)
                if total_specified > 0:
                    heights = [h * fig_height / total_specified for h in heights]
            except ValueError:
                logging.error(f"Invalid bottom_heights format")
                heights = [fig_height * 0.7, fig_height * 0.3]

            for h in heights:
                ten.add(pos="bottom", height=h)

            mat = tc.tl.extractContactRegions(clr=clr, row_regions=region_list)
            if args.vmin is not None or args.vmax is not None:
                mat = preprocess_matrix(mat, args.vmin, args.vmax)

            mapC_args = {
                'ax': ten.axs(0),
                'mat': mat.cmat,
                'map_type': args.map_type,
                'cmap': cmap,
                'ax_on': False,
            }
            if args.vmin is not None and args.vmax is not None:
                mapC_args['lim'] = [args.vmin, args.vmax]
            elif args.vmin is not None:
                mapC_args['lim'] = [args.vmin, np.nanmax(mat.cmat)]
            elif args.vmax is not None:
                mapC_args['lim'] = [np.nanmin(mat.cmat), args.vmax]

            tc.pl.mapC(**mapC_args)
            scale_track_args = {
                'ax': ten.axs(1), 
                'regions': region_list, 
                'scale_adjust': args.scale_adjust,
                'intervals': args.intervals,
                'tick_rotation': args.tick_rotation,
                'tick_fontsize': effective_tick_fontsize,
            }
            tc.pl.multi_scale_track(**scale_track_args)

            try:
                tc.savefig(str(out_path))
                logging.info(f"Saved figure using trackc: {out_path}")
            except Exception as e:
                logging.warning(f"trackc savefig failed, using matplotlib: {e}")
                plt.tight_layout()
                plt.savefig(str(out_path), dpi=300, bbox_inches='tight')
                plt.close()
        else:
            logging.warning("trackc not available, using matplotlib fallback")
            if args.auto_size:
                fig_width, fig_height = calculate_figure_size(
                    region_list, 
                    is_combined=(len(region_list) > 1 and not args.single)
                )
            else:
                fig_width, fig_height = args.fig_width, args.fig_height
            font_sizes = calculate_font_sizes(fig_width, fig_height)
            fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
            matrices = []
            for region in region_list:
                mat = clr.matrix(balance=True).fetch(region)
                matrices.append(mat)
            if len(matrices) == 1:
                mat = matrices[0]
                mask = np.triu(np.ones_like(mat, dtype=bool), k=1)
                mat_tri = np.ma.array(mat, mask=mask)
                im = ax.imshow(mat_tri, cmap=cmap, aspect='auto',
                              vmin=args.vmin, vmax=args.vmax)
                plt.colorbar(im, ax=ax, label='Contact frequency')
            else:
                for i, mat in enumerate(matrices):
                    mask = np.triu(np.ones_like(mat, dtype=bool), k=1)
                    mat_tri = np.ma.array(mat, mask=mask)
                    im = ax.imshow(mat_tri, cmap=cmap, aspect='auto',
                                  vmin=args.vmin, vmax=args.vmax)
                plt.colorbar(im, ax=ax, label='Contact frequency')
            ax.set_title(f"Hi-C Contact Map: {', '.join(region_list)}", fontsize=font_sizes['title'])
            ax.set_xlabel('Genomic Position', fontsize=font_sizes['label'])
            ax.set_ylabel('Genomic Position', fontsize=font_sizes['label'])
            ax.tick_params(axis='both', which='major', labelsize=font_sizes['tick'])
            plt.tight_layout()
            plt.savefig(str(out_path), dpi=300, bbox_inches='tight')
            plt.close()
            logging.info(f"Saved figure using matplotlib fallback: {out_path}")
    except Exception as e:
        logging.error(f"Failed to plot region {region_list}: {e}")
        raise

# Argument validation
def validate_args(args):
    if not os.path.isfile(args.mcool):
        raise FileNotFoundError(f"mcool file not found: {args.mcool}")
    cooler_ref = f"{args.mcool}::/resolutions/{args.resolution}"
    try:
        clr = cooler.Cooler(cooler_ref)
        logging.info(f"Loaded cooler with resolution {args.resolution}")
        test_region = clr.chromnames[0] + ":1-10000"
        _ = clr.matrix().fetch(test_region)
    except Exception as e:
        raise ValueError(f"Cannot access resolution {args.resolution} in {args.mcool}: {e}")
    outdir = Path(args.outdir)
    try:
        outdir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise ValueError(f"Cannot create output directory {outdir}: {e}")
    if args.fig_width <= 0 or args.fig_height <= 0:
        raise ValueError("Figure dimensions must be positive")
    if args.regions and args.select_chroms:
        logging.warning("Both --regions and --select_chroms provided, using --select_chroms")
    if not args.regions and not args.select_chroms:
        raise ValueError("Either --regions or --select_chroms must be provided")
    return True

# Main plotting function
def plot_hic_triangular(mcool_file, resolution, regions, output_dir, 
                       single_plots=False, select_chroms=None, map_type="triangle",
                       cmap="fruitpunch3", vmin=None, vmax=None,
                       fig_width=6.0, fig_height=4.0, auto_size=True,
                       bottom_heights="3,0.4", scale_adjust="Mb", intervals=1, 
                       tick_rotation=0.0, tick_fontsize=8.0, verbose=False):
    setup_logging(verbose)
    args_dict = {
        'mcool': mcool_file, 'resolution': resolution, 'regions': regions,
        'outdir': output_dir, 'single': single_plots,
        'select_chroms': select_chroms, 'map_type': map_type,
        'cmap': cmap, 'vmin': vmin, 'vmax': vmax,
        'fig_width': fig_width, 'fig_height': fig_height,
        'auto_size': auto_size, 'bottom_heights': bottom_heights,
        'scale_adjust': scale_adjust, 'intervals': intervals,
        'tick_rotation': tick_rotation, 'tick_fontsize': tick_fontsize,
        'verbose': verbose
    }
    class Args:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    args = Args(**args_dict)
    
    try:
        validate_args(args)
        cooler_ref = f"{args.mcool}::/resolutions/{args.resolution}"
        clr = cooler.Cooler(cooler_ref)
        logging.info(f"Successfully loaded cooler: {cooler_ref}")
        logging.info(f"Available chromosomes: {clr.chromnames}")

        if args.select_chroms:
            select_chroms_list = [chrom.strip() for chrom in args.select_chroms.split(",")]
            logging.info(f"Selected chromosomes: {select_chroms_list}")
            region_items = generate_chromosome_regions(clr, select_chroms_list)
            chromsizes = dict(zip(clr.chromnames, clr.chromsizes))
            region_items = [(adjust_region_to_bounds(r[0], chromsizes), r[1]) for r in region_items]
            region_list = [r[0] for r in region_items]
            chrom_names = "_".join(select_chroms_list)
            outname = Path(args.outdir) / f"selected_chromosomes_{chrom_names}.triangle_hic.pdf"
            logging.info(f"Generating combined plot for selected chromosomes: {select_chroms_list}")
            plot_one_region(clr, region_list, str(outname), args)
            return
        else:
            region_items, is_bed = parse_regions(args.regions, args.select_chroms)
            chromsizes = dict(zip(clr.chromnames, clr.chromsizes))
            region_items = [(adjust_region_to_bounds(r[0], chromsizes), r[1]) for r in region_items]

        if not TRACKC_AVAILABLE:
            logging.warning("trackc is not available. Using matplotlib fallback.")

        if not args.single:
            region_list = [r[0] for r in region_items]
            outname = Path(args.outdir) / "combined_triangle_hic.pdf"
            logging.info(f"Generating combined plot with {len(region_list)} regions")
            plot_one_region(clr, region_list, str(outname), args)
            return

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

# Command line interface
def main():
    parser = argparse.ArgumentParser(
        description="EasyCen Hi-C Plotting v1.0",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--mcool", required=True, help="Input .mcool file path")
    parser.add_argument("--resolution", required=True, type=int, help="Resolution for Hi-C data")
    parser.add_argument("--regions", help="Regions string or BED file path")
    parser.add_argument("--select_chroms", help="Select specific chromosomes to plot together (comma-separated)")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--single", action="store_true", help="Generate separate plot for each region")
    parser.add_argument("--map_type", default="triangle", choices=['square','squ','triangle','tri','rectangle','rec'],
                        help="Map type for heatmap display")
    parser.add_argument("--cmap", default="fruitpunch3", help="Colormap for heatmap")
    parser.add_argument("--vmin", type=float, default=None, help="Minimum value for colormap")
    parser.add_argument("--vmax", type=float, default=None, help="Maximum value for colormap")
    parser.add_argument("--fig_width", type=float, default=6.0, help="Figure width in inches")
    parser.add_argument("--fig_height", type=float, default=4.0, help="Figure height in inches")
    parser.add_argument("--auto_size", action="store_true", default=True, help="Automatically adjust figure size")
    parser.add_argument("--no_auto_size", action="store_false", dest="auto_size", help="Disable automatic figure sizing")
    parser.add_argument("--bottom_heights", default="3,0.4", help="Heights for bottom panels (comma separated)")
    parser.add_argument("--scale_adjust", default="Mb", help="Scale adjustment for track")
    parser.add_argument("--intervals", type=int, default=1, help="Number of intervals for track")
    parser.add_argument("--tick_rotation", type=float, default=0.0, help="Rotation angle for ticks")
    parser.add_argument("--tick_fontsize", type=float, default=8.0, help="Font size for ticks")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    if args.map_type in ['squ','que']:
        args.map_type = 'square'
    elif args.map_type in ['tri']:
        args.map_type = 'triangle'
    elif args.map_type in ['rec']:
        args.map_type = 'rectangle'
    
    if not args.regions and not args.select_chroms:
        print("Error: Either --regions or --select_chroms must be provided")
        sys.exit(1)
    
    print("=" * 60)
    print("EASYCEN Hi-C PLOTTING MODULE v1.0")
    print("=" * 60)
    print(f"Input file:    {args.mcool}")
    print(f"Resolution:    {args.resolution}")
    print(f"Output dir:    {args.outdir}")
    if args.select_chroms:
        print(f"Selected chromosomes: {args.select_chroms}")
    print(f"Trackc available: {TRACKC_AVAILABLE}")
    print("=" * 60)
    
    try:
        plot_hic_triangular(
            mcool_file=args.mcool,
            resolution=args.resolution,
            regions=args.regions,
            output_dir=args.outdir,
            single_plots=args.single,
            select_chroms=args.select_chroms,
            map_type=args.map_type,
            cmap=args.cmap,
            vmin=args.vmin,
            vmax=args.vmax,
            fig_width=args.fig_width,
            fig_height=args.fig_height,
            auto_size=args.auto_size,
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