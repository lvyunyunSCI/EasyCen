#!/usr/bin/env python3
"""
EasyCen Visualization Module
Centromere visualization with boundary optimization and statistical analysis

Author: Yunyun Lv
Email: lvyunyun_sci@foxmail.com
Version: 1.0.0
License: MIT
"""

import argparse
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, FancyBboxPatch, Patch
import seaborn as sns
from pathlib import Path
import re
from matplotlib.colors import LinearSegmentedColormap, to_rgba
import warnings
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm, normaltest
from scipy.signal import find_peaks
import random
warnings.filterwarnings('ignore')

# Set publication-quality styling
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'figure.figsize': (7, 5),
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.4,
    'lines.linewidth': 1.2,
    'patch.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
})

class CentromereOptimizer:
    """Centromere boundary optimization using multi-threshold randomized Gaussian distribution with adaptive sigma and dynamic extension"""
    
    def __init__(self, kmer_weight=0.6, feature_weight=0.4, extension_bp=100000,  # Changed default to 100k
                 smoothing_sigma=2.0, distribution_threshold=0.05,
                 random_sampling_times=100000, sample_size=None,
                 peak_prominence=0.1, min_region_size=50000,
                 dynamic_extension=True, target_mean=0.5, mean_tolerance=0.01,  # New parameters for dynamic extension
                 max_extension_factor=5.0, extension_increment=50000):  # Maximum extension and increment
        self.kmer_weight = kmer_weight
        self.feature_weight = feature_weight
        self.extension_bp = extension_bp
        self.smoothing_sigma = smoothing_sigma
        self.distribution_threshold = distribution_threshold
        self.random_sampling_times = random_sampling_times
        self.sample_size = sample_size
        self.peak_prominence = peak_prominence
        self.min_region_size = min_region_size
        
        # Dynamic extension parameters
        self.dynamic_extension = dynamic_extension
        self.target_mean = target_mean
        self.mean_tolerance = mean_tolerance
        self.max_extension_factor = max_extension_factor
        self.extension_increment = extension_increment
        
    def optimize_boundaries(self, known_centromere, kmer_data, feature_data, chrom_length):
        """
        Optimize centromere boundaries using composite scoring with adaptive sigma and dynamic extension
        
        Args:
            known_centromere: Dictionary with known centromere coordinates
            kmer_data: K-mer density track data
            feature_data: Feature percentage track data
            chrom_length: Chromosome length in bp
            
        Returns:
            Dictionary with optimized boundary information
        """
        start = known_centromere['start']
        end = known_centromere['end']
        center = (start + end) // 2
        
        # Apply dynamic extension if enabled
        if self.dynamic_extension:
            extension_result = self._apply_dynamic_extension(
                start, end, kmer_data, feature_data, chrom_length
            )
            search_start = extension_result['search_start']
            search_end = extension_result['search_end']
            final_extension = extension_result['final_extension']
            iterations = extension_result['iterations']
            mean_history = extension_result['mean_history']
        else:
            # Use fixed extension
            search_start = max(0, start - self.extension_bp)
            search_end = min(chrom_length, end + self.extension_bp)
            final_extension = self.extension_bp
            iterations = 1
            mean_history = []
        
        # Extract data from the final search region
        search_positions = []
        kmer_values = []
        feature_values = []
        
        for kmer_point in kmer_data:
            if search_start <= kmer_point['start'] <= search_end:
                search_positions.append(kmer_point['start'])
                kmer_values.append(kmer_point['value'])
                feature_val = 0
                for feature_point in feature_data:
                    if feature_point['start'] == kmer_point['start']:
                        feature_val = feature_point['value']
                        break
                feature_values.append(feature_val)
        
        if not search_positions:
            return known_centromere
        
        positions = np.array(search_positions)
        kmer_vals = np.array(kmer_values)
        feature_vals = np.array(feature_values)
        
        kmer_norm = self._normalize_signal(kmer_vals)
        feature_norm = 1 - self._normalize_signal(feature_vals)
        
        composite_scores = (self.kmer_weight * kmer_norm + 
                          self.feature_weight * feature_norm)
        
        # Adaptive sigma calculation
        adaptive_sigma = self._calculate_adaptive_sigma(composite_scores, positions)
        composite_smoothed = gaussian_filter1d(composite_scores, adaptive_sigma)
        
        # Peak detection with adaptive sigma
        if len(composite_smoothed) > 1:
            distance = max(1, len(composite_smoothed) // 20)
            peaks, peak_properties = find_peaks(composite_smoothed, 
                                              prominence=self.peak_prominence,
                                              distance=distance)
        else:
            peaks, peak_properties = np.array([]), {}
        
        optimization_result = self._find_boundary_aggressive_multithreshold(
            positions, composite_scores, composite_smoothed, peaks, start, end, chrom_length, adaptive_sigma
        )
        
        left_boundary = optimization_result['left_boundary']
        right_boundary = optimization_result['right_boundary']
        threshold_info = optimization_result['threshold_info']
        
        # Add dynamic extension info to threshold_info
        threshold_info['dynamic_extension'] = {
            'enabled': self.dynamic_extension,
            'final_extension': final_extension,
            'iterations': iterations,
            'target_mean': self.target_mean,
            'mean_tolerance': self.mean_tolerance,
            'mean_history': mean_history,
            'initial_extension': self.extension_bp
        }
        
        min_centromere_size = 50000
        if right_boundary - left_boundary < min_centromere_size:
            center_new = (left_boundary + right_boundary) // 2
            left_boundary = max(0, center_new - min_centromere_size // 2)
            right_boundary = min(chrom_length, center_new + min_centromere_size // 2)
        
        optimized = known_centromere.copy()
        optimized.update({
            'optimized_start': left_boundary,
            'optimized_end': right_boundary,
            'optimized_length': right_boundary - left_boundary,
            'optimized_start_mb': left_boundary / 1e6,
            'optimized_end_mb': right_boundary / 1e6,
            'optimized_center_mb': (left_boundary + right_boundary) / 2 / 1e6,
            'composite_scores': composite_scores.tolist(),
            'composite_smoothed': composite_smoothed.tolist(),
            'search_positions': positions.tolist(),
            'peaks': peaks.tolist() if peaks is not None else [],
            'peak_properties': peak_properties,
            'was_optimized': True,
            'optimization_method': 'adaptive_sigma_multi_threshold_dynamic_extension',
            'threshold_info': threshold_info,
            'adaptive_sigma': adaptive_sigma,
            'dynamic_extension_info': {
                'final_extension': final_extension,
                'iterations': iterations,
                'mean_history': mean_history
            }
        })
        
        return optimized
    
    def _apply_dynamic_extension(self, start, end, kmer_data, feature_data, chrom_length):
        """
        Apply dynamic extension to ensure sampling mean is near target mean
        
        Args:
            start: Known centromere start
            end: Known centromere end
            kmer_data: K-mer density data
            feature_data: Feature percentage data
            chrom_length: Chromosome length
            
        Returns:
            Dictionary with final search region and extension info
        """
        current_extension = self.extension_bp
        max_extension = min(chrom_length // 2, int(self.extension_bp * self.max_extension_factor))
        
        iterations = 0
        max_iterations = 1000
        mean_history = []
        
        # Store the best extension found so far
        best_extension = current_extension
        best_mean_diff = float('inf')
        
        while iterations < max_iterations:
            # Calculate current search region
            search_start = max(0, start - current_extension)
            search_end = min(chrom_length, end + current_extension)
            
            # Extract data from current search region
            search_positions, kmer_values, feature_values = self._extract_region_data(
                search_start, search_end, kmer_data, feature_data
            )
            
            if not search_positions:
                # No data in this region, extend further
                current_extension += self.extension_increment
                iterations += 1
                continue
            
            # Calculate composite scores for the entire search region
            positions = np.array(search_positions)
            kmer_vals = np.array(kmer_values)
            feature_vals = np.array(feature_values)
            
            if len(kmer_vals) == 0 or len(feature_vals) == 0:
                current_extension += self.extension_increment
                iterations += 1
                continue
            
            kmer_norm = self._normalize_signal(kmer_vals)
            feature_norm = 1 - self._normalize_signal(feature_vals)
            
            composite_scores = (self.kmer_weight * kmer_norm + 
                              self.feature_weight * feature_norm)
            
            # Calculate mean of ALL composite scores in the search region
            region_mean = np.mean(composite_scores)
            mean_diff = abs(region_mean - self.target_mean)
            
            mean_history.append({
                'iteration': iterations,
                'extension': current_extension,
                'mean': region_mean,
                'mean_diff': mean_diff,
                'search_start': search_start,
                'search_end': search_end,
                'n_points': len(composite_scores)
            })
            
            print(f"Iteration {iterations}: Extension={current_extension/1000:.1f}kb, Mean={region_mean:.3f}, Diff={mean_diff:.3f}, Points={len(composite_scores)}")
            
            # Update best extension found
            if mean_diff < best_mean_diff:
                best_mean_diff = mean_diff
                best_extension = current_extension
            
            # Check if mean is within tolerance
            if mean_diff <= self.mean_tolerance:
                print(f"Dynamic extension converged after {iterations} iterations")
                break
            
            # Adjust extension based on mean difference
            # If mean is too high (>0.5), we need to extend more to include more low-score regions
            # If mean is too low (<0.5), we need to reduce extension to focus on higher-score regions
            if region_mean > self.target_mean:
                # Mean is too high - extend more to include more low-score flanking regions
                extension_adjustment = self.extension_increment
            else:
                # Mean is too low - reduce extension to focus on higher-score regions
                extension_adjustment = -self.extension_increment
            
            current_extension += extension_adjustment
            
            # Ensure we don't exceed maximum extension or go below minimum
            if current_extension > max_extension:
                current_extension = max_extension
                print(f"Reached maximum extension: {max_extension/1000:.1f}kb")
                # Use the best extension we found
                if best_mean_diff < float('inf'):
                    current_extension = best_extension
                    print(f"Using best extension found: {best_extension/1000:.1f}kb (mean_diff={best_mean_diff:.3f})")
                break
                
            if current_extension < self.extension_bp:
                current_extension = self.extension_bp
                print(f"Reached minimum extension: {current_extension/1000:.1f}kb")
                break
                
            iterations += 1
        
        if iterations >= max_iterations:
            print(f"Dynamic extension reached maximum iterations ({max_iterations}), using best extension")
            current_extension = best_extension
        
        final_search_start = max(0, start - current_extension)
        final_search_end = min(chrom_length, end + current_extension)
        
        return {
            'search_start': final_search_start,
            'search_end': final_search_end,
            'final_extension': current_extension,
            'iterations': iterations,
            'mean_history': mean_history,
            'best_mean_diff': best_mean_diff
        }
    
    def _extract_region_data(self, search_start, search_end, kmer_data, feature_data):
        """Extract data from specified region"""
        search_positions = []
        kmer_values = []
        feature_values = []
        
        for kmer_point in kmer_data:
            if search_start <= kmer_point['start'] <= search_end:
                search_positions.append(kmer_point['start'])
                kmer_values.append(kmer_point['value'])
                feature_val = 0
                for feature_point in feature_data:
                    if feature_point['start'] == kmer_point['start']:
                        feature_val = feature_point['value']
                        break
                feature_values.append(feature_val)
        
        return search_positions, kmer_values, feature_values
    
    def _calculate_adaptive_sigma(self, composite_scores, positions):
        """Calculate adaptive sigma based on peak count"""
        # Initial smoothing with sigma=2
        initial_sigma = 2.0
        initial_smoothed = gaussian_filter1d(composite_scores, initial_sigma)
        
        # Detect initial peaks
        if len(initial_smoothed) > 1:
            distance = max(1, len(initial_smoothed) // 10)
            initial_peaks, _ = find_peaks(initial_smoothed, 
                                        prominence=self.peak_prominence,
                                        distance=distance)
            n_peaks = len(initial_peaks)
        else:
            n_peaks = 0
        
        # Calculate new sigma based on peak count: sigma = min(2**n, 25)
        t = (n_peaks - 1) / 9.0
        h = 15.0
        l = 2
        p = 0.3
        adaptive_sigma = l + (h - l) * (t ** p)
        
        print(f"Detected {n_peaks} peaks, using adaptive sigma: {adaptive_sigma}")
        return adaptive_sigma
    
    def _normalize_signal(self, signal_data):
        """Normalize signal data to 0-1 range"""
        min_val = np.min(signal_data)
        max_val = np.max(signal_data)
        if max_val - min_val == 0:
            return np.zeros_like(signal_data)
        return (signal_data - min_val) / (max_val - min_val)
    
    def _find_boundary_aggressive_multithreshold(self, positions, scores, smoothed_scores, peaks, original_start, original_end, chrom_length, adaptive_sigma):
        """Find boundaries using multiple threshold levels with adaptive peak handling - FIXED SAMPLING LOGIC"""
        if len(smoothed_scores) < 10:
            return {
                'left_boundary': original_start,
                'right_boundary': original_end,
                'threshold_info': {'error': 'insufficient_data'}
            }
        
        # Boundary determination for multiple peaks
        if len(peaks) > 1:
            print(f"Multiple peaks detected ({len(peaks)}), using multi-peak boundary determination")
            multi_peak_result = self._handle_multiple_peaks(positions, smoothed_scores, peaks, original_start, original_end)
            if multi_peak_result:
                return multi_peak_result
        
        # Correction: sample from original composite scores, not smoothed scores
        if self.sample_size is None:
            if 100 < len(scores):
                sample_size = 100
            else:
                sample_size = len(scores) // 2
        else:
            sample_size = min(self.sample_size, len(scores))
        
        if sample_size < 5:
            return {
                'left_boundary': original_start,
                'right_boundary': original_end,
                'threshold_info': {'error': 'sample_size_too_small'}
            }
        
        # Corrected random sampling logic - sample from original composite score distribution
        print(f"Performing {self.random_sampling_times} random samplings with sample size {sample_size}")
        sample_means = []
        for i in range(self.random_sampling_times):
            # Random sampling from original composite scores, not smoothed ones
            random_sample = np.random.choice(scores, size=sample_size, replace=True)
            sample_mean = np.mean(random_sample)
            sample_means.append(sample_mean)
            
            # Show progress every 10000 samples
            if (i + 1) % 10000 == 0:
                print(f"Completed {i + 1}/{self.random_sampling_times} samplings...")
        
        # Initialize variables
        mu_sampling = 0.0
        std_sampling = 1.0
        normality_p_value = 1.0
        is_normal = True
        
        # Validate normality of sampling distribution
        try:
            mu_sampling, std_sampling = norm.fit(sample_means)
            
            # Normality test
            if len(sample_means) > 3:
                try:
                    _, normality_p_value = normaltest(sample_means)
                    is_normal = normality_p_value > 0.05
                except:
                    normality_p_value = 1.0
                    is_normal = True
            else:
                normality_p_value = 1.0
                is_normal = True
            
            print(f"Sampling distribution: mean={mu_sampling:.4f}, std={std_sampling:.4f}, n={len(sample_means)}")
            print(f"Normality test p-value: {normality_p_value:.4f} ({'normal' if is_normal else 'non-normal'})")
            
            # If distribution is not normal, issue warning but continue processing
            if not is_normal:
                print("Warning: Sampling distribution may not be perfectly normal, but continuing with threshold calculation")
                
        except Exception as e:
            print(f"Error fitting sampling distribution: {e}")
            try:
                # Fallback to using statistics from original composite scores if sampling fitting fails
                mu_sampling, std_sampling = norm.fit(scores)
                print(f"Fallback to original scores distribution: mean={mu_sampling:.4f}, std={std_sampling:.4f}")
                normality_p_value = 1.0
                is_normal = True
            except:
                print("Failed to fit any distribution, using default values")
                mu_sampling = np.mean(scores) if len(scores) > 0 else 0.5
                std_sampling = np.std(scores) if len(scores) > 0 else 0.1
                normality_p_value = 1.0
                is_normal = True
        
        # Calculate thresholds based on sampling distribution
        zvalue=norm.ppf(1 - self.distribution_threshold/2)
        threshold_levels = [
            mu_sampling - zvalue * 1 * std_sampling,
            mu_sampling - zvalue * 0.95 * std_sampling,
            mu_sampling - zvalue * 0.5 * std_sampling,
            mu_sampling - 0.01 * std_sampling
        ]
        
        threshold_names = ['very_aggressive(select)', 'aggressive', 'standard', 'conservative']
        
        all_boundaries = []
        for i, threshold in enumerate(threshold_levels):
            boundaries = self._find_boundaries_at_threshold(positions, smoothed_scores, threshold, peaks)
            if boundaries:
                all_boundaries.append({
                    'threshold_level': threshold_names[i],
                    'threshold_value': threshold,
                    'boundaries': boundaries
                })
        
        selected_boundaries = self._select_optimal_boundaries(all_boundaries, original_start, original_end, chrom_length)
        
        # Ensure all necessary keys exist
        threshold_info = {
            'sample_size': sample_size,
            'sampling_times': self.random_sampling_times,
            'sampling_mean': mu_sampling,
            'sampling_std': std_sampling,
            'threshold_levels': dict(zip(threshold_names, threshold_levels)),
            'all_boundaries': all_boundaries,
            'selected_threshold': selected_boundaries.get('threshold_level', 'standard'),
            'adaptive_sigma': adaptive_sigma,
            'peak_count': len(peaks),
            'sampling_distribution_stats': {
                'min': float(np.min(sample_means)) if len(sample_means) > 0 else 0.0,
                'max': float(np.max(sample_means)) if len(sample_means) > 0 else 1.0,
                'median': float(np.median(sample_means)) if len(sample_means) > 0 else 0.5,
                'skewness': float(pd.Series(sample_means).skew()) if len(sample_means) > 2 else 0,
                'normality_p_value': float(normality_p_value),
                'is_normal': is_normal,
                'sample_means': sample_means  # Save sample means for visualization
            }
        }
        
        return {
            'left_boundary': selected_boundaries['left_boundary'],
            'right_boundary': selected_boundaries['right_boundary'],
            'threshold_info': threshold_info
        }
    
    def _handle_multiple_peaks(self, positions, smoothed_scores, peaks, original_start, original_end):
        """Handle boundary determination in multi-peak scenarios"""
        peak_positions = positions[peaks]
        peak_values = smoothed_scores[peaks]
        
        # Sort peaks by position
        sorted_indices = np.argsort(peak_positions)
        sorted_peaks = peak_positions[sorted_indices]
        sorted_values = peak_values[sorted_indices]
        
        # Find leftmost and rightmost peaks
        leftmost_peak = sorted_peaks[0]
        rightmost_peak = sorted_peaks[-1]
        
        # Calculate minimum distance between peaks
        if len(sorted_peaks) > 1:
            peak_distances = np.diff(sorted_peaks)
            min_peak_distance = np.min(peak_distances)
        else:
            min_peak_distance = 100000  # Default value
        
        # Find boundaries in multi-peak region
        left_boundary = self._find_boundary_near_peak(positions, smoothed_scores, leftmost_peak, 'left')
        right_boundary = self._find_boundary_near_peak(positions, smoothed_scores, rightmost_peak, 'right')
        
        # Ensure boundary reasonableness
        if left_boundary >= right_boundary:
            # Use original boundaries if boundaries cross
            return {
                'left_boundary': original_start,
                'right_boundary': original_end,
                'threshold_info': {'error': 'boundary_crossing_in_multi_peak'}
            }
        
        # Check region size
        region_size = right_boundary - left_boundary
        if region_size < self.min_region_size:
            # Extend region to minimum size
            center = (left_boundary + right_boundary) // 2
            left_boundary = max(0, center - self.min_region_size // 2)
            right_boundary = left_boundary + self.min_region_size
        
        return {
            'left_boundary': left_boundary,
            'right_boundary': right_boundary,
            'threshold_info': {
                'multi_peak_handling': True,
                'peak_count': len(peaks),
                'leftmost_peak': leftmost_peak,
                'rightmost_peak': rightmost_peak,
                'min_peak_distance': min_peak_distance,
                'region_size': region_size
            }
        }
    
    def _find_boundary_near_peak(self, positions, smoothed_scores, peak_position, direction='left'):
        """Find boundary point near peak"""
        peak_idx = np.where(positions == peak_position)[0][0]
        
        if direction == 'left':
            # Search left for boundary
            search_range = range(peak_idx, -1, -1)
        else:
            # Search right for boundary
            search_range = range(peak_idx, len(positions))
        
        # Calculate gradient changes
        gradients = np.gradient(smoothed_scores)
        
        boundary_idx = peak_idx
        for i in search_range:
            if direction == 'left':
                # Left search: find position with significant gradient change
                if i > 0 and abs(gradients[i]) < 0.001 and smoothed_scores[i] < smoothed_scores[peak_idx] * 0.3:
                    boundary_idx = i
                    break
            else:
                # Right search: find position with significant gradient change
                if i < len(positions) - 1 and abs(gradients[i]) < 0.001 and smoothed_scores[i] < smoothed_scores[peak_idx] * 0.3:
                    boundary_idx = i
                    break
        
        return positions[boundary_idx]
    
    def _find_boundaries_at_threshold(self, positions, smoothed_scores, threshold, peaks):
        """Find regions above specified threshold"""
        above_threshold = smoothed_scores >= threshold
        if not np.any(above_threshold):
            return []
        
        regions = []
        in_region = False
        region_start = None
        
        for i, (pos, above) in enumerate(zip(positions, above_threshold)):
            if above and not in_region:
                in_region = True
                region_start = pos
            elif not above and in_region:
                in_region = False
                regions.append((region_start, positions[i-1]))
        
        if in_region:
            regions.append((region_start, positions[-1]))
        
        regions = [r for r in regions if r[1] - r[0] >= self.min_region_size]
        
        if len(peaks) > 0 and len(regions) > 0:
            peak_positions = positions[peaks]
            regions_with_peaks = []
            
            for region in regions:
                region_has_peak = any(region[0] <= peak_pos <= region[1] for peak_pos in peak_positions)
                if region_has_peak:
                    regions_with_peaks.append(region)
            
            if regions_with_peaks:
                regions = regions_with_peaks
        
        return regions
    
    def _select_optimal_boundaries(self, all_boundaries, original_start, original_end, chrom_length):
        """Select optimal boundaries from multiple threshold levels"""
        original_center = (original_start + original_end) // 2
        original_size = original_end - original_start
        
        best_boundaries = None
        best_score = -float('inf')
        
        for threshold_data in all_boundaries:
            for boundaries in threshold_data['boundaries']:
                left, right = boundaries
                size = right - left
                
                size_score = 1.0 - abs(size - original_size) / (original_size + 1)
                center_score = 1.0 - abs((left + right)//2 - original_center) / (chrom_length + 1)
                coverage_score = self._calculate_coverage_score(left, right, original_start, original_end)
                
                total_score = 0.4 * size_score + 0.3 * center_score + 0.3 * coverage_score
                
                if threshold_data['threshold_level'] in ['very_aggressive', 'aggressive']:
                    total_score *= 1.2
                
                if total_score > best_score:
                    best_score = total_score
                    best_boundaries = {
                        'left_boundary': left,
                        'right_boundary': right,
                        'threshold_level': threshold_data['threshold_level'],
                        'score': total_score
                    }
        
        if best_boundaries is None:
            best_boundaries = {
                'left_boundary': original_start,
                'right_boundary': original_end,
                'threshold_level': 'fallback',
                'score': 0
            }
        
        return best_boundaries
    
    def _calculate_coverage_score(self, left, right, original_start, original_end):
        """Calculate coverage score for boundary evaluation"""
        overlap_start = max(left, original_start)
        overlap_end = min(right, original_end)
        
        if overlap_end <= overlap_start:
            return 0.0
        
        overlap_size = overlap_end - overlap_start
        original_size = original_end - original_start
        
        return overlap_size / original_size

class CentromereVisualizer:
    """EasyCen visualization toolkit with boundary optimization"""
    
    def __init__(self, results_dir, known_centromeres_file=None, kmer_weight=0.4, 
                 feature_weight=0.6, optimization_extension=100000,  # Changed default to 100k
                 heatmap_colormap='viridis', heatmap_height=0.3, output_dir=None,
                 smoothing_sigma=2.0, distribution_threshold=0.05,
                 random_sampling_times=100000, sample_size=None, compare_centromeres_file=None,
                 peak_prominence=0.1, min_region_size=50000, dynamic_extension=True,
                 target_mean=0.5, mean_tolerance=0.01, max_extension_factor=5.0,
                 extension_increment=50000):
        self.results_dir = Path(results_dir)
        self.known_centromeres_file = known_centromeres_file
        self.output_dir = Path(output_dir) if output_dir else self.results_dir / "visualization"
        self.chrom_data = {}
        self.centromere_data = {}
        self.known_centromeres = {}
        self.optimized_centromeres = {}
        self.compare_centromeres = {}
        self.heatmap_colormap = heatmap_colormap
        self.heatmap_height = heatmap_height
        
        self.optimizer = CentromereOptimizer(
            kmer_weight=kmer_weight,
            feature_weight=feature_weight,
            extension_bp=optimization_extension,
            smoothing_sigma=smoothing_sigma,
            distribution_threshold=distribution_threshold,
            random_sampling_times=random_sampling_times,
            sample_size=sample_size,
            peak_prominence=peak_prominence,
            min_region_size=min_region_size,
            dynamic_extension=dynamic_extension,
            target_mean=target_mean,
            mean_tolerance=mean_tolerance,
            max_extension_factor=max_extension_factor,
            extension_increment=extension_increment
        )
        
        self.colors = {
            'kmer_density': '#E74C3C',
            'gc_content': '#3498DB', 
            'cpg_density': '#2ECC71',
            'feature_percent': '#9B59B6',
            'centromere': '#F39C12',
            'background': '#F8F9FA',
            'telocentric': '#E74C3C',
            'metacentric': '#3498DB',  
            'submetacentric': '#2ECC71',
            'acrocentric': '#9B59B6',
            'holocentric': '#F39C12',
            'unknown': '#95A5A6',
            'known_centromere': '#F39C12',
            'optimized_centromere': '#E74C3C',
            'search_region': '#D7DBDD',
            'analysis_primary': '#3498DB',
            'analysis_candidate': '#9B59B6',
            'compare_centromere': '#27AE60',
            'highlight_primary_bg': (0.95, 0.95, 0.95),
            'highlight_border_primary': '#F39C12',
            'highlight_border_candidate': '#E74C3C',
            'marker_primary': '#F39C12',
            'marker_candidate': '#E74C3C',
            'distribution_raw': '#7D3C98',
            'distribution_smoothed': '#2E86C1',
            'sampling_distribution': '#E74C3C',
            'distribution_mean': '#E74C3C',
            'distribution_std': '#F39C12',
            'clt_threshold': '#27AE60',
            'dynamic_extension': '#E67E22',  # New color for dynamic extension visualization
        }
        
        self.heatmap_cmaps = {
            'viridis': 'viridis',
            'plasma': 'plasma',
            'inferno': 'inferno',
            'magma': 'magma',
            'coolwarm': 'coolwarm',
            'RdYlBu_r': 'RdYlBu_r',
            'Spectral_r': 'Spectral_r'
        }
        
        if compare_centromeres_file and os.path.exists(compare_centromeres_file):
            self._load_compare_centromeres(compare_centromeres_file)
    
    def load_data(self):
        """Load analysis results and prepare for visualization"""
        print("Loading EasyCen analysis results...")
        
        if self.known_centromeres_file and os.path.exists(self.known_centromeres_file):
            self._load_known_centromeres()
        
        if self.known_centromeres:
            self._create_centromere_data_from_known()
        else:
            self._load_centromere_summary()
        
        bedgraph_files = glob.glob(str(self.results_dir / "*_kmer.bedgraph"))
        
        for kmer_file in bedgraph_files:
            chrom_name = os.path.basename(kmer_file).replace('_kmer.bedgraph', '')
            
            chrom_data = {
                'kmer_density': self._load_bedgraph(kmer_file),
                'gc_content': self._load_bedgraph(kmer_file.replace('_kmer.bedgraph', '_GC.bedgraph')),
                'cpg_density': self._load_bedgraph(kmer_file.replace('_kmer.bedgraph', '_CpG.bedgraph')),
                'feature_percent': self._load_bedgraph(kmer_file.replace('_kmer.bedgraph', '_feature_percent.bedgraph'))
            }
            
            if chrom_data['kmer_density']:
                chrom_length = chrom_data['kmer_density'][-1]['end']
                chrom_data['length'] = chrom_length
                
                for track in chrom_data.values():
                    if isinstance(track, list):
                        for item in track:
                            item['pos_mb'] = (item['start'] + item['end']) / 2 / 1e6
                            
                self.chrom_data[chrom_name] = chrom_data
        
        if not self.known_centromeres:
            self._save_analysis_centromeres_bed()
        
        if self.known_centromeres:
            self._optimize_centromere_boundaries()
            self._save_optimized_centromeres_bed()
    
    def _load_bedgraph(self, filepath):
        """Load bedgraph file into list of dictionaries"""
        if not os.path.exists(filepath):
            return []
        
        data = []
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        data.append({
                            'chrom': parts[0],
                            'start': int(parts[1]),
                            'end': int(parts[2]),
                            'value': float(parts[3])
                        })
            return data
        except Exception:
            return []
    
    def _load_known_centromeres(self):
        """Load known centromere regions from BED file"""
        try:
            with open(self.known_centromeres_file, 'r') as f:
                for line in f:
                    if line.startswith('#') or not line.strip():
                        continue
                    
                    parts = line.strip().split('\t')
                    if len(parts) >= 4:
                        chrom = parts[0]
                        start = int(parts[1])
                        end = int(parts[2])
                        name = parts[3]
                        
                        if chrom not in self.known_centromeres:
                            self.known_centromeres[chrom] = []
                        
                        self.known_centromeres[chrom].append({
                            'start': start,
                            'end': end,
                            'name': name,
                            'start_mb': start / 1e6,
                            'end_mb': end / 1e6,
                            'center_mb': (start + end) / 2 / 1e6,
                            'length': end - start,
                            'is_primary': True,
                            'type': 'known',
                            'source': 'known_bed'
                        })
        except Exception:
            pass
    
    def _load_compare_centromeres(self, compare_file):
        """Load comparison centromere regions from BED file"""
        try:
            with open(compare_file, 'r') as f:
                for line in f:
                    if line.startswith('#') or not line.strip():
                        continue
                    
                    parts = line.strip().split('\t')
                    if len(parts) >= 4:
                        chrom = parts[0]
                        start = int(parts[1])
                        end = int(parts[2])
                        name = parts[3]
                        
                        if chrom not in self.compare_centromeres:
                            self.compare_centromeres[chrom] = []
                        
                        self.compare_centromeres[chrom].append({
                            'start': start,
                            'end': end,
                            'name': name,
                            'start_mb': start / 1e6,
                            'end_mb': end / 1e6,
                            'center_mb': (start + end) / 2 / 1e6,
                            'length': end - start,
                            'source': 'comparison_bed'
                        })
        except Exception:
            pass
    
    def _create_centromere_data_from_known(self):
        """Create centromere data structure from known centromeres"""
        for chrom, centromeres in self.known_centromeres.items():
            self.centromere_data[chrom] = centromeres
            for centromere in centromeres:
                centromere['is_primary'] = True
    
    def _save_analysis_centromeres_bed(self):
        """Save analysis centromere regions to BED file"""
        if not self.centromere_data:
            return
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        bed_file = self.output_dir / "analysis_centromeres.bed"
        
        with open(bed_file, 'w') as f:
            f.write("# EasyCen centromere regions from analysis results\n")
            f.write("# Format: chrom\tstart\tend\tname\tscore\tstrand\n")
            
            for chrom in self._natural_sort(self.centromere_data.keys()):
                for centromere in self.centromere_data[chrom]:
                    if centromere.get('is_primary', False):
                        name = f"CEN{chrom}"
                        score = 1000
                    else:
                        rank = centromere.get('rank', 'unknown')
                        name = f"CEN{chrom}_candidate_{rank}"
                        score = 500
                    
                    f.write(f"{chrom}\t{centromere['start']}\t{centromere['end']}\t"
                           f"{name}\t{score}\t.\n")
    
    def _load_centromere_summary(self):
        """Load centromere summary from analysis results"""
        summary_file = self.results_dir / "centromere_summary.txt"
        if not summary_file.exists():
            return
        
        try:
            with open(summary_file, 'r') as f:
                content = f.read()
        except Exception:
            return
        
        sections = content.split("DETAILED CANDIDATE REGIONS:")
        primary_section = sections[0]
        detailed_section = sections[1] if len(sections) > 1 else ""
        
        primary_lines = primary_section.split('\n')
        in_primary_table = False
        
        for line in primary_lines:
            line = line.strip()
            
            if line.startswith('Chromosome') and 'Start' in line and 'End' in line:
                in_primary_table = True
                continue
            elif line.startswith('---'):
                continue
            elif line.startswith('Total primary') or not line:
                in_primary_table = False
                continue
            
            if in_primary_table and line:
                parts = re.split(r'\s+', line)
                if len(parts) >= 8:
                    try:
                        chrom = parts[0]
                        centromere_type = parts[1].lower()
                        start = int(parts[2].replace(',', ''))
                        end = int(parts[3].replace(',', ''))
                        length = int(parts[4].replace(',', ''))
                        
                        if chrom not in self.centromere_data:
                            self.centromere_data[chrom] = []
                        
                        centromere_info = {
                            'start': start,
                            'end': end,
                            'length': length,
                            'start_mb': start / 1e6,
                            'end_mb': end / 1e6,
                            'center_mb': (start + end) / 2 / 1e6,
                            'is_primary': True,
                            'type': centromere_type
                        }
                        
                        if len(parts) >= 6:
                            try:
                                centromere_info['avg_kmer_density'] = float(parts[5])
                            except (ValueError, IndexError):
                                pass
                        
                        if len(parts) >= 8:
                            try:
                                centromere_info['score'] = float(parts[7])
                            except (ValueError, IndexError):
                                pass
                        
                        self.centromere_data[chrom].append(centromere_info)
                        
                    except (ValueError, IndexError):
                        continue
        
        if detailed_section:
            chrom_sections = re.split(r'(\w+)\s*\(Length:\s*[\d,]+ bp\):', detailed_section)
            
            for i in range(1, len(chrom_sections), 2):
                chrom_name = chrom_sections[i].strip()
                chrom_content = chrom_sections[i+1] if i+1 < len(chrom_sections) else ""
                
                lines = chrom_content.split('\n')
                in_table = False
                
                for line in lines:
                    line = line.strip()
                    
                    if line.startswith('Rank') and 'Start' in line and 'End' in line:
                        in_table = True
                        continue
                    elif line.startswith('---'):
                        continue
                    elif not line:
                        in_table = False
                        continue
                    
                    if in_table and line:
                        parts = re.split(r'\s+', line)
                        if len(parts) >= 8:
                            if parts[0] == 'PRIMARY' or parts[0] == '#1':
                                continue
                            
                            try:
                                centromere_type = parts[1].lower()
                                start = int(parts[2].replace(',', ''))
                                end = int(parts[3].replace(',', ''))
                                length = int(parts[4].replace(',', ''))
                                
                                if chrom_name not in self.centromere_data:
                                    self.centromere_data[chrom_name] = []
                                
                                candidate_info = {
                                    'start': start,
                                    'end': end,
                                    'length': length,
                                    'start_mb': start / 1e6,
                                    'end_mb': end / 1e6,
                                    'center_mb': (start + end) / 2 / 1e6,
                                    'is_primary': False,
                                    'rank': parts[0],
                                    'type': centromere_type
                                }
                                
                                if len(parts) >= 6:
                                    try:
                                        candidate_info['avg_kmer_density'] = float(parts[5])
                                    except (ValueError, IndexError):
                                        pass
                                
                                if len(parts) >= 8:
                                    try:
                                        candidate_info['score'] = float(parts[7])
                                    except (ValueError, IndexError):
                                        pass
                                
                                self.centromere_data[chrom_name].append(candidate_info)
                                
                            except (ValueError, IndexError):
                                continue

    def _optimize_centromere_boundaries(self):
        """Optimize centromere boundaries for known centromeres"""
        for chrom, known_centromeres in self.known_centromeres.items():
            if chrom not in self.chrom_data:
                continue
                
            chrom_data = self.chrom_data[chrom]
            self.optimized_centromeres[chrom] = []
            
            for known_centromere in known_centromeres:
                print(f"Optimizing boundaries for {chrom}...")
                optimized = self.optimizer.optimize_boundaries(
                    known_centromere,
                    chrom_data['kmer_density'],
                    chrom_data['feature_percent'],
                    chrom_data['length']
                )
                self.optimized_centromeres[chrom].append(optimized)

    def _save_optimized_centromeres_bed(self):
        """Save optimized centromere boundaries to BED file"""
        if not self.optimized_centromeres:
            return
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        bed_file = self.output_dir / f"optimized_centromeres.bed"
        
        with open(bed_file, 'w') as f:
            f.write(f"# EasyCen optimized centromere boundaries\n")
            f.write(f"# Generated from: {self.known_centromeres_file}\n")
            f.write(f"# Optimization parameters: kmer_weight={self.optimizer.kmer_weight}, "
                   f"feature_weight={self.optimizer.feature_weight}, "
                   f"extension={self.optimizer.extension_bp}bp\n")
            f.write(f"# Dynamic extension: {self.optimizer.dynamic_extension}\n")
            f.write(f"# Format: chrom\\tstart\\tend\\tname\\tscore\\tstrand\n")
            
            for chrom in self._natural_sort(self.optimized_centromeres.keys()):
                for optimized in self.optimized_centromeres[chrom]:
                    original_size = optimized['end'] - optimized['start']
                    optimized_size = optimized['optimized_length']
                    
                    if original_size > 0:
                        size_change_ratio = optimized_size / original_size
                    else:
                        size_change_ratio = 1.0
                    
                    original_name = optimized.get('name', f'CEN{chrom}')
                    optimized_name = f"{original_name}"
                    
                    score = min(1000, int(1000 * (1 - abs(1 - size_change_ratio))))
                    
                    f.write(f"{chrom}\t{optimized['optimized_start']}\t{optimized['optimized_end']}\t"
                           f"{optimized_name}\t{score}\t.\n")
        
        self._save_centromere_comparison_bed()
    
    def _save_centromere_comparison_bed(self):
        """Save centromere comparison BED file"""
        if not self.optimized_centromeres:
            return
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        comparison_file = self.output_dir / "centromere_comparison.bed"
        
        with open(comparison_file, 'w') as f:
            f.write("# EasyCen centromere comparison: original vs optimized\n")
            f.write("# track name=CentromereComparison description=\"Centromere boundary comparison\" "
                   "visibility=2 itemRgb=On\n")
            
            for chrom in self._natural_sort(self.optimized_centromeres.keys()):
                for optimized in self.optimized_centromeres[chrom]:
                    original_name = optimized.get('name', f'CEN{chrom}')
                    f.write(f"{chrom}\t{optimized['start']}\t{optimized['end']}\t"
                           f"{original_name}_original\t1000\t.\t{optimized['start']}\t{optimized['end']}\t255,0,0\n")
                    
                    optimized_name = f"{original_name}_optimized"
                    f.write(f"{chrom}\t{optimized['optimized_start']}\t{optimized['optimized_end']}\t"
                           f"{optimized_name}\t1000\t.\t{optimized['optimized_start']}\t{optimized['optimized_end']}\t0,255,0\n")

    def _natural_sort(self, l):
        """Natural sort for chromosome names"""
        def convert(text):
            return int(text) if text.isdigit() else text.lower()
        
        def alphanum_key(key):
            return [convert(c) for c in re.split('([0-9]+)', key)]
        
        return sorted(l, key=alphanum_key)

    def create_genome_overview(self, output_file=None):
        """Create genome-wide centromere overview plot"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if not output_file:
            output_file = self.output_dir / "genome_centromere_overview.pdf"
        else:
            output_file = Path(output_file)
        
        chroms = self._natural_sort(self.chrom_data.keys())
        n_chroms = len(chroms)
        
        if n_chroms == 0:
            print("No chromosome data available for visualization")
            return
        
        fig = plt.figure(figsize=(16, 10), constrained_layout=True)
        gs = gridspec.GridSpec(2, 1, height_ratios=[5, 1], figure=fig)
        
        ax_main = plt.subplot(gs[0])
        
        y_positions = np.arange(n_chroms)
        chrom_lengths = [self.chrom_data[chrom]['length'] for chrom in chroms]
        max_length = max(chrom_lengths) / 1e6
        
        all_types = set()
        for chrom in chroms:
            if chrom in self.centromere_data:
                for centromere in self.centromere_data[chrom]:
                    ctype = centromere.get('type', 'unknown')
                    all_types.add(ctype)
        
        for i, chrom in enumerate(chroms):
            chrom_length = self.chrom_data[chrom]['length'] / 1e6
            
            ax_main.plot([0, chrom_length], [i, i], 
                        color='black', linewidth=4, alpha=0.9, solid_capstyle='round')
            
            ax_main.text(-max_length * 0.04, i, chrom, 
                        ha='right', va='center', fontweight='bold', fontsize=12)
            
            if chrom in self.centromere_data:
                primary_regions = [r for r in self.centromere_data[chrom] if r.get('is_primary', False)]
                candidate_regions = [r for r in self.centromere_data[chrom] if not r.get('is_primary', False)]
                
                for centromere in candidate_regions:
                    start_mb = centromere['start_mb']
                    end_mb = centromere['end_mb']
                    centromere_type = centromere.get('type', 'unknown')
                    
                    if self.known_centromeres:
                        color = self.colors.get(centromere_type, self.colors['unknown'])
                    else:
                        color = self.colors['analysis_candidate']
                    
                    min_width = 0.01 * chrom_length
                    actual_width = end_mb - start_mb
                    display_width = max(actual_width, min_width)
                    
                    if actual_width < min_width:
                        center = (start_mb + end_mb) / 2
                        start_mb_display = center - display_width / 2
                        end_mb_display = center + display_width / 2
                    else:
                        start_mb_display = start_mb
                        end_mb_display = end_mb
                    
                    rect = Rectangle((start_mb_display, i-0.2), display_width, 0.4,
                                   facecolor=color, alpha=0.7, 
                                   edgecolor='darkred', linewidth=1.0)
                    ax_main.add_patch(rect)
                
                for centromere in primary_regions:
                    start_mb = centromere['start_mb']
                    end_mb = centromere['end_mb']
                    centromere_type = centromere.get('type', 'unknown')
                    
                    if self.known_centromeres:
                        color = self.colors.get(centromere_type, self.colors['unknown'])
                        edge_color = 'darkorange'
                    else:
                        color = self.colors['analysis_primary']
                        edge_color = 'darkblue'
                    
                    min_width = 0.01 * chrom_length
                    actual_width = end_mb - start_mb
                    display_width = max(actual_width, min_width)
                    
                    if actual_width < min_width:
                        center = (start_mb + end_mb) / 2
                        start_mb_display = center - display_width / 2
                        end_mb_display = center + display_width / 2
                    else:
                        start_mb_display = start_mb
                        end_mb_display = end_mb
                    
                    rect = Rectangle((start_mb_display, i-0.4), display_width, 0.8,
                                   facecolor=color, alpha=0.9, 
                                   edgecolor=edge_color, linewidth=2.0)
                    ax_main.add_patch(rect)
                    
                    center_mb = centromere['center_mb']
                    ax_main.plot(center_mb, i, 'o', color='white', markersize=6, 
                               markeredgecolor=edge_color, markeredgewidth=2)
            
            if chrom in self.optimized_centromeres:
                for optimized in self.optimized_centromeres[chrom]:
                    start_mb = optimized['optimized_start_mb']
                    end_mb = optimized['optimized_end_mb']
                    
                    rect = Rectangle((start_mb, i-0.5), end_mb - start_mb, 1.0,
                                   facecolor='none', alpha=0.8,
                                   edgecolor=self.colors['optimized_centromere'], 
                                   linewidth=2.0, linestyle='--')
                    ax_main.add_patch(rect)
            
            if chrom in self.compare_centromeres:
                for compare_cent in self.compare_centromeres[chrom]:
                    start_mb = compare_cent['start_mb']
                    end_mb = compare_cent['end_mb']
                    
                    rect = Rectangle((start_mb, i-0.6), end_mb - start_mb, 1.2,
                                   facecolor='none', alpha=0.7,
                                   edgecolor=self.colors['compare_centromere'], 
                                   linewidth=2.0, linestyle=':')
                    ax_main.add_patch(rect)
        
        ax_main.set_xlim(0, max_length * 1.05)
        ax_main.set_ylim(-0.7, n_chroms - 0.3)
        ax_main.set_ylabel('Chromosomes', fontweight='bold', fontsize=14)
        ax_main.set_xlabel('Chromosome Position (Mb)', fontweight='bold', fontsize=14)
        
        title = 'EasyCen - Genome-wide Centromere Distribution'
        if self.optimized_centromeres:
            title += ' (with Boundary Optimization)'
            if self.optimizer.dynamic_extension:
                title += ' [Dynamic Extension]'
        elif not self.known_centromeres:
            title += ' (Analysis Results)'
        ax_main.set_title(title, fontsize=18, fontweight='bold', pad=20)
        
        ax_main.tick_params(axis='x', labelsize=11)
        ax_main.tick_params(axis='y', labelsize=11)
        ax_main.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        
        legend_elements = []
        
        if self.known_centromeres:
            for ctype in sorted(all_types):
                color = self.colors.get(ctype, self.colors['unknown'])
                legend_elements.append(
                    Patch(facecolor=color, alpha=0.9, 
                          edgecolor='darkorange', linewidth=2.0, 
                          label=ctype.capitalize())
                )
        else:
            legend_elements.extend([
                Patch(facecolor=self.colors['analysis_primary'], alpha=0.9, 
                      edgecolor='darkblue', linewidth=2.0, 
                      label='Primary Centromere'),
                Patch(facecolor=self.colors['analysis_candidate'], alpha=0.7, 
                      edgecolor='darkred', linewidth=1.0, 
                      label='Candidate Region')
            ])
        
        legend_elements.extend([
            plt.Line2D([0], [0], color='black', linewidth=4, label='Chromosome')
        ])
        
        if self.optimized_centromeres:
            legend_elements.append(
                plt.Line2D([0], [0], color=self.colors['optimized_centromere'], 
                          linewidth=2, linestyle='--', label='Optimized Boundary')
            )
        
        if self.compare_centromeres:
            legend_elements.append(
                plt.Line2D([0], [0], color=self.colors['compare_centromere'], 
                          linewidth=2, linestyle=':', label='Published Centromere')
            )
        
        if self.optimizer.dynamic_extension:
            legend_elements.append(
                plt.Line2D([0], [0], color=self.colors['dynamic_extension'], 
                          linewidth=2, linestyle='-.', label='Dynamic Extension')
            )
        
        ax_main.legend(handles=legend_elements, loc='upper right', 
                      framealpha=0.95, fancybox=True, shadow=True, 
                      fontsize=10, frameon=True, ncol=2)
        
        ax_stats = plt.subplot(gs[1])
        
        primary_centromeres = []
        candidate_centromeres = []
        optimized_centromeres = []
        compare_centromeres = []
        type_counts = {}
        extension_history = []
        
        for chrom in chroms:
            if chrom in self.centromere_data:
                for centromere in self.centromere_data[chrom]:
                    ctype = centromere.get('type', 'unknown')
                    type_counts[ctype] = type_counts.get(ctype, 0) + 1
                    
                    if centromere.get('is_primary', False):
                        primary_centromeres.append(centromere['length'] / 1e6)
                    else:
                        candidate_centromeres.append(centromere['length'] / 1e6)
            
            if chrom in self.optimized_centromeres:
                for optimized in self.optimized_centromeres[chrom]:
                    optimized_centromeres.append(optimized['optimized_length'] / 1e6)
                    # Collect dynamic extension info
                    if 'dynamic_extension_info' in optimized:
                        ext_info = optimized['dynamic_extension_info']
                        extension_history.append({
                            'chrom': chrom,
                            'final_extension': ext_info['final_extension'],
                            'iterations': ext_info['iterations']
                        })
            
            if chrom in self.compare_centromeres:
                for compare_cent in self.compare_centromeres[chrom]:
                    compare_centromeres.append(compare_cent['length'] / 1e6)
        
        stats_text = [
            "EASYCEN GENOME CENTROMERE SUMMARY",
            "=" * 30,
            f"Total Chromosomes: {n_chroms}",
            f"Chromosomes with Centromeres: {sum(1 for chrom in chroms if chrom in self.centromere_data)}"
        ]
        
        if primary_centromeres:
            stats_text.extend([
                f"Primary Centromeres: {len(primary_centromeres)}",
                f"Mean Size: {np.mean(primary_centromeres):.2f} ± {np.std(primary_centromeres):.2f} Mb",
                f"Size Range: {min(primary_centromeres):.2f}-{max(primary_centromeres):.2f} Mb"
            ])
        
        if candidate_centromeres:
            stats_text.append(f"Candidate Regions: {len(candidate_centromeres)}")
        
        if optimized_centromeres:
            stats_text.extend([
                f"Optimized Centromeres: {len(optimized_centromeres)}",
                f"Optimized Mean Size: {np.mean(optimized_centromeres):.2f} ± {np.std(optimized_centromeres):.2f} Mb"
            ])
            
            if self.optimizer.dynamic_extension and extension_history:
                avg_extension = np.mean([e['final_extension'] for e in extension_history]) / 1000
                avg_iterations = np.mean([e['iterations'] for e in extension_history])
                stats_text.extend([
                    f"Dynamic Extension: Enabled",
                    f"Avg Extension: {avg_extension:.1f} kb",
                    f"Avg Iterations: {avg_iterations:.1f}"
                ])
        
        if compare_centromeres:
            stats_text.extend([
                f"Published Centromeres: {len(compare_centromeres)}",
                f"Published Mean Size: {np.mean(compare_centromeres):.2f} ± {np.std(compare_centromeres):.2f} Mb"
            ])
        
        ax_stats.text(0.02, 0.5, '\n'.join(stats_text), 
                     transform=ax_stats.transAxes, va='top', ha='left',
                     bbox=dict(boxstyle="round,pad=0.8", facecolor='white',
                             edgecolor='gray', alpha=0.9, linewidth=1.5),
                     fontfamily='monospace', fontsize=9, fontweight='bold')
        
        ax_stats.set_xlim(0, 1)
        ax_stats.set_ylim(0, 1)
        ax_stats.set_xticks([])
        ax_stats.set_yticks([])
        
        for spine in ax_stats.spines.values():
            spine.set_visible(False)
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Genome overview saved to: {output_file}")

    def create_chromosome_detail_plots(self, output_dir=None):
        """Create detailed plots for each chromosome"""
        if not output_dir:
            output_dir = self.output_dir / "chromosome_details"
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        chrom_count = 0
        for chrom in self._natural_sort(self.chrom_data.keys()):
            output_file = output_dir / f"{chrom}_detailed_analysis.pdf"
            try:
                self._create_single_chromosome_plot(chrom, output_file)
                chrom_count += 1
            except Exception as e:
                print(f"Error creating plot for {chrom}: {e}")
        
        print(f"Created detailed plots for {chrom_count} chromosomes in: {output_dir}")

    def _create_single_chromosome_plot(self, chrom, output_file):
        """Create detailed plot for a single chromosome"""
        chrom_data = self.chrom_data[chrom]
        chrom_length_mb = chrom_data['length'] / 1e6
        
        # Adjust figure height based on whether we have optimization data
        if chrom in self.optimized_centromeres:
            # Increased height for sampling distribution and dynamic extension visualization
            fig = plt.figure(figsize=(14, 20), constrained_layout=True)
            gs = gridspec.GridSpec(15, 1, height_ratios=[0.8, 0.15, 1, 0.3, 1, 0.3, 1, 0.3, 1, 0.3, 1.5, 1.0, 0.8, 0.5, 0.5], figure=fig)
        else:
            fig = plt.figure(figsize=(14, 14), constrained_layout=True)
            gs = gridspec.GridSpec(10, 1, height_ratios=[0.8, 0.15, 1, 0.3, 1, 0.3, 1, 0.3, 1, 0.3], figure=fig)
        
        ax_title = plt.subplot(gs[0])
        self._create_chromosome_header(ax_title, chrom, chrom_data)
        
        ax_marker = plt.subplot(gs[1])
        self._create_centromere_marker_track(ax_marker, chrom)
        
        ax_kmer_line = plt.subplot(gs[2])
        ax_kmer_heat = plt.subplot(gs[3])
        self._create_enhanced_track_plot(ax_kmer_line, ax_kmer_heat, chrom_data['kmer_density'], 
                                       'K-mer Density', self.colors['kmer_density'], 
                                       'Count', chrom, is_last=False, chrom_length_mb=chrom_length_mb)
        
        ax_gc_line = plt.subplot(gs[4])
        ax_gc_heat = plt.subplot(gs[5])
        self._create_enhanced_track_plot(ax_gc_line, ax_gc_heat, chrom_data['gc_content'], 
                                       'GC Content', self.colors['gc_content'], 
                                       'Percentage (%)', chrom, is_last=False, chrom_length_mb=chrom_length_mb)
        
        ax_cpg_line = plt.subplot(gs[6])
        ax_cpg_heat = plt.subplot(gs[7])
        self._create_enhanced_track_plot(ax_cpg_line, ax_cpg_heat, chrom_data['cpg_density'], 
                                       'CpG Density', self.colors['cpg_density'], 
                                       'Count', chrom, is_last=False, chrom_length_mb=chrom_length_mb)
        
        ax_feature_line = plt.subplot(gs[8])
        ax_feature_heat = plt.subplot(gs[9])
        self._create_enhanced_track_plot(ax_feature_line, ax_feature_heat, chrom_data['feature_percent'], 
                                       'Feature Percentage', self.colors['feature_percent'], 
                                       'Percentage (%)', chrom, is_last=not self.optimized_centromeres, chrom_length_mb=chrom_length_mb)
        
        if chrom in self.optimized_centromeres:
            ax_optimization = plt.subplot(gs[10])
            ax_sampling = plt.subplot(gs[11])
            ax_dynamic_extension = plt.subplot(gs[12])  # New panel for dynamic extension visualization
            ax_optimization_legend = plt.subplot(gs[13])
            ax_dynamic_legend = plt.subplot(gs[14])
            self._create_enhanced_optimization_panel(ax_optimization, ax_sampling, ax_dynamic_extension, 
                                                   ax_optimization_legend, ax_dynamic_legend, chrom, chrom_length_mb)
        
        for ax in [ax_kmer_line, ax_kmer_heat, ax_gc_line, ax_gc_heat, 
                  ax_cpg_line, ax_cpg_heat, ax_feature_line, ax_feature_heat]:
            self._add_elegant_centromere_annotations(ax, chrom)
            ax.set_xlim(0, chrom_length_mb)
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()

    def _create_enhanced_optimization_panel(self, ax, ax_sampling, ax_dynamic, ax_legend, ax_dynamic_legend, chrom, chrom_length_mb):
        """Create optimization visualization panel with sampling distribution and dynamic extension visualization"""
        if chrom not in self.optimized_centromeres:
            return
        
        optimized_data = self.optimized_centromeres[chrom][0]
        
        positions = np.array(optimized_data['search_positions'])
        composite_scores = np.array(optimized_data['composite_scores'])
        composite_smoothed = np.array(optimized_data.get('composite_smoothed', composite_scores))
        
        positions_mb = positions / 1e6
        known_start_mb = optimized_data['start'] / 1e6
        known_end_mb = optimized_data['end'] / 1e6
        opt_start_mb = optimized_data['optimized_start_mb']
        opt_end_mb = optimized_data['optimized_end_mb']
        
        search_start_mb = max(0, known_start_mb - self.optimizer.extension_bp / 1e6)
        search_end_mb = min(chrom_length_mb, known_end_mb + self.optimizer.extension_bp / 1e6)
        
        focus_start_mb = search_start_mb
        focus_end_mb = search_end_mb
        
        focus_mask = (positions_mb >= focus_start_mb) & (positions_mb <= focus_end_mb)
        focus_positions = positions_mb[focus_mask]
        focus_composite = composite_scores[focus_mask]
        focus_smoothed = composite_smoothed[focus_mask]
        
        # Main optimization panel
        ax.plot(focus_positions, focus_composite, color=self.colors['distribution_raw'], 
               label='Composite Score', linewidth=1.5, alpha=0.7)
        ax.plot(focus_positions, focus_smoothed, color=self.colors['distribution_smoothed'], 
               label='Smoothed Score', linewidth=2.5, alpha=0.9)
        
        if 'peaks' in optimized_data and optimized_data['peaks']:
            peak_positions = positions_mb[optimized_data['peaks']]
            peak_values = composite_smoothed[optimized_data['peaks']]
            ax.scatter(peak_positions, peak_values, color='red', s=50, zorder=5,
                     label=f'Peaks ({len(peak_positions)})')
        
        threshold_info = optimized_data.get('threshold_info', {})
        if 'threshold_levels' in threshold_info:
            colors = ['#FF0000', '#FF6600', '#FFCC00', '#00CC00']
            for i, (level_name, threshold_value) in enumerate(threshold_info['threshold_levels'].items()):
                if level_name == 'very_aggressive(select)':  # Only show selected threshold
                    ax.axhline(y=threshold_value, color=colors[i], linestyle='--', 
                              alpha=0.7, linewidth=1.5, label=f'Threshold')
        
        ax.axvspan(search_start_mb, search_end_mb, alpha=0.1, color=self.colors['search_region'],
                  label='Search Region')
        ax.axvspan(known_start_mb, known_end_mb, alpha=0.2, color=self.colors['known_centromere'],
                  label='Known Centromere')
        ax.axvspan(opt_start_mb, opt_end_mb, alpha=0.3, color=self.colors['optimized_centromere'],
                  label='Optimized Centromere')
        
        if chrom in self.compare_centromeres:
            for compare_cent in self.compare_centromeres[chrom]:
                compare_start_mb = compare_cent['start_mb']
                compare_end_mb = compare_cent['end_mb']
                ax.axvspan(compare_start_mb, compare_end_mb, alpha=0.2, 
                          color=self.colors['compare_centromere'],
                          label='Published Centromere')
        
        ax.axvline(known_start_mb, color=self.colors['known_centromere'], linestyle='-', alpha=0.7, linewidth=2)
        ax.axvline(known_end_mb, color=self.colors['known_centromere'], linestyle='-', alpha=0.7, linewidth=2)
        ax.axvline(opt_start_mb, color=self.colors['optimized_centromere'], linestyle='--', alpha=0.9, linewidth=2)
        ax.axvline(opt_end_mb, color=self.colors['optimized_centromere'], linestyle='--', alpha=0.9, linewidth=2)
        
        if chrom in self.compare_centromeres:
            for compare_cent in self.compare_centromeres[chrom]:
                compare_start_mb = compare_cent['start_mb']
                compare_end_mb = compare_cent['end_mb']
                ax.axvline(compare_start_mb, color=self.colors['compare_centromere'], 
                          linestyle=':', alpha=0.8, linewidth=2)
                ax.axvline(compare_end_mb, color=self.colors['compare_centromere'], 
                          linestyle=':', alpha=0.8, linewidth=2)
        
        ax.set_xlim(focus_start_mb, focus_end_mb)
        
        x_ticks = np.linspace(focus_start_mb, focus_end_mb, 20)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f'{x:.3f}' for x in x_ticks], rotation=45)
        
        ax.set_xlabel('Position in Search Region (Mb)', fontweight='bold')
        ax.set_ylabel('Composite Score', fontweight='bold')
        ax.set_title('EasyCen Centromere Boundary Optimization', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        
        stats_text = [
            f"Known: {known_start_mb:.3f}-{known_end_mb:.3f} Mb",
            f"Optimized: {opt_start_mb:.3f}-{opt_end_mb:.3f} Mb"
        ]
        
        if optimized_data['length'] > 0:
            size_change = ((optimized_data['optimized_length'] / optimized_data['length']) - 1) * 100
            stats_text.append(f"Size change: {size_change:+.1f}%")
        else:
            stats_text.append("Size change: N/A (original size 0)")
        
        if 'threshold_info' in optimized_data:
            t_info = optimized_data['threshold_info']
            stats_text.extend([
                f"Selected threshold: {t_info.get('selected_threshold', 'unknown')}",
                f"Detected peaks: {len(optimized_data.get('peaks', []))}",
                f"Sampling: {t_info.get('sampling_times', 0)} iterations"
            ])
        
        ax.text(0.02, 0.85, '\n'.join(stats_text), transform=ax.transAxes,
               va='top', ha='left', fontsize=8,
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # Sampling distribution panel
        if 'sampling_distribution_stats' in threshold_info:
            sampling_stats = threshold_info['sampling_distribution_stats']
            sample_means = sampling_stats.get('sample_means', [])
            
            if sample_means and len(sample_means) > 0:
                # Plot sampling distribution histogram
                n, bins, patches = ax_sampling.hist(sample_means, bins=50, alpha=0.7, 
                                                   color=self.colors['sampling_distribution'],
                                                   density=True, edgecolor='black', linewidth=0.5)
                
                # Plot fitted normal distribution curve
                xmin, xmax = ax_sampling.get_xlim()
                x = np.linspace(xmin, xmax, 100)
                
                # Ensure sampling_mean and sampling_std values exist
                sampling_mean = threshold_info.get('sampling_mean', 0.5)
                sampling_std = threshold_info.get('sampling_std', 0.1)
                
                p = norm.pdf(x, sampling_mean, sampling_std)
                ax_sampling.plot(x, p, 'k', linewidth=2, 
                               label=f'Normal fit (μ={sampling_mean:.3f}, σ={sampling_std:.3f})')
                
                # Mark threshold line
                if 'threshold_levels' in threshold_info:
                    for i, (level_name, threshold_value) in enumerate(threshold_info['threshold_levels'].items()):
                        if level_name == 'very_aggressive(select)':  # Only show selected threshold
                            ax_sampling.axvline(x=threshold_value, color='#FF0000', 
                                              linestyle='--', alpha=0.7, linewidth=1.5,
                                              label=f'Threshold')
                
                ax_sampling.set_xlabel('Sample Mean Value', fontweight='bold')
                ax_sampling.set_ylabel('Density', fontweight='bold')
                ax_sampling.set_title('Sampling Distribution of Means', 
                                    fontweight='bold', fontsize=12)
                ax_sampling.grid(True, alpha=0.3)
                ax_sampling.legend(fontsize=8)
                
                # Add sampling statistics
                sampling_info = [
                    "SAMPLING STATISTICS:",
                    f"Sample size: {threshold_info.get('sample_size', 'N/A')}",
                    f"Sampling iterations: {threshold_info.get('sampling_times', 'N/A')}",
                    f"Mean: {sampling_mean:.4f}",
                    f"Std: {sampling_std:.4f}",
                    f"Distribution: {'Normal' if sampling_stats.get('is_normal', True) else 'Non-normal'}"
                ]
                
                ax_sampling.text(0.02, 0.98, '\n'.join(sampling_info), transform=ax_sampling.transAxes,
                               va='top', ha='left', fontsize=8,
                               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            else:
                # Show message if no sampling data
                ax_sampling.text(0.5, 0.5, 'No sampling data available', 
                               transform=ax_sampling.transAxes, ha='center', va='center', fontsize=12)
                ax_sampling.set_xticks([])
                ax_sampling.set_yticks([])
        else:
            # Show message if no sampling distribution statistics
            ax_sampling.text(0.5, 0.5, 'No sampling distribution statistics available', 
                           transform=ax_sampling.transAxes, ha='center', va='center', fontsize=12)
            ax_sampling.set_xticks([])
            ax_sampling.set_yticks([])
        
        # Dynamic extension visualization panel
        if 'dynamic_extension_info' in optimized_data:
            dynamic_info = optimized_data['dynamic_extension_info']
            mean_history = dynamic_info.get('mean_history', [])
            
            if mean_history:
                iterations = [m['iteration'] for m in mean_history]
                extensions = [m['extension'] / 1000 for m in mean_history]  # Convert to kb
                means = [m['mean'] for m in mean_history]
                mean_diffs = [m.get('mean_diff', abs(m['mean'] - self.optimizer.target_mean)) for m in mean_history]
                
                # Create twin axes for extension and mean
                ax_ext = ax_dynamic
                ax_mean = ax_dynamic.twinx()
                
                # Plot extension history
                ext_line = ax_ext.plot(iterations, extensions, 'o-', 
                                     color=self.colors['dynamic_extension'], 
                                     linewidth=2, markersize=6, 
                                     label='Extension (kb)')
                
                # Plot mean history
                mean_line = ax_mean.plot(iterations, means, 's-', 
                                       color=self.colors['sampling_distribution'],
                                       linewidth=2, markersize=4,
                                       label='Region Mean')
                
                # Add target mean range
                ax_mean.axhspan(self.optimizer.target_mean - self.optimizer.mean_tolerance,
                              self.optimizer.target_mean + self.optimizer.mean_tolerance,
                              alpha=0.2, color='green', label='Target Range')
                
                ax_mean.axhline(y=self.optimizer.target_mean, color='green', 
                              linestyle='--', alpha=0.7, linewidth=1,
                              label='Target Mean')
                
                ax_ext.set_xlabel('Iteration', fontweight='bold')
                ax_ext.set_ylabel('Extension (kb)', fontweight='bold', 
                                color=self.colors['dynamic_extension'])
                ax_mean.set_ylabel('Region Mean', fontweight='bold',
                                 color=self.colors['sampling_distribution'])
                
                ax_ext.tick_params(axis='y', labelcolor=self.colors['dynamic_extension'])
                ax_mean.tick_params(axis='y', labelcolor=self.colors['sampling_distribution'])
                
                ax_dynamic.set_title('Dynamic Extension Optimization', 
                                   fontweight='bold', fontsize=12)
                ax_dynamic.grid(True, alpha=0.3)
                
                # Combine legends
                lines = ext_line + mean_line
                labels = [l.get_label() for l in lines]
                ax_dynamic.legend(lines, labels, loc='upper right', fontsize=8)
                
                # Add convergence info
                final_mean_diff = mean_diffs[-1] if mean_diffs else 0
                conv_info = [
                    "DYNAMIC EXTENSION:",
                    f"Final extension: {dynamic_info['final_extension']/1000:.1f} kb",
                    f"Total iterations: {dynamic_info['iterations']}",
                    f"Final mean: {means[-1]:.3f}",
                    f"Final mean diff: {final_mean_diff:.3f}",
                    f"Target mean: {self.optimizer.target_mean} ± {self.optimizer.mean_tolerance}"
                ]
                
                ax_dynamic.text(0.02, 0.98, '\n'.join(conv_info), 
                              transform=ax_dynamic.transAxes,
                              va='top', ha='left', fontsize=8,
                              bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            else:
                ax_dynamic.text(0.5, 0.5, 'No dynamic extension data available', 
                              transform=ax_dynamic.transAxes, ha='center', va='center', fontsize=12)
                ax_dynamic.set_xticks([])
                ax_dynamic.set_yticks([])
        else:
            ax_dynamic.text(0.5, 0.5, 'Dynamic extension not enabled', 
                          transform=ax_dynamic.transAxes, ha='center', va='center', fontsize=12)
            ax_dynamic.set_xticks([])
            ax_dynamic.set_yticks([])
        
        # Legend panel for optimization parameters
        ax_legend.set_xlim(0, 1)
        ax_legend.set_ylim(0, 1)
        ax_legend.set_xticks([])
        ax_legend.set_yticks([])
        for spine in ax_legend.spines.values():
            spine.set_visible(False)
        
        legend_text = [
            "OPTIMIZATION PARAMETERS:",
            f"K-mer weight: {self.optimizer.kmer_weight}",
            f"Feature weight: {self.optimizer.feature_weight}", 
            f"Initial extension: {self.optimizer.extension_bp/1000:.0f} kb",
            f"Smoothing sigma: {self.optimizer.smoothing_sigma}",
            f"Base threshold: {self.optimizer.distribution_threshold}",
            f"Min region size: {self.optimizer.min_region_size/1000:.0f} kb",
            f"Random sampling: {self.optimizer.random_sampling_times} times"
        ]
        
        ax_legend.text(0.02, 0.5, '\n'.join(legend_text), transform=ax_legend.transAxes,
                      va='center', ha='left', fontfamily='monospace', fontsize=9,
                      bbox=dict(boxstyle="round,pad=0.5", facecolor=self.colors['background']))
        
        # Dynamic extension parameters panel
        ax_dynamic_legend.set_xlim(0, 1)
        ax_dynamic_legend.set_ylim(0, 1)
        ax_dynamic_legend.set_xticks([])
        ax_dynamic_legend.set_yticks([])
        for spine in ax_dynamic_legend.spines.values():
            spine.set_visible(False)
        
        if self.optimizer.dynamic_extension:
            dynamic_legend_text = [
                "DYNAMIC EXTENSION PARAMETERS:",
                f"Enabled: {self.optimizer.dynamic_extension}",
                f"Target mean: {self.optimizer.target_mean}",
                f"Mean tolerance: {self.optimizer.mean_tolerance}",
                f"Max extension factor: {self.optimizer.max_extension_factor}",
                f"Extension increment: {self.optimizer.extension_increment/1000:.0f} kb",
                f"Max iterations: 50"
            ]
            
            ax_dynamic_legend.text(0.02, 0.5, '\n'.join(dynamic_legend_text), 
                                 transform=ax_dynamic_legend.transAxes,
                                 va='center', ha='left', fontfamily='monospace', fontsize=9,
                                 bbox=dict(boxstyle="round,pad=0.5", facecolor=self.colors['background']))

    def _create_centromere_marker_track(self, ax, chrom):
        """Create centromere marker track for chromosome plot"""
        if chrom not in self.centromere_data and chrom not in self.optimized_centromeres and chrom not in self.compare_centromeres:
            ax.text(0.5, 0.5, 'No centromere data', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            return
        
        chrom_length = self.chrom_data[chrom]['length'] / 1e6
        ax.set_xlim(0, chrom_length)
        ax.set_ylim(0, 1)
        
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        track_title = 'Centromere Regions'
        if self.optimized_centromeres:
            track_title += ' (with Boundary Optimization)'
            if self.optimizer.dynamic_extension:
                track_title += ' [Dynamic Extension]'
        elif not self.known_centromeres:
            track_title += ' (Analysis Results)'
        ax.text(0.01, 0.5, track_title, transform=ax.transAxes, 
               ha='left', va='center', fontsize=9, fontweight='bold')
        
        if chrom in self.centromere_data:
            primary_regions = [r for r in self.centromere_data[chrom] if r.get('is_primary', False)]
            candidate_regions = [r for r in self.centromere_data[chrom] if not r.get('is_primary', False)]
            
            for centromere in candidate_regions:
                start_mb = centromere['start_mb']
                end_mb = centromere['end_mb']
                
                if self.known_centromeres:
                    centromere_type = centromere.get('type', 'unknown')
                    color = self.colors.get(centromere_type, self.colors['unknown'])
                    edge_color = 'darkred'
                else:
                    color = self.colors['analysis_candidate']
                    edge_color = 'darkred'
                
                rect = Rectangle((start_mb, 0.1), end_mb - start_mb, 0.3,
                               facecolor=color, alpha=0.8, edgecolor=edge_color, linewidth=0.5)
                ax.add_patch(rect)
            
            for centromere in primary_regions:
                start_mb = centromere['start_mb']
                end_mb = centromere['end_mb']
                
                if self.known_centromeres:
                    centromere_type = centromere.get('type', 'unknown')
                    color = self.colors.get(centromere_type, self.colors['unknown'])
                    edge_color = 'darkorange'
                    label = 'CEN'
                else:
                    color = self.colors['analysis_primary']
                    edge_color = 'darkblue'
                    label = 'CEN'
                
                rect = Rectangle((start_mb, 0.5), end_mb - start_mb, 0.4,
                               facecolor=color, alpha=0.9, edgecolor=edge_color, linewidth=1)
                ax.add_patch(rect)
                
                center_mb = centromere['center_mb']
                ax.text(center_mb, 0.75, label, ha='center', va='bottom', 
                       fontsize=7, fontweight='bold', color=edge_color)
        
        if chrom in self.optimized_centromeres:
            for optimized in self.optimized_centromeres[chrom]:
                start_mb = optimized['optimized_start_mb']
                end_mb = optimized['optimized_end_mb']
                
                rect = Rectangle((start_mb, 0.0), end_mb - start_mb, 1.0,
                               facecolor='none', alpha=0.8,
                               edgecolor=self.colors['optimized_centromere'], 
                               linewidth=2.0, linestyle='--')
                ax.add_patch(rect)
                
                center_mb = optimized['optimized_center_mb']
                ax.text(center_mb, 0.25, 'OPT', ha='center', va='bottom',
                       fontsize=7, fontweight='bold', color=self.colors['optimized_centromere'])
        
        if chrom in self.compare_centromeres:
            for compare_cent in self.compare_centromeres[chrom]:
                start_mb = compare_cent['start_mb']
                end_mb = compare_cent['end_mb']
                
                rect = Rectangle((start_mb, 0.0), end_mb - start_mb, 1.0,
                               facecolor='none', alpha=0.7,
                               edgecolor=self.colors['compare_centromere'], 
                               linewidth=2.0, linestyle=':')
                ax.add_patch(rect)
                
                center_mb = compare_cent['center_mb']
                ax.text(center_mb, 0.1, 'PUB', ha='center', va='bottom',
                       fontsize=7, fontweight='bold', color=self.colors['compare_centromere'])

    def _add_elegant_centromere_annotations(self, ax, chrom):
        """Add centromere annotations to track plots"""
        if chrom not in self.centromere_data and chrom not in self.optimized_centromeres and chrom not in self.compare_centromeres:
            return
        
        ylim = ax.get_ylim()
        chrom_length = self.chrom_data[chrom]['length'] / 1e6
        
        if chrom in self.centromere_data:
            for centromere in self.centromere_data[chrom]:
                start_mb = centromere['start_mb']
                end_mb = centromere['end_mb']
                
                if self.known_centromeres:
                    centromere_type = centromere.get('type', 'unknown')
                    border_color = self.colors.get(centromere_type, self.colors['unknown'])
                else:
                    if centromere.get('is_primary', False):
                        border_color = self.colors['analysis_primary']
                    else:
                        border_color = self.colors['analysis_candidate']
                
                if centromere.get('is_primary', False):
                    line_style = '-'
                    line_width = 1.5
                else:
                    line_style = '--'
                    line_width = 1.0
                
                ax.axvspan(start_mb, end_mb, alpha=0.08, color=self.colors['highlight_primary_bg'], 
                          zorder=1)
                
                ax.axvline(start_mb, color=border_color, linestyle=line_style, 
                          linewidth=line_width, alpha=0.6, zorder=5)
                ax.axvline(end_mb, color=border_color, linestyle=line_style, 
                          linewidth=line_width, alpha=0.6, zorder=5)
        
        if chrom in self.optimized_centromeres:
            for optimized in self.optimized_centromeres[chrom]:
                start_mb = optimized['optimized_start_mb']
                end_mb = optimized['optimized_end_mb']
                
                ax.axvline(start_mb, color=self.colors['optimized_centromere'], 
                          linestyle='--', linewidth=2.0, alpha=0.8, zorder=6)
                ax.axvline(end_mb, color=self.colors['optimized_centromere'], 
                          linestyle='--', linewidth=2.0, alpha=0.8, zorder=6)
        
        if chrom in self.compare_centromeres:
            for compare_cent in self.compare_centromeres[chrom]:
                start_mb = compare_cent['start_mb']
                end_mb = compare_cent['end_mb']
                
                ax.axvline(start_mb, color=self.colors['compare_centromere'], 
                          linestyle=':', linewidth=2.0, alpha=0.7, zorder=7)
                ax.axvline(end_mb, color=self.colors['compare_centromere'], 
                          linestyle=':', linewidth=2.0, alpha=0.7, zorder=7)

    def _create_enhanced_track_plot(self, ax_line, ax_heat, track_data, title, color, ylabel, chrom, is_last=False, chrom_length_mb=None):
        """Create enhanced track plot with line and heatmap"""
        if not track_data:
            ax_line.text(0.5, 0.5, 'No data available', 
                       transform=ax_line.transAxes, ha='center', va='center')
            ax_heat.text(0.5, 0.5, 'No data available', 
                       transform=ax_heat.transAxes, ha='center', va='center')
            ax_line.set_xticks([])
            ax_line.set_yticks([])
            ax_heat.set_xticks([])
            ax_heat.set_yticks([])
            return
        
        positions = [item['pos_mb'] for item in track_data]
        values = [item['value'] for item in track_data]
        
        if chrom_length_mb is None:
            chrom_length_mb = self.chrom_data[chrom]['length'] / 1e6
        
        ax_line.fill_between(positions, values, alpha=0.3, color=color, zorder=3)
        ax_line.plot(positions, values, color=color, linewidth=1.2, alpha=0.9, zorder=4)
        
        ax_line.set_xlim(0, chrom_length_mb)
        ax_line.set_ylabel(ylabel, fontweight='bold', fontsize=11)
        ax_line.set_title(title, fontweight='bold', fontsize=12, pad=10)
        ax_line.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, zorder=1)
        ax_line.tick_params(axis='both', which='major', labelsize=9)
        
        if is_last or self.optimized_centromeres:
            ax_line.set_xlabel('Chromosome Position (Mb)', fontweight='bold', fontsize=11)
        else:
            ax_line.tick_params(axis='x', which='major', labelsize=8)
            ax_line.set_xticklabels([f'{x:.1f}' for x in ax_line.get_xticks()])
        
        self._create_heatmap(ax_heat, positions, values, chrom_length_mb)
        
        if is_last:
            ax_heat.set_xlabel('Chromosome Position (Mb)', fontweight='bold', fontsize=11)
        else:
            ax_heat.set_xticklabels([])
    
    def _create_heatmap(self, ax, positions, values, chrom_length, bins=200):
        """Create heatmap representation of track data"""
        if not positions or not values:
            return
        
        bins_positions = np.linspace(0, chrom_length, bins)
        digitized = np.digitize(positions, bins_positions)
        
        bin_means = [np.mean([values[i] for i in range(len(values)) if digitized[i] == bin_idx]) 
                    for bin_idx in range(1, len(bins_positions))]
        
        bin_means = np.nan_to_num(bin_means, nan=0.0)
        
        heatmap_data = np.array(bin_means).reshape(1, -1)
        
        im = ax.imshow(heatmap_data, aspect='auto', cmap=self.heatmap_cmaps.get(self.heatmap_colormap, 'viridis'),
                      extent=[0, chrom_length, 0, 1], interpolation='nearest', zorder=2)
        
        ax.set_yticks([])
        ax.set_ylim(0, 1)
        ax.set_xlim(0, chrom_length)
        
        if ax.get_subplotspec().rowspan.stop == (10 if not self.optimized_centromeres else 15):
            cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.1, shrink=0.8)
            cbar.set_label('Value Intensity', fontsize=9, fontweight='bold')
            cbar.ax.tick_params(labelsize=8)

    def _create_chromosome_header(self, ax, chrom, chrom_data):
        """Create chromosome header with summary information"""
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        chrom_length = chrom_data['length'] / 1e6
        chrom_info = [
            f"Chromosome: {chrom}",
            f"Length: {chrom_length:.2f} Mb",
            f"Analysis Window: 100 kb"
        ]
        
        if chrom in self.centromere_data:
            primary_centromeres = [c for c in self.centromere_data[chrom] 
                                 if c.get('is_primary', False)]
            candidate_centromeres = [c for c in self.centromere_data[chrom] 
                                   if not c.get('is_primary', False)]
            
            if primary_centromeres:
                centromere = primary_centromeres[0]
                centromere_type = centromere.get('type', 'unknown')
                
                if self.known_centromeres:
                    chrom_info.extend([
                        f"Primary Centromere: {centromere['start_mb']:.2f}-{centromere['end_mb']:.2f} Mb",
                        f"Centromere Size: {centromere['length']/1e6:.2f} Mb",
                        f"Position: {centromere['center_mb']:.2f} Mb",
                        f"Type: {centromere_type.capitalize()}"
                    ])
                else:
                    chrom_info.extend([
                        f"Detected Centromere: {centromere['start_mb']:.2f}-{centromere['end_mb']:.2f} Mb",
                        f"Centromere Size: {centromere['length']/1e6:.2f} Mb",
                        f"Position: {centromere['center_mb']:.2f} Mb"
                    ])
                
                if 'avg_kmer_density' in centromere:
                    chrom_info.append(f"Avg K-mer Density: {centromere['avg_kmer_density']:.2f}")
            
            if candidate_centromeres:
                chrom_info.append(f"Candidate Regions: {len(candidate_centromeres)}")
        
        if chrom in self.optimized_centromeres:
            optimized = self.optimized_centromeres[chrom][0]
            
            if optimized['length'] > 0:
                size_change = ((optimized['optimized_length'] / optimized['length']) - 1) * 100
                size_change_text = f"Size change: {size_change:+.1f}%"
            else:
                size_change_text = "Size change: N/A (original size 0)"
            
            chrom_info.extend([
                "",
                "BOUNDARY OPTIMIZATION:",
                f"Known: {optimized['start_mb']:.2f}-{optimized['end_mb']:.2f} Mb",
                f"Optimized: {optimized['optimized_start_mb']:.2f}-{optimized['optimized_end_mb']:.2f} Mb",
                size_change_text
            ])
            
            if 'dynamic_extension_info' in optimized:
                dyn_info = optimized['dynamic_extension_info']
                chrom_info.extend([
                    f"Dynamic Extension: {dyn_info['final_extension']/1000:.1f} kb",
                    f"Iterations: {dyn_info['iterations']}",
                    f"Final mean diff: {dyn_info.get('best_mean_diff', 0):.3f}"
                ])
        
        if chrom in self.compare_centromeres:
            chrom_info.extend([
                "",
                "PUBLISHED COMPARISON:"
            ])
            for i, compare_cent in enumerate(self.compare_centromeres[chrom]):
                chrom_info.append(f"  Published {i+1}: {compare_cent['start_mb']:.2f}-{compare_cent['end_mb']:.2f} Mb")
        
        bbox = FancyBboxPatch((0.02, 0.05), 0.96, 0.9,
                             boxstyle="round,pad=0.1",
                             facecolor=self.colors['background'],
                             edgecolor='gray',
                             linewidth=2,
                             alpha=0.9)
        ax.add_patch(bbox)
        
        ax.text(0.05, 0.8, '\n'.join(chrom_info), 
               transform=ax.transAxes, va='top', ha='left',
               fontfamily='monospace', fontsize=10,
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
        
        title_text = f'EasyCen - Chromosome {chrom} Genomic Feature Analysis'
        if self.optimized_centromeres:
            title_text += ' (with Boundary Optimization)'
            if self.optimizer.dynamic_extension:
                title_text += ' [Dynamic Extension]'
        elif not self.known_centromeres:
            title_text += ' (Analysis Results)'
        ax.text(0.5, 0.95, title_text,
               transform=ax.transAxes, ha='center', va='center',
               fontsize=16, fontweight='bold')

    def create_summary_statistics(self, output_file=None):
        """Create statistical summary of centromere analysis"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if not output_file:
            output_file = self.output_dir / "centromere_statistics.pdf"
        else:
            output_file = Path(output_file)
        
        stats_data = []
        for chrom in self._natural_sort(self.chrom_data.keys()):
            if chrom in self.centromere_data:
                primary_centromeres = [c for c in self.centromere_data[chrom] 
                                     if c.get('is_primary', False)]
                for centromere in primary_centromeres:
                    stat_entry = {
                        'Chromosome': chrom,
                        'Type': centromere.get('type', 'unknown'),
                        'Length_Mb': self.chrom_data[chrom]['length'] / 1e6,
                        'Centromere_Start_Mb': centromere['start_mb'],
                        'Centromere_End_Mb': centromere['end_mb'],
                        'Centromere_Size_Mb': centromere['length'] / 1e6,
                        'Centromere_Position_Mb': centromere['center_mb'],
                        'Relative_Position': centromere['center_mb'] / (self.chrom_data[chrom]['length'] / 1e6)
                    }
                    
                    if chrom in self.optimized_centromeres:
                        optimized = self.optimized_centromeres[chrom][0]
                        
                        if optimized['length'] > 0:
                            size_change_percent = ((optimized['optimized_length'] / centromere['length']) - 1) * 100
                        else:
                            size_change_percent = 0
                        
                        stat_entry.update({
                            'Optimized_Start_Mb': optimized['optimized_start_mb'],
                            'Optimized_End_Mb': optimized['optimized_end_mb'],
                            'Optimized_Size_Mb': optimized['optimized_length'] / 1e6,
                            'Size_Change_Percent': size_change_percent,
                            'Optimization_Method': optimized.get('optimization_method', 'unknown'),
                            'Sampling_Normality_P_Value': optimized.get('threshold_info', {}).get('sampling_distribution_stats', {}).get('normality_p_value', 1.0),
                            'Sampling_Is_Normal': optimized.get('threshold_info', {}).get('sampling_distribution_stats', {}).get('is_normal', True)
                        })
                        
                        if 'dynamic_extension_info' in optimized:
                            dyn_info = optimized['dynamic_extension_info']
                            stat_entry.update({
                                'Dynamic_Extension_Enabled': self.optimizer.dynamic_extension,
                                'Final_Extension_kb': dyn_info['final_extension'] / 1000,
                                'Extension_Iterations': dyn_info['iterations'],
                                'Best_Mean_Diff': dyn_info.get('best_mean_diff', 0)
                            })
                    
                    if chrom in self.compare_centromeres:
                        for i, compare_cent in enumerate(self.compare_centromeres[chrom]):
                            stat_entry[f'Published_{i+1}_Start_Mb'] = compare_cent['start_mb']
                            stat_entry[f'Published_{i+1}_End_Mb'] = compare_cent['end_mb']
                            stat_entry[f'Published_{i+1}_Size_Mb'] = compare_cent['length'] / 1e6
                    
                    stats_data.append(stat_entry)
        
        if not stats_data:
            print("No centromere data available for statistical analysis")
            return
        
        df = pd.DataFrame(stats_data)
        
        # Determine subplot layout based on available data
        n_plots = 2  # Base plots: size distribution and position vs size
        
        if self.optimized_centromeres:
            n_plots += 2  # Add size change and original vs optimized
        
        if self.compare_centromeres:
            n_plots += 1  # Add comparison plot
        
        if self.optimizer.dynamic_extension and 'Final_Extension_kb' in df.columns:
            n_plots += 1  # Add dynamic extension plot
        
        n_cols = 3
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows), constrained_layout=True)
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.reshape(-1)
        else:
            axes = axes.flatten()
            
        fig.suptitle('EasyCen Centromere Statistical Analysis' + 
                    (' (with Boundary Optimization)' if self.optimized_centromeres else 
                     ' (Analysis Results)' if not self.known_centromeres else '') +
                    (' [Dynamic Extension]' if self.optimizer.dynamic_extension else ''),
                    fontsize=18, fontweight='bold', y=0.95)
        
        plot_idx = 0
        
        # Plot 1: Centromere Size Distribution
        if len(df) > 0:
            if self.known_centromeres:
                colors = [self.colors.get(ctype, self.colors['unknown']) for ctype in df['Type']]
            else:
                colors = [self.colors['analysis_primary'] for _ in range(len(df))]
                
            axes[plot_idx].scatter(df['Centromere_Size_Mb'], range(len(df)), 
                             s=60, alpha=0.7, c=colors)
            axes[plot_idx].set_xlabel('Centromere Size (Mb)', fontweight='bold')
            axes[plot_idx].set_ylabel('Chromosome Index', fontweight='bold')
            axes[plot_idx].set_title('Centromere Size Distribution', fontweight='bold', fontsize=14)
            axes[plot_idx].grid(True, alpha=0.3)
            plot_idx += 1
        
        # Plot 2: Centromere Position vs Chromosome Size
        if len(df) > 0:
            if self.known_centromeres:
                colors = [self.colors.get(ctype, self.colors['unknown']) for ctype in df['Type']]
            else:
                colors = [self.colors['analysis_primary'] for _ in range(len(df))]
                
            axes[plot_idx].scatter(df['Length_Mb'], df['Relative_Position'], 
                             s=60, alpha=0.7, c=colors)
            axes[plot_idx].set_xlabel('Chromosome Length (Mb)', fontweight='bold')
            axes[plot_idx].set_ylabel('Relative Centromere Position', fontweight='bold')
            axes[plot_idx].set_title('Centromere Position vs Chromosome Size', fontweight='bold', fontsize=14)
            axes[plot_idx].grid(True, alpha=0.3)
            plot_idx += 1
        
        # Plot 3: Chromosome Size vs Centromere Size
        if len(df) > 0:
            if self.known_centromeres:
                colors = [self.colors.get(ctype, self.colors['unknown']) for ctype in df['Type']]
            else:
                colors = [self.colors['analysis_primary'] for _ in range(len(df))]
                
            axes[plot_idx].scatter(df['Length_Mb'], df['Centromere_Size_Mb'], 
                             s=60, alpha=0.7, c=colors)
            axes[plot_idx].set_xlabel('Chromosome Length (Mb)', fontweight='bold')
            axes[plot_idx].set_ylabel('Centromere Size (Mb)', fontweight='bold')
            axes[plot_idx].set_title('Chromosome Size vs Centromere Size', fontweight='bold', fontsize=14)
            axes[plot_idx].grid(True, alpha=0.3)
            plot_idx += 1
        
        # Additional plots for optimized centromeres
        if self.optimized_centromeres and 'Optimized_Size_Mb' in df.columns:
            # Size Change Histogram
            axes[plot_idx].hist(df['Size_Change_Percent'], bins=15, alpha=0.7, 
                          color=self.colors['optimized_centromere'], edgecolor='black')
            axes[plot_idx].set_xlabel('Size Change (%)', fontweight='bold')
            axes[plot_idx].set_ylabel('Frequency', fontweight='bold')
            axes[plot_idx].set_title('Centromere Size Change After Optimization', fontweight='bold', fontsize=14)
            axes[plot_idx].grid(True, alpha=0.3)
            axes[plot_idx].axvline(0, color='red', linestyle='--', alpha=0.7)
            plot_idx += 1
            
            # Original vs Optimized Sizes
            x = range(len(df))
            width = 0.35
            axes[plot_idx].bar([i - width/2 for i in x], df['Centromere_Size_Mb'], width, 
                         label='Original', alpha=0.7, color=self.colors['known_centromere'])
            axes[plot_idx].bar([i + width/2 for i in x], df['Optimized_Size_Mb'], width, 
                         label='Optimized', alpha=0.7, color=self.colors['optimized_centromere'])
            axes[plot_idx].set_xlabel('Chromosome Index', fontweight='bold')
            axes[plot_idx].set_ylabel('Centromere Size (Mb)', fontweight='bold')
            axes[plot_idx].set_title('Original vs Optimized Centromere Sizes', fontweight='bold', fontsize=14)
            axes[plot_idx].legend()
            axes[plot_idx].grid(True, alpha=0.3)
            plot_idx += 1
        
        # Dynamic Extension Visualization
        if self.optimizer.dynamic_extension and 'Final_Extension_kb' in df.columns:
            axes[plot_idx].scatter(df['Final_Extension_kb'], df['Size_Change_Percent'] if 'Size_Change_Percent' in df.columns else df['Centromere_Size_Mb'],
                             s=60, alpha=0.7, c=self.colors['dynamic_extension'])
            axes[plot_idx].set_xlabel('Final Extension (kb)', fontweight='bold')
            if 'Size_Change_Percent' in df.columns:
                axes[plot_idx].set_ylabel('Size Change (%)', fontweight='bold')
                axes[plot_idx].set_title('Dynamic Extension vs Size Change', fontweight='bold', fontsize=14)
            else:
                axes[plot_idx].set_ylabel('Centromere Size (Mb)', fontweight='bold')
                axes[plot_idx].set_title('Dynamic Extension vs Centromere Size', fontweight='bold', fontsize=14)
            axes[plot_idx].grid(True, alpha=0.3)
            plot_idx += 1
        
        # Comparison with published centromeres
        if self.compare_centromeres:
            published_cols = [col for col in df.columns if col.startswith('Published_') and col.endswith('_Size_Mb')]
            if published_cols:
                published_sizes = []
                for col in published_cols:
                    published_sizes.extend(df[col].dropna().tolist())
                
                if published_sizes:
                    axes[plot_idx].hist([df['Centromere_Size_Mb'], published_sizes], 
                                  bins=15, alpha=0.7, 
                                  label=['EasyCen', 'Published'],
                                  color=[self.colors['analysis_primary'], self.colors['compare_centromere']],
                                  edgecolor='black')
                    axes[plot_idx].set_xlabel('Centromere Size (Mb)', fontweight='bold')
                    axes[plot_idx].set_ylabel('Frequency', fontweight='bold')
                    axes[plot_idx].set_title('Size Distribution: EasyCen vs Published', fontweight='bold', fontsize=14)
                    axes[plot_idx].legend()
                    axes[plot_idx].grid(True, alpha=0.3)
                    plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)
        
        # Statistics text
        stats_text = [
            f"Total Chromosomes: {len(df)}",
            f"Mean Centromere Size: {df['Centromere_Size_Mb'].mean():.2f} ± {df['Centromere_Size_Mb'].std():.2f} Mb",
            f"Size Range: {df['Centromere_Size_Mb'].min():.2f}-{df['Centromere_Size_Mb'].max():.2f} Mb"
        ]
        
        if self.optimized_centromeres and 'Optimized_Size_Mb' in df.columns:
            stats_text.extend([
                f"Mean Optimized Size: {df['Optimized_Size_Mb'].mean():.2f} ± {df['Optimized_Size_Mb'].std():.2f} Mb",
                f"Mean Size Change: {df['Size_Change_Percent'].mean():.1f}% ± {df['Size_Change_Percent'].std():.1f}%"
            ])
            
            if self.optimizer.dynamic_extension and 'Final_Extension_kb' in df.columns:
                stats_text.extend([
                    f"Mean Final Extension: {df['Final_Extension_kb'].mean():.1f} ± {df['Final_Extension_kb'].std():.1f} kb",
                    f"Mean Iterations: {df['Extension_Iterations'].mean():.1f} ± {df['Extension_Iterations'].std():.1f}",
                    f"Mean Best Mean Diff: {df['Best_Mean_Diff'].mean():.3f} ± {df['Best_Mean_Diff'].std():.3f}"
                ])
        
        if self.compare_centromeres:
            published_cols = [col for col in df.columns if col.startswith('Published_') and col.endswith('_Size_Mb')]
            if published_cols:
                published_sizes = []
                for col in published_cols:
                    published_sizes.extend(df[col].dropna().tolist())
                
                if published_sizes:
                    stats_text.extend([
                        f"Published Centromeres: {len(published_sizes)}",
                        f"Mean Published Size: {np.mean(published_sizes):.2f} ± {np.std(published_sizes):.2f} Mb"
                    ])
        
        if self.known_centromeres:
            type_stats = df.groupby('Type')['Centromere_Size_Mb'].agg(['count', 'mean', 'std']).round(2)
            stats_text.append(f"Centromere Types:")
            for ctype, row in type_stats.iterrows():
                stats_text.append(f"  {ctype.capitalize()}: {row['count']} chromosomes, mean size: {row['mean']} ± {row['std']} Mb")
        
        fig.text(0.02, 0.02, '\n'.join(stats_text), 
                fontfamily='monospace', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.8", facecolor=self.colors['background'],
                        edgecolor='gray', linewidth=1.5))
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        csv_file = self.output_dir / "centromere_statistics.csv"
        df.to_csv(csv_file, index=False)
        
        print(f"Statistical summary saved to: {output_file}")
        print(f"Data table saved to: {csv_file}")

def visualize_results(results_dir, known_centromeres_file=None, output_dir=None,
                     kmer_weight=0.4, feature_weight=0.6, optimization_extension=100000,  # Changed default to 100k
                     genome_overview=True, chromosome_details=True, statistics=True,
                     heatmap_colormap='viridis', heatmap_height=0.3,
                     smoothing_sigma=2.0, distribution_threshold=0.05,
                     random_sampling_times=100000, sample_size=None, compare_centromeres_file=None,
                     peak_prominence=0.1, min_region_size=50000,
                     dynamic_extension=True, target_mean=0.5, mean_tolerance=0.01,  # New parameters
                     max_extension_factor=5.0, extension_increment=50000):
    """
    Main visualization function for EasyCen results
    
    Args:
        results_dir: Directory containing analysis results
        known_centromeres_file: BED file with known centromere regions
        output_dir: Output directory for visualization results
        kmer_weight: Weight for k-mer density in optimization
        feature_weight: Weight for feature percentage in optimization
        optimization_extension: Extension around centromeres for optimization (default: 100k)
        genome_overview: Generate genome overview plot
        chromosome_details: Generate detailed chromosome plots
        statistics: Generate statistical summary
        heatmap_colormap: Color map for heatmap tracks
        heatmap_height: Relative height of heatmap compared to line plot
        smoothing_sigma: Sigma parameter for Gaussian smoothing
        distribution_threshold: Base threshold for boundary detection
        random_sampling_times: Number of random samplings for distribution fitting
        sample_size: Sample size for each random sampling
        compare_centromeres_file: BED file with published centromere regions for comparison
        peak_prominence: Minimum prominence for peak detection
        min_region_size: Minimum size for candidate regions in bp
        dynamic_extension: Enable dynamic extension to achieve target sampling mean
        target_mean: Target sampling mean for dynamic extension (default: 0.5)
        mean_tolerance: Tolerance for target mean (default: 0.1)
        max_extension_factor: Maximum extension as multiple of initial extension
        extension_increment: Increment for extension adjustment in bp
    """
    
    if output_dir is None:
        output_dir = Path(results_dir) / "visualization"
    else:
        output_dir = Path(output_dir)
    
    print("=" * 60)
    print("EASYCEN VISUALIZATION")
    print("=" * 60)
    print(f"Results directory: {results_dir}")
    print(f"Known centromeres: {known_centromeres_file}")
    print(f"Output directory: {output_dir}")
    print(f"Optimization weights: kmer={kmer_weight}, feature={feature_weight}")
    print(f"Initial extension: {optimization_extension} bp")
    print(f"Dynamic extension: {dynamic_extension}")
    if dynamic_extension:
        print(f"Target mean: {target_mean} ± {mean_tolerance}")
        print(f"Max extension factor: {max_extension_factor}")
        print(f"Extension increment: {extension_increment} bp")
    print(f"Smoothing sigma: {smoothing_sigma}")
    print(f"Distribution threshold: {distribution_threshold}")
    print(f"Random sampling times: {random_sampling_times}")
    print(f"Peak prominence: {peak_prominence}")
    print(f"Min region size: {min_region_size}")
    print("=" * 60)
    
    visualizer = CentromereVisualizer(
        results_dir=results_dir,
        known_centromeres_file=known_centromeres_file,
        kmer_weight=kmer_weight,
        feature_weight=feature_weight,
        optimization_extension=optimization_extension,
        heatmap_colormap=heatmap_colormap,
        heatmap_height=heatmap_height,
        output_dir=output_dir,
        smoothing_sigma=smoothing_sigma,
        distribution_threshold=distribution_threshold,
        random_sampling_times=random_sampling_times,
        sample_size=sample_size,
        compare_centromeres_file=compare_centromeres_file,
        peak_prominence=peak_prominence,
        min_region_size=min_region_size,
        dynamic_extension=dynamic_extension,
        target_mean=target_mean,
        mean_tolerance=mean_tolerance,
        max_extension_factor=max_extension_factor,
        extension_increment=extension_increment
    )
    
    visualizer.load_data()
    
    if genome_overview:
        print("Creating genome overview...")
        visualizer.create_genome_overview()
    
    if chromosome_details:
        print("Creating chromosome detail plots...")
        visualizer.create_chromosome_detail_plots()
    
    if statistics:
        print("Creating statistical summary...")
        visualizer.create_summary_statistics()
    
    print("\n" + "=" * 60)
    print("EASYCEN VISUALIZATION COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {output_dir}")
    
    if not known_centromeres_file:
        analysis_bed = output_dir / "analysis_centromeres.bed"
        if analysis_bed.exists():
            print(f"Analysis results: {analysis_bed}")
    elif visualizer.optimized_centromeres:
        weight_suffix = f"_kmer{kmer_weight}_feature{feature_weight}"
        sampling_suffix = f"_samples{random_sampling_times}"
        dynamic_suffix = "_dynamic" if dynamic_extension else "_fixed"
        optimized_bed = output_dir / f"optimized_centromeres{weight_suffix}{sampling_suffix}{dynamic_suffix}.bed"
        if optimized_bed.exists():
            print(f"Optimized centromeres: {optimized_bed}")
    
    print(f"Total chromosomes processed: {len(visualizer.chrom_data)}")
    
    if visualizer.centromere_data:
        primary_count = sum(1 for chrom_data in visualizer.centromere_data.values() 
                          for region in chrom_data if region.get('is_primary', False))
        candidate_count = sum(1 for chrom_data in visualizer.centromere_data.values() 
                            for region in chrom_data if not region.get('is_primary', False))
        print(f"Primary centromeres: {primary_count}")
        print(f"Candidate regions: {candidate_count}")
    
    if visualizer.optimized_centromeres:
        optimized_count = sum(len(regions) for regions in visualizer.optimized_centromeres.values())
        print(f"Optimized centromeres: {optimized_count}")
        
        if dynamic_extension:
            # Calculate average extension statistics
            total_extensions = []
            total_iterations = []
            best_mean_diffs = []
            for chrom in visualizer.optimized_centromeres:
                for optimized in visualizer.optimized_centromeres[chrom]:
                    if 'dynamic_extension_info' in optimized:
                        ext_info = optimized['dynamic_extension_info']
                        total_extensions.append(ext_info['final_extension'])
                        total_iterations.append(ext_info['iterations'])
                        best_mean_diffs.append(ext_info.get('best_mean_diff', 0))
            
            if total_extensions:
                avg_extension = np.mean(total_extensions) / 1000  # Convert to kb
                avg_iterations = np.mean(total_iterations)
                avg_mean_diff = np.mean(best_mean_diffs)
                print(f"Average final extension: {avg_extension:.1f} kb")
                print(f"Average iterations: {avg_iterations:.1f}")
                print(f"Average best mean difference: {avg_mean_diff:.3f}")
                print(f"Success rate: {sum(1 for d in best_mean_diffs if d <= mean_tolerance)}/{len(best_mean_diffs)} chromosomes within target range")

def main():
    parser = argparse.ArgumentParser(
        description="EasyCen Visualization - Centromere visualization with boundary optimization and statistical analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--results-dir", required=True,
                       help="Directory containing EasyCen analysis results")
    parser.add_argument("--known-centromeres", 
                       help="BED file with known centromere regions")
    parser.add_argument("--compare", 
                       help="BED file with published centromere regions for comparison")
    parser.add_argument("--kmer-weight", type=float, default=0.4,
                       help="Weight for kmer density in optimization (default: 0.4)")
    parser.add_argument("--feature-weight", type=float, default=0.6,
                       help="Weight for feature percentage in optimization (default: 0.6)")
    parser.add_argument("--optimization-extension", type=int, default=100000,
                       help="Initial extension around known centromere for optimization search in bp (default: 100000)")
    parser.add_argument("--smoothing-sigma", type=float, default=2.0,
                       help="Sigma parameter for Gaussian smoothing of composite scores (default: 2.0)")
    parser.add_argument("--distribution-threshold", type=float, default=0.05,
                       help="Base threshold for boundary detection (default: 0.05)")
    parser.add_argument("--random-sampling-times", type=int, default=100000,
                       help="Number of random samplings for distribution fitting (default: 100000)")
    parser.add_argument("--sample-size", type=int, default=None,
                       help="Sample size for each random sampling (default: min(30, data_size//10))")
    parser.add_argument("--peak-prominence", type=float, default=0.1,
                       help="Minimum prominence for peak detection (default: 0.1)")
    parser.add_argument("--min-region-size", type=int, default=50000,
                       help="Minimum size for candidate regions in bp (default: 50000)")
    
    # New arguments for dynamic extension
    parser.add_argument("--dynamic-extension", action="store_true", default=True,
                       help="Enable dynamic extension to achieve target sampling mean (default: True)")
    parser.add_argument("--no-dynamic-extension", action="store_false", dest="dynamic_extension",
                       help="Disable dynamic extension")
    parser.add_argument("--target-mean", type=float, default=0.5,
                       help="Target sampling mean for dynamic extension (default: 0.5)")
    parser.add_argument("--mean-tolerance", type=float, default=0.01,
                       help="Tolerance for target mean in dynamic extension (default: 0.01)")
    parser.add_argument("--max-extension-factor", type=float, default=5.0,
                       help="Maximum extension as multiple of initial extension (default: 5.0)")
    parser.add_argument("--extension-increment", type=int, default=50000,
                       help="Increment for extension adjustment in bp (default: 50000)")
    
    parser.add_argument("--output-dir", 
                       help="Output directory for plots and BED files (default: results_dir/visualization)")
    parser.add_argument("--genome-overview", action="store_true", default=True,
                       help="Generate genome overview plot (default: True)")
    parser.add_argument("--chromosome-details", action="store_true", default=True,
                       help="Generate individual chromosome plots (default: True)")
    parser.add_argument("--statistics", action="store_true", default=True,
                       help="Generate statistical summary (default: True)")
    parser.add_argument("--heatmap-colormap", default="viridis",
                       choices=['viridis', 'plasma', 'inferno', 'magma', 'coolwarm', 'RdYlBu_r', 'Spectral_r'],
                       help="Color map for heatmap tracks (default: viridis)")
    parser.add_argument("--heatmap-height", type=float, default=0.3,
                       help="Relative height of heatmap compared to line plot (default: 0.3)")
    
    args = parser.parse_args()

    visualize_results(
        results_dir=args.results_dir,
        known_centromeres_file=args.known_centromeres,
        output_dir=args.output_dir,
        kmer_weight=args.kmer_weight,
        feature_weight=args.feature_weight,
        optimization_extension=args.optimization_extension,
        genome_overview=args.genome_overview,
        chromosome_details=args.chromosome_details,
        statistics=args.statistics,
        heatmap_colormap=args.heatmap_colormap,
        heatmap_height=args.heatmap_height,
        smoothing_sigma=args.smoothing_sigma,
        distribution_threshold=args.distribution_threshold,
        random_sampling_times=args.random_sampling_times,
        sample_size=args.sample_size,
        compare_centromeres_file=args.compare,
        peak_prominence=args.peak_prominence,
        min_region_size=args.min_region_size,
        dynamic_extension=args.dynamic_extension,
        target_mean=args.target_mean,
        mean_tolerance=args.mean_tolerance,
        max_extension_factor=args.max_extension_factor,
        extension_increment=args.extension_increment
    )

if __name__ == "__main__":
    main()
