#!/usr/bin/env python3
"""
EasyCen Visualization Module v1.0
Author: Yunyun Lv
Email: lvyunyun_sci@foxmail.com
"""

import argparse, os, glob, re, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, FancyBboxPatch, Patch as mpatches
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm, normaltest
from scipy.signal import find_peaks
warnings.filterwarnings('ignore')

# Publication-quality plot defaults
plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 10,
    'axes.labelsize': 11, 'axes.titlesize': 12,
    'legend.fontsize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'figure.figsize': (8, 10), 'axes.linewidth': 0.8,
    'grid.linewidth': 0.4, 'lines.linewidth': 1.2, 'patch.linewidth': 0.8,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
})


class CentromereOptimizer:
    def __init__(self, kmer_weight=0.7, feature_weight=0.4, extension_bp=100000,
                 smoothing_sigma=2.0, distribution_threshold=0.05,
                 random_sampling_times=100000, sample_size=None,
                 peak_prominence=0.1, min_region_size=50000,
                 dynamic_extension=True, target_mean=0.5, mean_tolerance=0.01,
                 max_extension_factor=5.0, extension_increment=50000):
        self.kmer_weight = kmer_weight
        self.feature_weight = feature_weight
        self.extension_bp = extension_bp
        self.smoothing_sigma = smoothing_sigma
        self.distribution_threshold = distribution_threshold
        self.random_sampling_times = random_sampling_times
        self.sample_size = sample_size
        self.peak_prominence = peak_prominence
        self.min_region_size = min_region_size
        self.dynamic_extension = dynamic_extension
        self.target_mean = target_mean
        self.mean_tolerance = mean_tolerance
        self.max_extension_factor = max_extension_factor
        self.extension_increment = extension_increment

    def optimize_boundaries(self, known_centromere, kmer_data, feature_data, chrom_length):
        start, end = known_centromere['start'], known_centromere['end']
        if self.dynamic_extension:
            ext_res = self._apply_dynamic_extension(start, end, kmer_data, feature_data, chrom_length)
            search_start, search_end = ext_res['search_start'], ext_res['search_end']
            final_extension, iterations = ext_res['final_extension'], ext_res['iterations']
            mean_history = ext_res['mean_history']
        else:
            search_start = max(0, start - self.extension_bp)
            search_end = min(chrom_length, end + self.extension_bp)
            final_extension, iterations = self.extension_bp, 1
            mean_history = []

        pos, kval, fval = [], [], []
        for kp in kmer_data:
            if search_start <= kp['start'] <= search_end:
                pos.append(kp['start'])
                kval.append(kp['value'])
                fv = 0.0
                for fp in feature_data:
                    if fp['start'] == kp['start']:
                        fv = fp['value']
                        break
                fval.append(fv)
        if not pos:
            return known_centromere

        pos = np.array(pos); kval = np.array(kval); fval = np.array(fval)
        knorm = self._normalize(kval)
        fnorm = 1 - self._normalize(fval)
        comp = self.kmer_weight * knorm + self.feature_weight * fnorm

        sigma = self._adaptive_sigma(comp)
        smooth = gaussian_filter1d(comp, sigma)
        if len(smooth) > 1:
            dist = max(1, len(smooth)//20)
            peaks, pprop = find_peaks(smooth, prominence=self.peak_prominence, distance=dist)
        else:
            peaks, pprop = np.array([]), {}

        opt_res = self._multithreshold(pos, comp, smooth, peaks, start, end, chrom_length, sigma)
        left, right = opt_res['left_boundary'], opt_res['right_boundary']
        tinfo = opt_res['threshold_info']
        tinfo['dynamic_extension'] = {
            'enabled': self.dynamic_extension, 'final_extension': final_extension,
            'iterations': iterations, 'target_mean': self.target_mean,
            'mean_tolerance': self.mean_tolerance, 'mean_history': mean_history,
            'initial_extension': self.extension_bp
        }

        overlap_len = max(0, min(right, end) - max(left, start))
        if overlap_len == 0:
            left = min(start, left)
            right = max(end, right)
            tinfo['union_taken'] = True
            tinfo['overlap_fallback'] = False
        else:
            tinfo['union_taken'] = False
            tinfo['overlap_fallback'] = False

        if right - left < self.min_region_size:
            ctr = (left + right)//2
            left = max(0, ctr - self.min_region_size//2)
            right = min(chrom_length, ctr + self.min_region_size//2)

        opt = known_centromere.copy()
        opt.update({
            'optimized_start': left, 'optimized_end': right,
            'optimized_length': right - left,
            'optimized_start_mb': left/1e6, 'optimized_end_mb': right/1e6,
            'optimized_center_mb': (left+right)/2/1e6,
            'composite_scores': comp.tolist(), 'composite_smoothed': smooth.tolist(),
            'search_positions': pos.tolist(), 'peaks': peaks.tolist() if len(peaks) else [],
            'peak_properties': pprop, 'was_optimized': True,
            'optimization_method': 'adaptive_sigma_multi_threshold_dynamic_extension',
            'threshold_info': tinfo, 'adaptive_sigma': sigma,
            'dynamic_extension_info': {'final_extension': final_extension, 'iterations': iterations,
                                       'mean_history': mean_history}
        })
        return opt

    def _apply_dynamic_extension(self, start, end, kmer_data, feature_data, chrom_length):
        cur_ext = self.extension_bp
        max_ext = min(chrom_length//2, int(self.extension_bp * self.max_extension_factor))
        iters = 0; max_iters = 1000; history = []
        best_ext, best_diff = cur_ext, float('inf')
        while iters < max_iters:
            ss = max(0, start - cur_ext); se = min(chrom_length, end + cur_ext)
            p, kv, fv = self._extract_region(ss, se, kmer_data, feature_data)
            if not p or len(kv)<3:
                cur_ext += self.extension_increment; iters += 1; continue
            kv, fv = np.array(kv), np.array(fv)
            kn = self._normalize(kv); fn = 1 - self._normalize(fv)
            comp = self.kmer_weight * kn + self.feature_weight * fn
            reg_mean = np.mean(comp); diff = abs(reg_mean - self.target_mean)
            history.append({'iteration': iters, 'extension': cur_ext, 'mean': reg_mean,
                            'mean_diff': diff, 'search_start': ss, 'search_end': se,
                            'n_points': len(comp)})
            if diff < best_diff: best_diff, best_ext = diff, cur_ext
            if diff <= self.mean_tolerance: break
            cur_ext += self.extension_increment if reg_mean > self.target_mean else -self.extension_increment
            if cur_ext > max_ext: cur_ext = best_ext if best_diff < float('inf') else max_ext; break
            if cur_ext < self.extension_bp: cur_ext = self.extension_bp; break
            iters += 1
        if iters >= max_iters: cur_ext = best_ext
        return {'search_start': max(0, start - cur_ext), 'search_end': min(chrom_length, end + cur_ext),
                'final_extension': cur_ext, 'iterations': iters, 'mean_history': history,
                'best_mean_diff': best_diff}

    def _extract_region(self, ss, se, kd, fd):
        pos, kv, fv = [], [], []
        for kp in kd:
            if ss <= kp['start'] <= se:
                pos.append(kp['start']); kv.append(kp['value'])
                f = 0
                for fp in fd:
                    if fp['start'] == kp['start']: f = fp['value']; break
                fv.append(f)
        return pos, kv, fv

    def _normalize(self, x):
        if len(x)==0: return np.array([])
        mn, mx = np.min(x), np.max(x)
        return np.zeros_like(x) if mx==mn else (x-mn)/(mx-mn)

    def _adaptive_sigma(self, comp):
        init = 2.0
        try:
            sm = gaussian_filter1d(comp, init)
            if len(sm)>1:
                dst = max(1, len(sm)//10)
                pks, _ = find_peaks(sm, prominence=self.peak_prominence, distance=dst)
                n = len(pks)
            else: n = 0
        except: n = 0
        if n == 0: return init
        t = max(0, (n-1)/9.0)
        return 2 + 13 * (t**0.3)

    def _multithreshold(self, pos, scores, smooth, peaks, ostart, oend, clen, sigma):
        if len(smooth)<10:
            return {'left_boundary': ostart, 'right_boundary': oend,
                    'threshold_info': {'error':'insufficient'}}
        if len(peaks)>1:
            mp = self._multi_peak(pos, smooth, peaks, ostart, oend)
            if mp: return mp
        ss = min(100, len(scores)) if self.sample_size is None else min(self.sample_size, len(scores))
        if ss<5:
            return {'left_boundary': ostart, 'right_boundary': oend,
                    'threshold_info': {'error':'small_sample'}}
        means = [np.mean(np.random.choice(scores, size=ss, replace=True)) for _ in range(self.random_sampling_times)]
        means = np.array(means)
        mu, std = np.mean(means), np.std(means) if np.std(means) else 0.1
        try:
            if len(means)>3: _, pv = normaltest(means); is_norm = pv>0.05
            else: is_norm = True; pv = 1.0
        except: is_norm = True; pv = 1.0
        z = norm.ppf(1 - self.distribution_threshold/2)
        thresh = [mu - z*1*std, mu - z*0.95*std, mu - z*0.5*std, mu - 0.01*std]
        names = ['very_aggressive(select)','aggressive','standard','conservative']
        all_bnd = []
        for i,th in enumerate(thresh):
            bds = self._regions_at_threshold(pos, smooth, th, peaks)
            if bds:
                all_bnd.append({'threshold_level': names[i], 'threshold_value': th,
                                'boundaries': bds})
        sel = self._select_best(all_bnd, ostart, oend, clen)
        info = {'sample_size': ss, 'sampling_times': self.random_sampling_times,
                'sampling_mean': mu, 'sampling_std': std,
                'threshold_levels': dict(zip(names, thresh)),
                'all_boundaries': all_bnd,
                'selected_threshold': sel.get('threshold_level','standard'),
                'adaptive_sigma': sigma, 'peak_count': len(peaks),
                'sampling_distribution_stats': {
                    'min': float(np.min(means)), 'max': float(np.max(means)),
                    'median': float(np.median(means)),
                    'skewness': float(pd.Series(means).skew()) if len(means)>2 else 0,
                    'normality_p_value': pv, 'is_normal': is_norm,
                    'sample_means': means.tolist()}}
        return {'left_boundary': sel['left_boundary'], 'right_boundary': sel['right_boundary'],
                'threshold_info': info}

    def _multi_peak(self, pos, smooth, peaks, ostart, oend):
        try:
            pp = pos[peaks]; si = np.argsort(pp)
            leftmost, rightmost = pp[si[0]], pp[si[-1]]
            lb = self._boundary_near_peak(pos, smooth, leftmost, 'left')
            rb = self._boundary_near_peak(pos, smooth, rightmost, 'right')
            if lb>=rb: return None
            sz = rb-lb
            if sz<self.min_region_size:
                ctr = (lb+rb)//2; lb = max(0, ctr-self.min_region_size//2)
                rb = lb+self.min_region_size
            return {'left_boundary': lb, 'right_boundary': rb,
                    'threshold_info': {'multi_peak': True, 'peak_count': len(peaks)}}
        except: return None

    def _boundary_near_peak(self, pos, smooth, pk, direction):
        try: idx = np.where(pos==pk)[0][0]
        except: return pk
        grad = np.gradient(smooth)
        if direction=='left':
            for i in range(idx, -1, -1):
                if i>0 and abs(grad[i])<0.001 and smooth[i]<smooth[idx]*0.3: return pos[i]
        else:
            for i in range(idx, len(pos)):
                if i<len(pos)-1 and abs(grad[i])<0.001 and smooth[i]<smooth[idx]*0.3: return pos[i]
        return pk

    def _regions_at_threshold(self, pos, smooth, th, peaks):
        above = smooth>=th
        if not np.any(above): return []
        regs = []; in_reg = False; st = None
        for i,(p,a) in enumerate(zip(pos,above)):
            if a and not in_reg: in_reg=True; st=p
            elif not a and in_reg: in_reg=False; regs.append((st, pos[i-1]))
        if in_reg: regs.append((st, pos[-1]))
        regs = [r for r in regs if r[1]-r[0]>=self.min_region_size]
        if len(peaks) and regs:
            ppos = pos[peaks]
            regs = [r for r in regs if any(r[0]<=pp<=r[1] for pp in ppos)]
        return regs

    def _select_best(self, all_bnd, ostart, oend, clen):
        oc = (ostart+oend)//2; osz = oend-ostart
        best, best_sc = None, -float('inf')
        for td in all_bnd:
            for l,r in td['boundaries']:
                sz = r-l
                s1 = 1 - abs(sz-osz)/(osz+1)
                s2 = 1 - abs((l+r)//2 - oc)/(clen+1)
                ov = max(0, min(r,oend)-max(l,ostart))
                cov = ov/(osz if osz else 1)
                sc = 0.4*s1 + 0.3*s2 + 0.3*cov
                if td['threshold_level'] in ['very_aggressive(select)','aggressive']: sc*=1.2
                if sc>best_sc: best_sc=sc; best={'left_boundary':l,'right_boundary':r,
                                                  'threshold_level':td['threshold_level'],'score':sc}
        return best if best else {'left_boundary':ostart,'right_boundary':oend,
                                  'threshold_level':'fallback','score':0}


class CentromereVisualizer:
    def __init__(self, results_dir, known_centromeres_file=None, kmer_weight=0.7,
                 feature_weight=0.4, optimization_extension=100000,
                 heatmap_colormap='viridis', heatmap_height=0.3,
                 output_dir=None, smoothing_sigma=2.0, distribution_threshold=0.05,
                 random_sampling_times=100000, sample_size=None,
                 compare_centromeres_file=None, peak_prominence=0.1,
                 min_region_size=50000, dynamic_extension=True,
                 target_mean=0.5, mean_tolerance=0.01,
                 max_extension_factor=5.0, extension_increment=50000):
        self.results_dir = Path(results_dir)
        self.known_centromeres_file = known_centromeres_file
        self.output_dir = Path(output_dir) if output_dir else self.results_dir / "visualization"
        self.chrom_data = {}
        self.final_regions = {}
        self.centromere_data = {}
        self.known_centromeres = {}
        self.optimized_centromeres = {}
        self.compare_centromeres = {}
        self.heatmap_colormap = heatmap_colormap
        self.heatmap_height = heatmap_height
        self.optimizer = CentromereOptimizer(
            kmer_weight=kmer_weight, feature_weight=feature_weight,
            extension_bp=optimization_extension, smoothing_sigma=smoothing_sigma,
            distribution_threshold=distribution_threshold,
            random_sampling_times=random_sampling_times, sample_size=sample_size,
            peak_prominence=peak_prominence, min_region_size=min_region_size,
            dynamic_extension=dynamic_extension, target_mean=target_mean,
            mean_tolerance=mean_tolerance, max_extension_factor=max_extension_factor,
            extension_increment=extension_increment)
        self.colors = {
            'kmer_density':'#E74C3C', 'gc_content':'#3498DB', 'cpg_density':'#2ECC71',
            'feature_percent':'#9B59B6', 'centromere':'#F39C12', 'background':'#F8F9FA',
            'telocentric':'#E74C3C', 'metacentric':'#3498DB',
            'submetacentric':'#2ECC71', 'acrocentric':'#9B59B6',
            'holocentric':'#F39C12', 'unknown':'#95A5A6',
            'known_centromere':'#F39C12', 'optimized_centromere':'#E74C3C',
            'search_region':'#D7DBDD', 'analysis_primary':'#3498DB',
            'analysis_candidate':'#9B59B6', 'compare_centromere':'#27AE60',
            'highlight_primary_bg':(0.95,0.95,0.95),
            'highlight_border_primary':'#F39C12', 'highlight_border_candidate':'#E74C3C',
            'marker_primary':'#F39C12', 'marker_candidate':'#E74C3C',
            'distribution_raw':'#7D3C98', 'distribution_smoothed':'#2E86C1',
            'sampling_distribution':'#E74C3C', 'distribution_mean':'#E74C3C',
            'distribution_std':'#F39C12', 'clt_threshold':'#27AE60',
            'dynamic_extension':'#E67E22'
        }
        self.heatmap_cmaps = {'viridis':'viridis', 'plasma':'plasma',
                              'inferno':'inferno', 'magma':'magma',
                              'coolwarm':'coolwarm', 'RdYlBu_r':'RdYlBu_r',
                              'Spectral_r':'Spectral_r'}
        if compare_centromeres_file and os.path.exists(compare_centromeres_file):
            self._load_compare_centromeres(compare_centromeres_file)

    # ---------- data loading ----------
    def load_data(self):
        print("Loading EasyCen results...")
        self._load_tracks()
        if self.known_centromeres_file and os.path.exists(self.known_centromeres_file):
            self._load_known_centromeres()
            self._create_centromere_data_from_known()
        else:
            self._load_centromere_summary()
        opt_bed = self.results_dir / "optimized_centromeres.bed"
        if opt_bed.exists():
            self._load_optimized_bed(opt_bed)
            print(f"Loaded optimized regions from {opt_bed}")
        elif self.known_centromeres:
            self._optimize_boundaries()
            self._save_optimized_bed()
            print("Optimization performed and saved.")
        else:
            print("No optimized BED found, will use primary regions.")
        self._set_final_regions()
        if not self.known_centromeres:
            self._save_analysis_bed()
        total_final = sum(len(v) for v in self.final_regions.values())
        print(f"Final regions to display: {total_final} across {len(self.final_regions)} chromosomes")

    def _load_tracks(self):
        wfiles = glob.glob(str(self.results_dir/"*_kmer_weighted.bedgraph"))
        if wfiles:
            kmer_suffix = '_kmer_weighted.bedgraph'
            bedfiles = wfiles
        else:
            kmer_suffix = '_kmer.bedgraph'
            bedfiles = glob.glob(str(self.results_dir/"*_kmer.bedgraph"))
        for bf in bedfiles:
            chrom = os.path.basename(bf).replace(kmer_suffix, '')
            cd = {
                'weighted_kmer': self._load_bed(bf),
                'kmer_density': self._load_bed(bf.replace(kmer_suffix, '_kmer.bedgraph')
                                               if kmer_suffix=='_kmer_weighted.bedgraph' else bf),
                'feature_percent': self._load_bed(bf.replace(kmer_suffix, '_feature_percent.bedgraph')),
                'gc_content': self._load_bed(bf.replace(kmer_suffix, '_GC.bedgraph')),
                'cpg_density': self._load_bed(bf.replace(kmer_suffix, '_CpG.bedgraph')),
                'periodicity': self._load_bed(bf.replace(kmer_suffix, '_periodicity.bedgraph')),
            }
            if cd['weighted_kmer']:
                clen = cd['weighted_kmer'][-1]['end']
                cd['length'] = clen
                for tr in cd.values():
                    if isinstance(tr, list):
                        for it in tr:
                            it['pos_mb'] = (it['start']+it['end'])/2/1e6
                self.chrom_data[chrom] = cd

    def _load_bed(self, path):
        if not os.path.exists(path): return []
        data = []
        with open(path) as f:
            for line in f:
                p = line.strip().split()
                if len(p)>=4:
                    data.append({'chrom':p[0],'start':int(p[1]),'end':int(p[2]),'value':float(p[3])})
        return data

    def _load_known_centromeres(self):
        try:
            with open(self.known_centromeres_file) as f:
                for line in f:
                    if line.startswith('#') or not line.strip(): continue
                    p = line.strip().split('\t')
                    if len(p)>=4:
                        chrom,start,end,name = p[0],int(p[1]),int(p[2]),p[3]
                        self.known_centromeres.setdefault(chrom,[]).append({
                            'start':start,'end':end,'name':name,
                            'start_mb':start/1e6,'end_mb':end/1e6,
                            'center_mb':(start+end)/2/1e6,'length':end-start,
                            'is_primary':True,'type':'known'})
        except:
            pass

    def _load_compare_centromeres(self, fpath):
        try:
            with open(fpath) as f:
                for line in f:
                    if line.startswith('#') or not line.strip(): continue
                    p = line.strip().split('\t')
                    if len(p)>=4:
                        chrom,start,end,name = p[0],int(p[1]),int(p[2]),p[3]
                        self.compare_centromeres.setdefault(chrom,[]).append({
                            'start':start,'end':end,'name':name,
                            'start_mb':start/1e6,'end_mb':end/1e6,
                            'center_mb':(start+end)/2/1e6,'length':end-start})
        except:
            pass

    def _create_centromere_data_from_known(self):
        for ch,cs in self.known_centromeres.items():
            self.centromere_data[ch] = cs
            for c in cs: c['is_primary'] = True

    def _save_analysis_bed(self):
        if not self.centromere_data: return
        self.output_dir.mkdir(parents=True,exist_ok=True)
        with open(self.output_dir/"analysis_centromeres.bed",'w') as f:
            f.write("# EasyCen centromere regions\n")
            for ch in self._natural_sort(self.centromere_data.keys()):
                for c in self.centromere_data[ch]:
                    nm = f"CEN{ch}" if c.get('is_primary') else f"CEN{ch}_candidate_{c.get('rank','?')}"
                    sc = 1000 if c.get('is_primary') else 500
                    start_1based = max(1, c['start'])
                    f.write(f"{ch}\t{start_1based}\t{c['end']}\t{nm}\t{sc}\t.\n")

    def _load_centromere_summary(self):
        sf = self.results_dir/"centromere_summary.txt"
        if not sf.exists():
            print("centromere_summary.txt not found")
            return
        with open(sf) as f:
            lines = f.readlines()
        in_primary = False
        for line in lines:
            line = line.strip()
            if 'PRIMARY CENTROMERE REGIONS' in line: in_primary = True; continue
            if in_primary and (line.startswith('Chromosome') or line.startswith('---') or not line): continue
            if in_primary and 'DETAILED CANDIDATE REGIONS' in line: break
            if in_primary:
                parts = line.split()
                if len(parts)>=7:
                    chrom = parts[0]; ctype = parts[1].lower()
                    try:
                        start = int(parts[2].replace(',',''))
                        end = int(parts[3].replace(',',''))
                        length = int(parts[4].replace(',',''))
                        self.centromere_data.setdefault(chrom,[]).append({
                            'start':start,'end':end,'length':length,
                            'start_mb':start/1e6,'end_mb':end/1e6,
                            'center_mb':(start+end)/2/1e6,'is_primary':True,'type':ctype})
                    except (ValueError, IndexError): continue
        in_detailed = False; current_chrom = None
        for line in lines:
            line = line.strip()
            if 'DETAILED CANDIDATE REGIONS' in line: in_detailed = True; continue
            if in_detailed and line.startswith('==='): continue
            m = re.match(r'(\S+)\s*\(Length:\s*[\d,]+ bp', line)
            if m: current_chrom = m.group(1); continue
            if in_detailed and current_chrom and (line.startswith('Rank') or line.startswith('---')): continue
            if in_detailed and current_chrom and line:
                parts = line.split()
                if len(parts)>=5 and parts[0] not in ['PRIMARY','#1']:
                    try:
                        start = int(parts[2].replace(',',''))
                        end = int(parts[3].replace(',',''))
                        length = int(parts[4].replace(',',''))
                        self.centromere_data.setdefault(current_chrom,[]).append({
                            'start':start,'end':end,'length':length,
                            'start_mb':start/1e6,'end_mb':end/1e6,
                            'center_mb':(start+end)/2/1e6,'is_primary':False,
                            'rank':parts[0],'type':'unknown'})
                    except (ValueError, IndexError): continue
        print(f"Loaded {sum(len(v) for v in self.centromere_data.values())} centromere entries")

    def _load_optimized_bed(self, bed_path):
        self.optimized_centromeres = {}
        try:
            with open(bed_path) as f:
                for line in f:
                    if line.startswith('#'): continue
                    parts = line.strip().split('\t')
                    if len(parts)>=4:
                        chrom,start,end,name = parts[0],int(parts[1]),int(parts[2]),parts[3]
                        region = {'chrom':chrom,'start':start,'end':end,
                                  'start_mb':start/1e6,'end_mb':end/1e6,
                                  'center_mb':(start+end)/2e6,'length':end-start,
                                  'name':name,'is_primary':True,'type':'optimized'}
                        self.optimized_centromeres.setdefault(chrom,[]).append(region)
            print(f"Loaded optimized regions for {len(self.optimized_centromeres)} chromosomes")
        except Exception as e:
            print(f"Warning: could not load optimized BED {bed_path}: {e}")

    def _optimize_boundaries(self):
        for ch, knowns in self.known_centromeres.items():
            if ch not in self.chrom_data: continue
            cd = self.chrom_data[ch]
            self.optimized_centromeres[ch] = []
            kmer_track = cd.get('weighted_kmer', cd.get('kmer_density', []))
            feat_track = cd.get('feature_percent', [])
            for k in knowns:
                opt = self.optimizer.optimize_boundaries(k, kmer_track, feat_track, cd['length'])
                region = {
                    'chrom': ch,
                    'start': opt['optimized_start'],
                    'end': opt['optimized_end'],
                    'start_mb': opt['optimized_start_mb'],
                    'end_mb': opt['optimized_end_mb'],
                    'center_mb': opt['optimized_center_mb'],
                    'length': opt['optimized_length'],
                    'name': opt.get('name', f'CEN{ch}'),
                    'is_primary': True,
                    'type': 'optimized',
                    'optimization_data': opt
                }
                self.optimized_centromeres[ch].append(region)

    def _save_optimized_bed(self):
        if not self.optimized_centromeres: return
        self.output_dir.mkdir(parents=True,exist_ok=True)
        with open(self.output_dir/"optimized_centromeres.bed",'w') as f:
            f.write("# Optimized centromeres\n")
            for ch in self._natural_sort(self.optimized_centromeres):
                for opt in self.optimized_centromeres[ch]:
                    start_1based = max(1, opt['start'])
                    f.write(f"{ch}\t{start_1based}\t{opt['end']}\t{opt.get('name',f'CEN{ch}')}\t1000\t.\n")

    def _set_final_regions(self):
        self.final_regions = {}
        for ch in self.chrom_data:
            if ch in self.optimized_centromeres and self.optimized_centromeres[ch]:
                self.final_regions[ch] = self.optimized_centromeres[ch][:]
            elif ch in self.centromere_data:
                prims = [c for c in self.centromere_data[ch] if c.get('is_primary')]
                if prims: self.final_regions[ch] = prims[:]
            else:
                self.final_regions[ch] = []

    def _natural_sort(self, lst):
        def conv(t): return int(t) if t.isdigit() else t.lower()
        return sorted(lst, key=lambda s: [conv(c) for c in re.split('([0-9]+)', s)])

    def _region_mean(self, chrom, start, end, track_name):
        if chrom not in self.chrom_data: return np.nan
        track = self.chrom_data[chrom].get(track_name, [])
        if not track: return np.nan
        values = [win['value'] for win in track if win['start']>=start and win['end']<=end]
        if not values: return np.nan
        return np.mean(values)

    def _chrom_mean(self, chrom, track_name):
        if chrom not in self.chrom_data: return np.nan
        track = self.chrom_data[chrom].get(track_name, [])
        if not track: return np.nan
        return np.mean([win['value'] for win in track])

    # ---------- plots ----------
    def create_genome_overview(self, out=None):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out = out or self.output_dir / "genome_centromere_overview.pdf"
        chroms = self._natural_sort(self.chrom_data.keys())
        n = len(chroms)
        if n==0:
            print("No chromosome data found.")
            return
        lengths = [self.chrom_data[c]['length']/1e6 for c in chroms]
        max_len = max(lengths)
        fig_height = max(6, n*0.6)
        fig, ax = plt.subplots(figsize=(12, fig_height), facecolor='white')
        ax.set_xlim(0, max_len*1.1)
        ax.set_ylim(-0.5, n+0.5)
        ax.set_yticks(range(n)); ax.set_yticklabels(chroms, fontweight='bold')
        ax.invert_yaxis()
        ax.set_xlabel('Position (Mb)', fontsize=12, fontweight='bold')
        ax.set_title('Genome-wide Centromere Distribution', fontsize=16, fontweight='bold', pad=20)
        xticks = np.arange(0, max_len+5, 5)
        ax.set_xticks(xticks); ax.set_xticklabels([f"{tick:.0f}" for tick in xticks], rotation=45, ha='right')
        bar_height = 0.5
        for i, (ch, l) in enumerate(zip(chroms, lengths)):
            rect = FancyBboxPatch((0, i-bar_height/2), l, bar_height,
                                  boxstyle="round,pad=0.02",
                                  facecolor='#E0E0E0', edgecolor='black', linewidth=1.5)
            ax.add_patch(rect)
            final_list = self.final_regions.get(ch, [])
            if final_list:
                region = final_list[0]
                smb, emb = region['start_mb'], region['end_mb']
                cen_rect = FancyBboxPatch((smb, i-bar_height/2+0.05), emb-smb, bar_height-0.1,
                                          boxstyle="round,pad=0.01",
                                          facecolor=self.colors['analysis_primary'], alpha=0.8,
                                          edgecolor='white', linewidth=1)
                ax.add_patch(cen_rect)
                ax.text((smb+emb)/2, i-bar_height/2-0.15,
                        f"{smb:.1f}-{emb:.1f} Mb",
                        ha='center', va='top', fontsize=7, rotation=0,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
            ax.text(l+0.5, i, f"{l:.1f} Mb", va='center', ha='left', fontsize=8, color='gray')
        legend_elements = [
            mpatches(facecolor='#E0E0E0', edgecolor='black', label='Chromosome'),
            mpatches(facecolor=self.colors['analysis_primary'], alpha=0.8, label='Centromere (final)'),
        ]
        if self.compare_centromeres:
            legend_elements.append(plt.Line2D([0],[0], color=self.colors['compare_centromere'],
                                              lw=2, ls=':', label='Published'))
        ax.legend(handles=legend_elements, loc='lower right', frameon=True, fontsize=9, framealpha=0.9)
        ax.grid(axis='x', linestyle='--', alpha=0.4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Genome overview saved: {out}")

    def create_summary_statistics(self, outfile=None):
        outfile = outfile or self.output_dir / "centromere_statistics.pdf"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stats = []
        for ch in self._natural_sort(self.chrom_data.keys()):
            final_list = self.final_regions.get(ch, [])
            if not final_list: continue
            region = final_list[0]
            chrom_len = self.chrom_data[ch]['length']
            start, end = region['start'], region['end']
            size_mb = (end-start)/1e6
            center_mb = (start+end)/2e6
            rel_pos = center_mb/(chrom_len/1e6)
            entry = {
                'Chromosome': ch,
                'Cent_Size_Mb': size_mb,
                'Rel_Pos': rel_pos,
                'GC_Content': self._region_mean(ch, start, end, 'gc_content'),
                'CpG_Density': self._region_mean(ch, start, end, 'cpg_density'),
                'Periodicity': self._region_mean(ch, start, end, 'periodicity'),
            }
            if ch in self.optimized_centromeres and self.optimized_centromeres[ch]:
                prim_list = self.centromere_data.get(ch, [])
                prim_primary = [p for p in prim_list if p.get('is_primary')]
                if prim_primary:
                    orig_size = (prim_primary[0]['end']-prim_primary[0]['start'])/1e6
                    entry['Size_Change_%'] = (size_mb/orig_size-1)*100
            stats.append(entry)
        if not stats:
            print("No centromere data found for statistics.")
            return
        df = pd.DataFrame(stats)
        fig, axes = plt.subplots(2,3, figsize=(12,10), constrained_layout=True)
        fig.suptitle('Centromere Statistics', fontsize=16, fontweight='bold')
        axes[0,0].boxplot(df['Cent_Size_Mb'], vert=False, widths=0.6, patch_artist=True,
                          boxprops=dict(facecolor='lightblue', edgecolor='black'))
        axes[0,0].scatter(df['Cent_Size_Mb'], np.random.normal(1,0.04,len(df)), alpha=0.6, c='red', s=30)
        axes[0,0].set_xlabel('Size (Mb)'); axes[0,0].set_title('Centromere Size Distribution')
        axes[0,0].grid(axis='x', linestyle='--', alpha=0.5)
        axes[0,1].hist(df['Rel_Pos'], bins=15, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0,1].axvline(0.5, color='red', linestyle='--', label='Metacentric (0.5)')
        axes[0,1].set_xlabel('Relative Position (0=end, 1=end)')
        axes[0,1].set_ylabel('Frequency'); axes[0,1].set_title('Centromere Position')
        axes[0,1].legend()
        chrom_gc = [self._chrom_mean(ch,'gc_content') for ch in df['Chromosome']]
        axes[0,2].scatter(chrom_gc, df['GC_Content'], alpha=0.7, c='green', edgecolors='black')
        axes[0,2].plot([min(chrom_gc), max(chrom_gc)], [min(chrom_gc), max(chrom_gc)], 'k--')
        axes[0,2].set_xlabel('Chromosome GC (%)'); axes[0,2].set_ylabel('Centromere GC (%)')
        axes[0,2].set_title('GC Content Comparison')
        chrom_cpg = [self._chrom_mean(ch,'cpg_density') for ch in df['Chromosome']]
        axes[1,0].scatter(chrom_cpg, df['CpG_Density'], alpha=0.7, c='orange', edgecolors='black')
        axes[1,0].plot([min(chrom_cpg), max(chrom_cpg)], [min(chrom_cpg), max(chrom_cpg)], 'k--')
        axes[1,0].set_xlabel('Chromosome CpG density'); axes[1,0].set_ylabel('Centromere CpG density')
        axes[1,0].set_title('CpG Density Comparison')
        axes[1,1].violinplot(df['Periodicity'], showmeans=True)
        axes[1,1].set_xticks([1]); axes[1,1].set_xticklabels(['Centromere'])
        axes[1,1].set_ylabel('Periodicity score'); axes[1,1].set_title('Periodicity Distribution')
        if 'Size_Change_%' in df.columns:
            axes[1,2].bar(df['Chromosome'], df['Size_Change_%'], color='lightcoral', edgecolor='black')
            axes[1,2].axhline(0, color='black', linestyle='-', linewidth=0.8)
            axes[1,2].set_xticklabels(df['Chromosome'], rotation=45, ha='right')
            axes[1,2].set_ylabel('Size change (%)'); axes[1,2].set_title('Optimization Effect')
        else:
            axes[1,2].text(0.5,0.5,'No optimization data', ha='center', transform=axes[1,2].transAxes)
            axes[1,2].set_title('No optimization')
        plt.savefig(outfile, dpi=300, bbox_inches='tight')
        plt.close()
        df.to_csv(self.output_dir/"centromere_statistics.csv", index=False)
        print(f"Statistics saved: {outfile}")

    def create_chromosome_detail_plots(self, outdir=None):
        outdir = outdir or self.output_dir / "chromosome_details"
        outdir.mkdir(parents=True, exist_ok=True)
        for ch in self._natural_sort(self.chrom_data.keys()):
            try:
                self._single_plot(ch, outdir / f"{ch}_detailed.pdf")
            except Exception as e:
                print(f"Error plotting {ch}: {e}")
        print(f"Chromosome detail plots saved to {outdir}")

    def _single_plot(self, ch, out):
        cd = self.chrom_data[ch]
        lmb = cd['length']/1e6
        final_list = self.final_regions.get(ch, [])
        final_region = final_list[0] if final_list else None
        has_opt = ch in self.optimized_centromeres and self.optimized_centromeres[ch]
        tracks = [
            (cd.get('weighted_kmer', cd.get('kmer_density', [])), 'Weighted K-mer Density',
             self.colors['kmer_density'], 'Score'),
            (cd.get('feature_percent', []), 'Feature %', self.colors['feature_percent'], '%'),
            (cd.get('periodicity', []), 'Periodicity', self.colors.get('periodicity','#D35400'), 'Score'),
            (cd.get('gc_content', []), 'GC Content', self.colors['gc_content'], '%'),
            (cd.get('cpg_density', []), 'CpG Density', self.colors['cpg_density'], 'Count')
        ]
        n_tracks = len(tracks)
        base_rows = 2 + 2*n_tracks
        if has_opt:
            total_rows = base_rows + 5
            height_ratios = [0.8, 0.15] + [1,0.3]*n_tracks + [1.5,1.0,0.8,0.5,0.5]
        else:
            total_rows = base_rows
            height_ratios = [0.8, 0.15] + [1,0.3]*n_tracks
        fig_height = 1.0*total_rows
        fig = plt.figure(figsize=(8, fig_height), constrained_layout=True)
        gs = gridspec.GridSpec(total_rows, 1, height_ratios=height_ratios, figure=fig)
        ax_idx = 0
        self._header(plt.subplot(gs[ax_idx]), ch, cd, final_region); ax_idx+=1
        self._marker_track(plt.subplot(gs[ax_idx]), ch, final_region); ax_idx+=1
        for i, (data, tit, col, yl) in enumerate(tracks):
            ax_l = plt.subplot(gs[ax_idx])
            ax_h = plt.subplot(gs[ax_idx+1]); ax_idx+=2
            is_last = (i==len(tracks)-1) and not has_opt
            self._enhanced_track(ax_l, ax_h, data, tit, col, yl, ch, is_last, lmb, final_region)
        if has_opt:
            ax_opt = plt.subplot(gs[ax_idx]); ax_samp=plt.subplot(gs[ax_idx+1])
            ax_dyn = plt.subplot(gs[ax_idx+2]); ax_leg=plt.subplot(gs[ax_idx+3])
            ax_dleg=plt.subplot(gs[ax_idx+4])
            self._optimization_panel(ax_opt, ax_samp, ax_dyn, ax_leg, ax_dleg, ch, lmb)
        for ax in fig.axes:
            if ax.get_xlabel():
                self._annotations(ax, ch, final_region)
                ax.set_xlim(0, lmb)
        plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    def _header(self, ax, ch, cd, final_region):
        ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
        lmb = cd['length']/1e6
        info = [f"Chromosome: {ch}", f"Length: {lmb:.2f} Mb"]
        if final_region:
            smb, emb = final_region['start_mb'], final_region['end_mb']
            size_mb = emb-smb
            info.extend([f"Centromere: {smb:.2f}-{emb:.2f} Mb", f"Size: {size_mb:.2f} Mb"])
            if 'optimization_data' in final_region:
                orig_len = final_region['optimization_data'].get('length', final_region['length'])
                chg = ((final_region['length']/orig_len)-1)*100 if orig_len>0 else 0
                info.append(f"Optimized: {chg:+.1f}% change")
        ax.text(0.05,0.8,'\n'.join(info), transform=ax.transAxes, va='top',
                fontfamily='monospace', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        ax.text(0.5,0.95,f'Chromosome {ch}', transform=ax.transAxes, ha='center', va='center',
                fontsize=16, fontweight='bold')

    def _marker_track(self, ax, ch, final_region):
        ax.set_xlim(0, self.chrom_data[ch]['length']/1e6); ax.set_ylim(0,1); ax.axis('off')
        if final_region:
            smb, emb = final_region['start_mb'], final_region['end_mb']
            rect = Rectangle((smb,0.2), emb-smb, 0.6,
                             facecolor=self.colors['analysis_primary'], alpha=0.8,
                             edgecolor='white', linewidth=1)
            ax.add_patch(rect)
            ax.text((smb+emb)/2,0.85, f"{smb:.1f}-{emb:.1f} Mb",
                    ha='center', va='bottom', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    def _annotations(self, ax, ch, final_region):
        if final_region:
            smb, emb = final_region['start_mb'], final_region['end_mb']
            ax.axvspan(smb, emb, alpha=0.08, color=self.colors['highlight_primary_bg'])
            ax.axvline(smb, color=self.colors['analysis_primary'], ls='-', lw=1.5)
            ax.axvline(emb, color=self.colors['analysis_primary'], ls='-', lw=1.5)
            ylim = ax.get_ylim()
            ax.text(smb, ylim[1]*0.95, f"{smb:.0f}", ha='center', va='top', fontsize=8,
                    color=self.colors['analysis_primary'])
            ax.text(emb, ylim[1]*0.95, f"{emb:.0f}", ha='center', va='top', fontsize=8,
                    color=self.colors['analysis_primary'])

    def _enhanced_track(self, ax_l, ax_h, data, tit, col, yl, ch, is_last, lmb, final_region):
        if not data:
            ax_l.text(0.5,0.5,'No data', ha='center'); ax_h.text(0.5,0.5,'No data', ha='center')
            ax_l.set_xticks([]); ax_h.set_xticks([])
            return
        pos = [d['pos_mb'] for d in data]
        val = [d['value'] for d in data]
        ax_l.fill_between(pos, val, alpha=0.3, color=col)
        ax_l.plot(pos, val, color=col, lw=1.2)
        ax_l.set_xlim(0,lmb); ax_l.set_ylabel(yl, fontweight='bold')
        ax_l.set_title(tit, fontweight='bold'); ax_l.grid(True, alpha=0.3)
        ticks = np.arange(0,lmb+1,5)
        ax_l.set_xticks(ticks)
        ax_l.set_xticklabels([f"{tick:.0f}" for tick in ticks], rotation=45, ha='right')
        if is_last: ax_l.set_xlabel('Position (Mb)', fontweight='bold')
        else: ax_l.tick_params(labelbottom=False)
        bins = np.linspace(0,lmb,200)
        digitized = np.digitize(pos, bins)
        bm = [np.mean([val[i] for i in range(len(val)) if digitized[i]==b]) for b in range(1,len(bins))]
        bm = np.nan_to_num(bm)
        im = ax_h.imshow(np.array(bm).reshape(1,-1), aspect='auto',
                         cmap=self.heatmap_cmaps.get(self.heatmap_colormap,'viridis'),
                         extent=[0,lmb,0,1])
        ax_h.set_yticks([]); ax_h.set_xlim(0,lmb)
        if is_last: ax_h.set_xlabel('Position (Mb)', fontweight='bold')
        else: ax_h.set_xticklabels([])

    def _optimization_panel(self, ax, ax_samp, ax_dyn, ax_leg, ax_dleg, ch, lmb):
        if ch not in self.optimized_centromeres: return
        opt_data = self.optimized_centromeres[ch][0].get('optimization_data', None)
        if opt_data is None: return
        pos = np.array(opt_data['search_positions'])
        comp = np.array(opt_data['composite_scores'])
        smooth = np.array(opt_data.get('composite_smoothed', comp))
        pos_mb = pos/1e6
        ks, ke = opt_data['start']/1e6, opt_data['end']/1e6
        os, oe = opt_data['optimized_start_mb'], opt_data['optimized_end_mb']
        ss, se = max(0, ks-self.optimizer.extension_bp/1e6), min(lmb, ke+self.optimizer.extension_bp/1e6)
        mask = (pos_mb>=ss)&(pos_mb<=se)
        ax.plot(pos_mb[mask], comp[mask], color=self.colors['distribution_raw'], lw=1.5, label='Composite')
        ax.plot(pos_mb[mask], smooth[mask], color=self.colors['distribution_smoothed'], lw=2.5, label='Smoothed')
        if 'peaks' in opt_data and opt_data['peaks']:
            ax.scatter(pos_mb[opt_data['peaks']], smooth[opt_data['peaks']], color='red', s=50, zorder=5,
                       label=f'Peaks ({len(opt_data["peaks"])})')
        ax.axvspan(ss, se, alpha=0.1, color=self.colors['search_region'], label='Search')
        ax.axvspan(ks, ke, alpha=0.2, color=self.colors['known_centromere'], label='Known')
        ax.axvspan(os, oe, alpha=0.3, color=self.colors['optimized_centromere'], label='Optimized')
        ax.set_xlim(ss, se); ax.set_xlabel('Position (Mb)'); ax.set_ylabel('Score')
        ax.set_title('Boundary Optimization'); ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        tinfo = opt_data.get('threshold_info', {})
        if 'sampling_distribution_stats' in tinfo:
            smeans = tinfo['sampling_distribution_stats'].get('sample_means', [])
            if smeans:
                ax_samp.hist(smeans, bins=50, alpha=0.7, color=self.colors['sampling_distribution'],
                             density=True, edgecolor='black')
                x = np.linspace(*ax_samp.get_xlim(),100)
                ax_samp.plot(x, norm.pdf(x, tinfo.get('sampling_mean',0.5), tinfo.get('sampling_std',0.1)), 'k', lw=2)
                ax_samp.set_xlabel('Sample Mean'); ax_samp.set_ylabel('Density')
                ax_samp.set_title('Sampling Distribution')
        if 'dynamic_extension_info' in opt_data:
            dyn = opt_data['dynamic_extension_info']
            hist = dyn.get('mean_history', [])
            if hist:
                iters = [m['iteration'] for m in hist]
                ext = [m['extension']/1000 for m in hist]
                means = [m['mean'] for m in hist]
                ax_ext = ax_dyn; ax_m = ax_dyn.twinx()
                ax_ext.plot(iters, ext, 'o-', color=self.colors['dynamic_extension'], lw=2, label='Extension (kb)')
                ax_m.plot(iters, means, 's-', color=self.colors['sampling_distribution'], lw=2, label='Mean')
                ax_m.axhline(self.optimizer.target_mean, color='green', ls='--', label='Target')
                ax_ext.set_xlabel('Iteration'); ax_ext.set_ylabel('Extension (kb)', color=self.colors['dynamic_extension'])
                ax_m.set_ylabel('Mean', color=self.colors['sampling_distribution'])
                ax_dyn.set_title('Dynamic Extension'); ax_dyn.legend(loc='upper right', fontsize=8)
        ax_leg.axis('off'); ax_dleg.axis('off')


def visualize_results(results_dir, known_centromeres_file=None, output_dir=None,
                     kmer_weight=0.7, feature_weight=0.4, optimization_extension=100000,
                     genome_overview=True, chromosome_details=True, statistics=True,
                     heatmap_colormap='viridis', heatmap_height=0.3,
                     smoothing_sigma=2.0, distribution_threshold=0.05,
                     random_sampling_times=100000, sample_size=None,
                     compare_centromeres_file=None, peak_prominence=0.1,
                     min_region_size=50000, dynamic_extension=True,
                     target_mean=0.5, mean_tolerance=0.01,
                     max_extension_factor=5.0, extension_increment=50000):
    output_dir = Path(output_dir) if output_dir else Path(results_dir)/"visualization"
    print(f"EasyCen Visualization v1.0\nResults: {results_dir}\nOutput: {output_dir}")
    viz = CentromereVisualizer(
        results_dir, known_centromeres_file, kmer_weight, feature_weight,
        optimization_extension, heatmap_colormap, heatmap_height, output_dir,
        smoothing_sigma, distribution_threshold, random_sampling_times,
        sample_size, compare_centromeres_file, peak_prominence, min_region_size,
        dynamic_extension, target_mean, mean_tolerance, max_extension_factor,
        extension_increment)
    viz.load_data()
    if genome_overview: viz.create_genome_overview()
    if chromosome_details: viz.create_chromosome_detail_plots()
    if statistics: viz.create_summary_statistics()
    print("Visualization complete.")


def main():
    parser = argparse.ArgumentParser(description="EasyCen Visualization v1.0")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--known-centromeres")
    parser.add_argument("--compare")
    parser.add_argument("--kmer-weight", type=float, default=0.7)
    parser.add_argument("--feature-weight", type=float, default=0.4)
    parser.add_argument("--optimization-extension", type=int, default=100000)
    parser.add_argument("--smoothing-sigma", type=float, default=2.0)
    parser.add_argument("--distribution-threshold", type=float, default=0.05)
    parser.add_argument("--random-sampling-times", type=int, default=100000)
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--peak-prominence", type=float, default=0.1)
    parser.add_argument("--min-region-size", type=int, default=50000)
    parser.add_argument("--dynamic-extension", action="store_true", default=True)
    parser.add_argument("--no-dynamic-extension", dest="dynamic_extension", action="store_false")
    parser.add_argument("--target-mean", type=float, default=0.5)
    parser.add_argument("--mean-tolerance", type=float, default=0.01)
    parser.add_argument("--max-extension-factor", type=float, default=5.0)
    parser.add_argument("--extension-increment", type=int, default=50000)
    parser.add_argument("--output-dir")
    parser.add_argument("--genome-overview", action="store_true", default=True)
    parser.add_argument("--chromosome-details", action="store_true", default=True)
    parser.add_argument("--statistics", action="store_true", default=True)
    parser.add_argument("--heatmap-colormap", default="viridis")
    parser.add_argument("--heatmap-height", type=float, default=0.3)
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
        extension_increment=args.extension_increment)


if __name__ == "__main__":
    main()