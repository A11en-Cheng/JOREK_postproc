"""
Boundary analysis module.

Handles the workflow for generating boundary analysis plots (plot_set),
which includes R-Z boundary contours and unfolded Phi-Theta heat flux maps.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional

from . import config as cfg
from . import (
    read_boundary_file,
    reshape_to_grid,
    get_device_geometry,
    PlottingConfig, 
)
# Note: plot_set is imported dynamically or we add it to __init__ later.
# For now, let's import directly from plotting to be safe
from .plotting import plot_set

def run_boundary_analysis(conf: cfg.ProcessingConfig):
    """
    Run the boundary analysis workflow using plot_set.
    """
    print("\n" + "="*70)
    print(f"Starting Boundary Analysis (Plot Set)")
    print(f"File: {conf.file_path}")
    print(f"Device: {conf.device}")
    print(f"Data: {conf.data_name}")
    print("="*70)
    
    # Check file
    if not os.path.exists(conf.file_path):
        raise FileNotFoundError(f"File not found: {conf.file_path}")
    
    # 1. Read file
    print("\n[1/3] Reading file...")
    try:
        col_names, blocks, t_mapping = read_boundary_file(conf.file_path, debug=conf.debug)
    except Exception as e:
        print(f"  ✗ Read failed: {e}")
        return
    
    print(f"  ✓ Read successful. Steps available: {len(blocks)}")
    
    # 2. Process each timestep
    for ts in conf.timesteps:
        ts_str = str(ts).zfill(6)
        print(f"\n[2/3] Processing timestep {ts_str}...")
        
        if ts_str not in blocks:
            print(f"  ✗ Timestep {ts_str} not found in file")
            continue
        
        block_data = blocks[ts_str]
        
        # Filter zero rows
        non_zero_mask = ~np.all(np.isclose(block_data, 0.0), axis=1)
        if np.sum(~non_zero_mask) > 0:
            if conf.debug: print(f"  Dropped {np.sum(~non_zero_mask)} zero rows")
            block_data = block_data[non_zero_mask]
            
        # Reshape to grid
        try:
            # We need 'theta' for plot_set, so explicitly check it
            # reshape_to_grid in reshaping.py should handle theta automatically now
            names = ['R', 'Z', 'phi', conf.data_name]
            
            xpoints = None
            if conf.xpoints is not None and len(conf.xpoints) > 0:
                 xpoints = np.array(conf.xpoints, dtype=float).reshape(-1, 2)
                 xpoints.sort(axis=0) # ensure consistent ordering
            
            grid_data = reshape_to_grid(
                block_data, col_names, names,
                iplane=conf.iplane,
                xpoints=xpoints,
                debug=conf.debug
            )
            
            # Assign time if available
            if t_mapping and ts_str in t_mapping:
                grid_data.time = t_mapping[ts_str]*conf.norm_factor
            
            # Additional check for theta
            if grid_data.theta is None:
                print("  ! Warning: Theta data missing. plot_set requires theta.")
                # We might proceed, but plot_set will likely skip the 2D map
            
        except Exception as e:
            print(f"  ✗ Reshape failed: {e}")
            if conf.debug:
                import traceback
                traceback.print_exc()
            continue

        # Get device info and masks
        device_geo = None
        try:
            device_geo = get_device_geometry(conf.device, grid_data.R, grid_data.Z, xpoints=xpoints, debug=conf.debug)
        except Exception as e:
            if conf.debug: print(f"  ! Device geometry warning: {e}")

        # 3. Plotting
        print(f"\n[3/3] Generating Plot Set...")
        
        # Output directory
        if conf.output_dir is None:
            output_dir = f"output_analysis_{conf.device}"
        else:
            output_dir = conf.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        plotting_config = PlottingConfig(
            log_norm=conf.log_norm,
            cmap='viridis', # Better for heat flux usually
            dpi=300,
            data_limits=conf.data_limits,
            find_max=conf.find_max,
            show_left_plot=conf.show_left_plot,
            show_right_plot=conf.show_right_plot
        )
        
        # Define regions based on device geometry masks
        regions = []
        if device_geo and device_geo.masks:
            # Map mask names to display labels and colors
            # Standard naming in geometry.py is 'mask_UO', 'mask_LO', etc.
            region_style = {
                'mask_UI': {'label': 'IU', 'color': 'red'},
                'mask_UO': {'label': 'OU', 'color': 'cyan'},
                'mask_LI': {'label': 'IL', 'color': 'green'},
                'mask_LO': {'label': 'OL', 'color': 'orange'} # Changed OL to purple to be distinct from IL blue
            }
            
            for mask_name, mask_array in device_geo.masks.items():
                if mask_name in region_style:
                    style = region_style[mask_name]
                    regions.append({
                        'label': style['label'],
                        'mask': mask_array,
                        'color': style['color']
                    })
                # else: could add generic masks here if needed
        
        # Determine filename suffix based on plot selection
        suffix = ""
        if conf.show_left_plot and not conf.show_right_plot:
            suffix = "_left"
        elif not conf.show_left_plot and conf.show_right_plot:
            suffix = "_right"

        if conf.log_norm:
            suffix += "_log"

        save_path = os.path.join(output_dir, f'plot_set_{conf.data_name}_{ts_str}{suffix}.png')

        try:
            plot_set(
                data=grid_data,
                config=plotting_config,
                save_path=save_path,
                regions=regions,
                debug=conf.debug
            )
            print(f"  ✓ Saved analysis plot to {save_path}")
        except Exception as e:
            print(f"  ✗ Plotting failed: {e}")
            if conf.debug:
                import traceback
                traceback.print_exc()
                
    print("\n" + "="*70)
    print("✓ Analysis Complete")
    print("="*70)
