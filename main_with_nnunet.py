#!/usr/bin/env python3
"""
Script that runs main.py to process vascular data, then automatically runs
create_nnunet.py to convert the output to nnUNet format.

This allows running both scripts in sequence with a single command.
"""

import argparse
import sys
import os
import subprocess
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent))

# Import io module only when needed (for loading config)
try:
    from modules import io
except ImportError:
    io = None


def run_main_py(args):
    """Run main.py with the provided arguments"""
    print("=" * 80)
    print("Running main.py...")
    print("=" * 80)
    
    # Build command for main.py
    cmd = [sys.executable, 'main.py']
    
    # Add all main.py arguments
    if args.outdir:
        cmd.extend(['--outdir', args.outdir])
    if args.config_name:
        cmd.extend(['--config_name', args.config_name])
    if args.perc_dataset is not None:
        cmd.extend(['--perc_dataset', str(args.perc_dataset)])
    if args.num_cores:
        cmd.extend(['--num_cores', str(args.num_cores)])
    if args.start_from is not None:
        cmd.extend(['--start_from', str(args.start_from)])
    if args.end_at is not None:
        cmd.extend(['--end_at', str(args.end_at)])
    if args.data_dir:
        cmd.extend(['--data_dir', args.data_dir])
    if args.testing:
        cmd.append('--testing')
    if args.validation_prop is not None:
        cmd.extend(['--validation_prop', str(args.validation_prop)])
    if args.max_samples is not None:
        cmd.extend(['--max_samples', str(args.max_samples)])
    if args.modality:
        cmd.extend(['--modality', args.modality])
    if args.truth_from_surface:
        cmd.append('--truth_from_surface')
    if args.truth_target_spacing is not None:
        cmd.extend(['--truth_target_spacing'] + [str(x) for x in args.truth_target_spacing])
    if args.truth_regenerate:
        cmd.append('--truth_regenerate')
    
    print(f"Command: {' '.join(cmd)}")
    print()
    
    # Run main.py
    result = subprocess.run(cmd, check=False)
    
    if result.returncode != 0:
        print(f"ERROR: main.py exited with code {result.returncode}")
        sys.exit(result.returncode)
    
    print()
    print("=" * 80)
    print("main.py completed successfully")
    print("=" * 80)
    print()


def run_create_nnunet(args, modality):
    """Run create_nnunet.py for a specific modality"""
    print("=" * 80)
    print(f"Running create_nnunet.py for modality: {modality}")
    print("=" * 80)
    
    # Build command for create_nnunet.py
    cmd = [sys.executable, 'dataset_dirs/create_nnunet.py']
    
    # Use the same output directory as main.py
    outdir = args.outdir if args.outdir else './extracted_data/'
    cmd.extend(['--outdir', outdir])
    cmd.extend(['--indir', outdir])  # Input is same as output from main.py
    
    # Add create_nnunet specific arguments
    if args.nnunet_name:
        cmd.extend(['--name', args.nnunet_name])
    if args.nnunet_dataset_number is not None:
        cmd.extend(['--dataset_number', str(args.nnunet_dataset_number)])
    if args.nnunet_start_from is not None:
        cmd.extend(['--start_from', str(args.nnunet_start_from)])
    
    # Add modality (required for create_nnunet)
    cmd.extend(['--modality', modality.lower()])
    
    print(f"Command: {' '.join(cmd)}")
    print()
    
    # Run create_nnunet.py
    result = subprocess.run(cmd, check=False)
    
    if result.returncode != 0:
        print(f"ERROR: create_nnunet.py exited with code {result.returncode}")
        return False
    
    print()
    print("=" * 80)
    print(f"create_nnunet.py completed successfully for {modality}")
    print("=" * 80)
    print()
    
    return True


def get_processed_modalities(outdir):
    """Get list of modalities that were processed by main.py"""
    modalities = []
    outdir_path = Path(outdir) if outdir else Path('./extracted_data/')
    
    if not outdir_path.exists():
        return modalities
    
    # Look for directories matching {modality}_train pattern
    for item in outdir_path.iterdir():
        if item.is_dir():
            name = item.name
            if '_train' in name:
                modality = name.replace('_train', '').upper()
                if modality not in modalities:
                    modalities.append(modality)
    
    return modalities


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run main.py and then create_nnunet.py to process vascular data and convert to nnUNet format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python3 main_with_nnunet.py --config_name global --data_dir /path/to/data \\
      --nnunet_name AORTAS --nnunet_dataset_number 1

  # With multiple modalities
  python3 main_with_nnunet.py --config_name global --data_dir /path/to/data \\
      --modality CT,MR --nnunet_name AORTAS --nnunet_dataset_number 1

  # Skip nnUNet conversion
  python3 main_with_nnunet.py --config_name global --data_dir /path/to/data \\
      --skip_nnunet
        """
    )
    
    # Arguments from main.py
    parser.add_argument('-outdir', '--outdir',
                        default='./extracted_data/',
                        type=str,
                        help='Output directory (default: ./extracted_data/)')
    parser.add_argument('-config_name', '--config_name',
                        type=str,
                        required=True,
                        help='Name of configuration file (without .yaml extension)')
    parser.add_argument('-perc_dataset', '--perc_dataset',
                        default=1.0,
                        type=float,
                        help='Percentage of dataset to use (default: 1.0)')
    parser.add_argument('-num_cores', '--num_cores',
                        default=1,
                        type=int,
                        help='Number of CPU cores to use (default: 1)')
    parser.add_argument('-start_from', '--start_from',
                        default=0,
                        type=int,
                        help='Start from case number (default: 0)')
    parser.add_argument('-end_at', '--end_at',
                        default=-1,
                        type=int,
                        help='End at case number, -1 for all cases (default: -1)')
    parser.add_argument('-data_dir', '--data_dir',
                        required=True,
                        type=str,
                        help='Directory where input data is stored')
    parser.add_argument('-testing', '--testing',
                        action='store_true',
                        help='Enable testing mode')
    parser.add_argument('-validation_prop', '--validation_prop',
                        type=float,
                        default=None,
                        help='Validation set proportion (0.0-1.0)')
    parser.add_argument('-max_samples', '--max_samples',
                        type=float,
                        default=None,
                        help='Maximum number of samples to extract')
    parser.add_argument('-modality', '--modality',
                        type=str,
                        default=None,
                        help='Imaging modality: CT, MR, or comma-separated list (CT,MR)')
    parser.add_argument('--truth_from_surface', '--seg_from_surface',
                        dest='truth_from_surface',
                        action='store_true',
                        help='Forward to main.py: rasterize surfaces/ to truths/ (global/create_seg_from_surf.py).')
    parser.add_argument('--truth_target_spacing',
                        type=float,
                        nargs=3,
                        metavar=('SX', 'SY', 'SZ'),
                        default=None,
                        help='Forward to main.py: truth voxel spacing in mm (optional).')
    parser.add_argument('--truth_regenerate',
                        action='store_true',
                        help='Forward to main.py: overwrite existing truths/ from surfaces.')
    
    # Arguments for create_nnunet.py
    parser.add_argument('-nnunet_name', '--nnunet_name',
                        type=str,
                        default='AORTAS',
                        help='Dataset name for nnUNet (default: AORTAS)')
    parser.add_argument('-nnunet_dataset_number', '--nnunet_dataset_number',
                        type=int,
                        default=1,
                        help='Dataset number for nnUNet (default: 1)')
    parser.add_argument('-nnunet_start_from', '--nnunet_start_from',
                        type=int,
                        default=0,
                        help='Starting number for nnUNet dataset (default: 0)')
    
    # Option to skip nnUNet conversion
    parser.add_argument('-skip_nnunet', '--skip_nnunet',
                        action='store_true',
                        help='Skip nnUNet conversion (only run main.py)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Vascular Segment Sampler with nnUNet Conversion")
    print("=" * 80)
    print()
    
    # Step 1: Run main.py
    try:
        run_main_py(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR running main.py: {e}")
        sys.exit(1)
    
    # Step 2: Run create_nnunet.py (if not skipped)
    if not args.skip_nnunet:
        # Get modalities from config or detect from output
        modalities = []
        if args.modality:
            modalities = [m.strip().upper() for m in args.modality.split(',')]
        else:
            # Load config to get modalities
            try:
                if io is None:
                    raise ImportError("modules.io not available")
                global_config = io.load_yaml(f"./config/{args.config_name}.yaml")
                if 'MODALITY' in global_config:
                    modalities = [m.upper() if isinstance(m, str) else m for m in global_config['MODALITY']]
                else:
                    # Try to detect from output directory
                    modalities = get_processed_modalities(args.outdir)
            except Exception as e:
                print(f"Warning: Could not load config, trying to detect modalities from output: {e}")
                modalities = get_processed_modalities(args.outdir)
        
        if not modalities:
            print("WARNING: No modalities detected. Skipping nnUNet conversion.")
            print("You may need to specify --modality explicitly.")
        else:
            print(f"Detected modalities: {modalities}")
            print()
            
            # Run create_nnunet for each modality
            for modality in modalities:
                try:
                    success = run_create_nnunet(args, modality)
                    if not success:
                        print(f"WARNING: nnUNet conversion failed for {modality}")
                except KeyboardInterrupt:
                    print(f"\nInterrupted during nnUNet conversion for {modality}")
                    sys.exit(1)
                except Exception as e:
                    print(f"\nERROR running create_nnunet.py for {modality}: {e}")
                    # Continue with other modalities
    
    print("=" * 80)
    print("All processing completed!")
    print("=" * 80)

