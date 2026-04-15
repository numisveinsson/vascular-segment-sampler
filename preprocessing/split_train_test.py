"""
Split dataset into training and testing sets.

This script takes a folder containing subfolders (images, centerlines, surfaces, truths, labels)
and randomly splits the cases into training and testing sets based on a specified percentage.
"""

import os
import sys
import shutil
import argparse
import random
from pathlib import Path

# Add project root to path so modules can be imported when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.logger import get_logger


def get_base_names(folder, extensions=['.mha', '.nii.gz', '.nii', '.vti', '.vtp', '.stl', '.mhd']):
    """Get base filenames without extensions from a folder."""
    if not os.path.exists(folder):
        return set()
    
    files = os.listdir(folder)
    base_names = set()
    
    for f in files:
        if f.startswith('.'):
            continue
        for ext in extensions:
            if f.endswith(ext):
                base_names.add(f.replace(ext, ''))
                break
    
    return base_names


DEFAULT_SUBFOLDERS = ['images', 'centerlines', 'surfaces', 'truths', 'labels']


def get_case_names(base_folder, subfolders=None):
    """
    Get case names from the dataset folder.
    
    Args:
        base_folder: Path to the base folder containing subfolders
        subfolders: List of subfolder names to require (default: all standard subfolders that exist)
    
    Returns:
        List of case names (base names without extensions)
    """
    if subfolders is None:
        subfolders = DEFAULT_SUBFOLDERS
    
    logger = get_logger(__name__)
    
    # Check which subfolders exist
    existing_subfolders = []
    name_sets = {}
    
    for subfolder in subfolders:
        subfolder_path = os.path.join(base_folder, subfolder)
        if os.path.exists(subfolder_path):
            existing_subfolders.append(subfolder)
            names = get_base_names(subfolder_path)
            name_sets[subfolder] = names
            logger.info(f"Found {len(names)} cases in {subfolder}/")
        else:
            logger.debug(f"Subfolder {subfolder}/ does not exist, skipping")
    
    if not existing_subfolders:
        raise ValueError(f"No valid subfolders found in {base_folder}. "
                        f"Expected one of: {subfolders}")
    
    # Use only cases present in ALL subfolders (intersection)
    case_names = list(set.intersection(*[name_sets[sf] for sf in existing_subfolders]))
    logger.info(f"Using cases present in all subfolders {existing_subfolders}: {len(case_names)} cases")
    
    if not case_names:
        raise ValueError(f"No cases found in {base_folder}")
    
    case_names.sort()
    return case_names, existing_subfolders


def split_dataset(base_folder, train_split, output_folder=None, seed=None, copy_files=True, subfolders=None):
    """
    Split dataset into training and testing sets.
    
    Args:
        base_folder: Path to folder containing subfolders (images, centerlines, etc.)
        train_split: Fraction of data for training (0.0 to 1.0)
        output_folder: Output folder for train/test splits (default: base_folder + '_split')
        seed: Random seed for reproducibility (optional)
        copy_files: If True, copy files; if False, move files
    
    Returns:
        Tuple of (train_cases, test_cases)
    """
    logger = get_logger(__name__)
    
    if not 0 < train_split < 1:
        raise ValueError(f"train_split must be between 0 and 1, got {train_split}")
    
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
        logger.info(f"Using random seed: {seed}")
    
    # Get case names
    case_names, existing_subfolders = get_case_names(base_folder, subfolders=subfolders)
    logger.info(f"Total cases found: {len(case_names)}")
    
    # Shuffle and split
    shuffled_cases = case_names.copy()
    random.shuffle(shuffled_cases)
    
    n_train = int(len(shuffled_cases) * train_split)
    train_cases = set(shuffled_cases[:n_train])
    test_cases = set(shuffled_cases[n_train:])
    
    logger.info(f"Training cases: {len(train_cases)} ({len(train_cases)/len(case_names)*100:.1f}%)")
    logger.info(f"Testing cases: {len(test_cases)} ({len(test_cases)/len(case_names)*100:.1f}%)")
    
    # Set output folder
    if output_folder is None:
        output_folder = base_folder.rstrip('/').rstrip('\\') + '_split'
    
    # Create output directory structure
    train_base = os.path.join(output_folder, 'train')
    test_base = os.path.join(output_folder, 'test')
    
    os.makedirs(train_base, exist_ok=True)
    os.makedirs(test_base, exist_ok=True)
    
    # Create subfolders for train and test
    for subfolder in existing_subfolders:
        os.makedirs(os.path.join(train_base, subfolder), exist_ok=True)
        os.makedirs(os.path.join(test_base, subfolder), exist_ok=True)
    
    # Copy/move files
    operation = "Copying" if copy_files else "Moving"
    logger.info(f"{operation} files to train/test folders...")
    
    for subfolder in existing_subfolders:
        subfolder_path = os.path.join(base_folder, subfolder)
        files = os.listdir(subfolder_path)
        
        for file in files:
            if file.startswith('.'):
                continue
            
            # Get base name (without extension)
            base_name = None
            for ext in ['.mha', '.nii.gz', '.nii', '.vti', '.vtp', '.stl', '.mhd']:
                if file.endswith(ext):
                    base_name = file.replace(ext, '')
                    break
            
            if base_name is None:
                logger.warning(f"Unknown file extension for {file}, skipping")
                continue
            
            # Determine if this case is in train or test
            if base_name in train_cases:
                dest_folder = os.path.join(train_base, subfolder)
            elif base_name in test_cases:
                dest_folder = os.path.join(test_base, subfolder)
            else:
                logger.warning(f"Case {base_name} not found in train or test sets, skipping")
                continue
            
            src_path = os.path.join(subfolder_path, file)
            dest_path = os.path.join(dest_folder, file)
            
            if copy_files:
                shutil.copy2(src_path, dest_path)
            else:
                shutil.move(src_path, dest_path)
        
        logger.info(f"  {operation.lower()} {subfolder}/ files...")
    
    # Save case lists
    train_list_path = os.path.join(output_folder, 'train_cases.txt')
    test_list_path = os.path.join(output_folder, 'test_cases.txt')
    
    with open(train_list_path, 'w') as f:
        for case in sorted(train_cases):
            f.write(f"{case}\n")
    
    with open(test_list_path, 'w') as f:
        for case in sorted(test_cases):
            f.write(f"{case}\n")
    
    logger.info(f"Saved case lists to {output_folder}/")
    logger.info(f"Split complete! Output folder: {output_folder}")
    
    return train_cases, test_cases


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Split dataset into training and testing sets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Split with 80% training, 20% testing (copy files)
  python preprocessing/split_train_test.py --folder /path/to/data --train_split 0.8
  
  # Use only cases present in images AND truths (more cases if surfaces has fewer)
  python preprocessing/split_train_test.py --folder /path/to/data --train_split 0.8 --subfolders images truths
  
  # Split with 70% training, 30% testing (move files)
  python preprocessing/split_train_test.py --folder /path/to/data --train_split 0.7 --move
  
  # Split with custom output folder and random seed
  python preprocessing/split_train_test.py --folder /path/to/data --train_split 0.8 --output /path/to/output --seed 42
        """
    )
    
    parser.add_argument('--folder', '--input_folder', '--input-folder',
                       type=str,
                       required=True,
                       help='Path to folder containing subfolders (images, centerlines, surfaces, truths, labels)')
    parser.add_argument('--subfolders',
                       type=str,
                       nargs='+',
                       default=None,
                       help='Subfolders that must all contain a case for it to be included (default: all existing). '
                            'E.g. --subfolders images truths to use only cases present in both')
    parser.add_argument('--train_split', '--train-split',
                       type=float,
                       required=True,
                       help='Fraction of data for training (0.0 to 1.0), e.g., 0.8 for 80%% training')
    parser.add_argument('--output', '--output_folder', '--output-folder',
                       type=str,
                       default=None,
                       help='Output folder for train/test splits (default: <input_folder>_split)')
    parser.add_argument('--seed',
                       type=int,
                       default=None,
                       help='Random seed for reproducibility (optional)')
    parser.add_argument('--move',
                       action='store_true',
                       default=False,
                       help='Move files instead of copying (default: copy files)')
    
    args = parser.parse_args()
    
    # Initialize logger
    logger = get_logger(__name__)
    
    # Validate input folder
    if not os.path.exists(args.folder):
        raise ValueError(f"Input folder not found: {args.folder}")
    
    # Validate train_split
    if not 0 < args.train_split < 1:
        raise ValueError(f"train_split must be between 0 and 1, got {args.train_split}")
    
    # Perform split
    try:
        train_cases, test_cases = split_dataset(
            base_folder=args.folder,
            train_split=args.train_split,
            output_folder=args.output,
            seed=args.seed,
            copy_files=not args.move,
            subfolders=args.subfolders
        )
        
        logger.info("=" * 80)
        logger.info("Split summary:")
        logger.info(f"  Total cases: {len(train_cases) + len(test_cases)}")
        logger.info(f"  Training cases: {len(train_cases)}")
        logger.info(f"  Testing cases: {len(test_cases)}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error during split: {e}")
        raise

