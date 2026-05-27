from datetime import datetime

import argparse
import importlib.util
import os
import random
import subprocess
import SimpleITK as sitk
import sys
sys.path.insert(0, './')

from modules import io
from modules.sampling_functions import *
from modules import sitk_functions as sf
from dataset_dirs.datasets import *
from preprocessing.change_img_resample import resample_image

now = datetime.now()
dt_string = now.strftime("_%d_%m_%Y_%H_%M_%S")
_CREATE_SEG_SURF_MOD = None


def _get_create_seg_surf_mod():
    """Load global/create_seg_from_surf.py once (directory name `global` is not a package)."""
    global _CREATE_SEG_SURF_MOD
    if _CREATE_SEG_SURF_MOD is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'create_seg_from_surf.py')
        spec = importlib.util.spec_from_file_location('create_seg_from_surf', path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _CREATE_SEG_SURF_MOD = mod
    return _CREATE_SEG_SURF_MOD


def _write_text_lines(filepath, lines):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
        if lines:
            f.write('\n')


if __name__ == '__main__':
    """
    Does same as gather_sampling_data_parallel.py but for global data
    So based on config/global.yaml, creates folders for training and testing
    Similarly, creates folders based on modalities
    This data then needs to be post-processed if to be used for eg nnUNet training
    
    Example:

    python3 global/global_gather_data.py \
            -outdir /Users/numisveins/Documents/datasets/ASOCA_dataset/global_trainset/ \
            -config_name global_original \
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-outdir', '--outdir',
                        type=str,
                        help='Output directory')
    parser.add_argument('--input_dir', '--input-dir',
                        type=str,
                        default=None,
                        help='Optional input dataset root directory. Overrides DATA_DIR in the config file.')
    parser.add_argument('-config_name', '--config_name',
                        default='global',
                        type=str,
                        help='Name of configuration file')
    parser.add_argument('-perc_dataset', '--perc_dataset',
                        default=1.0,
                        type=float,
                        help='Percentage of dataset to use')
    parser.add_argument('-testing', '--testing',
                        action='store_true',
                        help='Enable testing mode (uses TEST_CASES instead of training cases)')
    parser.add_argument('-validation_prop', '--validation_prop',
                        type=float,
                        default=None,
                        help='Validation set proportion (0.0-1.0). If not provided, uses config value.')
    parser.add_argument('--target_spacing', '--target-spacing',
                        type=float,
                        nargs=3,
                        default=None,
                        metavar=('SX', 'SY', 'SZ'),
                        help='Optional target spacing [sx sy sz] in mm. '
                             'If set, image is resampled to this spacing and segmentation '
                             'is regenerated from surface on that grid (no segmentation resampling).')
    parser.add_argument('--convert_nnunet', '--convert-nnunet',
                        action='store_true',
                        help='If set, run dataset_dirs/create_nnunet.py after writing global samples.')
    parser.add_argument('--nnunet_indir', '--nnunet-indir',
                        type=str,
                        default=None,
                        help='Input directory for nnU-Net conversion. Defaults to --outdir.')
    parser.add_argument('--nnunet_outdir', '--nnunet-outdir',
                        type=str,
                        default=None,
                        help='Output directory for nnU-Net conversion. Defaults to --outdir.')
    parser.add_argument('--nnunet_name', '--nnunet-name',
                        type=str,
                        default='AORTAS',
                        help='Dataset name prefix for nnU-Net conversion.')
    parser.add_argument('--nnunet_dataset_number', '--nnunet-dataset-number',
                        type=int,
                        default=1,
                        help='Dataset number for nnU-Net conversion.')
    parser.add_argument('--nnunet_start_from', '--nnunet-start-from',
                        type=int,
                        default=0,
                        help='Start index for nnU-Net naming (useful when appending).')
    args = parser.parse_args()

    print(args)

    global_config = io.load_yaml("./config/"+args.config_name+".yaml")
    if args.input_dir is not None:
        global_config['DATA_DIR'] = args.input_dir
    if args.testing:
        global_config['TESTING'] = True
    elif 'TESTING' not in global_config:
        global_config['TESTING'] = False
    if args.validation_prop is not None:
        global_config['VALIDATION_PROP'] = args.validation_prop
    elif 'VALIDATION_PROP' not in global_config:
        global_config['VALIDATION_PROP'] = 0.0
    modalities = global_config['MODALITY']

    out_dir = os.path.abspath(args.outdir)
    if not out_dir.endswith(os.sep):
        out_dir = out_dir + os.sep
    global_config['OUT_DIR'] = out_dir
    output_suffix = global_config.get('OUTPUT_SUFFIX', '')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # if not global_config['TESTING']:
    #     test_vars = [False]
    # else:
    #     test_vars = [False, True]

    # for testing in test_vars:

    testing = global_config['TESTING']

    for modality in modalities:

        cases = create_dataset(global_config, modality)

        # sort
        cases = sorted(cases)

        # shuffle cases
        # set random seed
        random.seed(42)
        random.shuffle(cases)
        cases_shuffled = list(cases)
        n_keep = int(args.perc_dataset * len(cases_shuffled))
        cases_after_perc = cases_shuffled[:n_keep]
        cases_excluded_perc = cases_shuffled[n_keep:]

        done_file_path = os.path.join(out_dir, f"done{output_suffix}.txt")
        if os.path.exists(done_file_path):
            with open(done_file_path, "r", encoding='utf-8') as f:
                done = f.read().splitlines()
            done_set = set(done)
            skipped_done = [c for c in cases_after_perc if c in done_set]
            for case in skipped_done:
                print(f"Skipping {case}")
            cases = [c for c in cases_after_perc if c not in done_set]
        else:
            skipped_done = []
            cases = cases_after_perc

        modality = modality.lower()
        info_file_name = "info" + '_' + modality + dt_string + output_suffix + ".txt"

        if cases_excluded_perc:
            _write_text_lines(
                os.path.join(out_dir, f"cases_excluded_perc_dataset_{modality}{dt_string}{output_suffix}.txt"),
                cases_excluded_perc,
            )
        if skipped_done:
            _write_text_lines(
                os.path.join(out_dir, f"cases_skipped_done_{modality}{dt_string}{output_suffix}.txt"),
                skipped_done,
            )
        _write_text_lines(
            os.path.join(out_dir, f"cases_in_run_{modality}{dt_string}{output_suffix}.txt"),
            cases,
        )

        create_directories(out_dir, modality, global_config)

        image_out_dir_train = os.path.join(out_dir, modality+'_train')
        seg_out_dir_train = os.path.join(out_dir, modality+'_train_masks')
        image_out_dir_val = os.path.join(out_dir, modality+'_val')
        seg_out_dir_val = os.path.join(out_dir, modality+'_val_masks')

        image_out_dir_test = os.path.join(out_dir, modality+'_test')
        seg_out_dir_test = os.path.join(out_dir, modality+'_test_masks')

        # cases = Dataset.sort_cases(testing, global_config['TEST_CASES'])
        # cases = Dataset.check_which_cases_in_image_dir(cases)

        for i in cases:
            print(f"Case: {i}")

        print_info_file(global_config, cases, global_config['TEST_CASES'], info_file_name)

        for i, case_fn in enumerate(cases):

            # Load data
            case_dict = get_case_dict_dir(global_config['DATA_DIR'], case_fn, global_config['IMG_EXT'])
            print(f"\n {i+1}/{len(cases)}: {case_dict['NAME']}")

            name = case_dict['NAME']
            # Choose destination directory
            image_out_dir, seg_out_dir, val_port = choose_destination(testing, global_config['VALIDATION_PROP'], image_out_dir_test, seg_out_dir_test, 
                                                                        image_out_dir_val, seg_out_dir_val, image_out_dir_train, seg_out_dir_train, ip = None)
            
            reader_im, origin_im, size_im, spacing_im = sf.import_image(case_dict['IMAGE'])

            # Load image and segmentation
            img = sitk.ReadImage(case_dict['IMAGE'])
            if args.target_spacing is not None:
                # Match main.py behavior: regenerate truth from surface on the target-spacing grid.
                data_dir_ts = os.path.abspath(os.path.expanduser(global_config['DATA_DIR']))
                surf_path = resolve_case_surface_path(data_dir_ts, case_dict['NAME'])
                if surf_path is None:
                    raise FileNotFoundError(
                        f"No surface mesh found for case {case_dict['NAME']} under "
                        f"{os.path.join(data_dir_ts, 'surfaces')} (.vtp or .stl)"
                    )
                mod = _get_create_seg_surf_mod()
                surface = mod.load_surface_polydata(surf_path)
                # Build the exact image grid first, then rasterize truth on this same grid.
                img = resample_image(img, target_spacing=list(args.target_spacing), order=3)
                seg = mod.seg_sitk_from_surface_polydata(
                    surface,
                    img,
                    target_spacing=None,
                    resample_order=3,
                )
                size_im = img.GetSize()
            else:
                data_dir = os.path.abspath(os.path.expanduser(global_config['DATA_DIR']))
                seg = None
                truths_dir = os.path.join(data_dir, 'truths')
                if os.path.isdir(truths_dir):
                    seg_path = case_dict['SEGMENTATION']
                    if os.path.isfile(seg_path):
                        try:
                            seg = sitk.ReadImage(seg_path)
                            seg = sitk.Cast(seg, sitk.sitkUInt8)
                        except Exception as e:
                            print(e)
                            seg = None
                if seg is None:
                    surf_path = resolve_case_surface_path(data_dir, case_dict['NAME'])
                    if surf_path is None:
                        raise FileNotFoundError(
                            f"No usable truth for case {case_dict['NAME']}: "
                            f"truths/ missing or empty, truth file missing or unreadable, and no surface under "
                            f"{os.path.join(data_dir, 'surfaces')} (.vtp or .stl)"
                        )
                    print(
                        f"{case_dict['NAME']}: segmentation from surface at native image spacing "
                        f"(truths dir or volume unavailable)"
                    )
                    mod = _get_create_seg_surf_mod()
                    surface = mod.load_surface_polydata(surf_path)
                    seg = mod.seg_sitk_from_surface_polydata(
                        surface,
                        img,
                        target_spacing=None,
                        resample_order=3,
                    )

            # check max and min values of img and seg
            max_val = sitk.GetArrayFromImage(img).max()
            min_val = sitk.GetArrayFromImage(img).min()
            print('Img Max value: ', max_val)
            print('Img Min value: ', min_val)
            if seg is not None:
                max_val = sitk.GetArrayFromImage(seg).max()
                min_val = sitk.GetArrayFromImage(seg).min()
                print('Seg Max value: ', max_val)
                print('Seg Min value: ', min_val)
                seg = sitk.Cast(seg, sitk.sitkUInt8)

            # If seg max value is over 1, then divide by max value
            if max_val > 1 and seg is not None and global_config['BINARIZE']:
                seg_np = sitk.GetArrayFromImage(seg)
                seg_np = seg_np/max_val
                seg = sitk.GetImageFromArray(seg_np)
                seg.SetSpacing(img.GetSpacing())
                seg.SetOrigin(img.GetOrigin())
                seg.SetDirection(img.GetDirection())
                print('Seg Max value: ', seg_np.max())
                print('Seg Min value: ', seg_np.min())
                seg = sitk.Cast(seg, sitk.sitkUInt8)

            if global_config['WRITE_SAMPLES']:
                sitk.WriteImage(img, os.path.join(image_out_dir, case_dict['NAME']+'.nii.gz'))
                if seg is not None:
                    sitk.WriteImage(seg, os.path.join(seg_out_dir, case_dict['NAME']+'.nii.gz'))
            if global_config['WRITE_VTK']:
                sitk.WriteImage(img, os.path.join(out_dir, 'vtk_data', 'vtk_'+case_dict['NAME']+'.mha'))
                if seg is not None:
                    sitk.WriteImage(seg*255, os.path.join(out_dir, 'vtk_data', 'vtk_mask_'+case_dict['NAME']+'.mha'))

            print(f"\n Finished: ' {case_dict['NAME']}, {size_im}")

            with open(done_file_path, "a", encoding='utf-8') as f_done:
                f_done.write(case_dict['NAME'] + '\n')

        if args.convert_nnunet:
            nnunet_indir = args.nnunet_indir or out_dir
            nnunet_outdir = args.nnunet_outdir or out_dir
            cmd = [
                sys.executable,
                os.path.join(os.path.dirname(__file__), '..', 'dataset_dirs', 'create_nnunet.py'),
                '--indir', nnunet_indir,
                '--outdir', nnunet_outdir,
                '--name', args.nnunet_name,
                '--dataset_number', str(args.nnunet_dataset_number),
                '--modality', modality,
                '--start_from', str(args.nnunet_start_from),
            ]
            print("Running nnU-Net conversion:", " ".join(cmd))
            subprocess.run(cmd, check=True)