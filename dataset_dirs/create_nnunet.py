import os
import shutil
import sys
import argparse

sys.stdout.flush()
sys.path.insert(0, '../..')
sys.path.insert(0, '..')
# from modules import io


SUPPORTED_EXTENSIONS = ('.nii.gz', '.nrrd')


def _get_extension(filename):
    """Return the file extension (.nii.gz or .nrrd) or None if not supported."""
    if filename.endswith('.nii.gz'):
        return '.nii.gz'
    if filename.endswith('.nrrd'):
        return '.nrrd'
    return None


def _get_base_name(filename):
    """Strip extension to get base name for matching image/label pairs."""
    ext = _get_extension(filename)
    if ext:
        return filename[:-len(ext)]
    return filename


def _filter_by_extensions(filelist):
    """Filter file list to supported extensions and return (filtered_list, detected_extension)."""
    filtered = [f for f in filelist if _get_extension(f) is not None]
    ext = _get_extension(filtered[0]) if filtered else '.nii.gz'
    return filtered, ext


def save_json(data, filename):
    """
    Save json file
    Args:
        data: data to save
        filename: filename
    Returns:
        json file
    """
    import json
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    """
    This script is used to create the nnUNet dataset names
    from a dataset that has the following structure:

    Format 1 (modality-based):
    ├── ct_train
    ├── ct_train_masks
    ├── ct_test
    ├── ct_test_masks

    Format 2 (images/labels):
    ├── images
    └── labels

    Format 3 (images/truths):
    ├── images
    └── truths

    (or mr instead of ct for Format 1)
    Supports both .nii.gz and .nrrd file formats.

    Into a dataset that has the following structure:

    Dataset
    ├── imagesTr
    ├── imagesTs
    └── labelsTr

    Example command:

    python3 dataset_dirs/create_nnunet.py \
            -outdir /Users/numisveins/Documents/datasets/ASOCA_dataset/global_trainset/ \
            -indir /Users/numisveins/Documents/datasets/ASOCA_dataset/global_trainset/ \
            -name AORTAS \
            -dataset_number 1 \
            -modality ct \
            -start_from 0

    example dataset.json
    dataset_json = { 
            "channel_names": {
                "0": "CT"
            },
            "labels": {
                "background": 0,
                "vessel": 1
            },
            "numTraining": 35362, 
            "file_ending": ".nii.gz"
            }
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-outdir', '--outdir',
                        type=str,
                        help='Output directory')
    parser.add_argument('-indir', '--indir',
                        type=str,
                        help='Input directory')
    parser.add_argument('-name', '--name',
                        default='AORTAS',
                        type=str,
                        help='Dataset name')
    parser.add_argument('-dataset_number', '--dataset_number',
                        default=1,
                        type=int,
                        help='Dataset number')
    parser.add_argument('-modality', '--modality',
                        type=str,
                        help='Modality')
    parser.add_argument('-start_from', '--start_from',
                        type=int,
                        default=0,
                        help='Number to start dataset from, use if adding to existing dataset')
    args = parser.parse_args()

    # global_config_file = "./config/global.yaml"
    # global_config = io.load_yaml(global_config_file)
    # modalities = global_config['MODALITY']

    directory = args.indir  # '/global/scratch/users/numi/vascular_data_3d/extraction_output/nnunet_only_one_aorta_vmr/'
    directory_out = args.outdir  # '/global/scratch/users/numi/vascular_data_3d/extraction_output/nnunet_only_one_aorta_vmr/'
    modality = args.modality  # 'mr' or 'ct'

    start_from = args.start_from  # 0 28707
    name = args.name  # 'SEQAORTASONE'
    dataset_number = args.dataset_number  # 1
    # make number 2 digits
    if dataset_number < 10:
        dataset_number = '0' + str(dataset_number)
    else:
        dataset_number = str(dataset_number)

    new_dir_dataset_name = 'Dataset0'+dataset_number+'_'+name+modality.upper()
    append = name.lower() + modality.lower()

    also_test = False

    out_data_dir = os.path.join(directory_out, new_dir_dataset_name)

    # create output directory if it doesn't exist
    os.makedirs(directory_out, exist_ok=True)

    # create new dataset directory
    try:
        os.mkdir(os.path.join(directory_out, new_dir_dataset_name))
    except FileExistsError:
        print(f'Directory {new_dir_dataset_name} already exists')

    # Detect input format: modality-based, images/labels, or images/truths
    if os.path.exists(os.path.join(directory, modality+'_train')) and os.path.exists(os.path.join(directory, modality+'_train_masks')):
        # Format 1: modality-based (ct_train, ct_train_masks)
        fns_in = [modality+'_train', modality+'_train_masks']
        if also_test:
            fns_in.extend([modality+'_test', modality+'_test_masks'])
        input_format = 'modality'
    elif os.path.exists(os.path.join(directory, 'images')) and os.path.exists(os.path.join(directory, 'labels')):
        # Format 2: images/labels
        fns_in = ['images', 'labels']
        input_format = 'images_labels'
    elif os.path.exists(os.path.join(directory, 'images')) and os.path.exists(os.path.join(directory, 'truths')):
        # Format 3: images/truths
        fns_in = ['images', 'truths']
        input_format = 'images_labels'
    else:
        raise FileNotFoundError(
            f"Could not find expected directory structure in {directory}. "
            f"Expected either: (1) {modality}_train and {modality}_train_masks, "
            "(2) images and labels, or (3) images and truths."
        )

    fns_out = ['imagesTr', 'labelsTr']
    if also_test and input_format == 'modality':
        fns_out.extend(['imagesTs', 'labelsTs'])

    for fn in fns_out:
        try:
            os.mkdir(os.path.join(directory_out, new_dir_dataset_name, fn))
        except FileExistsError:
            print(f'Directory {fn} already exists')

    file_ending = '.nii.gz'  # default
    name_mappings = []  # (nnunet_path, original_path) for .txt output

    if input_format == 'modality':
        # Original logic: each folder has its own file list, copy with sequential naming
        for fn in fns_in:
            if not os.path.exists(os.path.join(directory, fn)):
                print(f'{fn} does not exist')
                continue
            all_files = os.listdir(os.path.join(directory, fn))
            imgs, file_ext = _filter_by_extensions(all_files)
            imgs.sort()

            out_subfolder = fns_out[fns_in.index(fn)]
            for i, img in enumerate(imgs):
                ext = _get_extension(img)
                new_name = f'{append}_{(i+1+start_from):03d}_0000{ext}'
                if out_subfolder == 'labelsTr' or out_subfolder == 'labelsTs':
                    new_name = new_name.replace('_0000', '')
                name_mappings.append((f'{out_subfolder}/{new_name}', f'{fn}/{img}'))
                print(f'Copying {img} to {new_name}')
                if img != new_name:
                    shutil.copy(os.path.join(directory, fn, img), os.path.join(out_data_dir, out_subfolder, new_name))
            if fn == fns_in[0]:  # count from first (images) folder
                num_training = len(imgs)
                file_ending = file_ext
    else:
        # images/labels or images/truths: match by filename, copy with nnUNet naming
        images_dir = os.path.join(directory, 'images')
        labels_dir = os.path.join(directory, fns_in[1])  # 'labels' or 'truths'
        all_files = os.listdir(images_dir)
        imgs, file_ext = _filter_by_extensions(all_files)
        imgs.sort()
        file_ending = file_ext

        copy_count = 0
        for img in imgs:
            ext = _get_extension(img)
            base = _get_base_name(img)
            label_name = base + ext
            label_path = os.path.join(labels_dir, label_name)
            if not os.path.exists(label_path):
                print(f'Warning: no matching label for {img}, skipping')
                continue
            copy_count += 1
            idx = copy_count + start_from
            img_new = f'{append}_{idx:03d}_0000{ext}'
            label_new = f'{append}_{idx:03d}{ext}'
            name_mappings.append((f'imagesTr/{img_new}', f'images/{img}'))
            name_mappings.append((f'labelsTr/{label_new}', f'{fns_in[1]}/{label_name}'))
            print(f'Copying {img} to {img_new}')
            shutil.copy(os.path.join(images_dir, img), os.path.join(out_data_dir, 'imagesTr', img_new))
            print(f'Copying {label_name} to {label_new}')
            shutil.copy(label_path, os.path.join(out_data_dir, 'labelsTr', label_new))
        num_training = copy_count

    # Create a dataset.json file
    dataset_json = {
        "channel_names": {
            "0": modality.upper()
        },
        "labels": {
            "background": 0,
            "vessel": 1
        },
        "numTraining": num_training,
        "file_ending": file_ending
        }
    # Save dataset.json
    save_json(dataset_json, os.path.join(out_data_dir, 'dataset.json'))

    # Save name mapping (nnunet name -> original name)
    mapping_path = os.path.join(out_data_dir, 'name_mapping.txt')
    with open(mapping_path, 'w') as f:
        f.write('# nnunet_name -> original_path\n')
        for nnunet_path, original_path in name_mappings:
            f.write(f'{nnunet_path} -> {original_path}\n')
