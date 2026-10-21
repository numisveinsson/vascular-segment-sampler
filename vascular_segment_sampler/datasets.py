import os

import pandas as pd

# Longest-first so .nii.gz is matched before .nii.
IMAGE_EXTS = ('.nii.gz', '.mha', '.nrrd', '.nii', '.vti', '.vtk', '.mhd')


def _case_id_from_volume_filename(filename, img_ext=None):
    """Return case id by stripping a known (or given) volume suffix, else None."""
    name = os.path.basename(filename)
    if name.startswith('.'):
        return None
    suffixes = (img_ext,) if img_ext else IMAGE_EXTS
    for ext in sorted(suffixes, key=len, reverse=True):
        if ext and name.endswith(ext):
            return name[:-len(ext)]
    return None


def _resolve_volume_path(directory, subdir, model, img_ext=None):
    """Path to images/truths file for a case; probe known suffixes if img_ext is omitted."""
    folder = os.path.join(directory, subdir)
    if img_ext:
        return os.path.join(folder, model + img_ext)
    for ext in IMAGE_EXTS:
        path = os.path.join(folder, model + ext)
        if os.path.isfile(path):
            return path
    sibling = 'truths' if subdir == 'images' else 'images'
    for ext in IMAGE_EXTS:
        if os.path.isfile(os.path.join(directory, sibling, model + ext)):
            return os.path.join(folder, model + ext)
    return os.path.join(folder, model + '.mha')


def create_dataset(global_config, modality):

    dataset_name = global_config['DATASET_NAME']
    print(f'Reading in dataset')
    print(f'Make sure the dataset is in the correct directory: {global_config["DATA_DIR"]}\n')
    print(f'Dataset name: {dataset_name}')
    print(f'Subfolders need to be:\n images\n truths\n surfaces\n centerlines')

    test_cases = global_config.get('TEST_CASES') or []
    bad_cases = global_config.get('BAD_CASES') or []

    if dataset_name == 'vmr':
        img_mods = global_config.get('VMR_IMAGE_MODALITIES')
        Dataset = VMR_dataset(
            global_config['DATA_DIR'],
            [modality],
            global_config['ANATOMY'],
            image_modalities=img_mods,
        )
        cases = Dataset.sort_cases(global_config['TESTING'], test_cases)
        cases = Dataset.check_which_cases_in_image_dir(cases)
        cases = [f for f in cases if f not in bad_cases]
    elif dataset_name == 'other':
        cases = get_dataset_cases(
            global_config['DATA_DIR'],
            global_config.get('IMG_EXT'),
            test_cases,
            global_config['TESTING'],
        )
    else:
        print("Dataset not found")
        exit()

    return cases


def get_dataset_cases(directory, img_ext=None, test_cases=None, testing=False):

    """
    Returns a list of cases in the dataset
    Input: 
        directory (str)
        img_ext (str or None): optional suffix filter; if omitted, any IMAGE_EXTS
        test_cases (list of str)
    Output: cases (list of str)
    """
    if test_cases is None:
        test_cases = []
    directory = os.path.abspath(os.path.expanduser(directory))
    cases_im = []
    for f in os.listdir(os.path.join(directory, 'images')):
        cid = _case_id_from_volume_filename(f, img_ext)
        if cid is not None:
            cases_im.append(cid)
    if img_ext:
        print(f"Found {len(cases_im)} images with extension {img_ext}")
    else:
        print(f"Found {len(cases_im)} images")

    cases_cent = []
    for f in os.listdir(os.path.join(directory, 'centerlines')):
        # Skip AppleDouble / hidden files (``._case.vtp``). They are not cases.
        if f.startswith('.') or not f.endswith('.vtp'):
            continue
        cases_cent.append(f[: -len('.vtp')])
    print(f"Found {len(cases_cent)} centerlines")

    # if surface folder exists
    if os.path.exists(os.path.join(directory, 'surfaces')):
        cases_surf = []
        for f in os.listdir(os.path.join(directory, 'surfaces')):
            if f.startswith('.') or not f.endswith('.vtp'):
                continue
            stem = f[: -len('.vtp')]
            if stem.endswith('.seg'):
                stem = stem[: -len('.seg')]
            cases_surf.append(stem)
        print(f"Found {len(cases_surf)} surfaces")
    else:
        print("No surfaces found")
        cases_surf = []

    if testing:
        test_cases_w_img = [f for f in test_cases if f in cases_im]
        print(f"Found {len(test_cases_w_img)} test cases in cases dataset, returning them")
        # sort
        test_cases_w_img.sort()
        return test_cases_w_img
    else:
        # Image and centerline must share a stem. Extra .vtp files (and
        # ``._*.vtp`` copies) are not cases; reading them crashes whole-volume
        # gathering on a missing image.
        cent_set = set(cases_cent)
        im_set = set(cases_im)
        cases = [f for f in cases_im if f in cent_set and f not in test_cases]
        print(f"Found {len(cases)} training cases with image and centerline")
        images_without_cent = sorted(im_set - cent_set)
        cents_without_image = sorted(cent_set - im_set)
        if images_without_cent:
            print("Images with no centerline (skipped):")
            print(images_without_cent)
        if cents_without_image:
            print("Centerlines with no image (skipped):")
            print(cents_without_image)
        test_cases_removed = sorted(im_set & cent_set & set(test_cases))
        print(f"Found {len(test_cases_removed)} test cases, removing them")
        cases.sort()
        print(f"Returning {len(cases)} cases")
        return cases

class VMR_dataset:
    """
    Class for VMR dataset
    Assumes the data directory is organized as follows:
        Images (raw image data)
            Scaled Images (globally scaled images 0-1)
            Cropped Images (cropped around segmentation mask)
        Truths (segmentation masks)
        Surfaces (surface meshes)
        Centerlines (centerline meshes)
    """
    def __init__(self, directory, modality=None, anatomy=None, image_modalities=None):
        self.dir = os.path.abspath(os.path.expanduser(directory))
        self.modality = modality
        self.anatomy = anatomy
        try:
            self.df = get_vmr_dataset_names()
        except Exception as e:
            print(f'Could not get VMR dataset names from google sheet: {e}')
            print('Attempting to read from local CSV file...')
            try:
                self.df = get_vmr_dataset_names_local()
                print('Successfully loaded VMR dataset names from local CSV')
            except Exception as e2:
                print(f'Could not read local CSV file: {e2}')
                self.df = None
        # get the cases of specified modality and anatomy
        mod_list = image_modalities if image_modalities is not None else modality
        if mod_list is not None:
            self.df = get_vmr_names_modality(self.df, mod_list)
        if anatomy is not None:
            self.df = get_vmr_names_anatomy(self.df, anatomy)

    def sort_cases(self, testing, test_cases):
        """
        Sort the cases into train+val and test
        Input: test_cases (list of str)
        Output: cases (list of str)
        """
        if not testing:
            cases = self.df['Legacy Name'].values
            cases = [f for f in cases if f not in test_cases]
            cases.sort()
        else:
            cases = [f for f in self.df['Legacy Name'].values if f in test_cases]

        self.cases = cases
        print(f"Cases found from url: {cases}")

        return cases
    
    def check_which_cases_in_image_dir(self, cases_list):
        """
        Check which cases are in the data directory (Images)
        """
        img_dir = os.path.join(self.dir, 'images')
        images = []
        for f in os.listdir(img_dir):
            cid = _case_id_from_volume_filename(f)
            if cid is not None:
                images.append(cid)
        images = sorted(set(images))

        cases = [f for f in cases_list if f in images]

        # print the cases not in image dir
        print('Cases not in image dir:')
        imgs_mis = [f for f in cases_list if f not in images]
        for img in imgs_mis:
            print(img)

        return cases

    def compare_image_dir_and_csv(self):
        """
        Check which cases are in the data directory (Images)
        """
        img_dir = os.path.join(self.dir, 'images')
        images = []
        for f in os.listdir(img_dir):
            cid = _case_id_from_volume_filename(f)
            if cid is not None:
                images.append(cid)
        images = sorted(set(images))

        self.image_cases = images

        # check which cases are in the google sheet
        if self.df is not None:
            print('Cases in google sheet:')
            print(self.df['Legacy Name'].values)
            print(self.df['Name'].values)
        else:
            print('No google sheet dataframe found')
        # print the cases that are in image dir but not in google sheet
        print('Cases in image dir but not in google sheet:')
        print([f for f in images if f not in self.df['Legacy Name'].values and f not in self.df['Name'].values])
        # print the cases that are in google sheet but not in image dir
        print('Cases in google sheet but not in image dir:')
        print([f for f in self.df['Legacy Name'].values if f not in images and f not in self.df['Name'].values])

        # return list of cases in image dir that are in the google sheet
        return images
        
def resolve_case_surface_path(directory, model):
    """
    Return path to surfaces/{model}.vtp or surfaces/{model}.stl if it exists, else None.
    Prefer .vtp when both exist.
    """
    import os
    surf_dir = os.path.join(directory, 'surfaces')
    for ext in ('.vtp', '.stl'):
        p = os.path.join(surf_dir, model + ext)
        if os.path.exists(p):
            return p
    return None


def get_case_dict_dir(directory, model, img_ext=None):
    """
    Get the directories of the case
    Input: model (str)
    Output: dict of directories
    """
    dir_image, dir_seg, dir_cent, dir_surf = directories(directory, model, img_ext)
    dir_dict = {'NAME': model,
                'IMAGE': dir_image,
                'SEGMENTATION': dir_seg,
                'CENTERLINE': dir_cent,
                'SURFACE': dir_surf}
    return dir_dict

def build_sheet_url():
    doc_id = '1NCyC18ot2Vht5LlFX7Hq-g0jaLLwf5wirBcqOnWWn8s'
    sheet_id = 'svprojects'
    return f'https://docs.google.com/spreadsheets/d/{doc_id}/gviz/tq?tqx=out:csv&sheet={sheet_id}'

def write_df_to_local(df):
    file_path = './VMR_dataset_names.csv'
    df.to_csv(file_path)

def get_vmr_dataset_names():
    """
    Returns a pandas dataframe of the VMR dataset names
    """
    url = build_sheet_url()
    df = pd.read_csv(url)
    df = df[df['Legacy Name'].notna()]
    return df

def get_vmr_dataset_names_local():
    """
    Returns a pandas dataframe of the VMR dataset names from local file
    """
    import os
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'VMR_dataset_names.csv')
    df = pd.read_csv(file_path)
    df = df[df['Legacy Name'].notna()]
    return df

def get_vmr_names_modality(df, modality):
    """
    Returns a pandas dataframe of the VMR dataset names
    Input: modality (list of str)
    Possible values:
    'CT' (includes sheet rows labeled CTA)
    'MR'
    """
    modal = list(modality)
    if "CT" in modal and "CTA" not in modal:
        modal = modal + ["CTA"]
    return df[df["Image Modality"].isin(modal)]

def get_vmr_names_anatomy(df, anatomy):
    """
    Returns a pandas dataframe of the VMR dataset names
    Input: anatomy (list of str)
    Possible values: 
    'Aorta'
    'Abdominal Aorta'
    'Coronary'
    'Cerebral'
    'Pulmonary Fontan'
    'Pulmonary'
    'Pulmonary Glenn'
    """
    return df[df['Anatomy'].isin(anatomy)]

def get_vmr_names_disease(df, disease):
    """
    Returns a pandas dataframe of the VMR dataset names
    Input: disease (list of str)
    """
    return df[df['Disease'].isin(disease)]

def get_vmr_names_disease(df, image_mod):
    """
    Returns a pandas dataframe of the VMR dataset names
    Input: image_mod (list of str)
    """
    return df[df['Image Modality'].isin(image_mod)]

def directories(directory, model, img_ext=None):
    """
    Function to return the directories of
        Image Volume
        Segmentation Volume
        Centerline VTP
        Surface Mesh VTP
    """
    directory = os.path.abspath(os.path.expanduser(directory))
    dir_image = _resolve_volume_path(directory, 'images', model, img_ext)
    dir_seg = _resolve_volume_path(directory, 'truths', model, img_ext)
    dir_cent = os.path.join(directory, 'centerlines', model + '.vtp')
    dir_surf = os.path.join(directory, 'surfaces', model + '.vtp')

    return dir_image, dir_seg, dir_cent, dir_surf

def vmr_directories_old(directory, model, global_scale, dir_seg_exist, cropped, original):
    """
    Function to return the directories of
        Image Volume
        Segmentation Volume
        Centerline VTP
        Surface Mesh VTP
    for a specific model in the
    Vascular Model Repository
    """
    if not cropped:
        if global_scale:
            dir_image = directory +'scaled_images/'+model.replace('_aorta','')+'.vtk'
            dir_seg = directory +'truths/'+model.replace('_aorta','')+'.vtk'
        else:
            if original:
                dir_image = directory +'images/OSMSC' + model[0:4]+'/OSMSC'+model[0:4]+'-cm.mha'
                dir_seg = directory + 'images/OSMSC'+model[0:4]+'/'+model+'/'+model+'-cm.mha'
            else:
                dir_image = directory +'images/'+model.replace('_aorta','')+'.vtk'
                dir_seg = directory +'truths/'+model.replace('_aorta','')+'.vtk'
    else:
        dir_image = directory +'images/OSMSC' + model[0:4]+'/OSMSC'+model[0:4]+'-cm.mha'

    if not dir_seg_exist:
        dir_seg = None

    dir_cent = directory + 'centerlines/'+model+'.vtp'
    dir_surf = directory + 'surfaces/'+model+'.vtp'

    return dir_image, dir_seg, dir_cent, dir_surf

if __name__ == '__main__':

    df = get_vmr_dataset_names()
    # keep only where 'Legacy Name' is not NaN
    
    dataset = VMR_dataset('/Users/numisveins/Documents/vascular_data_3d/')
    import pdb; pdb.set_trace()
    dataset.check_which_cases_in_image_dir()
