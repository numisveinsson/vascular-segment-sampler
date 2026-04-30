import os

import pandas as pd


def create_dataset(global_config, modality):

    dataset_name = global_config['DATASET_NAME']
    print(f'Reading in dataset')
    print(f'Make sure the dataset is in the correct directory: {global_config["DATA_DIR"]}\n')
    print(f'Dataset name: {dataset_name}')
    print(f'Subfolders need to be:\n images\n truths\n surfaces\n centerlines')

    if dataset_name == 'vmr':
        img_mods = global_config.get('VMR_IMAGE_MODALITIES')
        Dataset = VMR_dataset(
            global_config['DATA_DIR'],
            [modality],
            global_config['ANATOMY'],
            image_modalities=img_mods,
        )
        cases = Dataset.sort_cases(global_config['TESTING'], global_config['TEST_CASES'])
        cases = Dataset.check_which_cases_in_image_dir(cases)
        cases = [f for f in cases if f not in global_config['BAD_CASES']]
    elif dataset_name == 'other':
        cases = get_dataset_cases(global_config['DATA_DIR'], global_config['IMG_EXT'], global_config['TEST_CASES'], global_config['TESTING'])
    else:
        print("Dataset not found")
        exit()

    return cases


def get_dataset_cases(directory, img_ext, test_cases, testing=False):

    """
    Returns a list of cases in the dataset
    Input: 
        directory (str)
        img_ext (str)
        test_cases (list of str)
    Output: cases (list of str)
    """
    directory = os.path.abspath(os.path.expanduser(directory))
    cases_im = os.listdir(os.path.join(directory, 'images'))
    cases_im = [f for f in cases_im if f.endswith(img_ext)]
    cases_im = [f.replace(img_ext, '') for f in cases_im]
    print(f"Found {len(cases_im)} images with extension {img_ext}")

    cases_cent = os.listdir(os.path.join(directory, 'centerlines'))
    cases_cent = [f for f in cases_cent if f.endswith('.vtp')]
    cases_cent = [f.replace('.vtp', '') for f in cases_cent]
    print(f"Found {len(cases_cent)} centerlines")

    # if surface folder exists
    if os.path.exists(os.path.join(directory, 'surfaces')):
        cases_surf = os.listdir(os.path.join(directory, 'surfaces'))
        cases_surf = [f for f in cases_surf if f.endswith('.vtp')]
        cases_surf = [f.replace('.vtp', '') for f in cases_surf]
        cases_surf = [f.replace('.seg', '') for f in cases_surf]
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
        # only keep cases that have all files
        cases = [f for f in cases_im if f in cases_cent]
        print(f"Found {len(cases)} training cases with all files")
        print(f"Returning {len(cases_cent)} cases, that all have centerlines")
        print(f"The cases that have images but no centerlines are:")
        print([f for f in cases_im if f not in cases_cent])

        # remove the test cases
        test_cases_in_cent = [f for f in cases_cent if f in test_cases]
        print(f"Found {len(test_cases_in_cent)} test cases in cases dataset, removing them")
        cases_cent = [f for f in cases_cent if f not in test_cases]
        print(f"Returning {len(cases_cent)} cases, that have all centerlines, and are not test cases")
        # sort the cases, so they are in the same order
        cases_cent.sort()
        return cases_cent

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
        images = os.listdir(img_dir)
        # ignore hidden files
        images = [f for f in images if f[0] != '.']
        # remove the file extension
        images = [f.split('.')[0] for f in images]
        # remove duplicates
        images = list(set(images))
        # sort
        images.sort()

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
        images = os.listdir(img_dir)
        # ignore hidden files
        images = [f for f in images if f[0] != '.']
        # remove the file extension
        images = [f.split('.')[0] for f in images]
        # remove duplicates
        images = list(set(images))
        # sort
        images.sort()

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


def get_case_dict_dir(directory, model, img_ext):
    """
    Get the directories of the case
    Input: model (str)
    Output: dict of directories
    """
    dir_image, dir_seg, dir_cent, dir_surf = directories( directory, model, img_ext)
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

def directories(directory, model, img_ext):
    """
    Function to return the directories of
        Image Volume
        Segmentation Volume
        Centerline VTP
        Surface Mesh VTP
    """
    directory = os.path.abspath(os.path.expanduser(directory))
    dir_image = os.path.join(directory, 'images', model + img_ext)
    dir_seg = os.path.join(directory, 'truths', model + img_ext)
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
