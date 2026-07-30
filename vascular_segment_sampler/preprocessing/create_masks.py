import os
# from ast import literal_eval

import sys
import SimpleITK as sitk
# sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'modules'))
# sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'dataset_dirs'))

from vascular_segment_sampler import io
from vascular_segment_sampler.sitk_functions import read_image, create_new, sitk_to_numpy, numpy_to_sitk, map_to_image, import_image
from vascular_segment_sampler.sampling.functions import sort_centerline
from vascular_segment_sampler.vtk_functions import read_geo
from vascular_segment_sampler.datasets import directories, create_dataset


import numpy as np

sys.path.insert(0, './')


class Mask:

    def __init__(self, case, image_file):
        self.name = case
        self.image_reader = read_image(image_file)

        new_img = create_new(self.image_reader)
        self.assembly = new_img

        self.number_updates = np.zeros(sitk_to_numpy(self.assembly).shape)

    def add_segmentation(self, index_extract, size_extract, volume_seg=None):

        # Load the volumes
        # np_arr = sitk_to_numpy(self.assembly).astype(float)
        # np_arr_add = sitk_to_numpy(volume_seg).astype(float)

        # Calculate boundaries
        cut = 0
        edges = (np.array(index_extract) + np.array(size_extract) - cut).astype(int)
        index_extract = (np.array(index_extract) + cut).astype(int)

        # Keep track of number of updates
        # curr_n = self.number_updates[index_extract[2]:edges[2], index_extract[1]:edges[1], index_extract[0]:edges[0]]

        # Isolate current subvolume of interest
        # curr_sub_section = np_arr[index_extract[2]:edges[2], index_extract[1]:edges[1], index_extract[0]:edges[0]]
        # np_arr_add = np_arr_add[cut:size_extract[2]-cut, cut:size_extract[1]-cut, cut:size_extract[0]-cut]

        # Find indexes where we need to average predictions
        # ind = curr_n > 0
        # Where this is the first update, copy directly
        # curr_sub_section[curr_n == 0] = np_arr_add[curr_n == 0]

        # Update those values, calculating an average
        # curr_sub_section[ind] = 1/(curr_n[ind]+1)*( np_arr_add[ind] + (curr_n[ind])*curr_sub_section[ind] )
        # Add to update counter for these voxels
        self.number_updates[index_extract[2]:edges[2], index_extract[1]:edges[1], index_extract[0]:edges[0]] += 1

        # Update the global volume
        #np_arr[index_extract[2]:edges[2], index_extract[1]:edges[1], index_extract[0]:edges[0]] = curr_sub_section
        #self.assembly = numpy_to_sitk(np_arr, self.image_reader)

    def create_mask(self):
        "Function to create a global image mask of areas that were segmented"
        mask = (self.number_updates > 0).astype(int)
        mask = numpy_to_sitk(mask,self.image_reader)

        # make binary
        mask = sitk.Cast(mask, sitk.sitkUInt8)
        # import pdb; pdb.set_trace()
        self.mask = mask

        return mask


def list_samples_cases(directory):

    samples = os.listdir(directory)
    samples = [f for f in samples if ('mask' not in f and 'vtk' not in f)]

    cases = []
    samples_cases = {}
    for f in samples:
        case_name = f[:9]
        if case_name not in cases:
            cases.append(case_name)
            samples_cases[str(case_name)] = []
        samples_cases[str(case_name)].append(f)
    # samples = [directory + f for f in samples]

    return samples_cases, cases


def create_mask_case(case, dir_global, rad_prop=5, img_ext='.mha'):

    dir_image, dir_seg, dir_cent, dir_surf = directories(dir_global, case, img_ext=img_ext)
    reader_im, origin_im, size_im, spacing_im = import_image(dir_image)

    assembly_mask = Mask(case, dir_image)

    global_centerline = read_geo(dir_cent).GetOutput()

    num_points, c_loc, radii, cent_ids, bifurc_id, num_cent = sort_centerline(global_centerline)
    # import pdb; pdb.set_trace()
    for i in range(num_points):

        point = c_loc[i]
        radius = radii[i]

        size_extract, index_extract, _, _ = map_to_image(point, radius, rad_prop, origin_im, spacing_im, size_im, prop=1)
        try:
            assembly_mask.add_segmentation(index_extract, size_extract)
            # print('Added segment', i)
        except:
            print('Error for segment: ', i)

    return assembly_mask.create_mask()


if __name__=='__main__':

    '''
    Note: this script is run to make masks for evaluation.
    Returns a binary mask of the areas that were segmented.
    rad_prop: proportion of radius to be used as volume size
    '''

    global_config_file = "./config/global.yaml"
    global_config = io.load_yaml(global_config_file)

    input_dir_global = global_config['DATA_DIR']
    modalities = global_config['MODALITY']    # create output dir for masks inside input_dir_local
    out_dir = global_config['DATA_DIR'] + 'global_masks/'

    rad_prop = 6

    try:
        os.mkdir(out_dir)
    except FileExistsError:
        print(f'Directory {out_dir} already exists')

    for modality in modalities:
        # Parameters
        Dataset = create_dataset(global_config, modality)
        # cases = Dataset.sort_cases(testing=global_config['TESTING'],
        #                            test_cases=global_config['TEST_CASES'])
        # cases = Dataset.check_which_cases_in_image_dir(cases)
        cases = Dataset
        cases.sort()
        # csv_file = modality+"_test_Sample_stats.csv"
        # csv_list = pandas.read_csv(input_dir_local+csv_file)
        # keep_values = ['NAME','INDEX', 'SIZE_EXTRACT', 'BIFURCATION', 'RADIUS']

        for case in cases:

            print('\nNext case: ', case)

            mask = create_mask_case(case, input_dir_global, rad_prop, img_ext=global_config['IMG_EXT'])

            # samples = samples_cases[case]
            # # only keep samples that have been '_0'
            # samples = [f for f in samples if '_0' in f]

            # N_tot = len(samples)
            # print('\nNumber of samples: ', N_tot)

            # samples_names = [f.replace('.nii.gz','') for f in samples]
            # csv_array = csv_list.loc[csv_list['NAME'].isin(samples_names)]
            # csv_list_small = csv_array[keep_values]

            # samples_names = [f.replace('.nii.gz','') for f in samples]

            # dir_image = input_dir_global + case + '.mha'

            # assembly_segs = Mask(case, dir_image)

            # for sample in samples:#[0:N_tot:20]:

            #     if len(samples) > 10:
            #         if sample in samples[1:N_tot:N_tot//10]: print('*', end=" ", flush=True)
            #     # Read in
            #     volume = sitk.ReadImage(input_dir_local+sample)
            #     vol_np = sitk.GetArrayFromImage(volume)

            #     stat_pd = csv_list_small.loc[csv_list_small['NAME'] == sample.replace('.vtk','')]
            #     index = literal_eval(stat_pd['INDEX'].values[0])
            #     size_extract = literal_eval(stat_pd['SIZE_EXTRACT'].values[0])

            #     assembly_segs.add_segmentation(vol_np, index, size_extract)

            # assembly_segs.create_mask()
            sitk.WriteImage(mask, out_dir +'mask_'+case+'.mha')
