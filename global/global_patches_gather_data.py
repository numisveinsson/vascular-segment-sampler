from gather_sampling_data import define_cases, create_directories, extract_subvolumes, print_info_file
from move_validation import random_files, move_files
from modules import sitk_functions as sf
from modules import io
import os
import SimpleITK as sitk
import numpy as np
import csv
import random

from datetime import datetime
now = datetime.now()
dt_string = now.strftime("_%d_%m_%Y_%H_%M_%S")


def define_bounds(size_im, volume_size, random_crops):
    """
    Defines indexing to split a global volume into 
        patches of size volume_size
    Includes stride where
        stride is volume_size/2
    """
    max_ind = size_im-volume_size
    min_ind = np.array([0,0,0])

    num_in_each_dim = np.ceil(np.array(size_im)/np.array(volume_size)).astype(int)

    indexes = []
    for i in range(num_in_each_dim[0]):
        for j in range(num_in_each_dim[1]):
            for k in range(num_in_each_dim[2]):
                for m in range(2):
                    index = np.array([i,j,k])*volume_size +m*1/2*volume_size
                    index = np.where(np.greater(index+volume_size, size_im), max_ind, index)
                    index = np.where(np.less(index, min_ind), min_ind, index)
                    indexes.append(index)
    # Now extract random crops from the image
    if random_crops:
        N = len(indexes)
        for i in range(N//2):
            rand_ind = np.random.rand(3)*min_ind//1
            if not (rand_ind+volume_size > max_ind).any():
                indexes.append(rand_ind)
    return indexes

if __name__=='__main__':

    random.seed(1)
    extract_volumes = True

    patches = False
    random_crops = False
    volume_size = [128, 128, 128]

    global_config_file = "./config/global.yaml"
    global_config = io.load_yaml(global_config_file)
    modalities = global_config['MODALITY']
    val_prop = global_config['VALIDATION_PROP']
    
    write_vtk_samples = global_config['WRITE_VTK']
    write_samples = global_config['WRITE_SAMPLES']

    trace_testing = global_config['TESTING']
    out_dir = global_config['OUT_DIR']
    #create_dir_sample_info(trace_testing, out_dir)

    for modality in modalities:

        modality = modality.lower()

        cases, test_samples, bad_samples = define_cases(global_config, modality)
        create_directories(out_dir, modality, trace_testing, write_vtk_samples)

        info_file_name = "info"+'_'+modality+dt_string+".txt"
        print_info_file(global_config, cases, test_samples, info_file_name)

        image_out_dir_train = os.path.join(out_dir, modality+'_train')
        seg_out_dir_train = os.path.join(out_dir, modality+'_train_masks')
        image_out_dir_val = os.path.join(out_dir, modality+'_val')
        seg_out_dir_val = os.path.join(out_dir, modality+'_val_masks')

        image_out_dir_test = os.path.join(out_dir, modality+'_test')
        seg_out_dir_test = os.path.join(out_dir, modality+'_test_masks')

        if trace_testing:
            image_out_dir = image_out_dir_test
            seg_out_dir = seg_out_dir_test
        else:
            image_out_dir = image_out_dir_train
            seg_out_dir = seg_out_dir_train

        info = {}
        csv_list = []
        N = 0

        cases_val = random.choices(cases, k= int(np.ceil(val_prop*len(cases)//1)))
        for case_fn in cases:
            if case_fn in cases_val and not trace_testing:
                image_out_dir = image_out_dir_val
                seg_out_dir = seg_out_dir_val
            elif not trace_testing: 
                image_out_dir = image_out_dir_train
                seg_out_dir = seg_out_dir_train
            
            case_dict = io.load_yaml(case_fn)
            reader_seg = sf.read_image(case_dict['SEGMENTATION'])

            reader_im, origin_im, size_im, spacing_im = sf.import_image(case_dict['IMAGE'])
            volume_size = np.array(volume_size)
            size_im = np.array(size_im)
            
            if patches:
                sizes_indexes = define_bounds(size_im, volume_size, random_crops)
                print(f"Number patches: {len(sizes_indexes)}")
                for i,index_extract in enumerate(sizes_indexes):

                    size_extract = np.where(np.greater(volume_size,size_im), size_im, volume_size)

                    if extract_volumes:
                        stats, new_img, new_seg, removed_seg, _ = extract_subvolumes(reader_im, reader_seg, index_extract, size_extract, origin_im, spacing_im, N, case_dict['NAME']+'_'+str(i), global_img=True)
                    else:
                        stats = {"No":N, "NAME": case_dict['NAME']+'_'+str(i), "RESOLUTION": size_extract,"ORIGIN": origin_im, "SPACING": spacing_im,}
                    stats.update({"NUM_VOX": size_extract[0]*size_extract[1]*size_extract[2]})

                    if write_samples:
                        sitk.WriteImage(new_img, os.path.join(image_out_dir, case_dict['NAME']+'_'+str(i)+'.nii.gz'))
                        sitk.WriteImage(removed_seg, os.path.join(seg_out_dir, case_dict['NAME']+'_'+str(i)+'.nii.gz'))
                    if write_vtk_samples:
                        sitk.WriteImage(new_img, os.path.join(out_dir, 'vtk_data', 'vtk_'+case_dict['NAME']+'_'+str(i)+'.vtk'))
                        sitk.WriteImage(removed_seg*255, os.path.join(out_dir, 'vtk_data', 'vtk_mask_'+case_dict['NAME']+'_'+str(i)+'.vtk'))

                    csv_list.append(stats)
                    N += 1
            else:
                if extract_volumes:
                    index_extract = np.array([0,0,0])
                    size_extract = size_im
                    stats, new_img, new_seg, removed_seg, _ = extract_subvolumes(reader_im, reader_seg, index_extract, size_extract, origin_im, spacing_im, N, case_dict['NAME'], global_img=True)
                else:
                    stats = {"No":N, "NAME": case_dict['NAME']+'_'+str(i), "RESOLUTION": size_extract,"ORIGIN": origin_im, "SPACING": spacing_im,}
                stats.update({"NUM_VOX": size_extract[0]*size_extract[1]*size_extract[2]})
                if write_samples:
                    sitk.WriteImage(new_img, os.path.join(image_out_dir, case_dict['NAME']+'.nii.gz'))
                    sitk.WriteImage(removed_seg, os.path.join(seg_out_dir, case_dict['NAME']+'.nii.gz'))
                csv_list.append(stats)

            print(f"\n Finished: ' {case_dict['NAME']}, {size_im}")
                

        csv_file = "_Sample_stats.csv"
        if trace_testing: csv_file = '_test'+csv_file

        csv_columns = ["No", "NAME", "SIZE","RESOLUTION", "ORIGIN", "SPACING", "POINT_CENT", "INDEX", "SIZE_EXTRACT", "VOL_CENT", "DIFF_CENT", "IM_MEAN",
        "IM_STD","IM_MAX","IM_MIN","BLOOD_MEAN","BLOOD_STD","BLOOD_MAX","BLOOD_MIN","GT_MEAN", "GT_STD", "GT_MAX", "GT_MIN",
        "LARGEST_MEAN","LARGEST_STD","LARGEST_MAX","LARGEST_MIN", 'RADIUS', 'TANGENTX', 'TANGENTY', 'TANGENTZ', 'BIFURCATION', 'NUM_VOX']
        with open(os.path.join(out_dir, modality+csv_file), 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in csv_list:
                writer.writerow(data)

        # if not trace_testing:
        #     data_set_percent_size = global_config['VALIDATION_PROP']
        #     random_img = random_files(image_out_dir_train, data_set_percent_size)
        #     random_seg = random_files(seg_out_dir_train, data_set_percent_size)
        #     print(f"Moving {data_set_percent_size*100}% files to validation")
        #     move_files(image_out_dir_train, image_out_dir_val, random_img)
        #     move_files(seg_out_dir_train, seg_out_dir_val, random_img)
