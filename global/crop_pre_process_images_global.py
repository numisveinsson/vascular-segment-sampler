import pdb
from modules import io
from modules.pre_process import rescale_intensity
from modules import sitk_functions as sf
import SimpleITK as sitk
import numpy as np
import os

def define_bounds(np_seg, dims, padding=0, template_size=None):
    """
    Function to find smallest possible bounds of GT seg
    vol_seg: sitk image
    dims: list of dims to get bounds of ['x','y']
    """
    max_val = np_seg.max()
    assert max_val == 1 or max_val == 255
    size = np_seg.shape
    boundsz, boundsy, boundsx = [0,size[0]-1],[0,size[1]-1],[0,size[2]-1]
    bounds = [boundsz, boundsy, boundsx]
    #print(f'Old bounds: {bounds}')

    if 'z' in dims:

        still_zero = True
        front_done, back_done = False, False
        count = 0
        while still_zero:
            seg_slice_front = np_seg[count,:,:]
            seg_slice_back = np_seg[size[0]-(count+1),:,:]
            max_front = seg_slice_front.max()
            max_back = seg_slice_back.max()
            if max_front == max_val and front_done == False:
                boundsz[0] = count
                front_done = True
                #print(f"front done")
            if max_back == max_val and back_done == False:
                boundsz[1] = size[0] - count-1
                back_done = True
                #print(f"back done")
            still_zero = front_done == False or back_done == False
            count+=1

    if 'y' in dims:
        still_zero = True
        front_done, back_done = False, False
        count = 0
        while still_zero:
            seg_slice_front = np_seg[:,count,:]
            seg_slice_back = np_seg[:,size[1]-(count+1),:]
            max_front = seg_slice_front.max()
            max_back = seg_slice_back.max()
            if max_front == max_val and front_done == False:
                boundsy[0] = count
                front_done = True
            if max_back == max_val and back_done == False:
                boundsy[1] = size[1] - count-1
                back_done = True
            still_zero = front_done == False or back_done == False
            count +=1

    if 'x' in dims:
        still_zero = True
        front_done, back_done = False, False
        count = 0
        while still_zero:
            seg_slice_front = np_seg[:,:,count]
            seg_slice_back = np_seg[:,:,size[2]-(count+1)]
            max_front = seg_slice_front.max()
            max_back = seg_slice_back.max()
            if max_front == max_val and front_done == False:
                boundsx[0] = count
                front_done = True
            if max_back == max_val and back_done == False:
                boundsx[1] = size[2] - count-1
                back_done = True
            still_zero = front_done == False or back_done == False
            count += 1

    bounds = [boundsz, boundsy, boundsx]

    if template_size:
        "Modify so divisible by this size, add equal on both sides"
        print("template size is on")

    # print(f"New bounds: {bounds}")
    return bounds

def crop_bounds(im_read, seg_read, bounds):
    """
    Function to crop global according to new bounds
    template_size: number that final volume should be
        divisible by (if it's later cut to patches)
    """

    # _, img_np = sf.read_image_numpy(img)
    # _, seg_np = sf.read_image_numpy(seg)
    boundsz = bounds[0]
    boundsy = bounds[1]
    boundsx = bounds[2]

    #new_img = img_np[boundsz[0]:boundsz[1]+1,boundsy[0]:boundsy[1]+1,boundsx[0]:boundsx[1]+1]
    #new_seg = seg_np[boundsz[0]:boundsz[1]+1,boundsy[0]:boundsy[1]+1,boundsx[0]:boundsx[1]+1]
    # new_img = sf.create_new_from_numpy(im_read, new_img)
    # new_seg = sf.create_new_from_numpy(seg_read, new_seg)

    index_extract = [boundsx[0], boundsy[0], boundsz[0]]
    size_extract = [boundsx[1]-boundsx[0]+1, boundsy[1]-boundsy[0]+1, boundsz[1]-boundsz[0]+1]

    new_img = sf.extract_volume(im_read, index_extract, size_extract)
    new_seg = sf.extract_volume(seg_read, index_extract, size_extract)

    return new_img, new_seg

def create_crop_dir(out_dir):
    name = 'cropped'
    img_dir = os.path.join(out_dir, name)
    seg_dir = os.path.join(out_dir, name+'_masks')
    try:
        os.mkdir(img_dir)
    except Exception as e: print(e)
    try:
        os.mkdir(seg_dir)
    except Exception as e: print(e)
    return img_dir, seg_dir

if __name__=='__main__':

    out_dir = '/Users/numisveins/Documents/vascular_data_3d/images/'
    config = io.load_yaml('./config/global.yaml')

    specific_folder = '/Users/numisveins/Documents/vascular_data_3d/'

    global_scale = False
    crop = False
    template_size = None

    dims = ['x','y','z']
    add_padding = 0

    if crop:
        img_dir, seg_dir = create_crop_dir(out_dir)


    cases_prefix = config['DATA_DIR']

    if not specific_folder:
        cases_dir = config['CASES_DIR']
        centerlines = open(os.path.join(cases_dir, 'centerlines.txt'))
        centerlines = [f.replace('\n','') for f in centerlines]
        centerlines = ['/centerlines/'+f for f in centerlines]

        images = open(os.path.join(cases_dir, 'images.txt')).readlines()
        images = [f.replace('\n','') for f in images]
        images = ['/images'+f for f in images]

        segs = open(os.path.join(cases_dir, 'truths.txt')).readlines()
        segs = [f.replace('\n','') for f in segs]
        segs = ['/images'+f for f in segs]

        modality = open(os.path.join(cases_dir, 'modality.txt'))
        modality  = [f.replace('\n','') for f in modality]

    else:
        images = os.listdir(os.path.join(cases_prefix, 'images'))
        images = [image for image in images if '.vtk' in image]
        images = ['images/'+image for image in images]
        centerlines = os.listdir(os.path.join(cases_prefix, 'centerlines'))
        centerlines = ['/centerlines/'+f for f in centerlines]
        modality = open(os.path.join(cases_prefix, 'modality.txt'))
        modality  = [f.replace('\n','') for f in modality]
        #segs = os.listdir(cases_prefix+'truths/')

    for img in images:
        print('\n',img)
    import pdb; pdb.set_trace()
    for i,image in enumerate(images):
        #print(f"Case: {image}")
        mod = modality[i].lower()
        if global_scale:
            if crop:
                img_reader, img_np = sf.read_image_numpy(os.path.join(cases_prefix, image))
                img_new_np = rescale_intensity(img_np, mod, [750, -750])
                img_new = sf.create_new_from_numpy(img_reader, img_new_np)
                sf.write_image(img_new, os.path.join(img_dir, centerlines[i][-13:-4]+'.vtk'))

                _, np_seg = sf.read_image_numpy(os.path.join(cases_prefix, segs[i]))
                new_bounds = define_bounds(np_seg, dims, add_padding, template_size)

                im_read = sf.read_image(os.path.join(img_dir, centerlines[i][-13:-4]+'.vtk'))
                seg_read = sf.read_image(os.path.join(cases_prefix, segs[i]))
                new_img, new_seg = crop_bounds(im_read, seg_read, new_bounds)

                sf.write_image(new_img, os.path.join(img_dir, centerlines[i][-13:-4]+'.vtk'))
                sf.write_image(new_seg, os.path.join(seg_dir, centerlines[i][-13:-4]+'.vtk'))
            else:
                img_reader, img_np = sf.read_image_numpy(os.path.join(cases_prefix, image))
                img_new_np = rescale_intensity(img_np, mod, [750, -750])
                img_new = sf.create_new_from_numpy(img_reader, img_new_np)
                sf.write_image(img_new, os.path.join(out_dir, image.replace('images/', '')))
            #print(f"Mean value: {img_new_np.mean()}")
        else:
            if crop:
                _, np_seg = sf.read_image_numpy(os.path.join(cases_prefix, segs[i]))
                new_bounds = define_bounds(np_seg, dims, add_padding, template_size)
                new_img, new_seg = crop_bounds(os.path.join(cases_prefix, image), os.path.join(cases_prefix, segs[i]), new_bounds)
                sf.write_image(new_img, os.path.join(img_dir, centerlines[i][-13:-4]+'.vtk'))
                sf.write_image(new_seg, os.path.join(seg_dir, centerlines[i][-13:-4]+'.vtk'))
            else:
                img = sf.read_image(os.path.join(cases_prefix, image))
                sf.write_image(img, os.path.join(out_dir, image.replace('images/', '')))

    pdb.set_trace()
