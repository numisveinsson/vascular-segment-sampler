import time
from datetime import datetime

import sys
import random
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from vascular_segment_sampler import vtk_functions as vf
from vascular_segment_sampler import sitk_functions as sf
from vascular_segment_sampler import io
from vascular_segment_sampler.sampling.functions import (
    create_vtk_dir, get_surf_caps, sort_centerline, choose_destination,
    get_tangent, rotate_volumes, calc_samples, extract_subvolumes,
    extract_surface, get_outlet_stats, write_2d_planes,
    write_subvolume_img, write_vtk, write_vtk_throwout, find_next_point,
    create_base_stats, add_tangent_stats, extract_centerline,
    discretize_centerline, write_surface, write_centerline, write_csv,
    write_csv_discrete_cent, write_csv_outlet_stats, write_pkl_outlet_stats,
    print_info_file, get_cross_sectional_planes,
    print_into_info, print_into_info_all_done,
    append_stats, create_directories, print_csv_stats,
    get_longest_centerline, sort_centerline_by_length, flip_radius,
    get_proj_traj
    )
from vascular_segment_sampler.pre_process import resample_spacing
from vascular_segment_sampler.datasets import get_case_dict_dir, create_dataset, resolve_case_surface_path
from vascular_segment_sampler.preprocessing.change_img_resample import resample_image
from vascular_segment_sampler import create_seg_from_surf as _create_seg_from_surf

import multiprocessing
import SimpleITK as sitk

start_time = time.time()
now = datetime.now()
dt_string = now.strftime("_%d_%m_%Y_%H_%M_%S")
sys.stdout.flush()


def _log(msg, *, verbose_only=False, verbose=False):
    """Write a line without breaking an active tqdm bar."""
    if verbose_only and not verbose:
        return
    tqdm.write(str(msg))


def _sample_case_star(args):
    return sample_case(*args)


def _get_create_seg_surf_mod():
    """Return create_seg_from_surf module (package import)."""
    return _create_seg_from_surf


def _load_surface_vtk(surface_path):
    """Load a triangular mesh from .vtp (project reader) or .stl."""
    import vtk
    if surface_path.endswith('.stl'):
        reader = vtk.vtkSTLReader()
        reader.SetFileName(surface_path)
        reader.Update()
        return reader.GetOutput()
    return vf.read_geo(surface_path).GetOutput()


def _ensure_truth_from_surface(global_config, case_dict):
    """
    Write case_dict['SEGMENTATION'] by rasterizing resolve_case_surface_path(...)
    via global/create_seg_from_surf.py (VTK stencil). Optional TRUTH_TARGET_SPACING
    resamples the truth grid (and main.py then resamples the image to match).
    """
    seg_path = case_dict['SEGMENTATION']
    truths_dir = os.path.dirname(seg_path)
    if truths_dir and not os.path.exists(truths_dir):
        os.makedirs(truths_dir, exist_ok=True)

    regen = global_config.get('TRUTH_REGENERATE', False)
    verbose = bool(global_config.get('VERBOSE', False))
    if not regen and os.path.isfile(seg_path):
        _log(
            f"Using existing segmentation: {seg_path}",
            verbose_only=True,
            verbose=verbose,
        )
        return False

    surf_path = resolve_case_surface_path(global_config['DATA_DIR'], case_dict['NAME'])
    if surf_path is None:
        raise FileNotFoundError(
            f"No surface mesh found for case {case_dict['NAME']} under "
            f"{global_config['DATA_DIR']}surfaces/ (.vtp or .stl)"
        )

    mod = _get_create_seg_surf_mod()
    surface = mod.load_surface_polydata(surf_path)
    img_sitk = sitk.ReadImage(case_dict['IMAGE'])
    ts = global_config.get('TRUTH_TARGET_SPACING')
    seg_out = mod.seg_sitk_from_surface_polydata(
        surface,
        img_sitk,
        target_spacing=ts,
        resample_order=3,
    )
    sitk.WriteImage(seg_out, seg_path)
    _log(
        f"Wrote segmentation from surface: {seg_path}",
        verbose_only=True,
        verbose=verbose,
    )
    return True


def sample_case(case_fn, global_config, out_dir, image_out_dir_train,
                seg_out_dir_train, image_out_dir_val, seg_out_dir_val,
                image_out_dir_test, seg_out_dir_test, info_file_name,
                modality):

    """ Sample a case and write out the results """

    verbose = bool(global_config.get('VERBOSE', False))
    suffix = global_config.get('OUTPUT_SUFFIX', '')

    # Check if case is in done.txt
    done_file = out_dir + "done" + suffix + ".txt"
    if os.path.exists(done_file):
        with open(done_file, "r") as f:
            done = f.read().splitlines()
            f.close()
        if case_fn in done:
            _log(f"Skipping {case_fn}", verbose_only=True, verbose=verbose)
            return (case_fn, [], [], [], [], [], [], [], 0)

    if global_config['WRITE_TRAJECTORIES']:
        traj_file = out_dir + "trajectories" + suffix + ".pkl"
        if os.path.exists(traj_file):
            df = pd.read_pickle(traj_file)
            num_trajs = df['metaId'].max() + 1
        else:
            num_trajs = 0
        traj_list = []
    else:
        traj_list = None
        num_trajs = 0

    N, M, K, O, skipped = 0, 0, 0, 0, 0
    csv_list, csv_list_val = [], []

    # If global_config['WRITE_DISCRETE_CENTERLINE']:
    csv_discrete_centerline, csv_discrete_centerline_val = [], []

    # if global_config['WRITE_OUTLET_STATS']:
    csv_outlet_stats, csv_outlet_stats_val = [], []
    total_num_examples, total_num_examples_pos = 0, 0

    # Load data
    case_dict = get_case_dict_dir(global_config['DATA_DIR'], case_fn,
                                  global_config.get('IMG_EXT'))

    _log(
        f"I am process {multiprocessing.current_process().name}",
        verbose_only=True,
        verbose=verbose,
    )
    _log(f"Starting case: {case_dict['NAME']}", verbose_only=True, verbose=verbose)
    time_now_case = time.time()

    if global_config['WRITE_VTK']:
        try:
            create_vtk_dir(out_dir, case_dict['NAME'], global_config['CAPFREE'],
                          suffix=suffix)
        except Exception as e:
            _log(e)

    # Read Image Metadata (optionally build truths/ from surfaces/ first)
    if global_config.get('SEG_FROM_SURFACE'):
        truth_regenerated = _ensure_truth_from_surface(global_config, case_dict)
        ts = global_config.get('TRUTH_TARGET_SPACING')
        if ts is not None:
            img_full = sitk.ReadImage(case_dict['IMAGE'])
            img_spacing_before = tuple(float(v) for v in img_full.GetSpacing())
            reader_im0 = resample_image(img_full, target_spacing=list(ts), order=3)
            reader_seg0 = sitk.ReadImage(case_dict['SEGMENTATION'])
            truth_spacing = tuple(float(v) for v in reader_seg0.GetSpacing())
            origin_im0 = np.array(list(reader_im0.GetOrigin()))
            size_im = np.array(list(reader_im0.GetSize()))
            spacing_im = np.array(list(reader_im0.GetSpacing()))
            if truth_regenerated:
                with open(out_dir + info_file_name, "a") as f:
                    f.write(
                        "\n TRUTH RESAMPLING (regenerated truth) - "
                        f"{case_dict['NAME']}: "
                        f"image spacing={img_spacing_before}, "
                        f"truth spacing={truth_spacing}\n"
                    )
        else:
            reader_seg0 = sf.read_image(case_dict['SEGMENTATION'])
            (reader_im0, origin_im0,
             size_im, spacing_im) = sf.import_image(case_dict['IMAGE'])
    else:
        reader_seg0 = sf.read_image(case_dict['SEGMENTATION'])
        (reader_im0, origin_im0,
         size_im, spacing_im) = sf.import_image(case_dict['IMAGE'])

    # Surface Caps
    if global_config['CAPFREE'] or global_config['WRITE_SURFACE']:
        surf_path = resolve_case_surface_path(global_config['DATA_DIR'], case_dict['NAME'])
        if surf_path is None:
            surf_path = case_dict['SURFACE']
        global_surface = _load_surface_vtk(surf_path)
        if global_config['CAPFREE']:
            cap_locs = get_surf_caps(global_surface)

    # Centerline
    global_centerline = vf.read_geo(case_dict['CENTERLINE']).GetOutput()
    (_, c_loc, radii, cent_ids,
     bifurc_id, num_cent) = sort_centerline(global_centerline)

    # Check radii and add if necessary
    radii += global_config['RADIUS_ADD']
    radii *= global_config['RADIUS_SCALE']

    ids_total = []
    m_old = M
    n_old = N
    k_old = K

    ip_longest = get_longest_centerline(cent_ids, c_loc)
    _log(
        f"Case: {case_fn}: Longest centerline is {ip_longest}"
        f" with {len(cent_ids[ip_longest])} points",
        verbose_only=True,
        verbose=verbose,
    )

    # Sort centerlines by length, starting with longest
    ips_sorted_length = sort_centerline_by_length(cent_ids, c_loc)

    # Make all cent_ids start where the radius is larger
    cent_ids = flip_radius(cent_ids, radii)

    # Workers leave the bar to the parent process.
    show_bars = multiprocessing.current_process().name == "MainProcess"
    cent_bar = tqdm(
        ips_sorted_length,
        desc=f"{case_dict['NAME']} centerlines",
        leave=False,
        disable=not show_bars,
    )
    warned_centerline = False

    # Loop over centerlines
    for ip in cent_bar:
        # Choose destination directory
        (image_out_dir, seg_out_dir,
         val_port) = choose_destination(global_config['TESTING'],
                                        global_config['VALIDATION_PROP'],
                                        image_out_dir_test, seg_out_dir_test,
                                        image_out_dir_val, seg_out_dir_val,
                                        image_out_dir_train, seg_out_dir_train,
                                        ip, case_dict['NAME'])
        # Get ids on this centerline
        ids = cent_ids[ip]
        # skip if empty
        if len(ids) == 0:
            continue
        # Get info of those ids
        # locations of those points, radii
        # and bifurcation ids at those locations
        locs, rads, bifurc = c_loc[ids], radii[ids], bifurc_id[ids]
        if len(locs) < 2:
            _log(
                f"Skipping centerline {ip} for {case_dict['NAME']}"
                f" (only {len(locs)} point)"
            )
            continue
        # Continue taking steps while still on centerline
        on_cent, count = True, 0  # count is the point along centerline
        _log(f"--- {case_dict['NAME']} ---", verbose_only=True, verbose=verbose)
        _log(f"--- Ip is {ip} / {num_cent} ---", verbose_only=True, verbose=verbose)
        while on_cent:
            # Only continue if we've not walked this centerline before
            if not (ids[count] in ids_total):

                if verbose:
                    _log('The point # along centerline is ' + str(count))
                    _log('The radius is ' + str(rads[count]))

                time_now = time.time()
                # check if we need to rotate the volume
                tangent = get_tangent(locs, count)
                if global_config['ROTATE_VOLUMES']:
                    _log("Rotating volume", verbose_only=True, verbose=verbose)
                    (reader_im, reader_seg,
                     origin_im, y_vec, z_vec,
                     rot_matrix) = rotate_volumes(
                         reader_im0, reader_seg0,
                         tangent, locs[count], outdir=out_dir)
                else:
                    reader_im, reader_seg = reader_im0, reader_seg0
                    origin_im = origin_im0
                    y_vec, z_vec, rot_matrix = None, None, None

                # Calculate centers and sizes of samples for this point
                (centers, sizes, save_bif,
                 n_samples, vec0) = calc_samples(count, bifurc, locs, rads,
                                                 global_config)

                sub = 0  # In the case of multiple samples at this point
                fixed_extract_size = global_config.get('FIXED_EXTRACT_SIZE')
                for sample in range(n_samples):
                    # Map each center and size to image data
                    center, size_r = centers[sample], sizes[sample]
                    # Get subvolume info
                    (size_extract, index_extract, voi_min,
                     voi_max) = sf.map_to_image(center, rads[count],
                                                size_r, origin_im,
                                                spacing_im, size_im,
                                                global_config['CAPFREE_PROP'],
                                                min_dim=global_config.get('MIN_DIM', 5),
                                                fixed_size=fixed_extract_size)
                    # Skip when fixed extraction would go out of bounds
                    if size_extract is None:
                        skipped += 1
                        continue

                    # Check if a surface cap is in volume
                    if global_config['CAPFREE']:
                        is_inside = vf.voi_contain_caps(voi_min, voi_max,
                                                        cap_locs)
                    else:
                        is_inside = False
                    # Continue if surface cap is not present
                    if not is_inside:
                        if verbose:
                            _log("*")
                        try:
                            name = (case_dict['NAME']+'_'+str(N-n_old)
                                    + '_'+str(sub))

                            # Extract volume
                            if global_config['EXTRACT_VOLUMES']:
                                (stats, new_img, removed_seg, O
                                 ) = extract_subvolumes(
                                     reader_im, reader_seg,
                                     index_extract,
                                     size_extract,
                                     origin_im, spacing_im,
                                     locs[count],
                                     rads[count], size_r, N,
                                     name, O,
                                     remove_others=global_config['REMOVE_OTHER'],
                                     # BINARIZE: Convert segmentation to binary (0,1) vs preserve multi-labels
                                     binarize=global_config['BINARIZE'],
                                     rotate=global_config['ROTATE_VOLUMES'],
                                     orig_im=reader_im0, orig_seg=reader_seg0,
                                     outdir=out_dir
                                     )

                                if global_config['WRITE_SURFACE']:
                                    surf_radius = (0.5 * np.linalg.norm(
                                        np.array(size_extract) * np.array(spacing_im))
                                        if fixed_extract_size else
                                        size_r*rads[count]/2)
                                    (stats_surf, new_surf_box, new_surf_sphere
                                     ) = extract_surface(
                                         new_img,
                                         global_surface,
                                         center,
                                         surf_radius)
                                    num_out = len(stats_surf['OUTLETS'])
                                    # print(f"Outlets are: {num_out}")
                                    stats.update(stats_surf)
                                else:
                                    num_out = 0
                            else:
                                stats = create_base_stats(N, name, size_r,
                                                          rads[count],
                                                          size_extract,
                                                          origin_im,
                                                          spacing_im,
                                                          index_extract,
                                                          center)
                                num_out = 0
                                # Initialize variables that might be used later
                                new_img = None
                                removed_seg = None
                                new_surf_box = None
                                new_surf_sphere = None

                            # Extract surface even if EXTRACT_VOLUMES is False
                            if (not global_config['EXTRACT_VOLUMES'] 
                                and global_config['WRITE_SURFACE']):
                                # Extract volume just for surface extraction
                                # Convert to list of ints (required by SimpleITK)
                                index_extract_list = index_extract.astype(int).tolist()
                                size_extract_list = size_extract.astype(int).tolist()
                                new_img = sf.extract_volume(reader_im, index_extract_list, size_extract_list)
                                surf_radius_alt = (0.5 * np.linalg.norm(
                                    np.array(size_extract) * np.array(spacing_im))
                                    if fixed_extract_size else size_r*rads[count]/2)
                                (stats_surf, new_surf_box, new_surf_sphere
                                 ) = extract_surface(
                                     new_img,
                                     global_surface,
                                     center,
                                     surf_radius_alt)
                                num_out = len(stats_surf['OUTLETS'])
                                # print(f"Outlets are: {num_out}")
                                stats.update(stats_surf)

                            stats = add_tangent_stats(stats, vec0, save_bif)

                            if global_config['WRITE_SAMPLES']:
                                # Write surface and centerline vtps
                                if ((global_config['WRITE_SURFACE']
                                    or global_config['WRITE_CENTERLINE'])
                                   and num_out in global_config['OUTLET_CLASSES']):
                                    if global_config['WRITE_SURFACE']:
                                        write_surface(new_surf_box,
                                                      new_surf_sphere,
                                                      seg_out_dir,
                                                      case_dict['NAME'],
                                                      N, n_old, sub)
                                    if global_config['WRITE_CENTERLINE']:
                                        if global_config['EXTRACT_VOLUMES']:
                                            _, new_cent = extract_centerline(
                                                new_img, global_centerline,
                                                verbose=verbose)
                                            write_centerline(new_cent,
                                                             seg_out_dir,
                                                             case_dict['NAME'],
                                                             N, n_old, sub)
                                        else:
                                            if not warned_centerline:
                                                _log(
                                                    "Warning: WRITE_CENTERLINE requires"
                                                    " EXTRACT_VOLUMES to be True,"
                                                    " skipping centerline extraction"
                                                )
                                                warned_centerline = True
                                if global_config['EXTRACT_VOLUMES'] and global_config['WRITE_IMG']:
                                    if global_config['RESAMPLE_VOLUMES']:
                                        removed_seg_re = resample_spacing(
                                            removed_seg,
                                            template_size=global_config['RESAMPLE_SIZE'],
                                            order=1)[0]
                                        new_img_re = resample_spacing(
                                            new_img,
                                            template_size=global_config['RESAMPLE_SIZE'],
                                            order=1)[0]
                                        # BINARIZE: Normalize seg values to [0,1] and enforce binary constraint
                                        write_subvolume_img(new_img_re, removed_seg_re,
                                                  image_out_dir, seg_out_dir,
                                                  case_dict['NAME'],
                                                  N, n_old, sub, global_config['BINARIZE'],
                                                  verbose=verbose)
                                    else:
                                        # BINARIZE: Normalize seg values to [0,1] and enforce binary constraint
                                        write_subvolume_img(new_img, removed_seg,
                                                  image_out_dir, seg_out_dir,
                                                  case_dict['NAME'],
                                                  N, n_old, sub, global_config['BINARIZE'],
                                                  verbose=verbose)

                                if global_config['EXTRACT_VOLUMES'] and global_config['WRITE_VTK']:
                                    write_vtk(new_img, removed_seg,
                                              out_dir, case_dict['NAME'],
                                              N, n_old, sub, suffix=suffix)

                            # Discretize centerline
                            if global_config['EXTRACT_VOLUMES'] and global_config['WRITE_DISCRETE_CENTERLINE']:
                                _, new_cent = extract_centerline(
                                    new_img,
                                    global_centerline,
                                    verbose=verbose)
                                cent_stats = discretize_centerline(
                                    new_cent,
                                    new_img,
                                    N-n_old,
                                    sub, name,
                                    out_dir,
                                    global_config['DISCRETE_CENTERLINE_N_POINTS'],
                                    suffix=suffix)
                                if val_port:
                                    csv_discrete_centerline_val.append(cent_stats)
                                else:
                                    csv_discrete_centerline.append(cent_stats)

                            # Outlet stats
                            if global_config['EXTRACT_VOLUMES'] and global_config['WRITE_OUTLET_STATS']:
                                (stats_out, planes, planes_seg, pos_example
                                 ) = get_outlet_stats(
                                     stats, new_img,
                                     removed_seg,
                                     upsample=global_config['UPSAMPLE_OUTLET_IMG'])

                                total_num_examples_pos += pos_example
                                total_num_examples += 6
                                _log(
                                    "Ratio of positive examples: "
                                    f"{total_num_examples_pos/total_num_examples * 100:.2f}",
                                    verbose_only=True,
                                    verbose=verbose,
                                )
                                if global_config['WRITE_OUTLET_IMG']:
                                    write_2d_planes(planes, stats_out,
                                                    image_out_dir)
                                    write_2d_planes(planes_seg, stats_out,
                                                    seg_out_dir)
                                if val_port:
                                    for out_stats in stats_out:
                                        csv_outlet_stats_val.append(out_stats)
                                else:
                                    for out_stats in stats_out:
                                        csv_outlet_stats.append(out_stats)

                            # Append stats to csv list
                            csv_list, csv_list_val = append_stats(stats, csv_list, csv_list_val, val_port)

                            # Cross sectional planes
                            if global_config['EXTRACT_VOLUMES'] and global_config['WRITE_CROSS_SECTIONAL']:
                                (stats_out, planes_img, planes_seg
                                 ) = get_cross_sectional_planes(
                                     stats, new_img, removed_seg,
                                     upsample=global_config['RESAMPLE_CROSS_IMG'])
                                # write cross sectional planes
                                write_2d_planes(planes_img, stats_out,
                                                image_out_dir, add='_cross_rot')
                                write_2d_planes(planes_seg, stats_out,
                                                seg_out_dir, add='_cross_rot')
                            if global_config['EXTRACT_VOLUMES'] and global_config['WRITE_TRAJECTORIES']:
                                (traj_list,
                                 num_trajs) = get_proj_traj(
                                    stats,
                                    new_img,
                                    removed_seg,
                                    global_centerline,
                                    traj_list,
                                    num_trajs,
                                    tangent=tangent,
                                    rot_point=locs[count],
                                    outdir=out_dir,
                                    visualize=True,
                                    suffix=suffix,
                                    img_size=global_config['RESAMPLE_CROSS_IMG'],
                                    n_slices=global_config['N_SLICES'],
                                    verbose=verbose)

                        except Exception as e:
                            _log(e)
                            _log(
                                "ERROR: did not save files for "
                                + case_dict['NAME'] + '_' + str(N-n_old)+'_'+str(sub)
                            )
                            K += 1
                            # import pdb; pdb.set_trace()
                    else:
                        _log(".", verbose_only=True, verbose=verbose)
                        # print(" No save - cap inside")
                        try:
                            if (global_config['WRITE_VTK']
                               and global_config['WRITE_VTK_THROWOUT']):
                                write_vtk_throwout(reader_seg, index_extract,
                                                   size_extract, out_dir,
                                                   case_dict['NAME'], N,
                                                   n_old, sub, suffix=suffix)
                            M += 1
                        except Exception as e:
                            _log(e)
                            # print("\n*****************************ERROR: did not save throwout for " +case_dict['NAME']+'_'+str(N-n_old)+'_'+str(sub))
                            K += 1
                    sub += 1

                _log(
                    'Finished: ' + case_dict['NAME'] + '_' + str(N-n_old),
                    verbose_only=True,
                    verbose=verbose,
                )
                _log(
                    f"Time for this point: {(time.time() - time_now):.2f} sec",
                    verbose_only=True,
                    verbose=verbose,
                )
                # print(" " + str(sub) + " variations")
                N += 1
                cent_bar.set_postfix(
                    saved=N - n_old,
                    throw=M - m_old,
                    err=K - k_old,
                    pt=count,
                    refresh=False,
                )

                if N*n_samples - skipped > int(float(global_config['MAX_SAMPLES'])):
                    _log("Max samples reached")
                    on_cent = False
                    break

            count, on_cent = find_next_point(count, locs, rads, bifurc,
                                             global_config, on_cent)
        # keep track of ids that have already been operated on   
        ids_total.extend(ids)

        if N*n_samples - skipped > int(float(global_config['MAX_SAMPLES'])):
            break

    cent_bar.close()
    elapsed_case = time.time() - time_now_case
    _log(
        f"{case_dict['NAME']}: {N-n_old} extractions, {M-m_old} throwouts, "
        f"{K-k_old} errors, {elapsed_case:.1f}s"
    )
    print_into_info(info_file_name, case_dict['NAME'], N, n_old, M, m_old, K,
                    k_old, out_dir)
    write_csv(csv_list, csv_list_val, modality, global_config)
    if global_config['WRITE_DISCRETE_CENTERLINE']:
        write_csv_discrete_cent(csv_discrete_centerline,
                                csv_discrete_centerline_val,
                                modality, global_config)

    if global_config['WRITE_OUTLET_STATS']:
        write_csv_outlet_stats(csv_outlet_stats, csv_outlet_stats_val,
                               modality, global_config)
        write_pkl_outlet_stats(csv_outlet_stats, csv_outlet_stats_val,
                               modality, global_config)

    # TODO: if trajectories, add to df file
    if global_config['WRITE_TRAJECTORIES']:
        _log(
            f"Number of trajectories for {case_dict['NAME']}: {num_trajs}",
            verbose_only=True,
            verbose=verbose,
        )
        column_names = ['frame', 'trackId', 'x', 'y', 'sceneId', 'metaId']
        # check if files exist
        if os.path.exists(traj_file):
            df = pd.read_pickle(traj_file)
            df = df.append(pd.DataFrame(traj_list, columns=column_names))
            df.to_pickle(traj_file)
        else:
            df = pd.DataFrame(traj_list, columns=column_names)
            df.to_pickle(traj_file)

        # test read in
        df = pd.read_pickle(traj_file)
        if verbose:
            _log(df[df['metaId'] == 0])

    # write to done.txt the name of the case
    with open(done_file, "a") as f:
        f.write(case_dict['NAME']+'\n')
        f.close()

    return (case_fn, csv_list, csv_list_val, csv_discrete_centerline,
            csv_discrete_centerline_val, csv_outlet_stats,
            csv_outlet_stats_val, traj_list, num_trajs)



def _load_config(config):
    """Accept a config dict or a path/name of a YAML file.

    Named configs overlay packaged ``default_mm.yaml``. Bare names like ``default_mm``
    must not match the repo's ``global/`` directory.
    """
    return io.load_config(config)


def extract_patches(
    data_dir,
    outdir="./extracted_data/",
    config="default_mm",
    *,
    perc_dataset=1.0,
    num_cores=1,
    start_from=0,
    end_at=-1,
    testing=False,
    validation_prop=None,
    max_samples=None,
    modality=None,
    output_suffix="",
    truth_from_surface=False,
    truth_target_spacing=None,
    truth_regenerate=False,
    yes=False,
    verbose=False,
):
    """
    Extract vascular segment patches from cases under ``data_dir``.

    Parameters
    ----------
    data_dir : str
        Directory with images/, truths/, centerlines/, surfaces/.
    outdir : str
        Output directory for extracted samples.
    config : dict | str
        Config dict or path/name of a YAML config (e.g. ``"default_mm"`` or
        ``"config/global.yaml"``). Overlay on packaged ``default_mm.yaml``.
    """
    global_config = _load_config(config)
    global_config["DATA_DIR"] = data_dir

    if testing:
        global_config["TESTING"] = True
    elif "TESTING" not in global_config:
        global_config["TESTING"] = False

    if validation_prop is not None:
        global_config["VALIDATION_PROP"] = validation_prop
    elif "VALIDATION_PROP" not in global_config:
        global_config["VALIDATION_PROP"] = 0.0

    if max_samples is not None:
        global_config["MAX_SAMPLES"] = max_samples
    elif "MAX_SAMPLES" not in global_config:
        global_config["MAX_SAMPLES"] = 1e6

    if modality is not None:
        modalities_list = [m.strip().upper() for m in str(modality).split(",")]
        global_config["MODALITY"] = modalities_list
    elif "MODALITY" not in global_config:
        global_config["MODALITY"] = ["CT"]

    if "MIN_DIM" not in global_config:
        global_config["MIN_DIM"] = 5

    if "LARGE_RADIUS_THRESH" not in global_config:
        global_config["LARGE_RADIUS_THRESH"] = 4.0

    global_config["TEST_CASES"] = global_config.get("TEST_CASES") or []
    global_config["BAD_CASES"] = global_config.get("BAD_CASES") or []

    output_suffix = output_suffix or ""
    global_config["OUTPUT_SUFFIX"] = output_suffix
    global_config["SEG_FROM_SURFACE"] = bool(truth_from_surface)
    global_config["TRUTH_TARGET_SPACING"] = truth_target_spacing
    global_config["TRUTH_REGENERATE"] = bool(truth_regenerate)
    global_config["VERBOSE"] = bool(verbose)

    modalities = global_config["MODALITY"]
    out_dir = outdir
    global_config["OUT_DIR"] = out_dir

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    modality_iter = (
        tqdm(modalities, desc="Modalities") if len(modalities) > 1 else modalities
    )
    for modality_name in modality_iter:
        cases = create_dataset(global_config, modality_name)

        random.seed(42)
        random.shuffle(cases)
        cases = cases[: int(perc_dataset * len(cases))]

        if end_at != -1:
            cases = cases[start_from:end_at]
        else:
            cases = cases[start_from:]

        done_file_path = out_dir + "done" + output_suffix + ".txt"
        done_names = set()
        if os.path.exists(done_file_path):
            with open(done_file_path, "r") as f:
                done_names = {line for line in f.read().splitlines() if line}
        n_done = sum(1 for case in cases if case in done_names)
        cases = [case for case in cases if case not in done_names]

        modality_l = modality_name.lower()
        suffix = output_suffix
        info_file_name = "info" + "_" + modality_l + dt_string + suffix + ".txt"

        create_directories(out_dir, modality_l, global_config)

        image_out_dir_train = out_dir + modality_l + "_train" + suffix + "/"
        seg_out_dir_train = out_dir + modality_l + "_train" + suffix + "_masks/"
        image_out_dir_val = out_dir + modality_l + "_val" + suffix + "/"
        seg_out_dir_val = out_dir + modality_l + "_val" + suffix + "_masks/"
        image_out_dir_test = out_dir + modality_l + "_test" + suffix + "/"
        seg_out_dir_test = out_dir + modality_l + "_test" + suffix + "_masks/"

        N, M, K, O = 0, 0, 0, 0

        _log(f"{modality_l}: {len(cases)} cases ({n_done} already done)")
        if verbose:
            for case in cases:
                _log(f"Case: {case}")

        if cases and not yes and not io.prompt_continue("Wish to continue? [y/n]: "):
            _log("Aborting before starting first case.")
            continue

        print_info_file(global_config, cases, global_config["TEST_CASES"], info_file_name)

        _log(
            f"Number of possible cores: {multiprocessing.cpu_count()}",
            verbose_only=True,
            verbose=verbose,
        )

        if num_cores > 1:
            sample_args = [
                (
                    case,
                    global_config,
                    out_dir,
                    image_out_dir_train,
                    seg_out_dir_train,
                    image_out_dir_val,
                    seg_out_dir_val,
                    image_out_dir_test,
                    seg_out_dir_test,
                    info_file_name,
                    modality_l,
                )
                for case in cases
            ]
            results = []
            with multiprocessing.Pool(num_cores) as pool:
                with tqdm(
                    total=len(cases), desc=f"{modality_l} cases", leave=True
                ) as case_bar:
                    for result in pool.imap_unordered(_sample_case_star, sample_args):
                        results.append(result)
                        case_bar.set_postfix(last=result[0], refresh=False)
                        case_bar.update(1)
        else:
            for case in tqdm(cases, desc=f"{modality_l} cases", leave=True):
                results = sample_case(
                    case,
                    global_config,
                    out_dir,
                    image_out_dir_train,
                    seg_out_dir_train,
                    image_out_dir_val,
                    seg_out_dir_val,
                    image_out_dir_test,
                    seg_out_dir_test,
                    info_file_name,
                    modality_l,
                )

        if global_config["WRITE_TRAJECTORIES"]:
            traj_list_all = []
            num_trajs = 0
            if num_cores > 1:
                for result in results:
                    (
                        case_fn,
                        csv_list,
                        csv_list_val,
                        csv_discrete_centerline,
                        csv_discrete_centerline_val,
                        csv_outlet_stats,
                        csv_outlet_stats_val,
                        traj_list,
                        num_trajs,
                    ) = result
            else:
                (
                    case_fn,
                    csv_list,
                    csv_list_val,
                    csv_discrete_centerline,
                    csv_discrete_centerline_val,
                    csv_outlet_stats,
                    csv_outlet_stats_val,
                    traj_list,
                    num_trajs,
                ) = results
                traj_list_all.extend(traj_list)
                num_trajs += num_trajs

        print_into_info_all_done(
            info_file_name, N, M, K, O, out_dir, start_time=start_time
        )
        elapsed = time.time() - start_time
        _log(f"--- {elapsed/60:.2f} min ({elapsed/3600:.2f} hours) ---")
        print_csv_stats(out_dir, global_config, modality_l)

    _log("All done")
    return out_dir
