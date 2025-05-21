#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline to take the MD and FA maps, downsample (resample) the label file into diffusion space,
then extract ROI statistics, save overlays, JSON stats, and maintain a master CSV of ROI values.
"""

import os
import glob
import json
import logging

import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dipy.io.image import load_nifti, save_nifti
from dipy.align import affine_registration
from dipy.align.imaffine import AffineMap


def configure_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "process_log.txt")
    logger = logging.getLogger("DatasetLogger")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def load_nifti_data(file_path):
    data, affine = load_nifti(file_path)
    return data, affine



def resample_labels_to_diffusion_space(label_filepath, diffusion_img, T2_img,
                                       diffusion_affine, T2_affine,
                                       trim_z=True, trim_slices=(2, -2),
                                       reg_params=None):
    """
    Resample a label image (in T2 space) into diffusion space using affine registration.
    
    Parameters
    ----------
    label_filepath : str
        Full path to the label image.
    diffusion_img : nibabel.Nifti1Image
        Diffusion image (used as moving image).
    T2_img : nibabel.Nifti1Image
        T2 image (static reference).
    diffusion_affine : (4,4) array
        Affine matrix for the diffusion image.
    T2_affine : (4,4) array
        Affine matrix for the T2 image.
    trim_z : bool, optional
        If True, trim slices along z-dimension (default: True).
    trim_slices : tuple, optional
        Slicing for the z-axis, e.g. (2, -2) to remove first two and last two slices.
    reg_params : dict, optional
        Registration parameters. If None, default values are used.
    
    Returns
    -------
    resampled_labels : ndarray
        Label image resampled into diffusion space.
    """
    # Load label image
    labels = nib.load(label_filepath)
    labels_data = labels.get_fdata()
    
    if trim_z:
        labels_data = labels_data[:, :, trim_slices[0]:trim_slices[1], ...]

    # Default registration parameters
    if reg_params is None:
        reg_params = {
            "nbins": 32,
            "level_iters": [10000, 1000, 100],
            "sigmas": [3.0, 1.0, 0.0],
            "factors": [4, 2, 1],
            "pipeline": ["center_of_mass", "translation", "rigid"]
        }

    # Extract image data
    diffusion_data = diffusion_img.get_fdata()
    T2_data = T2_img.get_fdata()

    # Use first diffusion volume for registration
    moving = diffusion_data[...,0]

    # Perform affine registration from diffusion (moving) to T2 (static)
    xformed_img, reg_affine = affine_registration(
        moving,
        T2_data,
        moving_affine=diffusion_affine,
        static_affine=T2_affine,
        nbins=reg_params["nbins"],
        metric='MI',
        pipeline=reg_params["pipeline"],
        level_iters=reg_params["level_iters"],
        sigmas=reg_params["sigmas"],
        factors=reg_params["factors"]
    )

    # Invert the transform to map from T2 → diffusion space
    inv_reg_affine = np.linalg.inv(reg_affine)

    # Use same first volume as reference
    diff_ref = diffusion_data[...,0]

    # Create affine map: source = T2, target = diffusion
    affine_map = AffineMap(
        inv_reg_affine,
        diff_ref.shape, diffusion_affine,
        T2_data.shape, T2_affine
    )

    # Swap axes if required (same as your example)
    switched_labels = np.swapaxes(labels_data, 1, 2)
    flipped_label = np.flip(switched_labels, axis=1)

    # Transform the label image using nearest-neighbour interpolation
    resampled_labels = affine_map.transform(flipped_label, interpolation='nearest')
    print(f" Resampled label image shape: {resampled_labels.shape}")

    return resampled_labels




def create_overlay(label_img, data_img, output_path, title_prefix="Overlay"):
    z = data_img.shape[2] // 2
    plt.figure(figsize=(6, 6))
    plt.imshow(label_img[:, :, z].T, cmap='gray', origin='lower')
    plt.imshow(data_img[:, :, z].T, cmap='hot', alpha=0.6, origin='lower')
    plt.title(f"{title_prefix} Slice {z}")
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def extract_regions_of_interest(label_image, fa_map, md_map):
    atlas_regions = {
        "Corpus Callosum Right": [8],
        "Corpus Callosum Left": [68],
        "Internal Capsule Left": [12],
        "Internal Capsule Right": [112],
        "Hippocampus Right": [6],
        "Hippocampus Left": [106],
        "Thalamus Left": [4],
        "Thalamus Right": [204],
        "Cortex Right": [209, 64, 130, 181],
        "Cortex Left": [230, 190, 164, 180],
    }
    stats = {}
    for name, labels in atlas_regions.items():
        mask = np.isin(label_image, labels)
        fa_vals = fa_map[mask]
        md_vals = md_map[mask]
        if fa_vals.size and md_vals.size:
            stats[name] = {
                "FA Mean": float(np.mean(fa_vals)),
                "FA Std": float(np.std(fa_vals)),
                "MD Mean": float(np.mean(md_vals)),
                "MD Std": float(np.std(md_vals)),
            }
        else:
            stats[name] = {k: None for k in ["FA Mean", "FA Std", "MD Mean", "MD Std"]}
    return stats


def process_dataset(warp_md_path, warp_fa_path, eddy_path, label_path, T2_path, output_dir):
    logger = configure_logging(output_dir)
    try:
        os.makedirs(output_dir, exist_ok=True)
        md_data, _ = load_nifti_data(warp_md_path)
        fa_data, _ = load_nifti_data(warp_fa_path)
        diffusion_img = nib.load(eddy_path)
        diffusion_affine = diffusion_img.affine
        T2_img = nib.load(T2_path)
        T2_affine = T2_img.affine
        resampled_labels = resample_labels_to_diffusion_space(
            label_path, diffusion_img, T2_img,
            diffusion_affine, T2_affine
        )
        save_nifti(
            os.path.join(output_dir, 'labels_resampled.nii.gz'),
            resampled_labels.astype(np.int16),
            diffusion_affine
        )
        create_overlay(
            resampled_labels, fa_data,
            os.path.join(output_dir, 'fa_overlay.png'), 'FA Overlay'
        )
        create_overlay(
            resampled_labels, md_data,
            os.path.join(output_dir, 'md_overlay.png'), 'MD Overlay'
        )
        stats = extract_regions_of_interest(resampled_labels, fa_data, md_data)
        with open(os.path.join(output_dir, 'roi_stats.json'), 'w') as f:
            json.dump(stats, f, indent=4)
        logger.info(f"Saved ROI stats and overlays in {output_dir}")
        return stats
    except Exception as e:
        logger.error(f"Error: {e}")
        return None

def batch_process(fa_md_dir, label_dir, t2_dir, output_base_dir):
    os.makedirs(output_base_dir, exist_ok=True)
    master_csv = os.path.join(output_base_dir, 'master_roi_stats.csv')

    if os.path.exists(master_csv):
        summary_df = pd.read_csv(master_csv)
    else:
        summary_df = pd.DataFrame(
            columns=['Subject','Region','FA Mean','FA Std','MD Mean','MD Std']
        )

    for ds in glob.glob(os.path.join(fa_md_dir, '*_loaded')):
        subj  = os.path.basename(ds)
        subid = subj.split('_')[2]

        # --- skip if already processed ---
        if subid in summary_df['Subject'].values:
            print(f"  {subj} already processed; skipping.")
            continue


        warp_md = os.path.join(ds, 'md_full.nii')
        warp_fa = os.path.join(ds, 'fa_full.nii.gz')
        eddy    = os.path.join(ds, 'eddy_corrected_full.nii.gz')
        label   = os.path.join(label_dir, subid,
                            'reorient_ex_vivo_strip_warped_label.nii.gz')
        t2_path = os.path.join(t2_dir, subj,
                            '2', 'pdata', '1', 'niiobj_1.nii.gz')

        # debug which are missing
        missing = [n for n,p in
                   {'warp_md':warp_md,'warp_fa':warp_fa,'label':label,'t2':t2_path}.items()
                   if not os.path.exists(p)]
        if missing:
            print(f"Skipping {subj}: missing {missing}")
            continue

        #  returns the stats dict
        stats = process_dataset(
            warp_md, warp_fa, eddy, label, t2_path,
            os.path.join(output_base_dir, subj)
        )
        if stats is None:
            continue

        # append/update master CSV
        records = []
        for region, vals in stats.items():
            records.append({
                'Subject': subid,
                'Region':  region,
                'FA Mean': vals.get('FA Mean'),
                'FA Std':  vals.get('FA Std'),
                'MD Mean': vals.get('MD Mean'),
                'MD Std':  vals.get('MD Std'),
            })
        # remove any old rows for this subject, then append new
        summary_df = pd.concat([summary_df, pd.DataFrame(records)], ignore_index=True)
        summary_df.to_csv(master_csv, index=False)
        print(f"Updated master CSV for {subj}")

if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser(
        description="Batch process FA/MD maps, resample labels, extract ROI stats and build master CSV."
    )
    p.add_argument('--fa_md_dir',       required=True,
                   help="Folder of *_loaded dirs containing md_nomotion.nii & fa_nomotion.nii.gz")
    p.add_argument('--label_dir',       required=True,
                   help="ex_vivo_final_data base directory for warped labels")
    p.add_argument('--t2_dir',          required=True,
                   help="sad_lab_data base directory where each subj has 2/pdata/1/niiobj_1.nii.gz")
    p.add_argument('--output_base_dir', required=True,
                   help="Where to write per-subject outputs and master_roi_stats.csv")

    args = p.parse_args()
    batch_process(
        fa_md_dir        = args.fa_md_dir,
        label_dir        = args.label_dir,
        t2_dir           = args.t2_dir,
        output_base_dir  = args.output_base_dir
    )
