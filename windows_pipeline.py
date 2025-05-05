#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline for pre-processsing of ex vivo data, testing three branches: eddy-only, motion-only and combined.
Produces tensor maps MD and FA for processed images and saves the nifti files after each processing step.

05/04/25
Nicola Mullane

"""

import os
import nibabel as nib
import numpy as np
import ants
import subprocess
import glob
import logging
from dipy.denoise.localpca import mppca
import dipy.reconst.dti as dti
from dipy.io.image import load_nifti, save_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
import dipy.denoise.noise_estimate as ne
from dipy.align import motion_correction
from dipy.viz import regtools
from dipy.align.imaffine import (AffineMap, MutualInformationMetric, AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D, RigidTransform3D, AffineTransform3D)
from dipy.align import affine_registration, register_dwi_to_template
from dipy.denoise.gibbs import gibbs_removal
from dipy.reconst.dti import fractional_anisotropy, color_fa
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import CCMetric
from dipy.denoise.patch2self import patch2self
from dipy.denoise.noise_estimate import piesno
import matplotlib.pyplot as plt
from scipy.stats import norm
from dipy.align import resample
from dipy.viz.regtools import overlay_slices

############################################
#   Get External Mask from Diffusion Folder
############################################

def get_external_mask_path(diffusion_path):
    """
    Compute the external mask path from the diffusion file path.
    
    Expects the diffusion file to be located at:
      <dataset_dir>/3/pdata/1/niiobj_1.nii/niiobj_1.nii
      
    and the external mask to be located at:
      <dataset_dir>/3/pdata/1/niiobj_1_refined_mask.nii
    """
    try:
        diffusion_dir = os.path.dirname(diffusion_path)
        parent_dir = os.path.dirname(diffusion_dir)
        mask_filename = "niiobj_1_refined_mask.nii"
        mask_path = os.path.join(parent_dir, "1", mask_filename)
        if not os.path.exists(mask_path):
            print(f"External mask not found at: {mask_path}")
            return None
        return mask_path
    except Exception as e:
        print(f"Error obtaining external mask: {e}")
        return None

############################################
#   Other Helper Functions
############################################

def configure_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "process_log.txt")
    logger = logging.getLogger("DatasetLogger")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

def load_nifti_data(file_path, return_img=False):
    try:
        if return_img:
            data, affine, nifti = load_nifti(file_path, return_img=True)
            return data, affine, nifti
        else:
            data, affine = load_nifti(file_path)
            return data, affine
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None, None

def load_method_file(method_path):
    try:
        method = np.load(method_path, allow_pickle=True).item()
        bval = method['PVM_DwEffBval']
        bvec = method['PVM_DwGradVec']
        return bval, bvec
    except Exception as e:
        print(f"Error loading method file {method_path}: {e}")
        return None, None

def perform_patch2self_denoising(data, bvals, nifti, save_path):
    try:
        image_denoised = mppca(data)
        nib.save(nib.Nifti1Image(image_denoised, nifti.affine, header=nifti.header), save_path)
        return image_denoised
    except Exception as e:
        print(f"Error during Patch2Self denoising: {e}")
        return None

def perform_gibbs_correction(input_path, save_path, nifti):
    try:
        image = nib.load(input_path).get_fdata()
        corrected = gibbs_removal(image, slice_axis=2, num_processes=-1)
        nib.save(nib.Nifti1Image(corrected, nifti.affine, header=nifti.header), save_path)
        return corrected
    except Exception as e:
        print(f"Error during Gibbs correction: {e}")
        return None

def perform_motion_correction_with_retries(nifti, bval, bvec, affine, motion_file_path):
    try:
        gtab = gradient_table(bval, bvec, atol=1)
        data = nifti.get_fdata()
        if data.ndim != 4 or data.shape[3] < 5:
            print("Error: Provided image must be 4D and have at least 5 volumes for b0 averaging.")
            return None, None
        b0_avg = np.mean(data[..., :5], axis=3)
        modified_data = data.copy()
        modified_data[..., 0] = b0_avg
        modified_nifti = nib.Nifti1Image(modified_data, affine, nifti.header)
        pipeline_variants = [
            ["center_of_mass", "translation", "rigid"],
            ["translation", "rigid"],
            ["rigid"]
        ]
        for attempt, pipeline in enumerate(pipeline_variants, start=1):
            try:
                print(f"Attempt {attempt}: Using pipeline {pipeline}")
                dwi_corrected, reg_affines = motion_correction(modified_nifti, gtab, affine, pipeline=pipeline)
                dwi_corrected_data = dwi_corrected.get_fdata()
                nib.save(nib.Nifti1Image(dwi_corrected_data, affine), motion_file_path)
                return dwi_corrected, reg_affines
            except Exception as e:
                print(f"Motion correction failed on attempt {attempt}: {e}")
        return None, None
    except Exception as e:
        print(f"Error during motion correction: {e}")
        return None, None

def prepare_eddy_files(method_path, output_dir):
    method = np.load(method_path, allow_pickle=True).item()
    bvals = method['PVM_DwEffBval']
    bvecs = method['PVM_DwGradVec']
    np.savetxt(os.path.join(output_dir, "bvals"), bvals, fmt='%d')
    np.savetxt(os.path.join(output_dir, "bvecs"), np.array(bvecs).T, fmt='%.6f')
    with open(os.path.join(output_dir, "acqparams.txt"), "w") as f:
        f.write("-1 0 0 0.068\n")
    with open(os.path.join(output_dir, "index.txt"), "w") as f:
        f.write("1 " * len(bvals))
    return os.path.join(output_dir, "acqparams.txt"), os.path.join(output_dir, "index.txt")

def convert_to_wsl_path(win_path):
    win_path = os.path.abspath(win_path)
    win_path = win_path.replace("\\", "/")
    if ":" in win_path:
        drive, path = win_path.split(":", 1)
        return f"/mnt/{drive.lower()}{path}"
    return win_path

def run_eddy_correction(dwi_path, mask_path, acqparams_path, index_path, bvecs_path, bvals_path, output_path):
    dwi_wsl = convert_to_wsl_path(dwi_path)
    mask_wsl = convert_to_wsl_path(mask_path)
    acqparams_wsl = convert_to_wsl_path(acqparams_path)
    index_wsl = convert_to_wsl_path(index_path)
    bvecs_wsl = convert_to_wsl_path(bvecs_path)
    bvals_wsl = convert_to_wsl_path(bvals_path)
    output_wsl = convert_to_wsl_path(output_path)
    print(f"Converted paths: {dwi_wsl}, {mask_wsl}, {acqparams_wsl}, {output_wsl}, {index_wsl}, {bvecs_wsl}, {bvals_wsl}")
    eddy_cmd = (
        f'wsl bash -ic "source /usr/local/fsl/etc/fslconf/fsl.sh && '
        f'eddy --imain={dwi_wsl} '
        f'--mask={mask_wsl} '
        f'--acqp={acqparams_wsl} '
        f'--index={index_wsl} '
        f'--bvecs={bvecs_wsl} '
        f'--bvals={bvals_wsl} '
        f'--fwhm=0 --flm=quadratic '
        f'--out={output_wsl}"'
    )
    print("Running eddy command:")
    print(eddy_cmd)
    env = os.environ.copy()
    env["FSLOUTPUTTYPE"] = "NIFTI"
    env["PATH"] += ":/usr/local/fsl/bin"
    result = subprocess.run(eddy_cmd, shell=True, capture_output=True, text=True, env=env)
    print("Eddy stdout:")
    print(result.stdout)
    print("Eddy stderr:")
    print(result.stderr)
    if result.returncode != 0:
        print("Eddy command failed with return code:", result.returncode)
    else:
        print("Eddy command executed successfully.")

def fix_affine_to_orthonormal(nifti_path):
    """
    Load a NIfTI image, compute an orthonormal affine using SVD,
    and save the fixed image to a new file.
    """
    img = nib.load(nifti_path)
    data = img.get_fdata()
    affine = img.affine
    U, _, Vt = np.linalg.svd(affine[:3, :3])
    ortho_affine = np.eye(4)
    ortho_affine[:3, :3] = U @ Vt
    ortho_affine[:3, 3] = affine[:3, 3]
    out_path = nifti_path.replace(".nii", "_fixed.nii")
    nib.save(nib.Nifti1Image(data, ortho_affine, img.header), out_path)
    return out_path

############################################
#   Updated Bias Field Correction Function
############################################
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import os
import numpy as np


def bias_field_correction(diffusion_data, diffusion_nifti, b0_path, dwi_path,
                          output_b0_corrected, output_dwi_corrected, mask_path):
    """
    Perform bias field correction on a 4D DWI image.
    
    Steps:
      1. Compute the b0 reference image (average of the first five volumes) and save it.
      2. Fix the affine of the b0 image to be orthonormal.
      3. If a mask is provided, fix and threshold it to form a binary mask.
      4. Perform ANTs' N4 bias field correction on the fixed b0 image.
      5. Compute the bias field and apply it to correct the full 4D DWI data.
    
    Returns:
        A nibabel Nifti1Image of the bias-corrected DWI.
    """
    # Compute the b0 reference image by averaging the first 5 volumes.
    reference_image = np.mean(diffusion_data[..., :5], axis=-1)
    reference_nifti = nib.Nifti1Image(reference_image, diffusion_nifti.affine, header=diffusion_nifti.header)
    nib.save(reference_nifti, b0_path)
    print(f" Saved b0 reference image as {b0_path}")
    
    # Load the b0 image and fix its affine.
    try:
        img = nib.load(b0_path)
    except Exception as e:
        print(f"Error loading b0 image from {b0_path}: {e}")
        return None
    affine = img.affine
    data = img.get_fdata()
    U, _, Vt = np.linalg.svd(affine[:3, :3])
    orth_affine = np.eye(4)
    orth_affine[:3, :3] = U @ Vt
    orth_affine[:3, 3] = affine[:3, 3]
    fixed_path = f"{b0_path}_fixed.nii"
    fixed_img = nib.Nifti1Image(data, orth_affine, header=img.header)
    nib.save(fixed_img, fixed_path)
    print(f" Saved orthonormalised b0 as {fixed_path}")
    
    # Prepare an optional mask for bias correction.
    ants_mask = None
    if mask_path is not None and os.path.exists(mask_path):
        fixed_mask_path = fix_affine_to_orthonormal(mask_path)
        ants_mask = ants.threshold_image(ants.image_read(fixed_mask_path), 0.5, 1.1, 1, 0)
        print(f"✅ Loaded and thresholded mask from {mask_path}")
    
    # Apply ANTs' N4 bias field correction on the fixed b0 image.
    try:
        b0_ants = ants.image_read(fixed_path)
    except Exception as e:
        print(f"Error reading fixed image with ANTs from {fixed_path}: {e}")
        return None
    
    b0_corrected = ants.n4_bias_field_correction(
        b0_ants,
        shrink_factor=2,
        convergence={'iters': [50, 50, 30, 20], 'tol': 1e-6},
        mask=ants_mask
    )
    b0_corrected.to_filename(output_b0_corrected)
    print(f" Saved bias-corrected b0 as {output_b0_corrected}")
    
    # Compute the bias field (ratio between original and corrected b0) and save it.
    bias_field = b0_ants / b0_corrected
    bias_field_filename = "bias_field.nii"
    bias_field.to_filename(bias_field_filename)
    print(f" Saved bias field as {bias_field_filename}")
    
    # Load the full 4D DWI image and apply the bias correction.
    if not os.path.exists(dwi_path):
        print(f"Error: DWI file not found at {dwi_path}")
        return None
    try:
        dwi_img = nib.load(dwi_path)
    except Exception as e:
        print(f"Error loading DWI image from {dwi_path}: {e}")
        return None
    dwi_data = dwi_img.get_fdata()
    if dwi_data.ndim != 4:
        print("Error: DWI image does not have 4 dimensions.")
        return None
    
    bias_data = nib.load(bias_field_filename).get_fdata()
    bias_4d = np.repeat(bias_data[..., np.newaxis], dwi_data.shape[3], axis=3)
    dwi_corrected = dwi_data / bias_4d
    
    dwi_img_corrected = nib.Nifti1Image(dwi_corrected, dwi_img.affine, dwi_img.header)
    nib.save(dwi_img_corrected, output_dwi_corrected)
    print(f" Saved corrected DWI: {output_dwi_corrected}")

    return dwi_img_corrected

def perform_tensor_model_fit(data_corrected, diffusion_affine, nifti, gtab,
                             save_path_fa, save_path_md, save_path_evals, save_path_evecs,
                             output_dir, logger, mask_path=None):
    try:
        if mask_path and os.path.exists(mask_path):
            mask_data = nib.load(mask_path).get_fdata()
            mask = mask_data > 0.5
        else:
            # Default to internal brain mask
            mask = data_corrected[..., 0] > 0.05

        tenmodel = dti.TensorModel(gtab)
        fit = tenmodel.fit(data_corrected, mask=mask)

        fa = fit.fa
        md = fit.md
        evals = fit.evals
        evecs = fit.evecs

        nib.save(nib.Nifti1Image(fa, diffusion_affine, header=nifti.header), save_path_fa)
        nib.save(nib.Nifti1Image(md, diffusion_affine, header=nifti.header), save_path_md)
        nib.save(nib.Nifti1Image(evals, diffusion_affine, header=nifti.header), save_path_evals)
        nib.save(nib.Nifti1Image(evecs, diffusion_affine, header=nifti.header), save_path_evecs)

        mean_fa = np.nanmean(fa[mask])
        mean_md = np.nanmean(md[mask])
        logger.info(f"The FA mean is: {mean_fa}")
        logger.info(f"The MD mean is: {mean_md}")
    except Exception as e:
        print(f"Error during tensor model fitting: {e}")


def compute_bias_field_stats(original_b0, corrected_b0, mask, output_dir):
    """
    Computes and compares statistics for original and bias-corrected b0 images,
    and plots an overlaid histogram showing intensity distributions inside the mask.
    """
    # Ensure mask is boolean
    mask = mask > 0

    # Extract voxel values inside the mask
    orig_inside = original_b0[mask]
    corr_inside = corrected_b0[mask]

    if orig_inside.size == 0 or corr_inside.size == 0:
        print(" No valid voxels inside mask for computing stats.")
        return

    # Compute statistics
    stats = {}
    for label, data in zip(["Original", "Corrected"], [orig_inside, corr_inside]):
        mean_val = np.mean(data)
        std_val = np.std(data)
        cov = std_val / mean_val
        stats[label] = {"mean": mean_val, "std": std_val, "cov": cov}
        print(f" {label} b0 — Mean: {mean_val:.4f}, Std: {std_val:.4f}, CoV: {cov:.4f}")

    # Determine common bin range
    all_data = np.concatenate([orig_inside, corr_inside])
    bins = np.linspace(np.min(all_data), np.max(all_data), 100)

    # Plot overlaid histograms
    plt.figure(figsize=(10, 6))
    plt.hist(orig_inside, bins=bins, alpha=0.5, label=f"Original (CoV={stats['Original']['cov']:.3f})", 
             color="royalblue", edgecolor="black")
    plt.hist(corr_inside, bins=bins, alpha=0.5, label=f"Corrected (CoV={stats['Corrected']['cov']:.3f})", 
             color="tomato", edgecolor="black")
    plt.title("Bias Field Intensity Histogram (Inside Mask)")
    plt.xlabel("Intensity Value")
    plt.ylabel("Voxel Count")
    plt.legend()
    plt.grid(True)

    hist_path = os.path.join(output_dir, "bias_field_overlay_histogram.png")
    plt.savefig(hist_path)
    plt.close()
    print(f" Saved overlaid bias field histogram to {hist_path}")




############################################
#   Modified Process Dataset Function (Three Branches)
############################################

def process_dataset(diffusion_path, T2_path, output_dir):
    logger = configure_logging(output_dir)
    try:
        # Define common file paths
        denoised_save_path = os.path.join(output_dir, "image_denoised.nii")
        gibbs_save_path = os.path.join(output_dir, "image_gibbs.nii")
        motion_save_path = os.path.join(output_dir, "motion.nii")
        mask_save_path = os.path.join(output_dir, "mask.nii") 
        os.makedirs(output_dir, exist_ok=True)

        # Load diffusion and T2 images
        diffusion_data, diffusion_affine, diffusion_nifti = load_nifti_data(diffusion_path, return_img=True)
        T2_data, T2_affine, T2_nifti = load_nifti_data(T2_path, return_img=True)

        # Get external mask from the diffusion folder
        external_mask_path = get_external_mask_path(diffusion_path)
        if not external_mask_path:
            logger.error("External mask not found. Skipping dataset.")
            return

        dataset_dir = os.path.abspath(os.path.join(diffusion_path, "../../../../"))
        method_file_path = os.path.join(dataset_dir, "3", "method.npy")
        if not os.path.exists(method_file_path):
            print(f"Method file not found in {method_file_path}. Skipping dataset.")
            return

        bval, bvec = load_method_file(method_file_path)
        if bval is None or bvec is None:
            logger.error("B-values or B-vectors are missing.")
            return

        gtab = gradient_table(bval, bvec, atol=1)
        bvals = np.array(bval)

        # Preprocessing: Denoising and Gibbs Correction
        denoised_data = perform_patch2self_denoising(diffusion_data, bvals, diffusion_nifti, denoised_save_path)
        if denoised_data is None:
            logger.error("Denoising failed. Skipping dataset.")
            return

        gibbs_corrected_data = perform_gibbs_correction(denoised_save_path, gibbs_save_path, diffusion_nifti)
        if gibbs_corrected_data is None:
            logger.error("Gibbs correction failed. Skipping dataset.")
            return

        # Motion Correction (if applicable)
        dwi_motion_corrected, reg_affines = perform_motion_correction_with_retries(
            nib.load(gibbs_save_path), bval, bvec, diffusion_affine, motion_save_path
        )
        if dwi_motion_corrected is None:
            logger.error("Motion correction failed. Skipping dataset.")
            return

        # Prepare eddy correction files 
        acqparams_path, index_path = prepare_eddy_files(method_file_path, output_dir)
        bvecs_path = os.path.join(output_dir, "bvecs")
        bvals_path = os.path.join(output_dir, "bvals")

        ############################
        # Branch 1: Full Pipeline (Motion + Eddy)
        ############################
        full_eddy_output_prefix = os.path.join(output_dir, "eddy_corrected_full")
        run_eddy_correction(motion_save_path, external_mask_path, acqparams_path, index_path,
                            bvecs_path, bvals_path, full_eddy_output_prefix)
        full_eddy_file = full_eddy_output_prefix + ".nii.gz"
        # Tensor fitting on eddy-corrected image before bias correction
        eddy_fa_path = os.path.join(output_dir, "fa_no_bias.nii.gz")
        eddy_md_path = os.path.join(output_dir, "md_no_bias.nii")

        perform_tensor_model_fit(
            nib.load(full_eddy_file).get_fdata(),
            diffusion_affine, diffusion_nifti, gtab,
            eddy_fa_path, eddy_md_path,
            os.path.join(output_dir, "evals_no_bias.nii"),
            os.path.join(output_dir, "evecs_no_bias.nii"),
            output_dir, logger
        )

        full_b0_path = os.path.join(output_dir, "b0_avg_full.nii")
        full_b0_corrected_path = os.path.join(output_dir, "b0_corrected_full.nii")
        full_bias_dwi_path = os.path.join(output_dir, "dwi_bias_corrected_full.nii")
        full_bias_corrected_img = bias_field_correction(nib.load(full_eddy_file).get_fdata(), diffusion_nifti,
                                                        full_b0_path, full_eddy_file, full_b0_corrected_path,
                                                        full_bias_dwi_path, external_mask_path)
        full_fa_path = os.path.join(output_dir, "fa_full.nii.gz")
        full_md_path = os.path.join(output_dir, "md_full.nii")
        full_evals_path = os.path.join(output_dir, "evals_full.nii")
        full_evecs_path = os.path.join(output_dir, "evecs_full.nii")
        perform_tensor_model_fit(full_bias_corrected_img.get_fdata(), diffusion_affine, diffusion_nifti, gtab,
                                   full_fa_path, full_md_path, full_evals_path, full_evecs_path, output_dir, logger)
        original_b0 = nib.load(full_b0_path).get_fdata()
        corrected_b0 = nib.load(full_b0_corrected_path).get_fdata()
        mask = nib.load(external_mask_path).get_fdata()
        compute_bias_field_stats(original_b0, corrected_b0, mask, output_dir)

        ############################
        # Branch 2: Eddy Uncorrected (After Motion, No Eddy)
        ############################
        noeddy_b0_path = os.path.join(output_dir, "b0_avg_noeddy.nii")
        noeddy_b0_corrected_path = os.path.join(output_dir, "b0_corrected_noeddy.nii")
        noeddy_bias_dwi_path = os.path.join(output_dir, "dwi_bias_corrected_noeddy.nii")
        noeddy_bias_corrected_img = bias_field_correction(nib.load(motion_save_path).get_fdata(), diffusion_nifti,
                                                          noeddy_b0_path, motion_save_path, noeddy_b0_corrected_path,
                                                          noeddy_bias_dwi_path, external_mask_path)
        noeddy_fa_path = os.path.join(output_dir, "fa_noeddy.nii.gz")
        noeddy_md_path = os.path.join(output_dir, "md_noeddy.nii")
        noeddy_evals_path = os.path.join(output_dir, "evals_noeddy.nii")
        noeddy_evecs_path = os.path.join(output_dir, "evecs_noeddy.nii")
        perform_tensor_model_fit(noeddy_bias_corrected_img.get_fdata(), diffusion_affine, diffusion_nifti, gtab,
                                   noeddy_fa_path, noeddy_md_path, noeddy_evals_path, noeddy_evecs_path, output_dir, logger)

        ############################
        # Branch 3: Without Motion (Skip Motion Correction)
        ############################
        nomotion_eddy_output_prefix = os.path.join(output_dir, "eddy_corrected_nomotion")
        run_eddy_correction(gibbs_save_path, external_mask_path, acqparams_path, index_path,
                            bvecs_path, bvals_path, nomotion_eddy_output_prefix)
        nomotion_eddy_file = nomotion_eddy_output_prefix + ".nii.gz"
        nomotion_b0_path = os.path.join(output_dir, "b0_avg_nomotion.nii")
        nomotion_b0_corrected_path = os.path.join(output_dir, "b0_corrected_nomotion.nii")
        nomotion_bias_dwi_path = os.path.join(output_dir, "dwi_bias_corrected_nomotion.nii")
        nomotion_bias_corrected_img = bias_field_correction(nib.load(nomotion_eddy_file).get_fdata(), diffusion_nifti,
                                                             nomotion_b0_path, nomotion_eddy_file, nomotion_b0_corrected_path,
                                                             nomotion_bias_dwi_path, external_mask_path)
        nomotion_fa_path = os.path.join(output_dir, "fa_nomotion.nii.gz")
        nomotion_md_path = os.path.join(output_dir, "md_nomotion.nii")
        nomotion_evals_path = os.path.join(output_dir, "evals_nomotion.nii")
        nomotion_evecs_path = os.path.join(output_dir, "evecs_nomotion.nii")
        perform_tensor_model_fit(nomotion_bias_corrected_img.get_fdata(), diffusion_affine, diffusion_nifti, gtab,
                                   nomotion_fa_path, nomotion_md_path, nomotion_evals_path, nomotion_evecs_path, output_dir, logger)

        logger.info("DTI processing complete for dataset.")
    except Exception as e:
        print(f"Error processing dataset {diffusion_path}: {e}")

def batch_process(root_dir):
    output_base_dir = os.path.join(root_dir, "low_snr_not_in_my_spin_echo")
    os.makedirs(output_base_dir, exist_ok=True)
    dataset_dirs = glob.glob(os.path.join(root_dir, "*_loaded"))

    for dataset_dir in dataset_dirs:
        diffusion_path = os.path.join(dataset_dir, "3", "pdata", "1", "niiobj_1.nii")
        T2_path = os.path.join(dataset_dir, "2", "pdata", "1", "niiobj_1.nii")

        if not os.path.exists(diffusion_path) or not os.path.exists(T2_path):
            print(f"Skipping dataset {dataset_dir}: Missing required files")
            continue

        output_dir = os.path.join(output_base_dir, os.path.basename(dataset_dir))
        fa_path = os.path.join(output_dir, "fa_full.nii.gz")  
        log_path = os.path.join(output_dir, "process_log.txt")

        # Skip already processed datasets
        if os.path.exists(fa_path) or os.path.exists(log_path):
            print(f"✅ Skipping already processed dataset: {os.path.basename(dataset_dir)}")
            continue

        print(f"🚀 Processing: {os.path.basename(dataset_dir)}")
        process_dataset(diffusion_path, T2_path, output_dir)


if __name__ == "__main__":
    default_root = "D:/sad_lab_data"
    print(f"Batch processing starting in directory: {default_root}")
    batch_process(default_root)
