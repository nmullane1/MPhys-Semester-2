#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 15:43:38 2025

@author: alexbralsford
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated pipeline wrapper for diffusion and T2 processing.
This script wraps your VSCode cells into an automated, batch‐processing pipeline.
It configures the environment, sets up logging, and iterates through multiple datasets.
"""

import os
import glob
import logging
import subprocess
import numpy as np
import nibabel as nib
import napari
import ants
import matplotlib.pyplot as plt
from scipy.stats import norm
from nibabel.processing import resample_from_to
from skimage.metrics import normalized_mutual_information as mi

# DIPY imports
from dipy.denoise.localpca import mppca
import dipy.reconst.dti as dti
from dipy.io.image import load_nifti, save_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
import dipy.denoise.noise_estimate as ne
from dipy.align import motion_correction, affine_registration
from dipy.denoise.gibbs import gibbs_removal
from dipy.denoise.patch2self import patch2self
from dipy.denoise.noise_estimate import piesno
from dipy.viz import regtools
from dipy.align.imaffine import AffineMap, MutualInformationMetric, AffineRegistration
from dipy.align.transforms import TranslationTransform3D, RigidTransform3D, AffineTransform3D
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import CCMetric






# External skull stripping function (assumed to be available)
from skull_strip import skull_strip

###############################################################################
# Logging and Environment Setup
###############################################################################
def configure_logging(output_dir):
    """
    Configure logging to file and console.
    """
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "process_log.txt")
    logger = logging.getLogger("PipelineLogger")
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

def establish_environment():
    """
    Set up the FSL environment variables and PATH.
    """
    os.environ["FSLDIR"] = "/Users/alexbralsford/fsl"
    os.environ["FSLOUTPUTTYPE"] = "NIFTI_GZ"
    fsl_paths = ["/Users/alexbralsford/fsl/bin", "/Users/alexbralsford/fsl/share/fsl/bin"]
    system_paths = ["/usr/local/bin", "/usr/bin", "/bin", "/usr/sbin", "/sbin"]
    os.environ["PATH"] = ":".join(fsl_paths + system_paths) + ":" + os.environ.get("PATH", "")
    print("FSLDIR:", os.environ.get("FSLDIR"))
    print("FSLOUTPUTTYPE:", os.environ.get("FSLOUTPUTTYPE"))
    print("Complete PATH:", os.environ["PATH"])

###############################################################################
# Data Loading Functions
###############################################################################
def load_data(diffusion_path, T2_path, output_dir, logger):
    """
    Load diffusion and T2 data from the provided paths.
    Saves a copy of the raw images and then trims the first slice along the z-axis.
    Returns the trimmed data along with updated affine transforms and NIfTI images.
    """
    try:
        # -------------------------
        # Load raw diffusion and T2 data
        # -------------------------
        # Load diffusion data (returns data, affine and the NIfTI image)
        diffusion_data, diffusion_affine, diffusion_nifti = load_nifti(diffusion_path, return_img=True)
        # Load T2 data (returns data, affine and the NIfTI image)
        T2_data, T2_affine, T2_nifti = load_nifti(T2_path, return_img=True)
        
        # Save raw images for reference
        raw_diffusion_path = os.path.join(output_dir, "raw_diffusion.nii.gz")
        raw_T2_path = os.path.join(output_dir, "raw_T2.nii.gz")
        nib.save(nib.Nifti1Image(diffusion_data, diffusion_affine, header=diffusion_nifti.header), raw_diffusion_path)
        nib.save(nib.Nifti1Image(T2_data, T2_affine, header=T2_nifti.header), raw_T2_path)
        
        # -------------------------
        # Remove the first slice along the z-axis
        # -------------------------
        
        # For diffusion data:
        # Compute new affine by shifting the origin by one slice thickness along z
        new_affine = diffusion_nifti.affine.copy()
        new_affine[:3, 3] += new_affine[:3, 2] * 1  # shift origin by one slice
        
        # Trim the diffusion data (assumes data shape is [X, Y, Z, ...])
        trimmed_diffusion = diffusion_data[:, :, 1:, ...]
        # Update header to match new shape
        new_header = diffusion_nifti.header.copy()
        new_header.set_data_shape(trimmed_diffusion.shape)
        # Create a new NIfTI image with the trimmed data
        diffusion_data = trimmed_diffusion
        diffusion_affine = new_affine
        diffusion_nifti = nib.Nifti1Image(diffusion_data, diffusion_affine, header=new_header)
        
        # For T2 data:
        t2_new_affine = T2_nifti.affine.copy()
        t2_new_affine[:3, 3] += t2_new_affine[:3, 2] * 1  # shift origin by one slice
        
        trimmed_T2 = T2_data[:, :, 1:, ...]
        T2_header = T2_nifti.header.copy()
        T2_header.set_data_shape(trimmed_T2.shape)
        T2_data = trimmed_T2
        T2_affine = t2_new_affine
        T2_nifti = nib.Nifti1Image(T2_data, T2_affine, header=T2_header)
        
        # -------------------------
        # Save the updated (trimmed) images
        # -------------------------
        trimmed_diffusion_path = os.path.join(output_dir, "trimmed_diffusion.nii.gz")
        trimmed_T2_path = os.path.join(output_dir, "trimmed_T2.nii.gz")
        nib.save(diffusion_nifti, trimmed_diffusion_path)
        nib.save(T2_nifti, trimmed_T2_path)
        
        logger.info("Loaded and trimmed diffusion and T2 data successfully.")
        return diffusion_data, diffusion_affine, diffusion_nifti, T2_data, T2_affine, T2_nifti
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None, None, None, None, None, None


def load_method_file(method_file_path, logger):
    """
    Load the method.npy file to extract B-values and B-vectors.
    Normalizes the bvecs so that each vector has unit length.
    """
    try:
        method = np.load(method_file_path, allow_pickle=True).item()
        # Convert to numpy arrays
        bval = np.array(method['PVM_DwEffBval'])
        bvec = np.array(method['PVM_DwGradVec'])
        # Transpose if needed: we want shape (N, 3)
        if bvec.shape[0] == 3:
            bvec = bvec.T
        # Normalize each b-vector to unit length using safe division
        norms = np.linalg.norm(bvec, axis=1, keepdims=True)
        bvec = bvec / np.clip(norms, 1e-8, np.inf)
        logger.info("Loaded method file successfully with normalized bvecs.")
        return bval, bvec
    except Exception as e:
        logger.error(f"Error loading method file from {method_file_path}: {e}")
        return None, None


###############################################################################
# Processing Functions
###############################################################################
def denoise_mppca(diffusion_data, diffusion_nifti, output_dir, logger):
    """
    Denoise diffusion data using MPPCA.
    """
    try:
        image_denoised = mppca(diffusion_data)
        denoised_path = os.path.join(output_dir, "image_denoised_mppca.nii.gz")
        nib.save(nib.Nifti1Image(image_denoised, diffusion_nifti.affine), denoised_path)
        image_denoised_img = nib.load(denoised_path)
        image_denoised = image_denoised_img.get_fdata()
        logger.info("MPPCA denoising completed.")
        return image_denoised
    except Exception as e:
        logger.error(f"Error in MPPCA denoising: {e}")
        return None

def apply_gibbs(image_denoised, diffusion_nifti, output_dir, logger):
    """
    Apply Gibbs removal to the denoised image.
    """
    try:
        gibbs_corrected = gibbs_removal(image_denoised, slice_axis=2, num_processes=-1)
        gibbs_path = os.path.join(output_dir, "gibbs_removal_mppca_ai.nii.gz")
        nib.save(nib.Nifti1Image(gibbs_corrected, diffusion_nifti.affine, header=diffusion_nifti.header), gibbs_path)
        logger.info("Gibbs removal completed.")
        return gibbs_corrected
    except Exception as e:
        logger.error(f"Error in Gibbs removal: {e}")
        return None

def perform_motion_correction(gibbs_corrected, gtab, affine, logger):
    """
    Perform motion correction using the average of the first five b0 volumes as reference.
    """
    try:
        modified_data = gibbs_corrected.copy()
        b0_avg = np.mean(gibbs_corrected[..., :5], axis=3)
        modified_data[..., 0] = b0_avg
        dwi_motion_corrected, reg_affines = motion_correction(modified_data, gtab, affine,
                                                              pipeline=["center_of_mass", "translation", "rigid"])
        logger.info("Motion correction completed.")
        return dwi_motion_corrected.get_fdata()
    except Exception as e:
        logger.error(f"Error in motion correction: {e}")
        return None

def perform_eddy_correction(diffusion_data, affine, base_output, logger, method_path):
    """
    Run eddy current correction using FSL's eddy.
    This version:
      - Saves the motion-corrected diffusion image.
      - Generates a b1000-based mask by averaging volumes 5 onward.
      - Skull-strips the b1000 average image (using "bet") to create a binary brain mask.
      - Prepares input files (bvals, bvecs, acqparams, index) for eddy.
      - Executes the eddy command using the new b1000-based mask.
    
    NOTE: bvals and bvecs are loaded using load_method_file, ensuring they are normalized.
    """
    try:
        # Save the motion-corrected diffusion image as the starting point.
        motion_file = os.path.join(base_output, "motion.nii.gz")
        nib.save(nib.Nifti1Image(diffusion_data, affine), motion_file)
        eddy_out = os.path.join(base_output, "eddy_corrected.nii.gz")
        
        # ------------------------------
        # Create b1000-based Mask
        # ------------------------------
        # Load the saved motion image.
        motion_img = nib.load(motion_file)
        diff_data = motion_img.get_fdata()         # Expected shape: (X, Y, Z, #volumes)
        diff_affine = motion_img.affine
        
        # Assuming the first 5 volumes are b0 images, average volumes 5 onward (b1000 images).
        b1000_avg = np.mean(diff_data[..., 5:], axis=3)
        b1000_avg_file = os.path.join(base_output, "b1000_avg.nii.gz")
        nib.save(nib.Nifti1Image(b1000_avg, diff_affine, header=motion_img.header), b1000_avg_file)
        logger.info(f"b1000 average image saved at: {b1000_avg_file}")
        
        # Skull-strip the b1000 average image to generate a mask (using "bet").
        logger.info("Skull stripping b1000 average image for mask generation...")
        skull_stripped_b1000 = skull_strip(b1000_avg_file, "bet")
        logger.info(f"Skull stripped b1000 image saved at: {skull_stripped_b1000}")
        
        # Load the skull-stripped image and create a binary mask.
        mask_img_b1000 = nib.load(skull_stripped_b1000)
        mask_b1000_data = (mask_img_b1000.get_fdata() > 0).astype(np.float64)
        mask_output = os.path.join(base_output, "mask_from_b1000.nii.gz")
        nib.save(nib.Nifti1Image(mask_b1000_data, diff_affine, header=mask_img_b1000.header), mask_output)
        logger.info(f"b1000-based mask saved at: {mask_output}")
        
        # ------------------------------
        # Prepare Eddy Correction Input Files
        # ------------------------------
        # Use the centralized load_method_file to obtain normalized bvals and bvecs.
        bvals, bvecs = load_method_file(method_path, logger)
        if bvals is None or bvecs is None:
            logger.error("Failed to load method file for eddy correction.")
            return None
        
        # Save the bvals and bvecs files.
        bvals_file = os.path.join(base_output, "bvals")
        bvecs_file = os.path.join(base_output, "bvecs")
        np.savetxt(bvals_file, bvals, fmt='%d')
        np.savetxt(bvecs_file, np.array(bvecs).T, fmt='%.6f')
        
        # Create the acqparams file.
        acqparams_path = os.path.join(base_output, "acqparams.txt")
        with open(acqparams_path, "w") as f:
            f.write("-1 0 0 0.068\n")
        
        # Create the index file.
        index_path = os.path.join(base_output, "index.txt")
        with open(index_path, "w") as f:
            f.write("1 " * len(bvals))
        
        # ------------------------------
        # Run Eddy Correction (FSL's eddy)
        # ------------------------------
        eddy_cmd = f"""/bin/bash -l -c \"
eddy --imain='{motion_file}' \\
     --mask='{mask_output}' \\
     --acqp='{acqparams_path}' \\
     --index='{index_path}' \\
     --bvecs='{bvecs_file}' \\
     --bvals='{bvals_file}' \\
     --fwhm=0 \\
     --flm=quadratic \\
     --out='{eddy_out}'
\""""
        logger.info("Running eddy correction with command:")
        logger.info(eddy_cmd)
        subprocess.run(eddy_cmd, shell=True, check=True, env=os.environ)
        logger.info(f"Eddy correction completed. Output saved at: {eddy_out}")
        return eddy_out
    except Exception as e:
        logger.error(f"Error in eddy correction: {e}")
        return None


def bias_field_correction(diffusion_data, diffusion_nifti, b0_path, dwi_path, output_b0_corrected, output_dwi_corrected):
    """
    Perform bias field correction using ANTs' N4 algorithm.
    This function implements the logic from your VSCode cell.
    """
    # Create reference image (average of the first 5 volumes)
    reference_image = np.mean(diffusion_data[..., :5], axis=-1)
    reference_nifti = nib.Nifti1Image(reference_image, affine=diffusion_nifti.affine, header=diffusion_nifti.header)
    nib.save(reference_nifti, b0_path)
    try:
        img = nib.load(b0_path)
    except Exception as e:
        print(f"Error loading b0 image from {b0_path}: {e}")
        return None
    affine_img = img.affine
    data = img.get_fdata()
    U, _, Vt = np.linalg.svd(affine_img[:3, :3])
    orth_affine = np.eye(4)
    orth_affine[:3, :3] = np.dot(U, Vt)
    orth_affine[:3, 3] = affine_img[:3, 3]
    fixed_path = f"{b0_path}_fixed.nii"
    fixed_img = nib.Nifti1Image(data, orth_affine, header=img.header)
    nib.save(fixed_img, fixed_path)
    print(f"Saved orthonormalised version as {fixed_path}")
    try:
        b0_ants = ants.image_read(fixed_path)
    except Exception as e:
        print(f"Error reading fixed image with ANTs from {fixed_path}: {e}")
        return None
    b0_corrected = ants.n4_bias_field_correction(b0_ants, shrink_factor=2,
                                                  convergence={'iters': [50, 50, 30, 20], 'tol': 1e-6})
    b0_corrected.to_filename(output_b0_corrected)
    b0_ants_np = b0_ants.numpy()
    b0_corrected_np = b0_corrected.numpy()
    denom = np.where(b0_corrected_np == 0, 1, b0_corrected_np)
    bias_field_np = b0_ants_np / denom
    bias_field = ants.from_numpy(
        bias_field_np,
        spacing=b0_ants.spacing,
        origin=b0_ants.origin,
        direction=b0_ants.direction
    )
    bias_field_filename = os.path.join(os.path.dirname(output_b0_corrected), "bias_field.nii")
    bias_field.to_filename(bias_field_filename)
    try:
        dwi_img = nib.load(dwi_path)
    except Exception as e:
        print(f"Error loading DWI image from {dwi_path}: {e}")
        return None
    dwi_data = dwi_img.get_fdata()
    try:
        bias_data = nib.load(bias_field_filename).get_fdata()
    except Exception as e:
        print(f"Error loading bias field from {bias_field_filename}: {e}")
        return None
    if dwi_data.ndim != 4:
        print("Error: DWI image does not have 4 dimensions.")
        return None
    # Repeat the bias field across the 4th dimension of the DWI data
    bias_4d = np.repeat(bias_data[:, :, :, np.newaxis], dwi_data.shape[3], axis=3)
    # Make division safe by clipping bias_4d to a minimum value
    bias_4d_safe = np.clip(bias_4d, a_min=1e-6, a_max=None)
    dwi_corrected = dwi_data / bias_4d_safe
    dwi_img_corrected = nib.Nifti1Image(dwi_corrected, dwi_img.affine, dwi_img.header)
    nib.save(dwi_img_corrected, output_dwi_corrected)
    return dwi_img_corrected

def run_bias_field_correction(input_dwi_file, diffusion_nifti, output_dir, logger, suffix=""):
    """
    Wrapper to run bias field correction on a diffusion dataset.
    The 'input_dwi_file' may be either the eddy-corrected data or the motion-corrected (eddy uncorrected) data.
    The 'suffix' is appended to output filenames so that the two pipeline branches are saved separately.
    """
    try:
        b0_path = os.path.join(output_dir, f"b0_avg{suffix}.nii")
        output_b0_corrected = os.path.join(output_dir, f"b0_corrected{suffix}.nii")
        output_dwi_corrected = os.path.join(output_dir, f"dwi_bias_corrected{suffix}.nii")
        
        # Load the input diffusion image.
        input_img = nib.load(input_dwi_file)
        input_data = input_img.get_fdata()
        logger.info(f"Input diffusion data shape for bias correction{suffix}: {input_data.shape}")
        
        # Run the bias field correction using your existing function.
        bias_corrected_img = bias_field_correction(
            diffusion_data=input_data,
            diffusion_nifti=input_img,
            b0_path=b0_path,
            dwi_path=input_dwi_file,
            output_b0_corrected=output_b0_corrected,
            output_dwi_corrected=output_dwi_corrected
        )
        if bias_corrected_img is not None:
            logger.info(f"Bias field correction{suffix} complete!")
            logger.info(f"Corrected DWI saved to: {output_dwi_corrected}")
            return output_dwi_corrected
        else:
            logger.error("Bias field correction failed.")
            return None
    except Exception as e:
        logger.error(f"Error in bias field correction: {e}")
        return None



def tensor_model_fit(diffusion_file, bvals, bvecs, affine, nifti_header, logger, suffix=""):
    """
    Fit the diffusion tensor model using DIPY's standard TensorModel.
    Assumes that bvals and bvecs are pre-processed and normalized.
    This version forces the data to float64 for numerical stability and
    ignores eddy-uncorrected data.
    Saves FA, MD, eigenvalues, and eigenvectors as NIfTI files with an appended suffix.
    
    Parameters:
      - diffusion_file: path to the diffusion dataset (bias-field corrected)
      - bvals, bvecs: normalized b-values and b-vectors
      - affine: affine transformation matrix to use in the output images
      - nifti_header: header from the original diffusion NIfTI image
      - logger: logging instance for progress reporting
      - suffix: optional string appended to the output filenames to distinguish branches
    """
    try:
        # Load the bias-corrected diffusion data
        data_img = nib.load(diffusion_file)
        data = data_img.get_fdata()
        logger.info(f"Data shape for tensor fitting (suffix {suffix}): {data.shape}")
        
        # Convert data to float64 for improved numerical stability.
        data = data.astype(np.float64)
        
        # Ensure bvecs are in the shape (N, 3)
        bvals = np.array(bvals)
        bvecs = np.array(bvecs)
        if bvecs.shape[0] == 3:
            bvecs = bvecs.T
        
        # Construct the gradient table using the pre-processed bvals and bvecs.
        gtab = gradient_table(bvals, bvecs, atol=1)
        
        # Fit the tensor model using DIPY's standard least-squares approach.
        tenmodel = dti.TensorModel(gtab)
        fit_bias = tenmodel.fit(data)
        
        # Extract tensor-derived parameters.
        fa = fit_bias.fa
        md = fit_bias.md
        evals = fit_bias.evals
        evecs = fit_bias.evecs
        
        # Define output file paths (append suffix to distinguish branches).
        base_dir = os.path.dirname(diffusion_file)
        fa_path = os.path.join(base_dir, f"fa_bias{suffix}.nii.gz")
        md_path = os.path.join(base_dir, f"md_bias{suffix}.nii.gz")
        evals_path = os.path.join(base_dir, f"evals_bias{suffix}.nii.gz")
        evecs_path = os.path.join(base_dir, f"evecs_bias{suffix}.nii.gz")
        
        # Save the computed maps as NIfTI images.
        nib.save(nib.Nifti1Image(fa, affine, header=nifti_header), fa_path)
        nib.save(nib.Nifti1Image(md, affine, header=nifti_header), md_path)
        nib.save(nib.Nifti1Image(evals, affine, header=nifti_header), evals_path)
        nib.save(nib.Nifti1Image(evecs, affine, header=nifti_header), evecs_path)
        
        logger.info(f"Tensor model fitting complete (suffix {suffix}). FA and MD maps saved.")
    except Exception as e:
        logger.error(f"Error in tensor model fitting (suffix {suffix}): {e}")




###############################################################################
# Dataset and Batch Processing
###############################################################################

def process_dataset(dataset_dir, output_base_dir, logger):
    """
    Process a single dataset and run all processing and analysis steps.

    Assumes the following folder structure within dataset_dir:
      - Diffusion data: {dataset_dir}/3/pdata/1/niiobj_1.nii
      - T2 data:        {dataset_dir}/2/pdata/1/niiobj_1.nii.gz
      - Method file:    {dataset_dir}/3/method.npy

    This pipeline now creates two branches:
      (1) Gibs → eddy → bias → tensor fitting (eddy-corrected branch)
      (2) Gibs → motion → bias → tensor fitting (motion-corrected branch)
    """
    
    import os
    from dipy.core.gradients import gradient_table
    output_dir = os.path.join(output_base_dir, os.path.basename(dataset_dir))
    fa_path = os.path.join(output_dir, "fa_bias_eddy.nii.gz")
    
    if os.path.exists(fa_path):
        logger.info(f"Skipping {dataset_dir} — already processed.")
        return

    # Define file paths based on the expected folder structure.
    diffusion_path = os.path.join(dataset_dir, "3", "pdata", "1", "niiobj_1.nii")
    T2_path = os.path.join(dataset_dir, "2", "pdata", "1", "niiobj_1.nii.gz")
    method_path = os.path.join(dataset_dir, "3", "method.npy")
    
    # Create an output folder for this dataset.
    dataset_output_dir = os.path.join(output_base_dir, os.path.basename(dataset_dir))
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    # Load the trimmed diffusion and T2 data.
    diffusion_data, diffusion_affine, diffusion_nifti, \
      T2_data, T2_affine, T2_nifti = load_data(diffusion_path, T2_path, dataset_output_dir, logger)
    if diffusion_data is None or T2_data is None:
        logger.error(f"Skipping dataset {dataset_dir} due to data loading error.")
        return
    
    # Load the method file (returns normalized bvals and bvecs).
    bvals, bvecs = load_method_file(method_path, logger)
    if bvals is None or bvecs is None:
        logger.error(f"Skipping dataset {dataset_dir} due to method file error.")
        return

    # Create the gradient table for processing where needed.
    gtab = gradient_table(bvals, bvecs, atol=1)
    
    # Run initial processing steps.
    image_denoised = denoise_mppca(diffusion_data, diffusion_nifti, dataset_output_dir, logger)
    if image_denoised is None:
        logger.error("Denoising failed, skipping dataset.")
        return
    
    gibbs_corrected = apply_gibbs(image_denoised, diffusion_nifti, dataset_output_dir, logger)
    if gibbs_corrected is None:
        logger.error("Gibbs removal failed, skipping dataset.")
        return

    # ----- Branch A: Eddy-corrected Pipeline -----
    # (Use gibbs-corrected data directly for eddy correction.)
    eddy_corrected_file = perform_eddy_correction(gibbs_corrected,
                                                  diffusion_affine,
                                                  dataset_output_dir, 
                                                  logger,
                                                  method_path)
    if eddy_corrected_file is None:
        logger.error("Eddy correction failed (Branch A), skipping dataset.")
        return

    # ----- Branch B: Motion-corrected Pipeline -----
    # (Apply motion correction to gibbs-corrected data.)
    dwi_motion_corrected = perform_motion_correction(gibbs_corrected, gtab, diffusion_affine, logger)
    if dwi_motion_corrected is None:
        logger.error("Motion correction failed (Branch B), skipping dataset.")
        return

    # -------------------------
    # Run bias field correction for Branch A and Branch B.
    # -------------------------
    # For eddy-corrected (Branch A) data.
    dwi_bias_corrected_eddy = run_bias_field_correction(eddy_corrected_file,
                                                        diffusion_nifti,
                                                        dataset_output_dir,
                                                        logger,
                                                        suffix="_eddy")
    # For motion-corrected (Branch B) data.
    # (Assumes that perform_motion_correction writes the output as "motion.nii.gz".)
    motion_file = os.path.join(dataset_output_dir, "motion.nii.gz")
    dwi_bias_corrected_motion = run_bias_field_correction(motion_file,
                                                          diffusion_nifti,
                                                          dataset_output_dir,
                                                          logger,
                                                          suffix="_motion")
    if dwi_bias_corrected_eddy is None or dwi_bias_corrected_motion is None:
        logger.error("Bias field correction failed for Branch A or Branch B.")
        return

    # Run tensor model fitting for Branch A and Branch B.
    tensor_model_fit(dwi_bias_corrected_eddy, bvals, bvecs,
                     diffusion_affine, diffusion_nifti.header, logger, suffix="_eddy")
    tensor_model_fit(dwi_bias_corrected_motion, bvals, bvecs,
                     diffusion_affine, diffusion_nifti.header, logger, suffix="_motion")

    # ----- Branch C: Motion then Eddy Pipeline -----
    # Use the motion-corrected data (already computed as dwi_motion_corrected) as input to eddy.
    # Create a separate folder for the Branch C outputs.
    branch_c_dir = os.path.join(dataset_output_dir, "motion_then_eddy")
    os.makedirs(branch_c_dir, exist_ok=True)
    motion_then_eddy_file = perform_eddy_correction(dwi_motion_corrected,
                                                     diffusion_affine,
                                                     branch_c_dir,
                                                     logger,
                                                     method_path)
    if motion_then_eddy_file is None:
        logger.error("Eddy correction on motion-corrected data (Branch C) failed, skipping Branch C.")
    else:
        # Run bias field correction for Branch C.
        dwi_bias_corrected_motion_eddy = run_bias_field_correction(motion_then_eddy_file,
                                                                   diffusion_nifti,
                                                                   branch_c_dir,
                                                                   logger,
                                                                   suffix="_motion_eddy")
        if dwi_bias_corrected_motion_eddy is None:
            logger.error("Bias field correction failed for Branch C.")
        else:
            # Run tensor model fitting for Branch C.
            tensor_model_fit(dwi_bias_corrected_motion_eddy, bvals, bvecs,
                             diffusion_affine, diffusion_nifti.header, logger, suffix="_motion_eddy")

    

def batch_process(root_dir, output_base_dir):
    """
    Batch process all datasets found in root_dir.
    
    Expects that each dataset folder's name ends with '_loaded'.
    """
    import glob, os
    dataset_dirs = glob.glob(os.path.join(root_dir, "*_loaded"))
    logger = configure_logging(output_base_dir)
    establish_environment()
    for dataset_dir in dataset_dirs:
        logger.info(f"Processing dataset: {dataset_dir}")
        process_dataset(dataset_dir, output_base_dir, logger)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Automated diffusion pipeline wrapper.")
    parser.add_argument("--root_dir", required=True, help="Root directory containing dataset folders (ending with _loaded)")
    parser.add_argument("--output_base_dir", required=True, help="Directory to save outputs")
    args = parser.parse_args()
    batch_process(args.root_dir, args.output_base_dir)
