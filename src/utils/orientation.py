import numpy as np
import nibabel as nib

def reorient_to_RAS(volume: np.ndarray, affine: np.ndarray):
    """
    Reorient volume to RAS+ and then standardize axis order to (S, A, R).
    
    This ensures that the array dimensions always correspond to:
    - Axis 0: Superior (Axial slices move along this)
    - Axis 1: Anterior (Coronal slices move along this)
    - Axis 2: Right (Sagittal slices move along this)
    
    Args:
        volume (np.ndarray): 3D or 4D image array
        affine (np.ndarray): 4x4 affine matrix
        
    Returns:
        volume_std (np.ndarray): Reoriented volume in (S, A, R) order
        affine_std (np.ndarray): Updated affine for the new volume
    """
    # 1. Create NIfTI image
    img = nib.Nifti1Image(volume, affine)
    
    # 2. Reorient to RAS+ (Standard NIfTI orientation: Right, Anterior, Superior)
    # This handles all flips and axis permutations automatically.
    # The resulting image has spatial axes ordered (R, A, S).
    img_ras = nib.as_closest_canonical(img)
    vol_ras = img_ras.get_fdata(dtype=np.float32)
    aff_ras = img_ras.affine
    
    # 3. Transpose axes from (R, A, S) -> (S, A, R)
    # We want Axis 0 to be S (which was Axis 2), 
    # Axis 1 to be A (which was Axis 1),
    # Axis 2 to be R (which was Axis 0).
    # Permutation for 3D: (2, 1, 0)
    # Permutation for 4D: (2, 1, 0, 3)
    if vol_ras.ndim == 4:
        vol_std = np.transpose(vol_ras, (2, 1, 0, 3))
    else:
        vol_std = np.transpose(vol_ras, (2, 1, 0))
    
    # 4. Update Affine Matrix
    # We need to map standard indices -> world coordinates.
    # world = aff_ras @ ras_indices
    # ras_indices = T @ std_indices
    # So: world = aff_ras @ T @ std_indices
    # New Affine = aff_ras @ T
    
    # Create transformation matrix T for permutation (2, 1, 0)
    # std[0] = ras[2]
    # std[1] = ras[1]
    # std[2] = ras[0]
    T = np.eye(4)
    T[0, :] = [0, 0, 1, 0] # std 0 comes from ras 2
    T[1, :] = [0, 1, 0, 0] # std 1 comes from ras 1
    T[2, :] = [1, 0, 0, 0] # std 2 comes from ras 0
    
    aff_std = aff_ras @ T
    
    return vol_std, aff_std