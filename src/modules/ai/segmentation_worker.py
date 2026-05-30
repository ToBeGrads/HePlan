import os
import gc
import subprocess
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from PIL import Image
import torch
import shutil
from PyQt5.QtCore import QThread, pyqtSignal
from ultralytics import YOLO
from src.ai.models.unet import UNet

class SegmentationWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(object)  # Returns the 3D numpy array mask
    error = pyqtSignal(str)

    def __init__(self, patient_data, vol_name="Unknown_Volume"):
        super().__init__()
        self.patient_data = patient_data
        self.vol_name = vol_name.replace(" ", "_").replace("/", "_")
        
        self.patient_name = getattr(patient_data, 'patient_name', 
                            getattr(patient_data, 'patient_id', 'Unknown_Patient'))
        self.patient_name = str(self.patient_name).replace(" ", "_").replace("/", "_")
        
        self.crop_size = 224
        
        # Paths
        self.weights_path = "/home/billal/pfe/DBS_CHUB/src/ai/weights/2DUnet.pth"
        self.yolo_path = "/home/billal/pfe/DBS_CHUB/src/ai/weights/Yolo11n.pt"
        self.hd_bet_cache = os.path.join(os.getcwd(), "hd_bet_weights")
        os.makedirs(self.hd_bet_cache, exist_ok=True)
        os.environ["HD_BET_CHECKPOINT_DIR"] = self.hd_bet_cache

    def run(self):
        try:
            # --- Setup Output Directories ---
            base_out_dir = os.path.join(os.getcwd(), "data", "segmentation", self.patient_name, self.vol_name)
            
            dir_skull = os.path.join(base_out_dir, "1_skull_stripped")
            dir_norm = os.path.join(base_out_dir, "2_normalized")
            dir_yolo = os.path.join(base_out_dir, "3_yolo_result")
            dir_unet = os.path.join(base_out_dir, "4_unet_result")
            dir_final = os.path.join(base_out_dir, "5_final_3d")
            img_dir = os.path.join(base_out_dir, "temp_slices") 

            for d in [dir_skull, dir_norm, dir_yolo, dir_unet, dir_final, img_dir]:
                os.makedirs(d, exist_ok=True)

            # Step 1: Save Initial Nifti & Skull Strip
            self.progress.emit("Phase 1/5: Saving initial NIfTI volume data...")
            raw_nii_path = os.path.join(base_out_dir, "mri_raw.nii.gz")
            nib.save(nib.Nifti1Image(self.patient_data.volume, self.patient_data.affine), raw_nii_path)

            original_path = getattr(self.patient_data, 'file_path', '')
            offline_skull_mask = None
            
            if original_path and os.path.exists(original_path):
                directory = os.path.dirname(original_path)
                base_name = os.path.splitext(os.path.basename(original_path))[0].replace('.nii', '')
                potential_sk_path = os.path.join(directory, f"{base_name}_sk.nii.gz")
                
                if os.path.exists(potential_sk_path):
                    offline_skull_mask = potential_sk_path
                    
            skull_path = os.path.join(dir_skull, "mri_skull.nii.gz")

            if offline_skull_mask:
                self.progress.emit("Phase 2/5: Offline skull-strip detected. Loading cache...")
                shutil.copy(offline_skull_mask, skull_path)
            else:
                self.progress.emit("Phase 2/5: Running skull stripping (HD-BET Brain Extraction)...")
                device_flag = "0" if torch.cuda.is_available() else "cpu"
                hd_bet_exe = shutil.which("hd-bet")
                
                if hd_bet_exe is None:
                    raise FileNotFoundError("hd-bet not found! Please run: pip install git+https://github.com/MIC-DKFZ/HD-BET.git")
                
                cmd = [hd_bet_exe, "-i", raw_nii_path, "-o", skull_path, "-device", device_flag, "--disable_tta"]
                
                try:
                    subprocess.run(cmd, check=True, capture_output=True, text=True)
                except subprocess.CalledProcessError as e:
                    raise Exception(f"Skull stripping failed: {e.stderr}")

            # Step 2: Bias Correction & Z-Score Normalization
            self.progress.emit("Phase 3/5: Applying N4 Bias Field Correction & Z-Score Normalization...")
            preproc_path = os.path.join(dir_norm, "mri_preproc.nii.gz")
            self._run_preprocessing(skull_path, preproc_path)

            # Step 3: Slice to 2D PNGs
            self.progress.emit("Phase 4/5: Slicing 3D volume into normalized 2D matrices...")
            depth = self._slice_to_png(preproc_path, img_dir)

            # Step 4: 2D UNet Inference & YOLO
            self.progress.emit("Phase 5/5: Processing object localizer and neural network inference...")
            self._run_inference(img_dir, dir_yolo, dir_unet, depth)

            # Step 5: 2D to 3D Reconstruction
            self.progress.emit("Finalizing: Reconstructing processed 2D masks into 3D space...")
            mask_3d = self._reconstruct_3d(dir_unet, preproc_path)

            # Save final 3D result
            final_nii_path = os.path.join(dir_final, "final_segmentation.nii.gz")
            nib.save(nib.Nifti1Image(mask_3d, self.patient_data.affine), final_nii_path)

            # Clean up temporary PNG slices
            shutil.rmtree(img_dir, ignore_errors=True)

            self.progress.emit("Complete!")
            self.finished.emit(mask_3d)

        except Exception as e:
            self.error.emit(f"Pipeline failed: {str(e)}")

    def _run_preprocessing(self, in_path, out_path):
        sitk_img = sitk.ReadImage(in_path, sitk.sitkFloat32)
        transformed = sitk.RescaleIntensity(sitk_img, 0, 255)
        mask = sitk.LiThreshold(transformed, 0, 1)

        shrink = 4
        img_small = sitk.Shrink(sitk_img, [shrink]*3)
        mask_small = sitk.Shrink(mask, [shrink]*3)

        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([30, 20, 10])
        corrector.Execute(img_small, mask_small)

        log_bias = corrector.GetLogBiasFieldAsImage(sitk_img)
        corrected = sitk_img / sitk.Exp(log_bias)

        corrected_np = np.transpose(sitk.GetArrayFromImage(corrected), (2, 1, 0))
        
        brain_mask = (corrected_np > np.percentile(corrected_np, 2)) & (corrected_np < np.percentile(corrected_np, 98))
        if np.sum(brain_mask) < 1000:
            brain_mask = corrected_np > np.percentile(corrected_np, 5)

        mu, sigma = np.mean(corrected_np[brain_mask]), np.std(corrected_np[brain_mask]) + 1e-8
        z = np.clip((corrected_np - mu) / sigma, -3, 3)
        normalized = ((z + 3) / 6).astype(np.float32)

        nib.save(nib.Nifti1Image(normalized, nib.load(in_path).affine), out_path)

    def _slice_to_png(self, mri_path, out_dir):
        mri = nib.load(mri_path).get_fdata().astype(np.float32)
        num_slices, h, w = mri.shape
        
        sy = (h - self.crop_size) // 2
        sx = (w - self.crop_size) // 2
        
        for z in range(num_slices):
            mri_slice = np.clip(mri[z, sy:sy+self.crop_size, sx:sx+self.crop_size], 0, 1)
            Image.fromarray((mri_slice * 255).astype(np.uint8)).save(
                os.path.join(out_dir, f"mri_z{z:02d}.png")
            )
        return num_slices

    def _reconstruct_3d(self, pred_dir, ref_nii_path):
        ref_nii = nib.load(ref_nii_path)
        ref_3d = ref_nii.get_fdata()
        num_slices, H, W = ref_3d.shape
        
        recon_3d = np.zeros((num_slices, H, W), dtype=np.uint8)
        sy = (H - self.crop_size) // 2
        sx = (W - self.crop_size) // 2

        for z in range(num_slices):
            p_path = os.path.join(pred_dir, f"unet_z{z:02d}.png")
            if os.path.exists(p_path):
                pred_slice = np.array(Image.open(p_path))
                recon_3d[z, sy:sy+self.crop_size, sx:sx+self.crop_size] = pred_slice
                
        return recon_3d

    def _run_inference(self, img_dir, yolo_out_dir, unet_out_dir, depth):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. Load Localizer YOLOv11n
        yolo_model = YOLO(self.yolo_path) 
        yolo_model.to(device)

        # 2. Load Segmenter UNet 
        unet_model = UNet(in_channels=1, num_classes=3, features=[64, 128, 256, 512])
        unet_model.load_state_dict(torch.load(self.weights_path, map_location=device))
        unet_model.to(device).eval()

        # Step 4a: Gather all YOLO hits WITH ORIENTATION MATCHING
        yolo_detected_slices = []
        for z in range(depth):
            img_path = os.path.join(img_dir, f"mri_z{z:02d}.png")
            if not os.path.exists(img_path): continue
            
            # Load image and rotate 90 CCW for YOLO inference
            img = Image.open(img_path).convert('RGB')
            img_ary = np.array(img)
            img_ary_rotated = np.rot90(img_ary, k=1).copy()
            
            results = yolo_model(img_ary_rotated, verbose=False, device=device)
            
            if len(results[0].boxes) > 0:
                yolo_detected_slices.append(z)
                
                # Get the plot image (which is currently sideways and in BGR format)
                res_img_array = results[0].plot()
                
                # Rotate 90 CW to bring the plot back to upright position
                res_img_upright = np.rot90(res_img_array, k=-1)
                
                # Convert BGR (OpenCV) to RGB (PIL) to prevent colors from swapping
                res_img_upright = res_img_upright[..., ::-1]
                
                # Save upright plot natively
                Image.fromarray(res_img_upright).save(os.path.join(yolo_out_dir, f"yolo_z{z:02d}.png"))

        # Step 4b: Contiguity filtering and single-slice gap filling
        if not yolo_detected_slices:
            free_gpu_memory()
            return

        yolo_detected_slices = sorted(list(set(yolo_detected_slices)))
        validated_slices = set()

        # Fill single slice gaps (e.g., if z11 and z13 are present, add z12)
        for idx in range(len(yolo_detected_slices) - 1):
            curr_s = yolo_detected_slices[idx]
            next_s = yolo_detected_slices[idx + 1]
            validated_slices.add(curr_s)
            validated_slices.add(next_s)
            if next_s - curr_s == 2:
                gap_slice = curr_s + 1
                validated_slices.add(gap_slice)
                # Copy an arbitrary adjacent plot prediction for visual consistency in the folder
                src_yolo = os.path.join(yolo_out_dir, f"yolo_z{curr_s:02d}.png")
                dst_yolo = os.path.join(yolo_out_dir, f"yolo_z{gap_slice:02d}.png")
                if os.path.exists(src_yolo):
                    shutil.copy(src_yolo, dst_yolo)

        # Remove single isolated noise slices (only keep if it has a neighbor within a distance of 1)
        final_slices = []
        for s in sorted(list(validated_slices)):
            has_left_neighbor = (s - 1) in validated_slices
            has_right_neighbor = (s + 1) in validated_slices
            if has_left_neighbor or has_right_neighbor:
                final_slices.append(s)
            else:
                # Remove isolated noise plots from folder
                bad_yolo_img = os.path.join(yolo_out_dir, f"yolo_z{s:02d}.png")
                if os.path.exists(bad_yolo_img):
                    os.remove(bad_yolo_img)

        # Step 4c: Run UNet on the clean, continuous collection of slices
        self.progress.emit(f"Phase 5/5: Running targeted UNet array evaluations on structural range...")
        
        for z in final_slices:
            img_path = os.path.join(img_dir, f"mri_z{z:02d}.png")
            if not os.path.exists(img_path): continue
            
            img = Image.open(img_path).convert('L')
            img_ary = np.array(img, dtype=np.float32) / 255.0
            
            # Rotate 90 CCW to match training orientation
            img_ary_rotated = np.rot90(img_ary, k=1)
            img_tensor = torch.from_numpy(img_ary_rotated.copy()).unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                pred = torch.argmax(unet_model(img_tensor), dim=1).squeeze(0).cpu().numpy()
            
            # Rotate back 90 CW to return upright
            pred_final = np.rot90(pred, k=-1)
            
            pred_img = Image.fromarray(pred_final.astype(np.uint8))
            pred_img.save(os.path.join(unet_out_dir, f"unet_z{z:02d}.png"))
            
        free_gpu_memory()

def free_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()