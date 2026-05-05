import os
import numpy as np
import SimpleITK as sitk
import nibabel as nib
from src.core.patient_data import PatientData
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

class PatientLoader:
    def load_dicom_series(self, series_folder: str, metadata: dict) -> PatientData:
        series_uid = metadata.get("series_uid")
        desc = metadata.get("description", "Unknown")
        logger.info(f"Loading series: {desc} (UID: {series_uid}) from {series_folder}")

        reader = sitk.ImageSeriesReader()
        
        if series_uid and series_uid != "UNKNOWN":
            dicom_names = reader.GetGDCMSeriesFileNames(series_folder, series_uid)
        else:
            dicom_names = reader.GetGDCMSeriesFileNames(series_folder)

        if not dicom_names:
            raise ValueError(f"No DICOM files found for '{desc}' in {series_folder}")

        print(f"📂 Loading {len(dicom_names)} files for '{desc}'...")
        reader.SetFileNames(dicom_names)
        reader.SetForceOrthogonalDirection(False)
        image = reader.Execute()

        modality = metadata.get("modality", "MRI")
        return self._sitk_to_patient_data(image, metadata, series_folder, modality)

    def load_nifti(self, file_path: str) -> PatientData:
        logger.info(f"Loading NIfTI file: {file_path}")
        nii = nib.load(file_path)
        volume = np.asanyarray(nii.dataobj, dtype=np.float32)
        affine = nii.affine

        spacing = tuple(np.sqrt(np.sum(affine[:3, :3]**2, axis=0)))
        origin = tuple(affine[:3, 3])
        direction = affine[:3, :3] / np.array(spacing)

        meta = {"Modality": "MRI", "FileName": os.path.basename(file_path)}
        return PatientData(
            volume=self._sanitize_volume(volume), spacing=spacing, origin=origin, direction=direction,
            affine=affine, metadata=meta, modality="MRI", file_path=file_path, is_loaded=True
        )

    def _sitk_to_patient_data(self, sitk_image: sitk.Image, metadata: dict, path: str, modality: str) -> PatientData:
        volume = sitk.GetArrayFromImage(sitk_image).astype(np.float32)
        spacing = sitk_image.GetSpacing()
        origin = sitk_image.GetOrigin()
        direction = np.array(sitk_image.GetDirection()).reshape(3, 3)

        affine = np.eye(4)
        affine[:3, :3] = direction * np.array(spacing)
        affine[:3, 3] = origin

        clean_meta = {
            "PatientID": metadata.get("patient_id", "Unknown"),
            "PatientName": metadata.get("patient_name", "Unknown"),
            "StudyDate": metadata.get("study_date", ""),
            "Modality": modality,
            "SeriesDescription": metadata.get("description", ""),
            "SeriesUID": metadata.get("series_uid", "")
        }

        return PatientData(
            volume=self._sanitize_volume(volume), spacing=spacing, origin=origin, direction=direction,
            affine=affine, metadata=clean_meta, modality=modality, file_path=path, is_loaded=True
        )

    def _sanitize_volume(self, vol: np.ndarray) -> np.ndarray:
        if vol.ndim == 4 and vol.shape[-1] == 1:
            vol = vol.squeeze(-1)
        if vol.ndim != 3:
            raise ValueError(f"Expected 3D volume, got shape {vol.shape}")

        vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)
        p1, p99 = np.percentile(vol, 1), np.percentile(vol, 99)
        vol = np.clip(vol, p1, p99)
        return vol.astype(np.float32)