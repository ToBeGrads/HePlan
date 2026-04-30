import os
import numpy as np
import SimpleITK as sitk
import nibabel as nib
from src.core.patient_data import PatientData
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

class PatientLoader:
    def load_dicom_series(self, series_folder: str, metadata: dict) -> PatientData:
        logger.info(f"Loading DICOM series from: {series_folder}")
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(series_folder)
        if not dicom_names:
            raise ValueError("No valid DICOM files found in the selected series folder.")

        reader.SetFileNames(dicom_names)
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

        metadata = {"Modality": "MRI", "FileName": os.path.basename(file_path)}
        return PatientData(
            volume=volume, spacing=spacing, origin=origin, direction=direction,
            affine=affine, metadata=metadata, modality="MRI", file_path=file_path, is_loaded=True
        )

    def _sitk_to_patient_data(self, sitk_image, metadata: dict, path: str, modality: str) -> PatientData:
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
            "SeriesDescription": metadata.get("description", "")
        }

        return PatientData(
            volume=volume, spacing=spacing, origin=origin, direction=direction,
            affine=affine, metadata=clean_meta, modality=modality, file_path=path, is_loaded=True
        )