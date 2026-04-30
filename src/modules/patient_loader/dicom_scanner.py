import os
import pydicom
from typing import Dict
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

def scan_dicom_directory(root_path: str) -> Dict:
    """Recursively scan DICOM dir and return hierarchical structure: Patient -> Study -> Series"""
    patients = {}

    for dirpath, _, filenames in os.walk(root_path):
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            try:
                ds = pydicom.dcmread(fpath, stop_before_pixels=True, force=True)
                if not hasattr(ds, 'SOPInstanceUID'):
                    continue

                pid = str(getattr(ds, 'PatientID', 'Unknown')).strip()
                pname = str(getattr(ds, 'PatientName', 'Unknown')).strip()
                sid = str(getattr(ds, 'StudyInstanceUID', 'Unknown')).strip()
                sdate = str(getattr(ds, 'StudyDate', '')).strip()
                ser_uid = str(getattr(ds, 'SeriesInstanceUID', 'Unknown')).strip()
                modality = str(getattr(ds, 'Modality', 'UNKNOWN')).strip()
                desc = str(getattr(ds, 'SeriesDescription', '')).strip()

                if pid not in patients:
                    patients[pid] = {"name": pname, "studies": {}}
                if sid not in patients[pid]["studies"]:
                    patients[pid]["studies"][sid] = {"date": sdate, "series": {}}
                if ser_uid not in patients[pid]["studies"][sid]["series"]:
                    patients[pid]["studies"][sid]["series"][ser_uid] = {
                        "modality": modality,
                        "description": desc,
                        "folder": dirpath,
                        "files": set()
                    }
                patients[pid]["studies"][sid]["series"][ser_uid]["files"].add(fpath)
            except Exception:
                continue

    # Clean up: convert file sets to counts, filter invalid series
    clean_patients = {}
    for pid, pdata in patients.items():
        clean_studies = {}
        for sid, sdata in pdata["studies"].items():
            clean_series = {}
            for ser_uid, ser_data in sdata["series"].items():
                if len(ser_data["files"]) >= 2:
                    clean_series[ser_uid] = {
                        "modality": ser_data["modality"],
                        "description": ser_data["description"],
                        "folder": ser_data["folder"],
                        "count": len(ser_data["files"])
                    }
            if clean_series:
                clean_studies[sid] = {"date": sdata["date"], "series": clean_series}
        if clean_studies:
            clean_patients[pid] = {"name": pdata["name"], "studies": clean_studies}

    logger.info(f"Scanned {root_path}: Found {len(clean_patients)} patients")
    return clean_patients