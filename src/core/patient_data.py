import numpy as np
from dataclasses import dataclass, field
from typing import Dict

@dataclass
class PatientData:
    volume: np.ndarray
    spacing: tuple
    origin: tuple
    direction: np.ndarray
    affine: np.ndarray
    metadata: Dict[str, str] = field(default_factory=dict)
    modality: str = "UNKNOWN"
    file_path: str = ""
    is_loaded: bool = False

    def get_shape(self):
        return self.volume.shape

    def voxel_to_world(self, ijk):
        ijk_h = np.array([*ijk, 1.0])
        world = self.affine @ ijk_h
        return tuple(world[:3])

    def world_to_voxel(self, xyz):
        xyz_h = np.array([*xyz, 1.0])
        ijk = np.linalg.inv(self.affine) @ xyz_h
        return tuple(np.round(ijk[:3]).astype(int))