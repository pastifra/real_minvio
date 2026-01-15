import numpy as np
import cv2
import scipy.io
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple
import pickle
import argparse

import os

WINGET_FFMPEG_BIN = (
    r"C:\Users\franc\AppData\Local\Microsoft\WinGet\Packages"
    r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\ffmpeg-8.0.1-full_build\bin"
)

def _sanitize_path_for_ffmpeg():
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    conda_libbin = os.path.join(conda_prefix, "Library", "bin").lower() if conda_prefix else ""
    parts = [p for p in os.environ.get("PATH", "").split(";") if p and p.lower() != conda_libbin]
    os.environ["PATH"] = WINGET_FFMPEG_BIN + ";" + ";".join(parts)

_sanitize_path_for_ffmpeg()

import skvideo
skvideo.setFFmpegPath(WINGET_FFMPEG_BIN)
import skvideo.io
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))
import utils

class LinearCameraModel:
    def __init__(self, v_poly: np.ndarray, undistortion_parameter_file: str | Path):
        """
        Initialize a linear camera model with calibration data.
        Assume the pose of the help camera is identical to the pose of the detectors.
        """
        with open(undistortion_parameter_file, "rb") as f:
            D = pickle.load(f)
            self.remap_params = D["remap_params"]
            self.undistort_img_size = D["undistort_img_size"]
            self.undistort_img_vert_fov = D["undistort_img_vert_fov"]
            self.undistort_img_horiz_fov = D["undistort_img_horiz_fov"]
            self.K_cam = D["K_cam"]
            self.K_undistort = D["K_undistort"]
            self.cam_img_size = D["cam_img_size"]
        
            self.output_img_size = self.remap_params["map1"].shape

        self.v_poly = v_poly
        self._create_inv_vignetting_mask()

    def _create_inv_vignetting_mask(self):
        r, c = np.meshgrid(
            np.arange(self.cam_img_size[0]),
            np.arange(self.cam_img_size[1]),
            indexing="ij"
        )
        xy = np.stack((c.ravel(), r.ravel()), axis=1) # Nx2
        radius = np.sqrt( ((xy - self.K_cam[:2,2][None,:])**2).sum(1) )

        v = np.polyval(self.v_poly, radius.ravel()).reshape(self.cam_img_size[:2])
        self.inv_v = 1 / (v + 1e-10)

    def remove_vignetting(self, img):
        assert img.shape[:2] == self.cam_img_size[:2]
        assert img.dtype == np.float32 or img.dtype == np.float64

        # Broadcast to color channels if necessary
        if img.ndim == 3:
            inv_v = self.inv_v[:,:,None]
        else:
            inv_v = self.inv_v

        return img * inv_v

    def undistort_image(self, img: np.ndarray):
        assert img.shape[:2] == self.cam_img_size[:2]
        img_undistort = cv2.remap(img.astype(np.float32), **self.remap_params)
        assert img_undistort.shape[:2] == self.undistort_img_size[:2]

        return img_undistort


def load_helper_camera_model():
    undistortion_parameter_file = utils.get_data_path() / \
        "camera-calibration" / "undistortion_params.pkl"

    v_poly_file = utils.get_data_path() / "camera-calibration" / \
        "vignetting_polynomial.mat"
    v_poly = scipy.io.loadmat(v_poly_file)["p"].squeeze()

    M = LinearCameraModel(v_poly, undistortion_parameter_file)

    return M


def convert_image():
    M = load_helper_camera_model()

    img_path = utils.get_data_path() / "70fov.bmp"

    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = img.astype(np.float32) / 255
    img = M.remove_vignetting(img)
    img = M.undistort_image(img)
    img = np.clip(img, 0, 1)

    fig, ax = plt.subplots()
    ax.imshow(utils.gamma_correct(img))    
    ax.axis("off")
    fig.show()

    plt.show()


def parse_args():
    args = argparse.ArgumentParser("Offline Odometry")
    args.add_argument("--name", "-n", type=str, required=True, default=None, help="experiment name in video-data/ folder (without .mp4)")
    return args.parse_args()

def convert_video():
    args = parse_args()

    input_video = utils.get_data_path() / "video-data" / (args.name + ".mp4")
    output_video = input_video.parent / (args.name + "-converted.mp4")

    M = load_helper_camera_model()

    # Fix numpy incompatibility with scipy
    np.float = float    
    np.int = int
    np.object = object
    np.bool = bool

    fps = 60
    crf = 22
    writer = skvideo.io.FFmpegWriter(
        output_video,
        inputdict={'-r': str(fps)},
        outputdict={'-vcodec': "libx264", 
                    '-crf': str(crf), 
                    '-threads': "32",
                    '-r': str(fps),
                    '-pix_fmt': 'yuv420p'})
    reader = skvideo.io.vreader(str(input_video))

    for frame in tqdm(reader):
        frame = frame.astype(np.float32) / 255
        frame = M.remove_vignetting(frame)
        frame = M.undistort_image(frame)
        frame = np.clip(frame, 0, 1)
        frame = (frame * 255).astype(np.uint8)
        frame = utils.gamma_correct(frame)

        writer.writeFrame(frame)
    
    reader.close()
    writer.close()


if __name__ == "__main__":
    #convert_image()
    convert_video()