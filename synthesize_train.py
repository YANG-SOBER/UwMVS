from __future__ import absolute_import, division, print_function

import os
import re
import cv2
import json as js
import PIL.Image as pil

from tqdm import tqdm
import random
from sea_thru import *

import skimage.io as skio


def estimate_backscattering(depths, B_inf, beta_B, J_prime, beta_D_prime):
    val = (B_inf * (1 - np.exp(-1 * beta_B * depths))) + (J_prime * np.exp(-1 * beta_D_prime * depths))
    # val = (B_inf * (1 - np.exp(-1 * beta_B * depths)))
    return val


def calculate_beta_D(depths, a, b, c, d):
    return (a * np.exp(b * depths)) + (c * np.exp(d * depths))


def degrade_image(img, depth, B, beta_D, wbalance):
    img[:, :, 0] /= wbalance[0]
    img[:, :, 1] /= wbalance[1]
    img[:, :, 2] /= wbalance[2]
    img = scale(img)
    # ratio_b = np.random.uniform(0.6, 0.8)
    # t = ratio_b * np.exp(-beta_D * np.expand_dims(depth, axis=2)) + 1 - ratio_b
    t = np.exp(-beta_D * np.expand_dims(depth, axis=2))

    # direct = cv2.GaussianBlur(t, (33, 33), 10) * 1.0
    # degrade = direct + B
    # degrade = img * direct + B *ratio_b
    degrade = img * t + B
    degrade = np.maximum(0.0, np.minimum(1.0, degrade))
    return degrade, t, B


def synthesizer(img, depth, coefs):
    Bcoefs_r, Bcoefs_g, Bcoefs_b = np.array(coefs["Bcoefs_r"]), np.array(coefs["Bcoefs_g"]), np.array(coefs["Bcoefs_b"])
    Dcoefs_r, Dcoefs_g, Dcoefs_b = np.array(coefs["Dcoefs_r"]), np.array(coefs["Dcoefs_g"]), np.array(coefs["Dcoefs_b"])
    wbalance = np.array(coefs['wbalance'])

    Br = estimate_backscattering(depth, *Bcoefs_r)
    Bg = estimate_backscattering(depth, *Bcoefs_g)
    Bb = estimate_backscattering(depth, *Bcoefs_b)

    beta_D_r = calculate_beta_D(depth, *Dcoefs_r) * 0.5  # TODO: check validity of 0.5
    beta_D_g = calculate_beta_D(depth, *Dcoefs_g) * 0.5
    beta_D_b = calculate_beta_D(depth, *Dcoefs_b) * 0.5

    B = np.stack([Br, Bg, Bb], axis=2)
    beta_D = np.stack([beta_D_r, beta_D_g, beta_D_b], axis=2)
    degraded, direct, backscatter = degrade_image(img, depth, B, beta_D, wbalance)
    sigma_est = estimate_sigma(degraded, multichannel=True, average_sigmas=True) / 10.0
    degraded = denoise_tv_chambolle(degraded, sigma_est, multichannel=True)

    return degraded, direct, backscatter

def read_pfm(file_path):
    with open(file_path, 'rb') as file:
        # Read the header
        header = file.readline().rstrip()
        if header == b'PF':
            color = True
        elif header == b'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        # Read the dimensions
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('latin-1'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        # Read the scale factor/endianness
        scale = float(file.readline().rstrip())
        if scale < 0: # Little-endian
            endian = '<'
            scale = -scale
        else: # Big-endian
            endian = '>'

        # Read the image data
        data = np.fromfile(file, endian + 'f')
        # Reshape the data into 3 channels for color images
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)  # Flip vertically

        return data, scale

def prepare_img(hr_img):
    # w1600-h1200-> 800-600 ; crop -> 640, 512; downsample 1/4 -> 160, 128

    # downsample
    h, w = hr_img.shape
    hr_img_ds = cv2.resize(hr_img, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)
    # crop
    h, w = hr_img_ds.shape
    target_h, target_w = 512, 640
    start_h, start_w = (h - target_h) // 2, (w - target_w) // 2
    hr_img_crop = hr_img_ds[start_h: start_h + target_h, start_w: start_w + target_w]

    # #downsample
    # lr_img = cv2.resize(hr_img_crop, (target_w//4, target_h//4), interpolation=cv2.INTER_NEAREST)

    return hr_img_crop


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/home/gdyang/Downloads/Data/UW_MVS/UwMVS')
    parser.add_argument('--image_dir', default='/home/gdyang/Downloads/Data/UW_MVS/UwMVS/Rectified_AIR')
    parser.add_argument('--coef_path', default='/home/gdyang/Downloads/Data/Underwater/UW_Types', help='underwater coefficients path')
    parser.add_argument('--depth_dir', default='/home/gdyang/Downloads/Data/UW_MVS/UwMVS/Mono_Depth')
    parser.add_argument('--output_dir', default='/home/gdyang/Downloads/Data/UW_MVS/UwMVS/Rectified_UW')
    #parser.add_argument('--depth_dir', default='/home/gdyang/Downloads/Data/Underwater/Underwater_Images_synthesis/Scene_4/depths', help='Input depth directory')
    #parser.add_argument('--coef_path', default='/home/gdyang/Downloads/Data/Underwater/Underwater_Images_synthesis/scripts/outputs/coeffs', help='Path to the coefficients')
    #parser.add_argument('--output_dir', default='/home/gdyang/Downloads/Data/Underwater/Underwater_Images_synthesis/Scene_4/greenish', help='Output directory')
    parser.add_argument('--f', type=float, default=2.0, help='f value (controls brightness)')
    parser.add_argument('--l', type=float, default=0.5, help='l value (controls balance of attenuation constants)')
    parser.add_argument('--p', type=float, default=0.01, help='p value (controls locality of illuminant map)')
    parser.add_argument('--min-depth', type=float, default=0.0,
                        help='Minimum depth value to use in estimations (range 0-1)')
    parser.add_argument('--max-depth', type=float, default=1.0,
                        help='Replacement depth percentile value for invalid depths (range 0-1)')
    parser.add_argument('--spread-data-fraction', type=float, default=0.05,
                        help='Require data to be this fraction of depth range away from each other in attenuation estimations')
    parser.add_argument('--size', type=int, default=320, help='Size to output')
    parser.add_argument('--monodepth-add-depth', type=float, default=2.0, help='Additive value for monodepth map')
    parser.add_argument('--monodepth-multiply-depth', type=float, default=10.0,
                        help='Multiplicative value for monodepth map')
    parser.add_argument('--model-name', type=str, default="mono_1024x320",
                        help='monodepth model name')
    parser.add_argument('--output-graphs', action='store_true', help='Output graphs')
    parser.add_argument('--raw', action='store_true', help='RAW image')
    parser.add_argument('--out_back', action='store_true', help='whether to output backscattering')
    args = parser.parse_args()

    num_viewpoint = 49
    light = 7 # 7 different levels of underwater degradation
    ############################################################################
    with open(os.path.join(args.coef_path, 'UW_Scene_Type.txt')) as f:
        scenes = f.readlines()
        scenes = [line.rstrip() for line in scenes]

        for scene in scenes:
            print("***************** Synthesizing for {} Scene ******************".format(scene))
            ####### Randomly sample coefs
            coef_path = os.path.join(args.coef_path, scene, "outputs_metric_k/coeffs")
            coefs = [js.load(open(os.path.join(coef_path, file), 'r')) for file in sorted(os.listdir(coef_path))]
            print(sorted(os.listdir(coef_path)))
            with open(os.path.join(args.data_dir, 'train.txt')) as f:
                scans = f.readlines()
                scans = [line.rstrip() for line in scans]
                #print(scans)
                for scan in tqdm(scans, total=len(scans)):
                    print("--------- Syn {} Scene for {} Scan ---------".format(scene, scan))
                    output_dir = os.path.join(args.output_dir, scene, scan)
                    os.makedirs(output_dir, exist_ok=True)

                    for light_idx in range(light):
                        coef = coefs[light_idx]
                        for vid in range(num_viewpoint):
                            img_filename = os.path.join(args.image_dir, '{}_train/rect_{:0>3}_{}_r5000.png'.format(scan, vid + 1, light_idx))
                            depth_filename = os.path.join(args.depth_dir, '{}_train/rect_{:0>3}_{}_r5000_depth.pfm'.format(scan, vid + 1, light_idx))

                            img = cv2.cvtColor(cv2.imread(img_filename), cv2.COLOR_BGR2RGB)
                            depth = np.array(read_pfm(depth_filename)[0], dtype=np.float32)
                            #print(img.shape)
                            #print(depth.shape)
                            h, w = int(depth.shape[0]), int(depth.shape[1])
                            img = cv2.resize(img, (w, h))

                            degraded, direct, backscatter = synthesizer(img / 255.0, depth, coef)
                            sigma_est = estimate_sigma(degraded, multichannel=True, average_sigmas=True) / 10.0 # Estimated noise standard deviation(s)
                            degraded = denoise_tv_chambolle(degraded, sigma_est, multichannel=True) # Perform total variation denoising

                            # output images and coefficients
                            im = Image.fromarray((np.clip(np.round(degraded * 255.0), 0, 255)).astype(np.uint8))
                            im.save(os.path.join(output_dir, 'rect_{:0>3}_{}_r5000.png'.format(vid + 1, light_idx)), format='png')
