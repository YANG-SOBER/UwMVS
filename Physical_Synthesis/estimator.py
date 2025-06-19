import collections
import sys
import os
import argparse
import numpy as np
import sklearn as sk
import scipy as sp
import scipy.optimize
import scipy.stats
import math
import cv2
from datetime import datetime
from PIL import Image
from tqdm import tqdm
import rawpy
import json as js
import matplotlib
from matplotlib import pyplot as plt
from skimage import exposure
from skimage.restoration import denoise_bilateral, denoise_tv_chambolle, estimate_sigma
from skimage.morphology import closing, opening, erosion, dilation, disk, diamond, square
import skimage.io as skio
import re
import time

'''
Finds points for which to estimate backscatter
by partitioning the image into different depth
ranges and taking the darkest RGB triplets
from that set as estimations of the backscatter
'''
def find_backscatter_estimation_points(img, depths, num_bins=10, fraction=0.01, max_vals=20, min_depth_percent=0.0):
    z_max, z_min = np.max(depths), np.min(depths)
    min_depth = z_min + (min_depth_percent * (z_max - z_min))
    z_ranges = np.linspace(z_min, z_max, num_bins + 1)
    img_norms = np.mean(img, axis=2)
    points_r = []
    points_g = []
    points_b = []
    for i in range(len(z_ranges) - 1):
        a, b = z_ranges[i], z_ranges[i+1]
        locs = np.where(np.logical_and(depths > min_depth, np.logical_and(depths >= a, depths <= b)))
        norms_in_range, px_in_range, depths_in_range = img_norms[locs], img[locs], depths[locs]
        arr = sorted(zip(norms_in_range, px_in_range, depths_in_range), key=lambda x: x[0])
        points = arr[:min(math.ceil(fraction * len(arr)), max_vals)]
        points_r.extend([(z, p[0]) for n, p, z in points])
        points_g.extend([(z, p[1]) for n, p, z in points])
        points_b.extend([(z, p[2]) for n, p, z in points])
    return np.array(points_r), np.array(points_g), np.array(points_b)

'''
Estimates coefficients for the backscatter curve
based on the backscatter point values and their depths
'''
def find_backscatter_values(B_pts, depths, restarts=10, max_mean_loss_fraction=0.1):
    B_vals, B_depths = B_pts[:, 1], B_pts[:, 0]
    z_max, z_min = np.max(depths), np.min(depths)
    max_mean_loss = max_mean_loss_fraction * (z_max - z_min)
    coefs = None
    best_loss = np.inf
    def estimate(depths, B_inf, beta_B, J_prime, beta_D_prime):
        val = (B_inf * (1 - np.exp(-1 * beta_B * depths))) + (J_prime * np.exp(-1 * beta_D_prime * depths))
        return val
    def loss(B_inf, beta_B, J_prime, beta_D_prime):
        val = np.mean(np.abs(B_vals - estimate(B_depths, B_inf, beta_B, J_prime, beta_D_prime)))
        return val
    bounds_lower = [0,0,0,0]
    bounds_upper = [1,5,1,5]
    for _ in range(restarts):
        # try:
        optp, pcov = sp.optimize.curve_fit(
            f=estimate,
            xdata=B_depths,
            ydata=B_vals,
            p0=np.random.random(4) * bounds_upper,
            bounds=(bounds_lower, bounds_upper),
        )
        l = loss(*optp)
        if l < best_loss:
            best_loss = l
            coefs = optp
        # except RuntimeError as re:
        #     print(re, file=sys.stderr)
    if best_loss > max_mean_loss:
        print('Warning: could not find accurate reconstruction. Switching to linear model.', flush=True)
        slope, intercept, r_value, p_value, std_err = sp.stats.linregress(B_depths, B_vals)
        BD = (slope * depths) + intercept
        return BD, np.array([slope, intercept])
    return estimate(depths, *coefs), coefs

'''
Estimate illumination map from local color space averaging
'''
def estimate_illumination(img, B, neighborhood_map, num_neighborhoods, p=0.5, f=2.0, max_iters=100, tol=1E-5):
    D = img - B
    avg_cs = np.zeros_like(img)
    avg_cs_prime = np.copy(avg_cs)
    sizes = np.zeros(num_neighborhoods)
    locs_list = [None] * num_neighborhoods
    for label in range(1, num_neighborhoods + 1):
        locs_list[label - 1] = np.where(neighborhood_map == label)
        sizes[label - 1] = np.size(locs_list[label - 1][0])
    for _ in range(max_iters):
        for label in range(1, num_neighborhoods + 1):
            locs = locs_list[label - 1]
            size = sizes[label - 1] - 1
            avg_cs_prime[locs] = (1 / size) * (np.sum(avg_cs[locs]) - avg_cs[locs])
        new_avg_cs = (D * p) + (avg_cs_prime * (1 - p))
        if(np.max(np.abs(avg_cs - new_avg_cs)) < tol):
            break
        avg_cs = new_avg_cs
    return f * denoise_bilateral(np.maximum(0, avg_cs))

'''
Estimate values for beta_D
'''
def estimate_wideband_attentuation(depths, illum, radius = 6, max_val = 10.0):
    eps = 1E-8
    BD = np.minimum(max_val, -np.log(illum + eps) / (np.maximum(0, depths) + eps))
    mask = np.where(np.logical_and(depths > eps, illum > eps), 1, 0)
    refined_attenuations = denoise_bilateral(closing(np.maximum(0, BD * mask), disk(radius)))
    return refined_attenuations

'''
Calculate the values of beta_D for an image from the depths, illuminations, and constants
'''
def calculate_beta_D(depths, a, b, c, d):
    return (a * np.exp(b * depths)) + (c * np.exp(d * depths))

def filter_data(X, Y, radius_fraction=0.01):
    idxs = np.argsort(X)
    X_s = X[idxs]
    Y_s = Y[idxs]
    x_max, x_min = np.max(X), np.min(X)
    radius = (radius_fraction * (x_max - x_min))
    ds = np.cumsum(X_s - np.roll(X_s, (1,)))
    dX = [X_s[0]]
    dY = [Y_s[0]]
    tempX = []
    tempY = []
    pos = 0
    for i in range(1, ds.shape[0]):
        if ds[i] - ds[pos] >= radius:
            tempX.append(X_s[i])
            tempY.append(Y_s[i])
            idxs = np.argsort(tempY)
            med_idx = len(idxs) // 2
            dX.append(tempX[med_idx])
            dY.append(tempY[med_idx])
            pos = i
        else:
            tempX.append(X_s[i])
            tempY.append(Y_s[i])
    return np.array(dX), np.array(dY)

'''
Estimate coefficients for the 2-term exponential
describing the wideband attenuation
'''
def refine_wideband_attentuation(depths, illum, estimation, restarts=10, min_depth_fraction = 0.1, max_mean_loss_fraction=np.inf, l=1.0, radius_fraction=0.01):
    eps = 1E-8
    z_max, z_min = np.max(depths), np.min(depths)
    min_depth = z_min + (min_depth_fraction * (z_max - z_min))
    max_mean_loss = max_mean_loss_fraction * (z_max - z_min)
    coefs = None
    best_loss = np.inf
    locs = np.where(np.logical_and(illum > 0, np.logical_and(depths > min_depth, estimation > eps)))
    def calculate_reconstructed_depths(depths, illum, a, b, c, d):
        eps = 1E-5
        res = -np.log(illum + eps) / (calculate_beta_D(depths, a, b, c, d) + eps)
        return res
    def loss(a, b, c, d):
        return np.mean(np.abs(depths[locs] - calculate_reconstructed_depths(depths[locs], illum[locs], a, b, c, d)))
    dX, dY = filter_data(depths[locs], estimation[locs], radius_fraction)
    for _ in range(restarts):
        # try:
        optp, pcov = sp.optimize.curve_fit(
            f=calculate_beta_D,
            xdata=dX,
            ydata=dY,
            p0=np.abs(np.random.random(4)) * np.array([1., -1., 1., -1.]),
            bounds=([0, -100, 0, -100], [100, 0, 100, 0]))
        L = loss(*optp)
        if L < best_loss:
            best_loss = L
            coefs = optp
        # except RuntimeError as re:
        #     print(re, file=sys.stderr)
    # # Uncomment to see the regression
    # plt.clf()
    # plt.scatter(depths[locs], estimation[locs])
    # plt.plot(np.sort(depths[locs]), calculate_beta_D(np.sort(depths[locs]), *coefs))
    # plt.show()
    if best_loss > max_mean_loss:
        print('Warning: could not find accurate reconstruction. Switching to linear model.', flush=True)
        slope, intercept, r_value, p_value, std_err = sp.stats.linregress(depths[locs], estimation[locs])
        BD = (slope * depths + intercept)
        return l * BD, np.array([slope, intercept])
    # print(f'Found best loss {best_loss}', flush=True)
    BD = l * calculate_beta_D(depths, *coefs)
    return BD, coefs

'''
Reconstruct the scene and globally white balance
based the Gray World Hypothesis
'''
def recover_image(img, depths, B, beta_D, nmap):
    res = (img - B) * np.exp(beta_D * np.expand_dims(depths, axis=2))
    res = np.maximum(0.0, np.minimum(1.0, res))
    res[nmap == 0] = 0
    # res = scale(wbalance_no_red_10p(res))
    # res[nmap == 0] = img[nmap == 0]
    return res

'''
Reconstruct the scene and globally white balance
'''
def recover_image_S4(img, B, illum, nmap):
    eps = 1E-8
    res = (img - B) / (illum + eps)
    res = np.maximum(0.0, np.minimum(1.0, res))
    res[nmap == 0] = img[nmap == 0]
    return scale(wbalance_no_red_gw(res))

'''
Constructs a neighborhood map from depths and
epsilon
'''
def construct_neighborhood_map(depths, epsilon=0.05):
    eps = (np.max(depths) - np.min(depths)) * epsilon
    nmap = np.zeros_like(depths).astype(np.int32)
    n_neighborhoods = 1
    while np.any(nmap == 0):
        locs_x, locs_y = np.where(nmap == 0)
        start_index = np.random.randint(0, len(locs_x))
        start_x, start_y = locs_x[start_index], locs_y[start_index]
        q = collections.deque()
        q.append((start_x, start_y))
        while not len(q) == 0:
            x, y = q.pop()
            if np.abs(depths[x, y] - depths[start_x, start_y]) <= eps:
                nmap[x, y] = n_neighborhoods
                if 0 <= x < depths.shape[0] - 1:
                    x2, y2 = x + 1, y
                    if nmap[x2, y2] == 0:
                        q.append((x2, y2))
                if 1 <= x < depths.shape[0]:
                    x2, y2 = x - 1, y
                    if nmap[x2, y2] == 0:
                        q.append((x2, y2))
                if 0 <= y < depths.shape[1] - 1:
                    x2, y2 = x, y + 1
                    if nmap[x2, y2] == 0:
                        q.append((x2, y2))
                if 1 <= y < depths.shape[1]:
                    x2, y2 = x, y - 1
                    if nmap[x2, y2] == 0:
                        q.append((x2, y2))
        n_neighborhoods += 1
    zeros_size_arr = sorted(zip(*np.unique(nmap[depths == 0], return_counts=True)), key=lambda x: x[1], reverse=True)
    if len(zeros_size_arr) > 0:
        nmap[nmap == zeros_size_arr[0][0]] = 0 #reset largest background to 0
    return nmap, n_neighborhoods - 1

'''
Finds the closest nonzero label to a location
'''
def find_closest_label(nmap, start_x, start_y):
    mask = np.zeros_like(nmap).astype(np.bool)
    q = collections.deque()
    q.append((start_x, start_y))
    while not len(q) == 0:
        x, y = q.pop()
        if 0 <= x < nmap.shape[0] and 0 <= y < nmap.shape[1]:
            if nmap[x, y] != 0:
                return nmap[x, y]
            mask[x, y] = True
            if 0 <= x < nmap.shape[0] - 1:
                x2, y2 = x + 1, y
                if not mask[x2, y2]:
                    q.append((x2, y2))
            if 1 <= x < nmap.shape[0]:
                x2, y2 = x - 1, y
                if not mask[x2, y2]:
                    q.append((x2, y2))
            if 0 <= y < nmap.shape[1] - 1:
                x2, y2 = x, y + 1
                if not mask[x2, y2]:
                    q.append((x2, y2))
            if 1 <= y < nmap.shape[1]:
                x2, y2 = x, y - 1
                if not mask[x2, y2]:
                    q.append((x2, y2))

'''
Refines the neighborhood map to remove artifacts
'''
def refine_neighborhood_map(nmap, min_size = 10, radius = 3):
    refined_nmap = np.zeros_like(nmap)
    vals, counts = np.unique(nmap, return_counts=True)
    neighborhood_sizes = sorted(zip(vals, counts), key=lambda x: x[1], reverse=True)
    num_labels = 1
    for label, size in neighborhood_sizes:
        if size >= min_size and label != 0:
            refined_nmap[nmap == label] = num_labels
            num_labels += 1
    for label, size in neighborhood_sizes:
        if size < min_size and label != 0:
            for x, y in zip(*np.where(nmap == label)):
                refined_nmap[x, y] = find_closest_label(refined_nmap, x, y)
    refined_nmap = closing(refined_nmap, square(radius))
    return refined_nmap, num_labels - 1

def load_image_and_depth_map(img_fname, depths_fname, size_limit = 1024):
    depths = Image.open(depths_fname)
    img = Image.fromarray(rawpy.imread(img_fname).postprocess())
    img.thumbnail((size_limit, size_limit), Image.ANTIALIAS)
    depths = depths.resize(img.size, Image.ANTIALIAS)
    return np.float32(img) / 255.0, np.array(depths)

'''
White balance with 'grey world' hypothesis
'''
def wbalance_gw(img):
    dr = 1.0 / np.mean(img[:, :, 0])
    dg = 1.0 / np.mean(img[:, :, 1])
    db = 1.0 / np.mean(img[:, :, 2])
    dsum = dr + dg + db
    dr = dr / dsum * 3.
    dg = dg / dsum * 3.
    db = db / dsum * 3.

    img[:, :, 0] *= dr
    img[:, :, 1] *= dg
    img[:, :, 2] *= db
    return img

def get_wbalance(img):
    dr = 1.0 / np.mean(np.sort(img[:, :, 0], axis=None)[int(round(-1 * np.size(img[:, :, 0]) * 0.1)):])
    dg = 1.0 / np.mean(np.sort(img[:, :, 1], axis=None)[int(round(-1 * np.size(img[:, :, 0]) * 0.1)):])
    db = 1.0 / np.mean(np.sort(img[:, :, 2], axis=None)[int(round(-1 * np.size(img[:, :, 0]) * 0.1)):])
    dsum = dg + db + db
    dr = dr / dsum * 3.
    dg = dg / dsum * 3.
    db = db / dsum * 3.
    return dr, dg, db

'''
White balance based on top 10% average values of each channel
'''
def wbalance_10p(img):
    dr = 1.0 / np.mean(np.sort(img[:, :, 0], axis=None)[int(round(-1 * np.size(img[:, :, 0]) * 0.1)):])
    dg = 1.0 / np.mean(np.sort(img[:, :, 1], axis=None)[int(round(-1 * np.size(img[:, :, 0]) * 0.1)):])
    db = 1.0 / np.mean(np.sort(img[:, :, 2], axis=None)[int(round(-1 * np.size(img[:, :, 0]) * 0.1)):])
    dsum = dr + dg + db
    dr = dr / dsum * 3.
    dg = dg / dsum * 3.
    db = db / dsum * 3.

    img[:, :, 0] *= dr
    img[:, :, 1] *= dg
    img[:, :, 2] *= db
    return img

'''
White balance based on top 10% average values of blue and green channel
'''
def wbalance_no_red_10p(img):
    dg = 1.0 / np.mean(np.sort(img[:, :, 1], axis=None)[int(round(-1 * np.size(img[:, :, 0]) * 0.1)):])
    db = 1.0 / np.mean(np.sort(img[:, :, 2], axis=None)[int(round(-1 * np.size(img[:, :, 0]) * 0.1)):])
    dsum = dg + db
    dg = dg / dsum * 2.
    db = db / dsum * 2.
    img[:, :, 0] *= (db + dg) / 2
    img[:, :, 1] *= dg
    img[:, :, 2] *= db
    return img

'''
White balance with 'grey world' hypothesis
'''
def wbalance_no_red_gw(img):
    dg = 1.0 / np.mean(img[:, :, 1])
    db = 1.0 / np.mean(img[:, :, 2])
    dsum = dg + db
    dg = dg / dsum * 2.
    db = db / dsum * 2.

    img[:, :, 0] *= (db + dg) / 2
    img[:, :, 1] *= dg
    img[:, :, 2] *= db
    return img

def scale(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))

def preprocess_for_monodepth(img_fname, output_fname, size_limit=1024):
    img = Image.fromarray(rawpy.imread(img_fname).postprocess())
    img.thumbnail((size_limit, size_limit), Image.ANTIALIAS)
    img_adapteq = exposure.equalize_adapthist(np.array(img), clip_limit=0.03)
    Image.fromarray((np.round(img_adapteq * 255.0)).astype(np.uint8)).save(output_fname)

def preprocess_sfm_depth_map(depths, min_depth, max_depth):
    z_min = np.min(depths) + (min_depth * (np.max(depths) - np.min(depths)))
    z_max = np.min(depths) + (max_depth * (np.max(depths) - np.min(depths)))
    if max_depth != 0:
        depths[depths == 0] = z_max
    depths[depths < z_min] = 0
    return depths

def preprocess_monodepth_depth_map(depths, additive_depth, multiply_depth):
    depths = ((depths - np.min(depths)) / (
                np.max(depths) - np.min(depths))).astype(np.float32)
    # depths = (multiply_depth * (1.0 - depths)) + additive_depth
    depths = multiply_depth * depths + additive_depth
    return depths

'''
Run sea-thru pipeline, output reconstructed images and the estimated
forward illumination and backsacttering coefficients.
'''
def sea_thru(img, depths, args):

    # 0. Estimate backscattering
    pts_r, pts_g, pts_b = find_backscatter_estimation_points(img, depths, fraction=0.01, min_depth_percent=args.min_depth)

    Br, Bcoefs_r = find_backscatter_values(pts_r, depths, restarts=25)
    Bg, Bcoefs_g = find_backscatter_values(pts_g, depths, restarts=25)
    Bb, Bcoefs_b = find_backscatter_values(pts_b, depths, restarts=25)

    # 1. Constructing and refining neighborhood map based on depth map
    nmap, _ = construct_neighborhood_map(depths, 0.1)
    nmap, n = refine_neighborhood_map(nmap, 50)

    # 2. Estimating forward illumination
    illR = estimate_illumination(img[:, :, 0], Br, nmap, n, p=args.p, max_iters=100, tol=1e-5, f=args.f)
    illG = estimate_illumination(img[:, :, 1], Bg, nmap, n, p=args.p, max_iters=100, tol=1e-5, f=args.f)
    illB = estimate_illumination(img[:, :, 2], Bb, nmap, n, p=args.p, max_iters=100, tol=1e-5, f=args.f)

    # 3. Estimating wideband attenuation and its coefficients
    beta_D_r = estimate_wideband_attentuation(depths, illR)
    refined_beta_D_r, Dcoefs_r = refine_wideband_attentuation(depths, illR, beta_D_r, radius_fraction=args.spread_data_fraction, l=args.l)
    beta_D_g = estimate_wideband_attentuation(depths, illG)
    refined_beta_D_g, Dcoefs_g = refine_wideband_attentuation(depths, illG, beta_D_g, radius_fraction=args.spread_data_fraction, l=args.l)
    beta_D_b = estimate_wideband_attentuation(depths, illB)
    refined_beta_D_b, Dcoefs_b = refine_wideband_attentuation(depths, illB, beta_D_b, radius_fraction=args.spread_data_fraction, l=args.l)

    # 4. Reconstructing image
    B = np.stack([Br, Bg, Bb], axis=2)
    beta_D = np.stack([refined_beta_D_r, refined_beta_D_g, refined_beta_D_b], axis=2)
    recovered = recover_image(img, depths, B, beta_D, nmap)
    wb = get_wbalance(recovered)
    # recovered = scale(wbalance_10p(recovered)) #scale(wbalance_no_red_10p(recovered))
    recovered = scale(wbalance_no_red_10p(recovered)) #scale(wbalance_no_red_10p(recovered))
    recovered[nmap == 0] = img[nmap == 0]

    output_dict = {"Bcoefs_r": Bcoefs_r.tolist(),
                   "Bcoefs_g": Bcoefs_g.tolist(),
                   "Bcoefs_b": Bcoefs_b.tolist(),
                   "Dcoefs_r": Dcoefs_r.tolist(),
                   "Dcoefs_g": Dcoefs_g.tolist(),
                   "Dcoefs_b": Dcoefs_b.tolist(),
                   "wbalance": list(wb)
                   }

    ###############################
    t = np.exp(-beta_D * np.expand_dims(depths, axis=2)) # transmission map
    return recovered, output_dict, t, B

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/home/gdyang/Downloads/Data/Underwater/UW_Types_test', help='Input images directory')
    parser.add_argument('--test_list', default='test.txt', help='Greenish, BLuish, Hazy, Low-light')
    #parser.add_argument('--depth_dir', default='/home/gdyang/Downloads/Data/Underwater/Underwater_Images_synthesis/scripts/depths', help='depth or depth images directory')
    #parser.add_argument('--output_dir', default='/home/gdyang/Downloads/Data/Underwater/Underwater_Images_synthesis/scripts/outputs', help='output directory')
    parser.add_argument('--coeffs_dir', default='/home/gdyang/Downloads/Data/Underwater/Underwater_Images_synthesis/scripts/outputs/coeffs', help='output coeffs directory')
    parser.add_argument('--edge', type=int, default=5, help='image edge to ignore for estimating coefficients')
    parser.add_argument('--f', type=float, default=2.0, help='f value (controls brightness)')
    parser.add_argument('--l', type=float, default=0.5, help='l value (controls balance of attenuation constants)')
    parser.add_argument('--p', type=float, default=0.01, help='p value (controls locality of illuminant map)')
    parser.add_argument('--min-depth', type=float, default=0.1,
                        help='Minimum depth value to use in estimations (range 0-1)')
    parser.add_argument('--max-depth', type=float, default=1.0,
                        help='Replacement depth percentile value for invalid depths (range 0-1)')
    parser.add_argument('--spread-data-fraction', type=float, default=0.01,
                        help='Require data to be this fraction of depth range away from each other in attenuation estimations')
    parser.add_argument('--size', type=int, default=320, help='Size to output')
    parser.add_argument('--output-graphs', action='store_true', help='Output graphs')
    parser.add_argument('--preprocess-for-monodepth', action='store_true', help='Preprocess for monodepth depth maps')
    parser.add_argument('--monodepth', action='store_true', help='Preprocess for monodepth')
    parser.add_argument('--monodepth-add-depth', type=float, default=0.0, help='Additive value for monodepth map')
    parser.add_argument('--monodepth-multiply-depth', type=float, default=10.0,
                        help='Multiplicative value for monodepth map')
    parser.add_argument('--equalize-image', action='store_true', help='Histogram equalization for final output')
    args = parser.parse_args()

    ############################################################################
    with open(os.path.join(args.data_dir, args.test_list)) as f:
        scans = f.readlines()
        scans = [line.rstrip() for line in scans]

        for scan in scans:
            print("----------- Computing Coefs of {} Scene --------------".format(scan))

            image_dir = os.path.join(args.data_dir, scan, "images")
            depth_dir = os.path.join(args.data_dir, scan, "depths_k")
            output_dir = os.path.join(args.data_dir, scan, "outputs_metric_k")
            coeffs_dir = os.path.join(args.data_dir, scan, "outputs_metric_k/coeffs")
            os.makedirs(coeffs_dir, exist_ok=True)

    ############################################################################

            img_files = os.listdir(image_dir)
            img_files.sort()
            dep_files = os.listdir(depth_dir)
            dep_files.sort()
            '''
            if args.output_dir is None:
                current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
                args.output_dir = os.path.join('./output', current_time)
            os.makedirs(args.output_dir, exist_ok=True)
            os.makedirs(args.coeffs_dir, exist_ok=True)
            '''
            outputs = {}
            for i, (img_f, dep_f) in tqdm(enumerate(zip(img_files, dep_files)), total=len(img_files)):
                if img_f.endswith('.png') or img_f.endswith('.jpg') or img_f.endswith('.tif'):
                    # load image
                    fmt= img_f[-4:] # get image format
                    img = cv2.cvtColor(cv2.imread(os.path.join(image_dir, img_f)), cv2.COLOR_BGR2RGB)

                    # load depth
                    if dep_f.endswith('.png') or dep_f.endswith('.jpg'):
                        depth = cv2.imread(os.path.join(depth_dir, dep_f), cv2.IMREAD_GRAYSCALE) * 1.0
                        depth = 255 - depth
                    elif dep_f.endswith('.pfm'):
                        depth = np.array(read_pfm(os.path.join(depth_dir, dep_f))[0], dtype=np.float32)
                    else:
                        depth = skio.imread(os.path.join(depth_dir, dep_f), plugin="tifffile")
                    #depth = ((depth - np.min(depth)) / (np.max(depth) - np.min(depth))).astype(np.float32)

                    h, w = int(depth.shape[0]), int(depth.shape[1])
                    img = cv2.resize(img, (w, h))

                    # img = cv2.resize(img, (1027, 768))
                    try:
                        #multiply_depth = np.random.uniform(6, args.monodepth_multiply_depth, size=1)[0]
                        #depth = preprocess_monodepth_depth_map(depth, args.monodepth_add_depth, args.monodepth_multiply_depth)
                        #print(depth.min(), depth.max())
                        img_basename = img_f.split(fmt)[0]
                        tic = time.time()
                        recovered, output_dict, direct, back = sea_thru(img[args.edge:-args.edge, args.edge:-args.edge, :] / 255.0,
                                                                depth[args.edge:-args.edge, args.edge:-args.edge], args)
                        tok = time.time()
                        print("The Parameter Estimation Time is {}".format(tok - tic))
                        sigma_est = estimate_sigma(recovered, multichannel=True, average_sigmas=True) / 10.0
                        recovered = denoise_tv_chambolle(recovered, sigma_est, multichannel=True)

                        # output images and coefficients
                        im = Image.fromarray((np.round(recovered * 255.0)).astype(np.uint8))
                        im.save(os.path.join(output_dir, img_basename+'.png'), format='png')

                        im = Image.fromarray((np.round(direct * 255.0)).astype(np.uint8))
                        im.save(os.path.join(output_dir, img_basename+'_direct_uw.png'), format='png')
                        #
                        im = Image.fromarray((np.round(back * 255.0)).astype(np.uint8))
                        im.save(os.path.join(output_dir, img_basename+'_back_uw.png'), format='png')
                        outputs.setdefault(img_basename, output_dict)
                        js.dump(output_dict, open(os.path.join(coeffs_dir, f'{img_basename}.json'), 'w'), indent=True)
                    except Exception as e:
                        print(e)
            # js.dump(outputs, open('coeffs.json', 'w'), indent=True)
