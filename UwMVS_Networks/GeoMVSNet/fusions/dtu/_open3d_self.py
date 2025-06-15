import torch
import numpy as np
import sys
import argparse
import errno, os
import glob
import os.path as osp
import re
import cv2
from PIL import Image
import gc
import open3d as o3d

import torch
import torch.nn.functional as F
import numpy as np


parser = argparse.ArgumentParser(description='Depth fusion with consistency check.') # 创建一个新的 ArgumentParser 对象。所有的参数都应当作为关键字参数传入
parser.add_argument('--root_path', type=str, default='') # test set data path
parser.add_argument('--depth_path', type=str, default='') # depth estimation path, i.e., output path
parser.add_argument('--data_list', type=str, default='') # test.txt
parser.add_argument('--ply_path', type=str, default='') # point cloud save path
parser.add_argument('--dist_thresh', type=float, default=0.2) # distance threshold
parser.add_argument('--prob_thresh', type=float, default=0.3) # probability threshold
parser.add_argument('--num_consist', type=int, default=4) # number of consistent views
parser.add_argument('--device', type=str, default='cuda') # device type

# parse arguments 将参数字符串转换为对象并将其设为命名空间的属性。返回带有成员的命名空间。
args = parser.parse_args()

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def parse_cameras(path):
    cam_txt = open(path).readlines()
    f = lambda xs: list(map(lambda x: list(map(float, x.strip().split())), xs))

    extr_mat = f(cam_txt[1:5]) # [4, 4]
    intr_mat = f(cam_txt[7:10]) # [3, 3]

    extr_mat = np.array(extr_mat, np.float32)
    intr_mat = np.array(intr_mat, np.float32)

    return extr_mat, intr_mat

def read_pfm(filename):

    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file')

    dim_match = re.match(r'^(\d+)\s(\d+)$', f.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header')

    scale = float(file.readline().rstrip())
    if scale < 0:
        endian = "<" # little endian
        scale = -scale
    else:
        endian = ">"

    data = np.fromfile(file, endian + "f") # float32 and little endian
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()

    return data, scale

def load_data(root_path, depth_path, scene_name, thresh):

    """
    [OUTPUT Params]:
        1. depths: [49, 1, h, w] tensor, i.e. depth map estimation
        2. projs: [49, 4, 4] tensor, i.e. projection matrix [KR Kt]
        3. rgbs: list: 49 views of [h, w, 3] img ndarray
    """

    depths = []
    projs = []
    rgbs = []

    for view in range(49):
        img_filename = "{}/{}/images/{:08d}.jpg".format(depth_path, scene_name, view)
        cam_filename = "{}/{}/cams/{:08d}_cam.txt".format(depth_path, scene_name, view)
        depth_filename = "{}/{}/depth_est/{:08d}.pfm".format(depth_path, scene_name, view)
        confidence_filename = "{}/{}/confidence/{:08d}.pfm".format(depth_path, scene_name, view)

        # [4,4], [3,3]
        extr_mat, intr_mat = parse_cameras(cam_filename)
        ######################## IMPORTANT DETAILS #############################
        proj_mat = np.eye(4)
        proj_mat[:3, :4] = np.dot(intr_mat[:3, :3], extr_mat[:3, :4]) # [4, 4]
        ########################################################################
        projs.append(torch.from_numpy(proj_mat)) # list: 49 views of [4, 4] projection matrices, tensor

        dep_map, _ = read_pfm(depth_filename) # depth map estimation [h, w]
        h, w = dep_map.shape
        conf_map, _ = read_pfm(confidence_filename) # photometric confidence [h, w]
        conf_map = cv2.resize(conf_map, (w, h), interpolation=cv2.INTER_LINEAR)

        ####################### IMPORTANT DETAILS ##############################
        depth_map = depth_map * (conf_map > thresh).astype(np.float32) # apply probability threshold
        ########################################################################
        depths.append(torch.from_numpy(depth_map).unsqueeze(0)) # list: 49 views of [1, h, w] depth map tensors which meet the probability threshold

        rgb = np.array(Image.open(img_filename))
        rgbs.append(rgb) # list: 49 views of [h, w, 3] img ndarray

    depths = torch.stack(depths).float() # [49, 1, h, w] tensor
    projs = torch.stack(projs).float() # [49, 4, 4] tensor

    if args.device == "cuda" and torch.cuda.is_available():
        depths = depths.cuda()
        projs = projs.cuda()

    return depths, projs, rgbs

def generate_points_from_depth(depth, proj):
    '''
    [INPUT Params]
        1. depth: [B, 1, h, w] ref-view or multiple src-view depth maps
        2. proj: [B, 4, 4] ref-view or multiple src-view projection matrices

    [OUTPUT Params]
        proj_xyz: [B, 3, H, w] back-projected point cloud
    '''
    batch, height, width = depth.shape[0], depth.shape[2], depth.shape[3]
    inv_proj = torch.inverse(proj) # [B, 4, 4]

    rot = inv_proj[:, :3, :3] # (KR)^(-1) [B, 3, 3]
    trans = inv_proj[:, :3, 3:4] # X_0 = - R^(-1)T inhomogeneous coordinates of the camera center in the world coordinates [B, 3, 1]

    y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=depth.device),
                           torch.arange(0, width, dtype=torch.float32, device=depth.device)])
    y, x = y.contiguous(), x.contiguous() # [h, w]
    y, x = y.view(height * width), x.view(height * width) # [h*w,]
    xyz = torch.stack((x, y, torch.ones_like(x))) # [3, h*w]
    xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1) # [B, 3, h*w]
    rot_xyz = torch.matmul(rot, xyz) # [B, 3, h*w]
    rot_depth_xyz = rot_xyz * depth.view(batch, 1, -1) # [B, 3, h*w]
    proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1) # [B, 3, h*w]
    proj_xyz = proj_xyz.view(batch, 3, height, width) # [B, 3, h, w]

    return proj_xyz

def homo_warping(src_fea, src_proj, ref_proj, depth_values):
    """
    [INPUT Params]
        1. src_fea: src_pcs, [B, 3, h, w] src-view point clouds back-projected via src-view depth maps and projection matrices
        2. src_proj: src_projs, [B, 4, 4] src-view projection matrices
        3. ref_proj: ref_proj, [1, 4, 4] ref-view projection matrices
        4. depth_values: ref_depth [1, 1, h, w] ref-view depth map estimation

    [OUTPUT Params]
        warped_src_fea: [B, 3, h, w] i.e., aligned_pcs
    """
    batch, channels = src_fea.shape[0], src_fea.shape[1] # B, 3
    height, width = src_fea.shape[2], src_fea.shape[3] # h, w

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj)) # [B, 4, 4]
        rot = proj[:, :3, :3] # [B, 3, 3]
        trans = proj[:, :3, 3:4] # [B, 3, 1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)]) # [h, w], torch.float32, cuda
        y, x = y.contiguous(), x.contiguous() # [h, w]
        y, x = y.view(height * width), x.view(height * width) # [h*w,]
        xyz = torch.stack((x, y, torch.ones_like(x))) # [3, h*w]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1) # [B, 3, h*w]
        rot_xyz = torch.matmul(rot, xyz) # [B, 3, h*w]

        # rot_xyz.unsqueeze(2) [B, 3, 1, h*w] ORI: [B, 3, D, h*w]
        # depth_values.view(-1, 1, 1, height*width) [B, 1, 1, h*w] ORI: [B, 1, D, h*w]
        rot_depth_xyz = rot_xyz.unsqueeze(2) * depth_values.view(-1, 1, 1, height*width) # [B, 3, 1, h*w]

        # trans.view(batch, 3, 1, 1) [B, 3, 1, 1]
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1) # [B, 3, 1, h*w] src-view homogeneous coordinates
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :] # [B, 2, 1, h*w] src-view inhomogeneous coordinates
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1 # [-1, 1] [B, 1, h*w]
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1 # [-1, 1] [B, 1, h*w]
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3) # [B, 1, h*w, 2]

    # src_fea: [B, 3, h, w]
    # grid.view(): [B, h, w, 2]
    # warped_src_fea: [B, 3, h, w]
    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, height, width, 2), mode='bilinear',
                                   padding_mode='zeros')
    warped_src_fea = warped_src_fea.view(batch, channels, height, width) # [B, 3, h, w]

    return warped_src_fea

def filter_depth(ref_depth, src_depths, ref_proj, src_projs):
    '''
    [INPUT Params]:
        1. ref_depth: [1, 1, h, w] torch.Tensor
        2. src_depths: [B, 1, h, w] torch.Tensor
        3. ref_proj: [1, 4, 4] torch.Tensor
        4. src_projs: [B, 4, 4] torch.Tensor

    [OUTPUT Params]:
        1. ref_pc: [1, 3, h, w] ref-view point cloud back-projected via ref-view depth map and projection matrix
        2. aligned_pcs: [B, 3, h, w] src-view point clouds back-projected via src-view depth maps and projection matrices aligned to the ref-view
        3. dist: [B, 1, h, w] point-wise distance between ref-view back-projected point cloud and src-view back-projected point cloud
    '''

    ref_pc = generate_points_from_depth(ref_depth, ref_proj) # [1, 3, h, w]
    src_pcs = generate_points_from_depth(src_depths, src_projs) # [B, 3, h, w]

    aligned_pcs = homo_warping(src_pcs, src_projs, ref_proj, ref_depth) # [B, 3, h, w]

    x_2 = (ref_pc[:, 0] - aligned_pcs[:, 0]) ** 2 # [B, h, w]
    y_2 = (ref_pc[:, 1] - aligned_pcs[:, 1]) ** 2 # [B, h, w]
    z_2 = (ref_pc[:, 2] - aligned_pcs[:, 2]) ** 2 # [B, h, w]
    dist = torch.sqrt(x_2 + y_2 + z_2).unsqueeze(1) # [B, 1, h, w]

    return ref_pc, aligned_pcs, dist

def extract_points(pc, mask, rgb):
    """
    [INPUT Params]
        1. pc: avg_points [h, w, 3] torch.Tensor
        2. mask: final_mask [h, w] torch.Tensor
        3. rgb: rgbs[i] [h, w, 3] numpy ndarray

    [OUTPUT Params]
        points_with_color: [h_valid * w_valid, 3]
    """
    pc = pc.cpu().numpy()
    mask = mask.cpu().numpy()

    mask = np.reshape(mask, (-1,)) # [h*w, ]
    pc = np.reshape(pc, (-1, 3)) # [h*w, 3] (x, y, z)
    rgb = np.reshape(rgb, (-1, 3)) # [h*w, 3] (r, g, b)

    points = pc[np.where(mask)] # [h_valid * w_valid, 3] (x, y, z)
    colors = rgb[np.where(mask)] # [h_valid * w_valid, 3] (r, g, b)

    points_with_color = np.concatenate([points, colors], axis=1) # [h_valid * w_valid, 6]

    return points_with_color

########################### For Visualization ##################################
def extract_points_wo_mask(pc, rgb):
    """
    [INPUT Params]
        1. pc: avg_points [h, w, 3] torch.Tensor
        2. rgb: rgbs[i] [h, w, 3] numpy ndarray

    [OUTPUT Params]
        points_with_color: [h * w, 3]
    """
    pc = pc.cpu().numpy()

    points = np.reshape(pc, (-1, 3)) # [h*w, 3] (x, y, z)
    colors = np.reshape(rgb, (-1, 3)) # [h*w, 3] (r, g, b)

    points_with_color = np.concatenate([points, colors], axis=1) # [h_valid * w_valid, 6]

    return points_with_color
################################################################################

def write_ply(file, points):
    pcd = o3d.geometry.PointCloud() # PointCloud class. A point cloud consists of point coordinates, and optionally point colors and point normals
    pcd.points = o3d.utility.Vector3dVector(points[:, :3]) # Convert float64 numpy array of shape (n, 3) to Open3D format
    pcd.colors = o3d.utility.Vector3dVector(points[:, 3:] / 255.) # Convert float64 numpy array of shape (n, 3) to Open3D format
    o3d.io.write_point_cloud(file, pcd, write_ascii=False) # Function to write PointCloud to file

def open3d_filter():
    with torch.no_grad():
        mkdir_p(args.ply_path)
        all_scenes = open(args.data_list, 'r').readlines()
        all_scenes = list(map(str.strip, all_scenes))

        # iterate over all test scans
        for i, scene in enumerate(all_scenes):
            print("{}/{} {}:".format(i, len(all_scenes), scene), '----------------------')

            # depths: [49, 1, h, w] 49 views of depth estimation, torch.Tensor
            # projs: [49, 4, 4] 49 views of projection matrix, torch.Tensor
            # rgbs: 49-length list of [h, w, 3], numpy.ndarray
            depths, projs, rgbs = load_data(args.root_path, args.depth_path, scene, args.prob_thresh)
            tot_frame = depths.shape[0] # 49
            height, width = depths.shape[2], depths.shape[3]
            points = []

            print("Scene: {} total: {} frames".format(scene, tot_frame))
            ##################### For visualization ############################
            ply_id = int(scene[4:])
            ####################################################################

            # Iterate over each ref view (each view is regarded as reference view respectively.)
            for i in range(tot_frame):
                pc_buff = torch.zeros((3, height, width), device=depths.device, dtype=depths.dtype) # "cuda", torch.float32
                val_cnt = torch.zeros((1, height, width), device=depths.device, dtype=depths.dtype) # "cuda", torch.float32
                j = 0
                batch_size = 20 # number of source views for geometric consistency check

                while True:
                    # ref_pc: [1, 3, h, w] ref-view point cloud back-projected via ref-view depth map and projection matrix
                    # pcs: [B, 3, h, w] src-view point clouds back-projected via src-view depth maps and projection matrices aligned to the ref-view
                    # ref_pc and pcs should be identical as they represent the world coordinates of the same scene
                    # dist: [B, 1, h, w] point-wise distance between ref_pc and pcs
                    ref_pc, pcs, dist = filter_depth(ref_depth=depths[i:i+1], src_depths=depths[j:min(j+batch_size, tot_frame)],
                                                    ref_proj=projs[i:i+1], src_projs=projs[j:min(j+batch_size, tot_frame)])

                    masks = (dist < args.dist_thresh).float() # [B, 1, h, w]
                    masked_pc = pcs * masks # [B, 3, h, w]
                    pc_buff += masked_pc.sum(dim=0, keepdim=False) # [3, h, w] valid point world coordinates sum over all source views
                    val_cnt += masks.sum(dim=0, keepdim=False) # [1, h, w] number of views meet the point distance threshold

                    j += batch_size
                    if j >= tot_frame:
                        break
                    ####################### For Visualization ##################
                    if i == 0:
                        ref_pc_visual = ref_pc.squeeze(0).permute(1, 2, 0) # [h, w, 3]
                        ref_view_pc = extract_points_wo_mask(ref_pc_visual, rgbs[i]) # [h, w, 6]
                        write_ply('{}/mvsnet{:03d}_ref.ply'.format(args.ply_path, ply_id), ref_view_pc)

                        src_pc_visual = pcs[2].permute(1, 2, 0) # [h, w, 3]
                        src_pc_mask = masks[2].squeeze(0) # [h, w]
                        src_view_pc = extract_points(src_pc_visual, src_pc_mask, rgbs[i]) # [h_valid, w_valid, 6]
                        write_ply('{}/mvsnet{:03d}_src.ply'.format(args.ply_path, ply_id), src_view_pc)
                    #############################################################

                final_mask = (val_cnt >= args.num_consist).squeeze(0) # [h, w] pixel locations meeting the number of consistent views
                avg_points = torch.div(pc_buff, val_cnt).permute(1, 2, 0) # [h, w, 3]

                final_pc = extract_points(avg_points, final_mask, rgbs[i]) # [h_valid * w_valid, 6]
                points.append(final_pc)
                if i == 0 or i == tot_frame - 1:
                    print("Processing {} {}/{} ...".format(scene, i+1, tot_frame))

            ply_id = int(scene[4:])
            write_ply('{}/mvsnet{:03d}.ply'.format(args.ply_path, ply_id), np.concatenate(points, axis=0)) # [49 * h_valid * w_valid, 6]
            del points, depths, rgbs, projs

            gc.collect()

            print('Save {} / mvsnet{:03d}.ply successful.'.format(args.ply_path, ply_id))

if __name__ == '__main__':
    open3d_filter()
