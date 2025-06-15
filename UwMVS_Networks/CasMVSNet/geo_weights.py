## Imports
import argparse
import os
import sys
import cv2
import numpy as np
from .geo_utils import *
import torch
import matplotlib.pyplot as plt

## Geometric weight generation methods
## Consistency based:
##      - direct_joint_inconsistency: value-based mask
##      - geo_consistency_to_inconsistency: {0,1} binary mask converted to Normal, Average, and Inverse mask
##      - projected_pixel_displacement: Value-based mask
##      - projected_depth_difference: Value-based mask
## Inconsistency based:
##      - joint_inconsistency_mask: {0,1} binary mask converted to Normal and Average weights
##      - displacement_inconsistency_mask:{0,1} binary mask converted to Normal and Average weights
##      - difference_inconsistency_mask: {0,1} binary mask converted to Normal and Average weights

class GeometricWeights:
    def __init__(self, args):
        incosistency_base_masks = ["joint_inconsistency_mask", "displacement_inconsistency_mask", "difference_inconsistency_mask"]
        self.mask_type = args.mask_type
        self.geo_mask_sum_th = args.geo_mask_sum_thresh
        self.photo_mask_th = args.photo_mask_thresh
        self.avg_weight_gap = args.avg_weight_gap
        if self.mask_type == "direct_joint_inconsistency":
            self.lambda_1 = args.joint_lambda_1
            self.lambda_2 = args.joint_lambda_2
            self.dist_max_th = [float(e) for e in args.dist_max_thresh.split(",") if e]
            self.dist_min_th = [float(e) for e in args.dist_min_thresh.split(",") if e]
            self.depth_min_th = [float(e) for e in args.relative_depth_diff_min_thresh.split(",") if e]
            self.depth_max_th = [float(e) for e in args.relative_depth_diff_max_thresh.split(",") if e]
        elif self.mask_type in ["geo_consistency_to_inconsistency"] + incosistency_base_masks:
            self.dist_th = [float(e) for e in args.dist_thresh.split(",") if e]
            self.depth_min_th = [float(e) for e in args.relative_depth_diff_min_thresh.split(",") if e]
            self.cons2incon_type = args.cons2incon_type
        elif self.mask_type == "projected_pixel_displacement":
            self.dist_max_th = [float(e) for e in args.dist_max_thresh.split(",") if e]
            self.dist_min_th = [float(e) for e in args.dist_min_thresh.split(",") if e]
        elif self.mask_type == "projected_depth_difference":
            self.depth_min_th = [float(e) for e in args.relative_depth_diff_min_thresh.split(",") if e]
            self.depth_max_th = [float(e) for e in args.relative_depth_diff_max_thresh.split(",") if e]
        else:
            raise ValueError ('mask type not provided')
    ############################################################################

    def generate_points_from_depth(self, depth, proj):
        '''
        [INPUT Params]

        1. depth: (B, 1, H, W)
        2. proj: (B, 4, 4)

        [OUTPUT Params]
        proj_xyz: point cloud (B, 3, H, W)
        '''
        batch, height, width = depth.shape[0], depth.shape[2], depth.shape[3]
        inv_proj = torch.inverse(proj)

        rot = inv_proj[:, :3, :3]  # [B,3,3]
        trans = inv_proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=depth.device),
                               torch.arange(0, width, dtype=torch.float32, device=depth.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz * depth.view(batch, 1, -1)
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1)  # [B, 3, H*W]
        proj_xyz = proj_xyz.view(batch, 3, height, width)

        return proj_xyz

    def homo_warping(self, src_fea, src_proj, ref_proj, depth_values):
        '''
        [INPUT Params]
            1. src_fea: i.e. src_pcs [B, 3, h, w], point clouds back-projected by the src-view depth map and projection matrices.
            2. src_proj: i.e. src_projs [B, 4, 4], src-view projection matrices
            3. ref_proj: i.e. ref_proj [1, 4, 4], ref-view projection matrix
            4. depth_values: i.e. ref_depth [1, 1, H, W], ref-view depth estimation

        [OUTPUT Params]
            warped_src_fea: i.e. aligned_pcs [B, 3, h, w]
        '''
        batch, channels = src_fea.shape[0], src_fea.shape[1]
        height, width = src_fea.shape[2], src_fea.shape[3]

        with torch.no_grad():
            proj = torch.matmul(src_proj, torch.inverse(ref_proj))
            rot = proj[:, :3, :3]  # [B,3,3]
            trans = proj[:, :3, 3:4]  # [B,3,1]

            y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                                   torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
            y, x = y.contiguous(), x.contiguous()
            y, x = y.view(height * width), x.view(height * width)
            xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
            xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
            rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]

            rot_depth_xyz = rot_xyz.unsqueeze(2) * depth_values.view(-1, 1, 1, height*width)  # [B, 3, 1, H*W]

            proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
            proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
            proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
            proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
            proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
            grid = proj_xy

        warped_src_fea = F.grid_sample(src_fea, grid.view(batch,  height, width, 2), mode='bilinear',
                                       padding_mode='zeros')
        warped_src_fea = warped_src_fea.view(batch, channels, height, width)

        return warped_src_fea

    def filter_depth(self, ref_depth, src_depths, ref_proj, src_projs):
        '''
        [INPUT Params]

        1. ref_depth: (1, 1, H, W)
        2. src_depths: (B, 1, H, W)
        3. ref_proj: (1, 4, 4)
        4. src_projs: (B, 4, 4)

        [OUTPUT Params]
        1. ref_pc: (1, 3, H, W)
        2. aligned_pcs: (B, 3, H, W)
        3. dist: (B, 1, H, W)
        '''
        # [1, 3, H, W]
        ref_pc = self.generate_points_from_depth(ref_depth, ref_proj)
        # [B, 3, H, W]
        src_pcs = self.generate_points_from_depth(src_depths, src_projs)
        # [B, 3, H, W]
        # src-view point clouds aligned to the reference view
        aligned_pcs = self.homo_warping(src_pcs, src_projs, ref_proj, ref_depth)

        # [B, h, w]
        x_2 = (ref_pc[:, 0] - aligned_pcs[:, 0]) ** 2
        y_2 = (ref_pc[:, 1] - aligned_pcs[:, 1]) ** 2
        z_2 = (ref_pc[:, 2] - aligned_pcs[:, 2]) ** 2

        dist = torch.sqrt(x_2 + y_2 + z_2).unsqueeze(1) # [B, 1, h, w]

        return dist

    def generate_geometric_weights_3D_points(self, confidence, depth_est, p_mats, src_gt, stage_idx):
        """
        1. confidence: [B, h, w] ref-view photometric confidence
        2. depth_est: [B, h, w] ref-view depth estimation
        3. p_mats: [B, M+1, 2, 4, 4] projection matrix for M src views, 1 ref views
        4. src_gt: [M, B, h, w] ground-truth depth map for M source views
        5. stage_idx: 0, 1, 2 int
        """
        ## loop variables
        batch_size, _, _ = depth_est.shape
        total_src_views = len(src_gt) # M

        batch_geo_mask = []

        ## process each elements of a batch one-by-one
        for batch_idx in range(batch_size):
            ref_depth_est = depth_est[batch_idx, :,:]  # ref-view depth map estimation [h, w]
            ref_extrinsics = p_mats[batch_idx, 0, 0, :4, :4] # ref-view extrinsics [4, 4]
            ref_intrinsics = p_mats[batch_idx, 0, 1, :3, :3] # # ref-view intrinsics [3, 3]
            ####################################################################
            ref_proj = torch.eye(4).to(device='cuda') # [4, 4]
            ref_proj[:3, :4] = torch.matmul(ref_intrinsics, ref_extrinsics[:3, :4])
            ####################################################################

            ## init geo inconsistency mask sum
            # The Geometric Consistency Module is initialized with a `geometric inconsistency mask_sum`
            # This mask sum accumulates the inconsistency of each pixel in reference-view depth estimation across M source views
            mask_sum = torch.zeros(ref_depth_est.shape).to(device='cuda') # [h, w]

            ## Iterate over each source view to generate mask for it
            # For each source view, the Geometric Consistency Module performs forward-backward projection of reference-view depth estimation
            # to generate the penalty and then adds it to the mask sum
            for src_idx in range(total_src_views):
                src_depth_gt = src_gt[src_idx][batch_idx, :,:] # [h, w]
                src_extrinsics = p_mats[batch_idx, src_idx + 1, 0, :4, :4] # [4, 4]
                src_intrinsics = p_mats[batch_idx, src_idx + 1, 1, :3, :3] # [3, 3]
                ################################################################
                src_proj = torch.eye(4).to(device='cuda') # [4, 4]
                src_proj[:3, :4] = torch.matmul(src_intrinsics, src_extrinsics[:3, :4])
                ################################################################

                # ref_pc: [1, 3, h, w]
                # aligned_pcs: [1, 3, h, w]
                # dist: [1, 1, H, W] -> [h, w]
                dist = self.filter_depth(ref_depth=ref_depth_est.unsqueeze(0).unsqueeze(0),
                                         src_depths=src_depth_gt.unsqueeze(0).unsqueeze(0),
                                         ref_proj=ref_proj.unsqueeze(0),
                                         src_projs=src_proj.unsqueeze(0)).squeeze(0).squeeze(0)

                mask_in = (dist > 0.2).float() # [h, w]

                mask_sum += mask_in.to(device='cuda')

            ## Convert geo consistency mask sum to final geo weights

            avg_weight_controller = total_src_views if self.avg_weight_gap=="0.1" else total_src_views/2
            geo_inconsistent_mask = 1.0 + mask_sum/avg_weight_controller # [h, w] ranges from [1, 3]
            ## collect generated geo mask in list for batch manipulation
            batch_geo_mask.append(geo_inconsistent_mask.to(device='cuda'))

        output = (torch.stack(batch_geo_mask, dim=0),) # # ([B, h, w],)

        return output
    ############################################################################

    def pixel_displacement_error(self, x2d_reprojected, y2d_reprojected, x_ref, y_ref):
        """check ||p_reproj-p_1|| L2 norm"""
        return torch.sqrt((x2d_reprojected - x_ref.to(device='cuda')) ** 2 + (y2d_reprojected - y_ref.to(device='cuda')) ** 2)

    def depth_estimate_error(self, depth_reprojected, depth_ref, relative_=True):
        """check |d_reproj-d_1| / d_1"""
        depth_diff = torch.abs(depth_reprojected - depth_ref)
        if relative_:
            relative_depth_diff = depth_diff / depth_ref
        return relative_depth_diff if relative_ else depth_diff

    # project the reference point cloud into the source view, then project back
    def reproject_with_depth(self, depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
        width, height = depth_ref.shape[1], depth_ref.shape[0]
        ## step1. project reference pixels to the source view
        # reference view x, y
        x_ref, y_ref = torch.meshgrid(torch.arange(0, width), torch.arange(0, height), indexing='xy')
        x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])

        # reference 3D space
        xyz_ref = torch.matmul(torch.linalg.inv(intrinsics_ref),
                            torch.vstack((x_ref.to(device='cuda'),
                                          y_ref.to(device='cuda'),
                                          torch.ones_like(x_ref, device=torch.device('cuda')))) * depth_ref.reshape([-1]))
        # source 3D space
        xyz_src = torch.matmul(torch.matmul(extrinsics_src, torch.linalg.inv(extrinsics_ref)),
                            torch.vstack((xyz_ref.to(device='cuda'),
                                          torch.ones_like(x_ref, device=torch.device('cuda')))))[:3]
        # source view x, y
        K_xyz_src = torch.matmul(intrinsics_src, xyz_src)
        xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

        ## reproject the source view points with source view depth estimation
        # find the depth estimation of the source view
        x_src = xy_src[0].reshape([height, width]).cpu().detach().numpy()
        y_src = xy_src[1].reshape([height, width]).cpu().detach().numpy()
        sampled_depth_src = cv2.remap(np.squeeze(depth_src.cpu().detach().numpy()),
                                      x_src,
                                      y_src,
                                      interpolation=cv2.INTER_LINEAR)
        sampled_depth_src = torch.from_numpy(sampled_depth_src)

        # source 3D space
        # NOTE that we should use sampled source-view depth_here to project back
        xyz_src = torch.matmul(torch.linalg.inv(intrinsics_src),
                               torch.vstack((xy_src, torch.ones_like(x_ref, device=torch.device('cuda')))) * sampled_depth_src.reshape([-1]).to(device='cuda'))
        # reference 3D space
        xyz_reprojected = torch.matmul(torch.matmul(extrinsics_ref, torch.linalg.inv(extrinsics_src)),
                                    torch.vstack((xyz_src.to(device='cuda'),
                                                  torch.ones_like(x_ref, device=torch.device('cuda')))))[:3]
        # source view x, y, depth
        depth_reprojected = xyz_reprojected[2].reshape([height, width])
        K_xyz_reprojected = torch.matmul(intrinsics_ref, xyz_reprojected)
        xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
        x_reprojected = xy_reprojected[0].reshape([height, width])
        y_reprojected = xy_reprojected[1].reshape([height, width])

        ## put back to cuda
        x_src = torch.from_numpy(x_src)
        y_src = torch.from_numpy(y_src)

        return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src


    def geometric_inconsistency_mask(self, depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src, stage_idx):
        """
        Performs inconsistency check across different source views for three different
        methods and returns geometric inconsistency mask sum

        [INPUT Params]
            1. depth_ref: ref-view depth map estimation [h, w]
            2. intrinsics_ref: ref-view intrinsics [3, 3]
            3. extrinsics_ref: ref-view extrinsics [4, 4]
            4. depth_src: src-view gt depth map [h, w]
            5. intrinsics_src: src-view intrinsics [3, 3]
            6. extrinsics_src: src-view extrinsics [4, 4]
            7. stage_idx: 0, 1, 2

        [OUTPUT Params]
            mask: geometric inconsistency mask [h, w]

        """
        width, height = depth_ref.shape[1], depth_ref.shape[0]
        x_ref, y_ref = torch.meshgrid(torch.arange(0, width), torch.arange(0, height), indexing='xy')

        # all [h, w]
        depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = self.reproject_with_depth(depth_ref,
                                                                                                          intrinsics_ref,
                                                                                                          extrinsics_ref,
                                                                                                          depth_src,
                                                                                                          intrinsics_src,
                                                                                                          extrinsics_src)
        ## apply threshold on dist and depth, then take logical OR
        # pixel displacement error (PDE)
        dist = self.pixel_displacement_error(x2d_reprojected, y2d_reprojected, x_ref, y_ref)
        # relative depth difference (RDD)
        relative_depth_diff = self.depth_estimate_error(depth_reprojected, depth_ref, relative_=True)

        ## Apply threshold and take logical OR
        dist = dist > self.dist_th[stage_idx] # [1, 1, 1] for coarse, intermediate, and refinement stage
        relative_depth_diff = relative_depth_diff > self.depth_min_th[stage_idx] # [0.01, 0.01, 0.01]
        # # Geometric inconsistency mask: either exceed PDE threshold or RDD threshold
        mask = torch.logical_or(dist, relative_depth_diff).to(torch.float32)

        return mask # [h, w]


    def generate_geometric_weights(self, confidence, depth_est, p_mats, src_gt, stage_idx):
        """
        1. confidence: [B, h, w] ref-view photometric confidence
        2. depth_est: [B, h, w] ref-view depth estimation
        3. p_mats: [B, M+1, 2, 4, 4] projection matrix for M src views, 1 ref views
        4. src_gt: [M, B, h, w] ground-truth depth map for M source views
        5. stage_idx: 0, 1, 2 int
        """
        ## loop variables
        batch_size, _, _ = depth_est.shape
        total_src_views = len(src_gt) # M

        batch_geo_mask = []

        ## process each elements of a batch one-by-one
        for batch_idx in range(batch_size):
            ref_depth_est = depth_est[batch_idx, :,:]  # ref-view depth map estimation [h, w]
            ref_extrinsics = p_mats[batch_idx, 0, 0, :4, :4] # ref-view extrinsics [4, 4]
            ref_intrinsics = p_mats[batch_idx, 0, 1, :3, :3] # # ref-view intrinsics [3, 3]

            ## init geo inconsistency mask sum
            # The Geometric Consistency Module is initialized with a `geometric inconsistency mask_sum`
            # This mask sum accumulates the inconsistency of each pixel in reference-view depth estimation across M source views
            mask_sum = torch.zeros(ref_depth_est.shape).to(device='cuda') # [h, w]

            ## Iterate over each source view to generate mask for it
            # For each source view, the Geometric Consistency Module performs forward-backward projection of reference-view depth estimation
            # to generate the penalty and then adds it to the mask sum
            for src_idx in range(total_src_views):
                src_depth_est = src_gt[src_idx][batch_idx, :,:] # [h, w]
                src_extrinsics = p_mats[batch_idx, src_idx + 1, 0, :4, :4] # [4, 4]
                src_intrinsics = p_mats[batch_idx, src_idx + 1, 1, :3, :3] # [3, 3]

                # [h, w]
                mask_in = self.geometric_inconsistency_mask(ref_depth_est, ref_intrinsics, ref_extrinsics,
                                                               src_depth_est, src_intrinsics, src_extrinsics,
                                                               stage_idx) # [h, w]
                mask_sum += mask_in.to(device='cuda')

            ## Convert geo consistency mask sum to final geo weights

            avg_weight_controller = total_src_views if self.avg_weight_gap=="0.1" else total_src_views/2
            geo_inconsistent_mask = 1.0 + mask_sum/avg_weight_controller # [h, w] ranges from [1, 3]
            ## collect generated geo mask in list for batch manipulation
            batch_geo_mask.append(geo_inconsistent_mask.to(device='cuda'))

        output = (torch.stack(batch_geo_mask, dim=0),) # # ([B, h, w],)

        return output
