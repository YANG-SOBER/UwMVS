import os
import time
import torch
import numpy as np
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from zoedepth.utils.misc import colorize, colorize_conf
from data_io_mvs import save_pfm


# ZoeD_N
conf = get_config("zoedepth", "infer")
model_zoe_n = build_model(conf)

##### sample prediction
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
zoe = model_zoe_n.to(DEVICE)


# Local file
from PIL import Image

data_path = "/home/gdyang/Downloads/Data/mvs_training/dtu"
train_list_file = "train.txt"
depth_save_path = "/home/gdyang/Downloads/Data/mvs_training/dtu/depth_mono"

os.makedirs(depth_save_path, exist_ok=True)

with open(os.path.join(data_path, train_list_file)) as f:

    ############################################################################
    scans = f.readlines()
    scans = [line.rstrip() for line in scans]

    for scan in scans:
        print("-----Processing {}-------".format(scan))
        metas = []
        pair_file = "Cameras/pair.txt"

        with open(os.path.join(data_path, pair_file)) as f:
            num_viewpoint = int(f.readline()) # 1st row of "pair.txt" is the number of viewpoints, 49 or 64, dependent on the paticular scan
            # viewpoints (49)
            # iterate over different viewpoints
            for view_idx in range(num_viewpoint): # 0-48
                ref_view = int(f.readline().rstrip())
                src_views_scores = f.readline().rstrip().split()
                src_views = [int(x) for x in src_views_scores[1::2]]
                for light_idx in range(7):
                    # metas.append((scan, light_idx, ref_view, src_views))
                    img_filename = os.path.join(data_path, 'Rectified_for_mono/{}_train/rect_{:0>3}_{}_r5000.png'.format(scan, view_idx+1, light_idx))
                    image = Image.open(img_filename).convert("RGB")
                    t0 = time.time()
                    depth_est, conf_est = zoe.infer_pil(image)
                    t1 = time.time()
                    print("Time per Image: {:.3f}".format(t1-t0))
                    print(f"depth_est_min: {np.min(depth_est)}, max: {np.max(depth_est)}")
                    depth_filename = os.path.join(depth_save_path, "depth_est_n/{}_train/rect_{:0>3}_{}_r5000_depth.pfm".format(scan, view_idx+1, light_idx))
                    os.makedirs(depth_filename.rsplit('/', 1)[0], exist_ok=True)
                    save_pfm(depth_filename, depth_est)

                    colored = colorize(depth_est)
                    depth_img_filename = os.path.join(depth_save_path, "depth_est_n/{}_train/rect_{:0>3}_{}_r5000_depth.png".format(scan, view_idx+1, light_idx))
                    Image.fromarray(colored).save(depth_img_filename)
