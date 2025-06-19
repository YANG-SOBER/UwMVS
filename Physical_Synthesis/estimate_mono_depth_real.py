import os
import time
import torch
import numpy as np
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from zoedepth.utils.misc import colorize
from data_io_mvs import save_pfm
from tqdm import tqdm


# ZoeD_K
conf = get_config("zoedepth", "infer", config_version="kitti")
model_zoe_k = build_model(conf)


##### sample prediction
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
zoe = model_zoe_k.to(DEVICE)


# Local file
from PIL import Image

data_path = "/home/gdyang/Downloads/Data/Underwater/UW_Types"
pair_file = "pair.txt"
scene_type_file = "UW_Scene_Type.txt"
depth_save_path = "/home/gdyang/Downloads/Data/Underwater/UW_Types"

with open(os.path.join(data_path, scene_type_file)) as f:
    scenes = f.readlines()
    scenes = [line.rstrip() for line in scenes]

    for scene in tqdm(scenes, total=len(scenes)):

        with open(os.path.join(data_path, pair_file)) as f:
            num_viewpoint = int(f.readline())
            for view_idx in range(num_viewpoint):
                try:
                    image_filename = os.path.join(data_path, scene, "images/{:0>4}.png").format(view_idx)
                    image = Image.open(image_filename).convert("RGB")  # load "BGR" -> "RGB"
                except:
                    try:
                        image_filename = os.path.join(data_path, scene, "images/{:0>4}.jpg").format(view_idx)
                        image = Image.open(image_filename).convert("RGB")  # load "BGR" -> "RGB"
                    except:
                        image_filename = os.path.join(data_path, scene, "images/{:0>4}.tif").format(view_idx)
                        image = Image.open(image_filename).convert("RGB")  # load "
                # image = image.resize((1027, 768))
                # image.save(image_filename)
                #print(image.size)
                t0 = time.time()
                depth_est, conf_est = zoe.infer_pil(image)  # as numpy
                # depth_est = (depth_est - np.min(depth_est))  / (np.max(depth_est)-np.min(depth_est))
                # depth_est = (935 - 425) * depth_est + 425
                t1 = time.time()
                # print("Time per Image: {:.3f}".format(t1-t0))
                print(np.min(depth_est), np.max(depth_est))

                depth_filename = os.path.join(depth_save_path, scene, "depths_k/{:0>4}.pfm").format(view_idx)
                os.makedirs(depth_filename.rsplit('/', 1)[0], exist_ok=True)
                save_pfm(depth_filename, depth_est)

                colored = colorize(depth_est)
                depth_img_filename = os.path.join(depth_save_path, scene, "depths_k/{:0>4}.png").format(view_idx)
                Image.fromarray(colored).save(depth_img_filename)
