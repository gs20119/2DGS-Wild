#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state, PILtoTorch
from argparse import ArgumentParser,Namespace
from arguments import ModelParams, PipelineParams, get_combined_args,args_init
from gaussian_renderer import GaussianModel
import copy,pickle,time
from utils.general_utils import *
import imageio

from utils.mesh_utils import GaussianExtractor, to_cam_open3d, post_process_mesh
from utils.render_utils import generate_path, create_videos
import open3d as o3d


def render_interpolate(path, views, gaussians, pipeline, background): # select small portion of views
    inter_path = os.path.join(path, "intrinsic_dynamic_interpolate")
    makedirs(inter_path, exist_ok=True)
    inter_weights=[i*0.1 for i in range(0,21)]
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        for inter_weight in inter_weights:
            gaussians.colornet_inter_weight=inter_weight
            rendering = render(view, gaussians, pipeline, background)["render"]
            torchvision.utils.save_image(rendering, os.path.join(inter_path, f"{idx}_{inter_weight:.2f}.png"))
    gaussians.colornet_inter_weight=1.0

def render_multiview(path, views, gaussians, pipeline, background): # select small portion of views
    origin_views = copy.deepcopy(views)
    multiview_path = os.path.join(path, "multiview")
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        sub_multiview_path=os.path.join(multiview_path,f"{idx}")
        makedirs(sub_multiview_path, exist_ok=True)
        for o_idx, o_view in enumerate(tqdm(origin_views, desc="Rendering progress")):
            rendering = render(view, gaussians, pipeline, background, other_viewpoint_camera=o_view)["render"]
            torchvision.utils.save_image(rendering, os.path.join(sub_multiview_path, f"{idx}_{o_idx}" + ".png"))

def render_intrinsic(path, views, gaussians, pipeline, background):
    intrinsic_path = os.path.join(path, "render_intrinsic")
    makedirs(intrinsic_path, exist_ok=True)
    gaussians.colornet_inter_weight=0.0
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]       
        torchvision.utils.save_image(rendering, os.path.join(intrinsic_path, '{0:05d}'.format(idx) + ".png"))
    gaussians.colornet_inter_weight=1.0


def test_rendering_speed(views, gaussians, pipeline,background,use_cache=False): # don't use
    views=copy.deepcopy(views)
    length=min(1000,len(views))
    for idx in range(length):
        view=views[idx] 
        view.original_image=torch.nn.functional.interpolate(view.original_image.unsqueeze(0),size=(800,800)).squeeze()
        view.image_height,view.image_width=800,800
    if not use_cache:
        rendering = render(views[0], gaussians, pipeline, background)["render"]
        start_time=time.time()
        for idx in tqdm(range(length), desc="Rendering progress"):
            view=views[idx]
            rendering = render(view, gaussians, pipeline, background)["render"]
        end_time=time.time()
        
        avg_rendering_speed=(end_time-start_time)/length
        print(f"rendering speed:{avg_rendering_speed}s/image")
        return avg_rendering_speed
    else:
        for i in range(100):
            views[i+1].image_height,views[i+1].image_width=view.image_height,view.image_width
        rendering = render(views[0], gaussians, pipeline, background,store_cache=True)["render"]
        start_time=time.time()
        rendering = render(view, gaussians, pipeline, background,store_cache=True)["render"]
        #for idx, view in enumerate(tqdm(views[1:], desc="Rendering progress")):
        for idx in tqdm(range(length), desc="Rendering progress"):
            view=views[idx+1]
            rendering = render(view, gaussians, pipeline, background,use_cache=True)["render"]       
        end_time=time.time()
        avg_rendering_speed=(end_time-start_time)/length
        print(f"rendering speed using cache:{avg_rendering_speed}s/image")
        return avg_rendering_speed
    
        
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_mesh", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_interpolate", action="store_true",default=False)
    parser.add_argument("--render_multiview_video", action="store_true",default=False)
    #parser.add_argument("--data_perturb", nargs="+", type=str, default=[]) #for lego ["color","occ"]
    parser.add_argument("--voxel_size", default=-1.0, type=float, help='Mesh: voxel size for TSDF') # From 2DGS
    parser.add_argument("--depth_trunc", default=-1.0, type=float, help='Mesh: Max depth range for TSDF')
    parser.add_argument("--sdf_trunc", default=-1.0, type=float, help='Mesh: truncation value for TSDF')
    parser.add_argument("--num_cluster", default=50, type=int, help='Mesh: number of connected clusters to export')
    parser.add_argument("--unbounded", action="store_true", help='Mesh: using unbounded mode for meshing')
    parser.add_argument("--mesh_res", default=1024, type=int, help='Mesh: resolution for unbounded mesh extraction')
    
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    
    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
    safe_state(args.quiet) # ??

    gaussians = GaussianModel(dataset.sh_degree,args) 
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False) 
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    train_dir = os.path.join(args.model_path, 'train', "ours_{}".format(scene.loaded_iter))
    test_dir = os.path.join(args.model_path, 'test', "ours_{}".format(scene.loaded_iter))
    gaussExtractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg_color)    
    gaussians.set_eval(True)

    if not args.skip_train:
        print("export training images ...")
        train_cameras=scene.getTrainCameras()
        os.makedirs(train_dir, exist_ok=True)
        gaussExtractor.reconstruction(train_cameras)
        gaussExtractor.export_image(train_dir)  
        render_intrinsic(train_dir, train_cameras, gaussians, pipe, background) 

    if not args.skip_test and (len(scene.getTestCameras()) > 0):
        print("export rendered testing images ...")
        test_cameras=scene.getTestCameras()
        os.makedirs(test_dir, exist_ok=True)
        gaussExtractor.reconstruction(test_cameras)
        gaussExtractor.export_image(test_dir)
        render_intrinsic(test_dir, test_cameras, gaussians, pipe, background) 

    if args.render_multiview_video: 
        render_multiview(test_dir, scene.getTestCameras(), gaussians, pipe, background)

    if args.render_interpolate: 
        render_interpolate(test_dir, scene.getTrainCameras(), gaussians, pipe, background)

    if not args.skip_mesh:
        print("export mesh ...")
        os.makedirs(train_dir, exist_ok=True)
        # export only diffuse texture
        # gaussExtractor.gaussians.active_sh_degree = 0
        gaussExtractor.gaussians.colornet_inter_weight = 0.0
        gaussExtractor.reconstruction(scene.getTrainCameras())

        # extract the mesh and save
        if args.unbounded:
            name = 'fuse_unbounded.ply'
            mesh = gaussExtractor.extract_mesh_unbounded(resolution=args.mesh_res)
        else:
            name = 'fuse.ply'
            depth_trunc = (gaussExtractor.radius * 2.0) if args.depth_trunc < 0  else args.depth_trunc
            voxel_size = (depth_trunc / args.mesh_res) if args.voxel_size < 0 else args.voxel_size
            sdf_trunc = 5.0 * voxel_size if args.sdf_trunc < 0 else args.sdf_trunc
            mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)
        
        gaussExtractor.gaussians.colornet_inter_weight = 1.0
        o3d.io.write_triangle_mesh(os.path.join(train_dir, name), mesh)
        print("mesh saved at {}".format(os.path.join(train_dir, name)))
        
        # post-process the mesh and save, saving the largest N clusters
        mesh_post = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
        o3d.io.write_triangle_mesh(os.path.join(train_dir, name.replace('.ply', '_post.ply')), mesh_post)
        print("mesh post processed saved at {}".format(os.path.join(train_dir, name.replace('.ply', '_post.ply'))))

    # All done
    gaussians.set_eval(False)
    print("\nRendering complete.")


