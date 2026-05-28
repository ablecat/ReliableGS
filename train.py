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

import os
import torch
import numpy as np
import random
from random import randint
from utils.loss_utils import l1_loss, ssim, lncc, get_img_grad_weight
from utils.graphics_utils import patch_offsets, patch_warp
from utils.camera_utils import gen_virtul_cam
from gaussian_renderer import render, render_depth
import sys
from scene import Scene, GaussianModel, SpecularModel
from utils.general_utils import safe_state
import cv2
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, erode
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.app_model import AppModel
from scene.cameras import Camera
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
import logging
from lpipsPyTorch import lpips
import json

from utils.mv_normal_utils import compute_mv_normal_loss, get_img_grad_weight
from utils.reliability_stats import update_reliability_buffer, get_reliability_weights, get_reliability_map
from utils.diagnostics import DiagnosticTracker  # [DIAG]

# launch tensorboard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer, logger = prepare_output_and_logger(dataset)
    
    # 记录命令行参数
    command = " ".join(sys.argv)
    logger.info(f"Command: {command}")
    
    # backup main code
    cmd = f'cp ./train.py {dataset.model_path}/'
    os.system(cmd)
    cmd = f'cp -rf ./arguments {dataset.model_path}/'
    os.system(cmd)
    cmd = f'cp -rf ./gaussian_renderer {dataset.model_path}/'
    os.system(cmd)
    cmd = f'cp -rf ./scene {dataset.model_path}/'
    os.system(cmd)
    cmd = f'cp -rf ./utils {dataset.model_path}/'
    os.system(cmd)

    if opt.use_asg:    
        specular_mlp = SpecularModel(dataset.is_real, dataset.is_indoor)
        specular_mlp.train_setting(opt)
        asg_degree = dataset.asg_degree
    else:
        specular_mlp = None
        asg_degree = None

    gaussians = GaussianModel(dataset.sh_degree, asg_degree)


    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    app_model = AppModel()
    app_model.train()
    app_model.cuda()
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        app_model.load_weights(scene.model_path)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_single_view_for_log = 0.0
    ema_multi_view_geo_for_log = 0.0
    ema_multi_view_pho_for_log = 0.0
    ema_sd_normal_loss_for_log = 0.0
    ema_alpha_loss_for_log = 0.0
    ema_transparency_loss_for_log = 0.0
    normal_loss, geo_loss, ncc_loss = None, None, None
    sd_normal_loss = None
    alpha_loss = None
    transparency_loss = None
    mv_normal_loss = None
    ema_mv_normal_loss_for_log = 0.0
    ema_reliability_for_log = 0.0
    reliability_weights_for_ncc = None
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    debug_path = os.path.join(scene.model_path, "debug")
    os.makedirs(debug_path, exist_ok=True)

    # Reliability buffer: per-view reliability maps (Innovation 1 - multi-view statistics)
    reliability_buffer = {}

    # [DIAG] Initialize diagnostic tracker
    _diag = DiagnosticTracker(debug_path, tb_writer, enabled=opt.use_reliability)

    logger.info(f"opt.wo_depth_normal_detach: {opt.wo_depth_normal_detach}")

    for iteration in range(first_iter, opt.iterations + 1):
        _diag.reset_iter()  # [DIAG]
        iter_start.record()
        # gaussians.update_learning_rate(iteration)
        gaussians.selective_learning_rate_control(iteration, 15000, nofix_position=opt.nofix_position, nofix_opacity=opt.nofix_opacity, nofix_param=opt.nofix_param, nofix_scaling=opt.nofix_scaling, nofix_rotation=opt.nofix_rotation)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        gt_image, gt_image_gray, gt_image_delight, gt_image_normal, transparencies_map = viewpoint_cam.get_image()

        # ── Periodic reliability buffer update (Innovation 1) ──
        if (opt.use_reliability
            and iteration >= opt.reliability_start_iter
            and iteration % opt.reliability_update_every == 0):
            update_reliability_buffer(
                scene=scene,
                gaussians=gaussians,
                pipe=pipe,
                background=bg,
                render_fn=render,
                app_model=app_model,
                reliability_buffer=reliability_buffer,
                K=opt.reliability_K,
                tau_d=opt.reliability_tau_d,
                tau_n=opt.reliability_tau_n,
                pixel_noise_th=opt.reliability_pixel_noise_th,
                save_dir=debug_path,
                iteration=iteration,
                adaptive_tau=opt.reliability_adaptive_tau,
                tau_quantile=opt.reliability_tau_quantile,
                tau_alpha=opt.reliability_tau_alpha,
                tau_d_floor=opt.reliability_tau_d_floor,
                tau_n_floor=opt.reliability_tau_n_floor,
            )
        # sd delight pass, replace gt_image with gt_image_delight, to build better geometry
        if args.delight and iteration < opt.delight_iterations:
            gt_image = gt_image_delight
        elif args.delight and iteration == opt.delight_iterations:
            # clear optimizer momentum when delight iterations end
            # after test, it influnces little on the final result, not necessary to use
            gaussians.clear_optimizer_momentum(clear_f_dc=opt.clear_f_dc, clear_f_rest=opt.clear_f_rest, clear_opacity=opt.clear_opacity, clear_scaling=opt.clear_scaling, clear_rotation=opt.clear_rotation)

        if iteration > 1000 and opt.exposure_compensation:
            gaussians.use_app = True

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        if opt.use_asg:
            if dataset.load2gpu_on_the_fly:
                viewpoint_cam.load2device()

            if iteration > 3000 + opt.delight_iterations:
                dir_pp = (gaussians.get_xyz - viewpoint_cam.camera_center.repeat(gaussians.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                normal = gaussians.get_normal_axis(dir_pp_normalized=dir_pp_normalized, return_delta=True)
                mlp_color = specular_mlp.step(gaussians.get_asg_features, dir_pp_normalized, normal.detach())
            else:
                mlp_color = 0
        else:
            mlp_color = None

        if iteration % 1000 == 0:
            logger.info(f"iteration: {iteration}")

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, app_model=app_model,
                            # return_plane=iteration>opt.single_view_weight_from_iter, 
                            return_plane=True,
                            # return_depth_normal=iteration>opt.single_view_weight_from_iter
                            return_depth_normal=True,
                            wo_depth_normal_detach=opt.wo_depth_normal_detach,
                            mlp_color=mlp_color,
                            T_threshold=opt.T_threshold,
                            observe_T_threshold=opt.observe_T_threshold,
                            )
        image, viewspace_point_tensor, visibility_filter, radii = \
            render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        # Loss
        ssim_loss = (1.0 - ssim(image, gt_image))
        if 'app_image' in render_pkg and ssim_loss < 0.5:
            app_image = render_pkg['app_image']
            Ll1 = l1_loss(app_image, gt_image)
        else:
            Ll1 = l1_loss(image, gt_image)
        image_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss
        loss = image_loss.clone()
        
        # scale loss
        if visibility_filter.sum() > 0:
            scale = gaussians.get_scaling[visibility_filter]
            sorted_scale, _ = torch.sort(scale, dim=-1)
            min_scale_loss = sorted_scale[...,0]
            loss += opt.scale_loss_weight * min_scale_loss.mean()
        # single-view loss
        if iteration > opt.single_view_weight_from_iter:
            weight = opt.single_view_weight
            normal = render_pkg["rendered_normal"]
            depth_normal = render_pkg["depth_normal"]

            image_weight = (1.0 - get_img_grad_weight(gt_image))
            image_weight = (image_weight).clamp(0,1).detach() ** 2
            if not opt.wo_image_weight:
                normal_loss = weight * (image_weight * (((depth_normal - normal)).abs().sum(0))).mean()
            else:
                if not opt.use_2dgsnormal_loss:
                    normal_loss = weight * (((depth_normal - normal)).abs().sum(0)).mean()
                else:
                    normal_error = (1 - (normal * depth_normal).sum(dim=0))[None]
                    normal_loss = weight * (normal_error).mean()
            loss += (normal_loss)

        # sd normal loss
        if args.normal:
            lambda_sd_normal = opt.lambda_sd_normal if iteration < opt.sd_normal_until_iter else 0.0
            normal = render_pkg["rendered_normal"]

            if iteration > args.normal_cos_threshold_iter:
                normal_threshold = 0.3
                cosine_similarity = (normal * (-gt_image_normal)).sum(dim=0)
                valid_normal_mask = (cosine_similarity > normal_threshold)
            else:
                valid_normal_mask = torch.ones(normal.shape[1:], device=normal.device)
            if not opt.use_2dgsnormal_loss:
                # 应用mask到L1损失
                normal_diff = ((-gt_image_normal - normal)).abs().sum(0)
                sd_normal_loss = lambda_sd_normal * (normal_diff * valid_normal_mask.float()).sum() / (valid_normal_mask.sum() + 1e-6)
            else:
                # 应用mask到余弦相似度损失
                sd_normal_error = (1 - cosine_similarity)[None]
                sd_normal_loss = lambda_sd_normal * (sd_normal_error * valid_normal_mask.float()[None]).sum() / (valid_normal_mask.sum() + 1e-6)
            loss += sd_normal_loss
            
            if opt.wo_depth_normal_detach:
                depth_normal_loss_item = lambda_sd_normal * ((((-gt_image_normal - render_pkg["depth_normal"])).abs().sum(0))).mean()
                sd_normal_loss += depth_normal_loss_item

            if viewpoint_cam.mask is not None:
                # mask & alpha loss
                lambda_alpha = 0.1
                gt_alpha = viewpoint_cam.mask.cuda()
                alpha_loss = lambda_alpha * F.binary_cross_entropy(render_pkg["rendered_alpha"], gt_alpha)
                loss += alpha_loss
                
            # calculate transparency loss
            lambda_transparency = 0.1
            if iteration > 3000 and transparencies_map is not None:          
                transparency_loss = lambda_transparency * F.binary_cross_entropy(render_pkg["out_transparency_map"], transparencies_map)
                loss += transparency_loss
                


        # multi-view loss
        if iteration > opt.multi_view_weight_from_iter:
            nearest_cam = None if len(viewpoint_cam.nearest_id) == 0 else scene.getTrainCameras()[random.sample(viewpoint_cam.nearest_id,1)[0]]
            use_virtul_cam = False
            if opt.use_virtul_cam and (np.random.random() < opt.virtul_cam_prob or nearest_cam is None):
                nearest_cam = gen_virtul_cam(viewpoint_cam, trans_noise=dataset.multi_view_max_dis, deg_noise=dataset.multi_view_max_angle)
                use_virtul_cam = True
            if nearest_cam is not None:
                patch_size = opt.multi_view_patch_size
                sample_num = opt.multi_view_sample_num
                pixel_noise_th = opt.multi_view_pixel_noise_th
                total_patch_size = (patch_size * 2 + 1) ** 2
                ncc_weight = opt.multi_view_ncc_weight
                geo_weight = opt.multi_view_geo_weight
                ## compute geometry consistency mask and loss
                H, W = render_pkg['plane_depth'].squeeze().shape
                ix, iy = torch.meshgrid(
                    torch.arange(W), torch.arange(H), indexing='xy')
                pixels = torch.stack([ix, iy], dim=-1).float().to(render_pkg['plane_depth'].device)
                if opt.use_asg:
                    if iteration > 3000 + opt.delight_iterations:
                        dir_pp = (scene.gaussians.get_xyz - nearest_cam.camera_center.repeat(
                            scene.gaussians.get_features.shape[0], 1))
                        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                        normal = scene.gaussians.get_normal_axis(dir_pp_normalized=dir_pp_normalized, return_delta=True)
                        mlp_color = specular_mlp.step(scene.gaussians.get_asg_features, dir_pp_normalized, normal)
                    else:
                        mlp_color = 0
                else:
                    mlp_color = None

                nearest_render_pkg = render(nearest_cam, gaussians, pipe, bg, app_model=app_model,
                                            return_plane=True,
                                            return_depth_normal=opt.use_mv_normal_consistency,
                                            mlp_color=mlp_color)

                pts = gaussians.get_points_from_depth(viewpoint_cam, render_pkg['plane_depth'])
                pts_in_nearest_cam = pts @ nearest_cam.world_view_transform[:3,:3] + nearest_cam.world_view_transform[3,:3]
                map_z, d_mask = gaussians.get_points_depth_in_depth_map(nearest_cam, nearest_render_pkg['plane_depth'], pts_in_nearest_cam)
                
                pts_in_nearest_cam = pts_in_nearest_cam / (pts_in_nearest_cam[:,2:3])
                pts_in_nearest_cam = pts_in_nearest_cam * map_z.squeeze()[...,None]
                R = torch.tensor(nearest_cam.R).float().cuda()
                T = torch.tensor(nearest_cam.T).float().cuda()
                pts_ = (pts_in_nearest_cam-T)@R.transpose(-1,-2)
                pts_in_view_cam = pts_ @ viewpoint_cam.world_view_transform[:3,:3] + viewpoint_cam.world_view_transform[3,:3]
                pts_projections = torch.stack(
                            [pts_in_view_cam[:,0] * viewpoint_cam.Fx / pts_in_view_cam[:,2] + viewpoint_cam.Cx,
                            pts_in_view_cam[:,1] * viewpoint_cam.Fy / pts_in_view_cam[:,2] + viewpoint_cam.Cy], -1).float()
                pixel_noise = torch.norm(pts_projections - pixels.reshape(*pts_projections.shape), dim=-1)
                if not opt.wo_use_geo_occ_aware:
                    d_mask = d_mask & (pixel_noise < pixel_noise_th)
                    weights = (1.0 / torch.exp(pixel_noise)).detach()
                    weights[~d_mask] = 0
                else:
                    d_mask = d_mask
                    weights = torch.ones_like(pixel_noise)
                    weights[~d_mask] = 0
                weights_baseline = weights.clone()  # save original weights for NCC (before Innovation 2)
                # Innovation 2: Dual-view reliability modulation for geo_loss only
                if opt.use_reliability and iteration >= opt.reliability_use_from_iter and not opt.reliability_no_loss_modulate:
                    _d_mask_before = d_mask.clone()  # [DIAG]
                    _weights_before_sum = weights[d_mask].sum().item()  # [DIAG]
                    with torch.no_grad():
                        r_ref_map = get_reliability_map(reliability_buffer, viewpoint_cam)
                        r_tgt_map = get_reliability_map(reliability_buffer, nearest_cam)
                        if r_ref_map is not None and r_tgt_map is not None:
                            r_ref_flat = r_ref_map.reshape(-1).to(weights.device)
                            # Sample r_tgt at nearest_cam pixel coords for each ref pixel
                            W_n, H_n = int(nearest_cam.image_width), int(nearest_cam.image_height)
                            z_safe = pts_in_nearest_cam[:, 2].clamp(min=0.1)
                            proj_x_n = pts_in_nearest_cam[:, 0] * nearest_cam.Fx / z_safe + nearest_cam.Cx
                            proj_y_n = pts_in_nearest_cam[:, 1] * nearest_cam.Fy / z_safe + nearest_cam.Cy
                            gx_n = 2.0 * proj_x_n / (W_n - 1) - 1.0
                            gy_n = 2.0 * proj_y_n / (H_n - 1) - 1.0
                            grid_n = torch.stack([gx_n, gy_n], dim=-1).reshape(1, 1, -1, 2)
                            r_tgt_sampled = F.grid_sample(
                                r_tgt_map.unsqueeze(0).unsqueeze(0).to(grid_n.device),
                                grid_n, mode='bilinear', padding_mode='zeros', align_corners=True
                            ).reshape(-1)
                            dual_rel = (r_ref_flat * r_tgt_sampled).clamp(min=0.05, max=1)
                            # NOTE: mean-normalization removed. It forced mean(dual_rel)=1.0
                            # which cancelled R-map suppression on transparent pixels.
                            # Plain multiplicative: dual_rel in [0.05,1] attenuates weights
                            # in low-R regions and keeps them near 1 in high-R regions.
                            weights = weights * dual_rel
                            # Tighten d_mask: exclude extremely low dual-reliability pixels
                            d_mask = d_mask & (dual_rel > 0.01)
                            weights[~d_mask] = 0
                            # [DIAG] Record Innovation 2 diagnostics
                            _diag.record_innov2(
                                _d_mask_before, d_mask,
                                _weights_before_sum, weights[d_mask].sum().item(),
                                dual_rel, r_ref_flat, r_tgt_sampled)
                with torch.no_grad():
                    if iteration % 5000 == 0:
                        gt_img_show = ((gt_image).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                        if 'app_image' in render_pkg:
                            img_show = ((render_pkg['app_image']).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                        else:
                            img_show = ((image).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                        normal = render_pkg["rendered_normal"]
                        normal_show = (((normal+1.0)*0.5).permute(1,2,0).clamp(0,1)*255).detach().cpu().numpy().astype(np.uint8)
                        depth_normal_show = (((depth_normal+1.0)*0.5).permute(1,2,0).clamp(0,1)*255).detach().cpu().numpy().astype(np.uint8)
                        # Original d_mask (before reliability modulation)
                        d_mask_show = (weights_baseline.float()*255).detach().cpu().numpy().astype(np.uint8).reshape(H,W)
                        d_mask_show_color = cv2.applyColorMap(d_mask_show, cv2.COLORMAP_JET)
                        depth = render_pkg['plane_depth'].squeeze().detach().cpu().numpy()
                        depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
                        depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
                        depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)
                        distance = render_pkg['rendered_distance'].squeeze().detach().cpu().numpy()
                        distance_i = (distance - distance.min()) / (distance.max() - distance.min() + 1e-20)
                        distance_i = (distance_i * 255).clip(0, 255).astype(np.uint8)
                        distance_color = cv2.applyColorMap(distance_i, cv2.COLORMAP_JET)
                        # R-map visualization
                        _r_map_debug = get_reliability_map(reliability_buffer, viewpoint_cam) if opt.use_reliability else None
                        if _r_map_debug is not None:
                            _r_np = (_r_map_debug.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                            rmap_color = cv2.applyColorMap(cv2.resize(_r_np, (W, H)), cv2.COLORMAP_JET)
                        else:
                            rmap_color = np.full((H, W, 3), 128, dtype=np.uint8)
                        # Image gradient weight (edge-aware SD normal weight)
                        _iw = image_weight.detach().cpu().numpy()
                        _iw = (_iw * 255).clip(0, 255).astype(np.uint8)
                        image_weight_color = cv2.applyColorMap(_iw, cv2.COLORMAP_JET)
                        # Transparency map visualization (GT annotation)
                        if transparencies_map is not None:
                            _tm = transparencies_map[0].cpu().numpy()
                            _tm = (_tm * 255).clip(0, 255).astype(np.uint8)
                            trans_color = cv2.applyColorMap(_tm, cv2.COLORMAP_JET)
                        else:
                            trans_color = np.full((H, W, 3), 128, dtype=np.uint8)
                        row0 = np.concatenate([gt_img_show, img_show, normal_show, distance_color, image_weight_color], axis=1)
                        row1 = np.concatenate([d_mask_show_color, rmap_color, depth_color, depth_normal_show, trans_color], axis=1)
                        image_to_show = np.concatenate([row0, row1], axis=0)
                        cv2.imwrite(os.path.join(debug_path, "%05d"%iteration + "_" + viewpoint_cam.image_name + ".jpg"), image_to_show)



                if d_mask.sum() > 0:
                    geo_loss = geo_weight * ((weights * pixel_noise)[d_mask]).mean()
                    loss += geo_loss
                    if use_virtul_cam is False:
                        with torch.no_grad():
                            ## sample mask
                            d_mask = d_mask.reshape(-1)
                            valid_indices = torch.arange(d_mask.shape[0], device=d_mask.device)[d_mask]
                            if d_mask.sum() > sample_num:
                                index = np.random.choice(d_mask.sum().cpu().numpy(), sample_num, replace = False)
                                valid_indices = valid_indices[index]

                            weights = weights.reshape(-1)[valid_indices]
                            weights_baseline = weights_baseline.reshape(-1)[valid_indices]
                            ## sample ref frame patch
                            pixels = pixels.reshape(-1,2)[valid_indices]
                            offsets = patch_offsets(patch_size, pixels.device)
                            ori_pixels_patch = pixels.reshape(-1, 1, 2) / viewpoint_cam.ncc_scale + offsets.float()
                            
                            H, W = gt_image_gray.squeeze().shape
                            pixels_patch = ori_pixels_patch.clone()
                            pixels_patch[:, :, 0] = 2 * pixels_patch[:, :, 0] / (W - 1) - 1.0
                            pixels_patch[:, :, 1] = 2 * pixels_patch[:, :, 1] / (H - 1) - 1.0
                            ref_gray_val = F.grid_sample(gt_image_gray.unsqueeze(1), pixels_patch.view(1, -1, 1, 2), align_corners=True)
                            ref_gray_val = ref_gray_val.reshape(-1, total_patch_size)

                            ref_to_neareast_r = nearest_cam.world_view_transform[:3,:3].transpose(-1,-2) @ viewpoint_cam.world_view_transform[:3,:3]
                            ref_to_neareast_t = -ref_to_neareast_r @ viewpoint_cam.world_view_transform[3,:3] + nearest_cam.world_view_transform[3,:3]

                        ## compute Homography
                        ref_local_n = render_pkg["rendered_normal"].permute(1,2,0)
                        ref_local_n = ref_local_n.reshape(-1,3)[valid_indices]

                        ref_local_d = render_pkg['rendered_distance'].squeeze()
                        # rays_d = viewpoint_cam.get_rays()
                        # rendered_normal2 = render_pkg["rendered_normal"].permute(1,2,0).reshape(-1,3)
                        # ref_local_d = render_pkg['plane_depth'].view(-1) * ((rendered_normal2 * rays_d.reshape(-1,3)).sum(-1).abs())
                        # ref_local_d = ref_local_d.reshape(*render_pkg['plane_depth'].shape)

                        ref_local_d = ref_local_d.reshape(-1)[valid_indices]
                        H_ref_to_neareast = ref_to_neareast_r[None] - \
                            torch.matmul(ref_to_neareast_t[None,:,None].expand(ref_local_d.shape[0],3,1), 
                                        ref_local_n[:,:,None].expand(ref_local_d.shape[0],3,1).permute(0, 2, 1))/ref_local_d[...,None,None]
                        H_ref_to_neareast = torch.matmul(nearest_cam.get_k(nearest_cam.ncc_scale)[None].expand(ref_local_d.shape[0], 3, 3), H_ref_to_neareast)
                        H_ref_to_neareast = H_ref_to_neareast @ viewpoint_cam.get_inv_k(viewpoint_cam.ncc_scale)
                        
                        ## compute neareast frame patch
                        grid = patch_warp(H_ref_to_neareast.reshape(-1,3,3), ori_pixels_patch)
                        grid[:, :, 0] = 2 * grid[:, :, 0] / (W - 1) - 1.0
                        grid[:, :, 1] = 2 * grid[:, :, 1] / (H - 1) - 1.0
                        _, nearest_image_gray, nearest_image_delight, nearest_image_normal, nearest_transparencies_map = nearest_cam.get_image()
                        sampled_gray_val = F.grid_sample(nearest_image_gray[None], grid.reshape(1, -1, 1, 2), align_corners=True)
                        sampled_gray_val = sampled_gray_val.reshape(-1, total_patch_size)
                        
                        ## apply reliability weights (Innovation 1 - multi-view statistics)
                        reliability_weights_for_ncc = None
                        if opt.use_reliability and iteration >= opt.reliability_use_from_iter and not opt.reliability_no_loss_modulate:
                            with torch.no_grad():
                                rel_w = get_reliability_weights(
                                    reliability_buffer, viewpoint_cam, valid_indices)
                                reliability_weights_for_ncc = rel_w
                                # weights = weights * rel_w  # Removed: dual-view R already applied upstream (Innovation 2)

                        # Save reliability heatmap (every 500 iterations)
                        if opt.use_reliability and iteration >= opt.reliability_use_from_iter and iteration % 5000 == 0:
                            R_map = get_reliability_map(reliability_buffer, viewpoint_cam)
                            if R_map is not None:
                                rel_img = R_map.cpu().numpy()
                                rel_show = (rel_img * 255).clip(0, 255).astype(np.uint8)
                                rel_color = cv2.applyColorMap(rel_show, cv2.COLORMAP_JET)
                                cv2.imwrite(os.path.join(debug_path, "%05d"%iteration + "_" + viewpoint_cam.image_name + "_reliability.jpg"), rel_color)

                        ## compute loss
                        if iteration > args.ncc_loss_from_iter:
                            ncc, ncc_mask = lncc(ref_gray_val, sampled_gray_val)
                            mask = ncc_mask.reshape(-1)
                            ncc = ncc.reshape(-1) * weights_baseline
                            # R-map photometric suppression: reduce NCC in low-reliability
                            # (transparent) regions where photometric matching is deceptive
                            if reliability_weights_for_ncc is not None:
                                _ncc_before = ncc[mask].mean().item() if mask.sum() > 0 else 0.0  # [DIAG]
                                ncc = ncc * reliability_weights_for_ncc
                                _ncc_after = ncc[mask].mean().item() if mask.sum() > 0 else 0.0  # [DIAG]
                                _diag.record_ncc_modulation(_ncc_before, _ncc_after, reliability_weights_for_ncc[mask])  # [DIAG]
                            ncc = ncc[mask].squeeze()

                            if mask.sum() > 0:
                                ncc_loss = ncc_weight * ncc.mean()
                                loss += ncc_loss
                        # Restore H, W to render resolution (ncc_scale may have
                        # reassigned them to the NCC gray-image resolution).
                        H, W = render_pkg['plane_depth'].squeeze().shape


                        # Multi-view normal consistency loss (MND-GS style)
                        if opt.use_mv_normal_consistency:
                            # Edge-aware weighting (MNE from MND-GS)
                            _edge_grad = get_img_grad_weight(gt_image)  # (H,W) edges->1
                            _edge_w_valid = (1.0 - _edge_grad).reshape(-1)[valid_indices]
                            # Inverse reliability sampling: pass R-map so MNE
                            # oversamples transparent (low-R) pixels
                            _r_for_mne = reliability_weights_for_ncc  # (M,) or None
                            mv_result = compute_mv_normal_loss(
                                viewpoint_cam=viewpoint_cam,
                                nearest_cam=nearest_cam,
                                render_pkg=render_pkg,
                                nearest_render_pkg=nearest_render_pkg,
                                valid_indices=valid_indices,
                                H_ref_to_neareast=H_ref_to_neareast,
                                ix=ix, iy=iy, H=H, W=W,
                                mv_normal_weight=opt.mv_normal_weight,
                                loss_type=opt.mv_normal_loss_type,
                                normal_key="depth_normal",
                                max_samples=opt.mv_normal_max_samples,
                                edge_weight=_edge_w_valid,
                                reliability_for_sampling=_r_for_mne,
                            )
                            if mv_result["has_valid"]:
                                mv_normal_loss = opt.mv_normal_weight * mv_result["loss"]
                                if not (torch.isnan(mv_normal_loss) or torch.isinf(mv_normal_loss)):
                                    loss += mv_normal_loss
                                else:
                                    mv_normal_loss = None
                            # [DIAG] Record MNE sampling diagnostics
                            _mne_sdiag = mv_result.get("sampling_diag", {})
                            if _mne_sdiag:
                                _mne_sdiag["n_valid"] = mv_result.get("n_valid", 0)
                                _mne_sdiag["loss"] = mv_result["loss"].item() if mv_result["has_valid"] else 0.0
                            _diag.record_mne_sampling(_mne_sdiag)

                            # ── MVN Dense Diagnostics (Layer 2 & 3) ──
                            if (iteration % 5000 == 0):
                                with torch.no_grad():
                                    # --- Dense MVN: recompute with ALL valid_indices (no max_samples) ---
                                    dense_mv = compute_mv_normal_loss(
                                        viewpoint_cam=viewpoint_cam,
                                        nearest_cam=nearest_cam,
                                        render_pkg=render_pkg,
                                        nearest_render_pkg=nearest_render_pkg,
                                        valid_indices=valid_indices,
                                        H_ref_to_neareast=H_ref_to_neareast,
                                        ix=ix, iy=iy, H=H, W=W,
                                        mv_normal_weight=opt.mv_normal_weight,
                                        loss_type=opt.mv_normal_loss_type,
                                        normal_key="depth_normal",
                                        max_samples=0,
                                        edge_weight=_edge_w_valid,
                                    )

                                    _n_total_pixels = H * W
                                    _n_dmask = int(d_mask.sum().item()) if d_mask is not None else 0
                                    _n_valid = valid_indices.numel()

                                    if dense_mv["has_valid"] and dense_mv["cos_sim"] is not None:
                                        _mv_cos   = dense_mv["cos_sim"]        # (M,)
                                        _mv_ok    = dense_mv["ok_mask"]        # (M,)
                                        _mv_lpp   = dense_mv["loss_per_pixel"] # (M_ok,)
                                        _mv_rw    = dense_mv["weights"]      # (M_ok,) or None
                                        _mv_pidx  = dense_mv["pixel_indices"]  # (M,)
                                        M_total   = _mv_pidx.numel()
                                        M_ok      = int(_mv_ok.sum().item())

                                        # --- Build full-image maps from dense results ---
                                        cos_map = torch.full((H * W,), float('nan'), device="cuda")
                                        cos_map[_mv_pidx] = _mv_cos
                                        cos_map = cos_map.reshape(H, W)

                                        loss_map = torch.zeros(H * W, device="cuda")
                                        ok_pidx = _mv_pidx[_mv_ok]
                                        loss_map[ok_pidx] = _mv_lpp
                                        loss_map = loss_map.reshape(H, W)

                                        rw_loss_map = torch.zeros(H * W, device="cuda")
                                        if _mv_rw is not None:
                                            rw_loss_map[ok_pidx] = _mv_lpp * _mv_rw
                                        rw_loss_map = rw_loss_map.reshape(H, W)

                                        # coverage mask: which pixels have MVN data
                                        coverage_map = torch.zeros(H * W, device="cuda")
                                        coverage_map[_mv_pidx] = 1.0
                                        coverage_map = coverage_map.reshape(H, W)
                                    else:
                                        cos_map = torch.full((H, W), float('nan'), device="cuda")
                                        loss_map = torch.zeros(H, W, device="cuda")
                                        rw_loss_map = torch.zeros(H, W, device="cuda")
                                        coverage_map = torch.zeros(H, W, device="cuda")
                                        M_total = 0; M_ok = 0
                                        ok_pidx = torch.tensor([], dtype=torch.long, device="cuda")

                                    # --- d_mask coverage map ---
                                    if d_mask is not None:
                                        _rH, _rW = render_pkg['plane_depth'].squeeze().shape
                                        dmask_map = d_mask.float().reshape(_rH, _rW)
                                        if (_rH, _rW) != (H, W):
                                            dmask_map = F.interpolate(dmask_map[None, None], (H, W), mode='nearest')[0, 0]
                                    else:
                                        dmask_map = torch.zeros(H, W, device="cuda")

                                    # --- Helper: tensor -> JET colormap image ---
                                    def _to_jet(t, vmin=None, vmax=None):
                                        arr = t.cpu().numpy()
                                        mask_nan = ~np.isfinite(arr)
                                        if vmin is None: vmin = np.nanmin(arr) if not mask_nan.all() else 0
                                        if vmax is None: vmax = np.nanmax(arr) if not mask_nan.all() else 1
                                        arr = np.clip((arr - vmin) / (vmax - vmin + 1e-10), 0, 1)
                                        arr = (arr * 255).astype(np.uint8)
                                        arr[mask_nan] = 0
                                        return cv2.applyColorMap(arr, cv2.COLORMAP_JET)

                                    # --- 3x3 diagnostic panel ---
                                    # Row 0: GT | d_mask coverage (white=pass) | Transparency
                                    gt_panel = ((gt_image).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).cpu().numpy().astype(np.uint8)
                                    dmask_panel = (dmask_map.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                                    dmask_panel = cv2.cvtColor(dmask_panel, cv2.COLOR_GRAY2BGR)
                                    trans_np = transparencies_map[0].cpu().numpy() if transparencies_map is not None else np.zeros((H, W))
                                    trans_panel = cv2.applyColorMap((trans_np * 255).clip(0,255).astype(np.uint8), cv2.COLORMAP_JET)

                                    # Row 1: cos_sim | MVN L1 loss | loss*edge_w
                                    cos_panel = _to_jet(cos_map, vmin=0.9, vmax=1.0)
                                    loss_panel = _to_jet(loss_map, vmin=0, vmax=0.2)
                                    rwloss_panel = _to_jet(rw_loss_map, vmin=0, vmax=0.2)

                                    # Row 2: R-map | render-GT residual | MVN coverage
                                    _r_diag = get_reliability_map(reliability_buffer, viewpoint_cam) if opt.use_reliability else None
                                    if _r_diag is not None:
                                        rmap_panel = cv2.applyColorMap((_r_diag.cpu().numpy() * 255).clip(0,255).astype(np.uint8), cv2.COLORMAP_JET)
                                        rmap_panel = cv2.resize(rmap_panel, (W, H))
                                    else:
                                        rmap_panel = np.full((H, W, 3), 128, dtype=np.uint8)

                                    render_img = render_pkg["render"]  # (3, H, W)
                                    residual = (gt_image - render_img).abs().mean(dim=0)  # (H, W)
                                    residual_panel = _to_jet(residual, vmin=0, vmax=0.1)

                                    coverage_panel = (coverage_map.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                                    coverage_panel = cv2.cvtColor(coverage_panel, cv2.COLOR_GRAY2BGR)

                                    row0 = np.concatenate([gt_panel, dmask_panel, trans_panel], axis=1)
                                    row1 = np.concatenate([cos_panel, loss_panel, rwloss_panel], axis=1)
                                    row2 = np.concatenate([rmap_panel, residual_panel, coverage_panel], axis=1)
                                    mvn_diag = np.concatenate([row0, row1, row2], axis=0)

                                    # Add labels with coverage stats
                                    font = cv2.FONT_HERSHEY_SIMPLEX
                                    labels = [
                                        "GT",
                                        f"d_mask ({_n_dmask}/{_n_total_pixels}={100*_n_dmask/_n_total_pixels:.1f}%)",
                                        "Transparency",
                                        f"cos_sim [0.9,1.0] (N={M_total})",
                                        f"MVN L1 [0,0.2] (ok={M_ok})",
                                        f"loss*edge [0,0.2]",
                                        "R-map",
                                        "render-GT residual",
                                        f"MVN coverage ({M_total}/{_n_total_pixels}={100*M_total/_n_total_pixels:.1f}%)",
                                    ]
                                    for i, lbl in enumerate(labels):
                                        col = (i % 3) * W + 5
                                        r = (i // 3) * H + 20
                                        cv2.putText(mvn_diag, lbl, (col, r), font, 0.4, (255,255,255), 1, cv2.LINE_AA)
                                    cv2.imwrite(os.path.join(debug_path, "%05d" % iteration + "_" + viewpoint_cam.image_name + "_mvn_diag.jpg"), mvn_diag)

                                    # --- Layer 3: Dense statistics ---
                                    if dense_mv["has_valid"] and transparencies_map is not None:
                                        trans_flat = transparencies_map[0].reshape(-1).to("cuda")
                                        ok_trans = trans_flat[ok_pidx]  # (M_ok,)
                                        is_transparent = ok_trans > 0.5
                                        is_opaque = ~is_transparent
                                        n_trans = is_transparent.sum().item()
                                        n_opaq  = is_opaque.sum().item()
                                        cos_ok = _mv_cos[_mv_ok]  # (M_ok,)

                                        # Quantile stats for cos_sim
                                        cos_q = torch.quantile(_mv_cos[_mv_ok].float(), torch.tensor([0.01, 0.05, 0.25, 0.50], device="cuda"))

                                        stats_lines = [
                                            f"[MVN-DENSE iter={iteration}] cam={viewpoint_cam.image_name}",
                                            f"  pixels: total={_n_total_pixels} d_mask={_n_dmask}({100*_n_dmask/_n_total_pixels:.1f}%) valid={_n_valid} mvn_dense={M_total} ok={M_ok}",
                                            f"  cos_sim quantiles: q01={cos_q[0]:.4f} q05={cos_q[1]:.4f} q25={cos_q[2]:.4f} q50={cos_q[3]:.4f}",
                                        ]
                                        if n_trans > 0:
                                            t_loss = _mv_lpp[is_transparent].mean().item()
                                            t_cos  = cos_ok[is_transparent].mean().item()
                                            t_rw   = _mv_rw[is_transparent].mean().item() if _mv_rw is not None else -1
                                            t_loss_max = _mv_lpp[is_transparent].max().item()
                                            stats_lines.append(f"  [TRANSPARENT n={n_trans}] mean_loss={t_loss:.6f} max_loss={t_loss_max:.6f} mean_cos={t_cos:.6f} mean_ew={t_rw:.4f}")
                                        if n_opaq > 0:
                                            o_loss = _mv_lpp[is_opaque].mean().item()
                                            o_cos  = cos_ok[is_opaque].mean().item()
                                            o_rw   = _mv_rw[is_opaque].mean().item() if _mv_rw is not None else -1
                                            o_loss_max = _mv_lpp[is_opaque].max().item()
                                            # Find top-loss opaque pixels
                                            _opq_losses = _mv_lpp[is_opaque]
                                            _opq_top5 = torch.topk(_opq_losses, min(5, _opq_losses.numel()), largest=True)
                                            stats_lines.append(f"  [OPAQUE n={n_opaq}] mean_loss={o_loss:.6f} max_loss={o_loss_max:.6f} mean_cos={o_cos:.6f} mean_ew={o_rw:.4f}")
                                            stats_lines.append(f"  [OPAQUE top5_losses] {_opq_top5.values.cpu().numpy()}")
                                        logger.info("\n".join(stats_lines))



        loss.backward()

        save_training_vis(viewpoint_cam, gaussians, background, render, pipe, opt, first_iter, iteration, pbr_kwargs=None, is_pbr=False)


        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * image_loss.item() + 0.6 * ema_loss_for_log
            ema_single_view_for_log = 0.4 * normal_loss.item() if normal_loss is not None else 0.0 + 0.6 * ema_single_view_for_log
            ema_multi_view_geo_for_log = 0.4 * geo_loss.item() if geo_loss is not None else 0.0 + 0.6 * ema_multi_view_geo_for_log
            ema_multi_view_pho_for_log = 0.4 * ncc_loss.item() if ncc_loss is not None else 0.0 + 0.6 * ema_multi_view_pho_for_log
            ema_sd_normal_loss_for_log = 0.4 * sd_normal_loss.item() if sd_normal_loss is not None else 0.0 + 0.6 * ema_sd_normal_loss_for_log
            ema_alpha_loss_for_log = 0.4 * alpha_loss.item() if alpha_loss is not None else 0.0 + 0.6 * ema_alpha_loss_for_log
            ema_transparency_loss_for_log = 0.4 * transparency_loss.item() if transparency_loss is not None else 0.0 + 0.6 * ema_transparency_loss_for_log
            ema_mv_normal_loss_for_log = 0.4 * mv_normal_loss.item() if mv_normal_loss is not None else 0.0 + 0.6 * ema_mv_normal_loss_for_log
            if opt.use_reliability and len(reliability_buffer) > 0:
                _cur_R = get_reliability_map(reliability_buffer, viewpoint_cam)
                if _cur_R is not None:
                    ema_reliability_for_log = 0.4 * _cur_R.mean().item() + 0.6 * ema_reliability_for_log
                    _diag.record_rmap(_cur_R)  # [DIAG]
            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "Single": f"{ema_single_view_for_log:.{5}f}",
                    "SD Normal": f"{ema_sd_normal_loss_for_log:.{5}f}",
                    "Geo": f"{ema_multi_view_geo_for_log:.{5}f}",
                    "Pho": f"{ema_multi_view_pho_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}",
                    "Alpha": f"{ema_alpha_loss_for_log:.{5}f}",
                    "Transparency": f"{ema_transparency_loss_for_log:.{5}f}",
                    "MVN": f"{ema_mv_normal_loss_for_log:.{5}f}",
                    "Rel": f"{ema_reliability_for_log:.{4}f}",
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
                # 每10次迭代记录一次详细的损失信息
                logger.info(f"Iteration {iteration}: Loss={ema_loss_for_log:.5f}, Single={ema_single_view_for_log:.5f}, SD Normal={ema_sd_normal_loss_for_log:.5f}, Geo={ema_multi_view_geo_for_log:.5f}, Pho={ema_multi_view_pho_for_log:.5f}, Points={len(gaussians.get_xyz)}, Alpha={ema_alpha_loss_for_log:.5f}, Transparency={ema_transparency_loss_for_log:.5f}, MVN={ema_mv_normal_loss_for_log:.5f}, Rel={ema_reliability_for_log:.4f}")
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), app_model, specular_mlp if iteration > 3000 + opt.delight_iterations else None, dataset.load2gpu_on_the_fly)
            if (iteration in saving_iterations):
                logger.info(f"[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)
                if specular_mlp is not None:
                    specular_mlp.save_weights(scene.model_path, iteration)
                    
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                mask = (render_pkg["out_observe"] > 0) & visibility_filter
                gaussians.max_radii2D[mask] = torch.max(gaussians.max_radii2D[mask], radii[mask])
                viewspace_point_tensor_abs = render_pkg["viewspace_points_abs"]
                # Innovation 3: compute per-Gaussian reliability from pixel R map
                per_gaussian_reliability = None
                if (opt.use_reliability and iteration >= opt.reliability_use_from_iter
                        and not opt.reliability_no_densify_modulate):
                    with torch.no_grad():
                        R_map_densify = get_reliability_map(reliability_buffer, viewpoint_cam)
                        if R_map_densify is not None:
                            xyz_cam = (gaussians.get_xyz.detach()
                                       @ viewpoint_cam.world_view_transform[:3,:3]
                                       + viewpoint_cam.world_view_transform[3,:3])
                            z_safe = xyz_cam[:, 2].clamp(min=0.1)
                            proj_x = xyz_cam[:, 0] * viewpoint_cam.Fx / z_safe + viewpoint_cam.Cx
                            proj_y = xyz_cam[:, 1] * viewpoint_cam.Fy / z_safe + viewpoint_cam.Cy
                            W_r = int(viewpoint_cam.image_width)
                            H_r = int(viewpoint_cam.image_height)
                            gx = 2.0 * proj_x / (W_r - 1) - 1.0
                            gy = 2.0 * proj_y / (H_r - 1) - 1.0
                            grid_pts = torch.stack([gx, gy], dim=-1).reshape(1, 1, -1, 2)
                            per_gaussian_reliability = F.grid_sample(
                                R_map_densify.unsqueeze(0).unsqueeze(0).to(grid_pts.device),
                                grid_pts, mode='bilinear', padding_mode='zeros', align_corners=True
                            ).reshape(-1, 1)
                # [DIAG] record per-Gaussian reliability sampling stats
                if per_gaussian_reliability is not None:
                    pgr = per_gaussian_reliability.detach().squeeze()
                    _diag.record_per_gs_reliability(pgr)
                gaussians.add_densification_stats(viewspace_point_tensor, viewspace_point_tensor_abs,
                                                  visibility_filter,
                                                  per_gaussian_reliability=per_gaussian_reliability)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    use_rel_mod = (opt.use_reliability and not opt.reliability_no_densify_modulate)
                    _densify_diag = gaussians.densify_and_prune(opt.densify_grad_threshold, opt.densify_abs_grad_threshold,  # [DIAG]
                                                opt.opacity_cull_threshold, scene.cameras_extent, size_threshold,
                                                use_reliability_modulate=use_rel_mod,
                                                floater_rel_th=opt.reliability_floater_rel_th,
                                                floater_opacity_th=opt.reliability_floater_opacity_th)
                    _diag.record_innov3_densify(_densify_diag)  # [DIAG]

                _diag.flush(iteration)  # [DIAG] — flush after all diagnostics collected
                    

            
            # multi-view observe trim
            if opt.use_multi_view_trim and iteration % 1000 == 0 and iteration < opt.densify_until_iter:
                observe_the = 2
                observe_cnt = torch.zeros_like(gaussians.get_opacity)
                for view in scene.getTrainCameras():
                    render_pkg_tmp = render(view, gaussians, pipe, bg, app_model=app_model, return_plane=False, return_depth_normal=False)
                    out_observe = render_pkg_tmp["out_observe"]
                    observe_cnt[out_observe > 0] += 1
                prune_mask = (observe_cnt < observe_the).squeeze()
                _mv_trim_n = prune_mask.sum().item()  # [DIAG]
                if _mv_trim_n > 0:
                    gaussians.prune_points(prune_mask)
                _diag.record_mv_trim(_mv_trim_n, gaussians._xyz.shape[0])  # [DIAG]

            # reset_opacity
            if iteration < opt.densify_until_iter:
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            _diag.flush(iteration)  # [DIAG] — guaranteed flush point

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                app_model.optimizer.step()
                if specular_mlp is not None:
                    specular_mlp.optimizer.step()
                # gaussians.update_learning_rate(iteration)
                # app_model.update_learning_rate(iteration)
                if specular_mlp is not None:
                    specular_mlp.update_learning_rate(iteration)
                gaussians.optimizer.zero_grad(set_to_none = True)
                app_model.optimizer.zero_grad(set_to_none = True)
                if specular_mlp is not None:
                    specular_mlp.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                logger.info(f"[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                app_model.save_weights(scene.model_path, iteration)
    
    app_model.save_weights(scene.model_path, opt.iterations)
    torch.cuda.empty_cache()

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
    
    # Set up output folder
    print(f"Output folder: {args.model_path}")  # 使用print而不是logging
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # 设置日志
    log_file = os.path.join(args.model_path, "training_log.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # 记录初始信息
    logger = logging.getLogger()
    logger.info(f"Output folder: {args.model_path}")
    
    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        logger.warning("Tensorboard not available: not logging progress")
    return tb_writer, logger

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, app_model, specular_mlp=None, load2gpu_on_the_fly=False):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    if specular_mlp is not None:
                        if load2gpu_on_the_fly:
                            viewpoint.load2device()
                        dir_pp = (scene.gaussians.get_xyz - viewpoint.camera_center.repeat(
                            scene.gaussians.get_features.shape[0], 1))
                        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                        normal = scene.gaussians.get_normal_axis(dir_pp_normalized=dir_pp_normalized, return_delta=True)
                        mlp_color = specular_mlp.step(scene.gaussians.get_asg_features, dir_pp_normalized, normal)
                    else:
                        mlp_color = None
                    out = renderFunc(viewpoint, scene.gaussians, *renderArgs, app_model=app_model, mlp_color=mlp_color)
                    image = out["render"]
                    if 'app_image' in out:
                        image = out['app_image']
                    image = torch.clamp(image, 0.0, 1.0)
                    gt_image, _, gt_delight, gt_normal, transparencies_map = viewpoint.get_image()
                    gt_image = torch.clamp(gt_image.to("cuda"), 0.0, 1.0)
                    if gt_delight is not None:
                        gt_delight = torch.clamp(gt_delight.to("cuda"), 0.0, 1.0)
                    if gt_normal is not None:
                        gt_normal = torch.clamp(gt_normal.to("cuda"), 0.0, 1.0)
                    if transparencies_map is not None:
                        transparencies_map = torch.clamp(transparencies_map.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                            if gt_delight is not None:
                                tb_writer.add_images(config['name'] + "_view_{}/ground_truth_delight".format(viewpoint.image_name), gt_delight[None], global_step=iteration)
                            if gt_normal is not None:
                                tb_writer.add_images(config['name'] + "_view_{}/ground_truth_normal".format(viewpoint.image_name), gt_normal[None], global_step=iteration)
                            if transparencies_map is not None:
                                tb_writer.add_images(config['name'] + "_view_{}/ground_truth_transparency".format(viewpoint.image_name), transparencies_map[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                    lpips_test += lpips(image, gt_image, net_type='vgg').mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                logging.info(f"[ITER {iteration}] Evaluating {config['name']}: L1 {l1_test} PSNR {psnr_test} SSIM {ssim_test} LPIPS {lpips_test}")
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)
                
                # save all metrics in json file
                results = {
                    "PSNR": psnr_test.item(),
                    "SSIM": ssim_test.item(),
                    "LPIPS": lpips_test.item(),
                }
                result_path = os.path.join(scene.model_path, 'results', config['name'])
                os.makedirs(result_path, exist_ok=True)
                with open(os.path.join(result_path, f"result_{iteration}.json"), 'w') as fp:
                    json.dump(results, fp, indent=True)
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

@torch.no_grad()
def save_training_vis(viewpoint_cam, gaussians, background, render_fn, pipe, opt, first_iter, iteration, pbr_kwargs=None, is_pbr=False):
    os.makedirs(os.path.join(args.model_path, "visualize"), exist_ok=True)
    if iteration % 5000 == 0 or iteration == first_iter + 1:
        render_pkg = render_fn(viewpoint_cam, gaussians, pipe, background,
                            )
        visualization_list = [
            render_pkg["render"],
            viewpoint_cam.original_image.cuda(),
            render_pkg["rendered_distance"].repeat(3, 1, 1),
            render_pkg["plane_depth"].repeat(3, 1, 1),
            render_pkg["rendered_normal"] * 0.5 + 0.5,
            render_pkg["depth_normal"] * 0.5 + 0.5,
            render_pkg["rendered_alpha"].repeat(3, 1, 1),
            # render_pkg["out_observe"],
            render_pkg["out_transparency_map"].repeat(3, 1, 1)
        ]
        if viewpoint_cam.mask is not None:
            visualization_list.append(viewpoint_cam.mask.cuda().repeat(3, 1, 1))
        
        if viewpoint_cam.normal is not None:
            visualization_list.append(-viewpoint_cam.normal.cuda() * 0.5 + 0.5)
            normal_threshold = 0.3
            cosine_similarity = (render_pkg["rendered_normal"] * (-viewpoint_cam.normal)).sum(dim=0)
            valid_normal_mask = (cosine_similarity > normal_threshold)
            visualization_list.append(valid_normal_mask.cuda().repeat(3, 1, 1))
        if viewpoint_cam.delight is not None:
            visualization_list.append(viewpoint_cam.delight.cuda())
        
        if "out_nearest_depth" in render_pkg:
            cpu_out_nearest_depth = render_pkg["out_nearest_depth"].squeeze().detach().cpu().numpy()
            mask = cpu_out_nearest_depth >= 1e2
            valid_depths = cpu_out_nearest_depth[~mask]
            if valid_depths.size > 0:
                median_depth = np.percentile(valid_depths, 50)
            else:
                median_depth = 0
            cpu_out_nearest_depth[mask] = median_depth
            
            normalized_out_nearest_depth = (cpu_out_nearest_depth - cpu_out_nearest_depth.min()) / (cpu_out_nearest_depth.max() - cpu_out_nearest_depth.min())
            colored_depth = cv2.applyColorMap((normalized_out_nearest_depth * 255).astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(args.model_path, "visualize", f"{iteration:06d}_out_nearest_depth.png"), colored_depth)
            cv2.imwrite(os.path.join(args.model_path, "visualize", f"{iteration:06d}_out_nearest_depth_mask.png"), mask.astype(np.uint8) * 255)

        grid = torch.stack(visualization_list, dim=0)
        grid = make_grid(grid, nrow=4)
        scale = grid.shape[-2] / 800
        grid = F.interpolate(grid[None], (int(grid.shape[-2]/scale), int(grid.shape[-1]/scale)))[0]
        save_image(grid, os.path.join(args.model_path, "visualize", f"{iteration:06d}.png"))


if __name__ == "__main__":
    torch.set_num_threads(8)
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6007)
    parser.add_argument('--debug_from', type=int, default=-100)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(sys.argv[1:])
    setup_seed(args.seed)
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
