import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
import json

# GLOBAL_POSE: if true, optimize global rotation, otherwise, only optimize head rotation (shoulder stays un-rotated)
# if GLOBAL_POSE is set to false, global translation is used.
GLOBAL_POSE = True
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg
from decalib.utils import lossfunc

import cv2
import argparse

np.random.seed(0)


def projection(points, K, w2c, no_intrinsics=False):
    rot = w2c[:, np.newaxis, :3, :3]
    points_cam = torch.sum(points[..., np.newaxis, :] * rot, -1) + w2c[:, np.newaxis, :3, 3]
    if no_intrinsics:
        return points_cam

    points_cam_projected = points_cam
    points_cam_projected[..., :2] /= points_cam[..., [2]]
    points_cam[..., [2]] *= -1

    i = points_cam_projected[..., 0] * K[0] + K[2]
    j = points_cam_projected[..., 1] * K[1] + K[3]
    points2d = torch.stack([i, j, points_cam_projected[..., -1]], dim=-1)
    return points2d


def inverse_projection(points2d, K, c2w):
    i = points2d[:, :, 0]
    j = points2d[:, :, 1]
    dirs = torch.stack([(i - K[2]) / K[0], (j - K[3]) / K[1], torch.ones_like(i) * -1], -1)
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:, np.newaxis, :3, :3], -1)
    rays_d = F.normalize(rays_d, dim=-1)
    rays_o = c2w[:, np.newaxis, :3, -1].expand(rays_d.shape)

    return rays_o, rays_d


class Optimizer(object):
    def __init__(self, device='cuda:0'):
        deca_cfg.model.use_tex = False
        # TODO: landmark_embedding.npy with eyes to optimize iris parameters
        deca_cfg.model.flame_lmk_embedding_path = os.path.join(deca_cfg.deca_dir, 'data',
                                                               'landmark_embedding_with_eyes.npy')
        deca_cfg.rasterizer_type = 'pytorch3d' # or 'standard'
        self.deca = DECA(config=deca_cfg, device=device)

    def optimize(self, shape, exp, landmark_all, pose, name, visualize_images_all, savefolders, intrinsics_all, json_path, size,
                 save_name):
        num_img = pose.shape[0]
        n_view = len(visualize_images_all)

        # we need to project to [-1, 1] instead of [0, size], hence modifying the cam_intrinsics as below
        cam_intrinsics_all = []
        for intrinsics in intrinsics_all:
            cam_intirinsics = torch.tensor(
                [-1 * intrinsics[0] / size * 2, intrinsics[1] / size * 2, intrinsics[2] / size * 2 - 1,
                 intrinsics[3] / size * 2 - 1]).float().cuda()
            cam_intrinsics_all.append(cam_intirinsics)
            
        # cam_intrinsics = torch.tensor(
        #     [-1 * intrinsics[0] / size * 2, intrinsics[1] / size * 2, intrinsics[2] / size * 2 - 1,
        #      intrinsics[3] / size * 2 - 1]).float().cuda()

        translation_p_all = []
        if GLOBAL_POSE:
            # translation_p_0 = torch.tensor([0, 0, -4]).float().cuda()
            # translation_p_1 = torch.tensor([0, 0, -4]).float().cuda()
            # translation_p_2 = torch.tensor([0, 0, -4]).float().cuda()
            # translation_p_3 = torch.tensor([0, 0, -4]).float().cuda()
            for i in range(n_view):
                translation_p_all.append(torch.tensor([0, 0, -4]).float().cuda())
        else:
            # translation_p_0 = torch.tensor([0, 0, -4]).unsqueeze(0).expand(num_img, -1).float().cuda()
            # translation_p_1 = torch.tensor([0, 0, -4]).unsqueeze(0).expand(num_img, -1).float().cuda()
            # translation_p_2 = torch.tensor([0, 0, -4]).unsqueeze(0).expand(num_img, -1).float().cuda()
            # translation_p_3 = torch.tensor([0, 0, -4]).unsqueeze(0).expand(num_img, -1).float().cuda()
            for i in range(n_view):
                translation_p_all.append(torch.tensor([0, 0, -4]).unsqueeze(0).expand(num_img, -1).float().cuda())
        # convert translation_p_all from list to tensor
        translation_p_all = torch.stack(translation_p_all, dim=0)
        
        if GLOBAL_POSE:
            pose = torch.cat([torch.zeros_like(pose[:, :3]), pose], dim=1)
        
        use_iris = False
        if landmark_all[0].shape[1] == 70:
            # use iris landmarks, optimize gaze direction
            use_iris = True

        if use_iris:
            pose = torch.cat([pose, torch.zeros_like(pose[:, :6])], dim=1)

        pose_all = []
        for i in range(n_view):
            pose_all.append(pose)
        pose_all = torch.stack(pose_all, dim=0)
        
        # translation_p = nn.Parameter(translation_p)
        translation_p_all = nn.Parameter(translation_p_all)
        # pose = nn.Parameter(pose)
        pose_all = nn.Parameter(pose_all)

        exp = nn.Parameter(exp)
        shape = nn.Parameter(shape)

        # set optimizer
        if json_path is None:
            opt_p = torch.optim.Adam(
                [translation_p_all, pose_all, exp, shape],
                lr=1e-2)
        else:
            opt_p = torch.optim.Adam(
                [translation_p_all, pose_all, exp],
                lr=1e-2)

        # optimization steps
        len_landmark = landmark_all[0].shape[1]
        for k in range(1001):
            full_pose_all = []
            verts_p_all = []
            landmarks2d_p_all = []
            for i in range(n_view):
                full_pose = pose_all[i]
                if not use_iris:
                    full_pose = torch.cat([full_pose, torch.zeros_like(full_pose[..., :6])], dim=1)
                if not GLOBAL_POSE:
                    full_pose = torch.cat([torch.zeros_like(full_pose[:, :3]), full_pose], dim=1)
                verts_p, landmarks2d_p, landmarks3d_p = self.deca.flame(shape_params=shape.expand(num_img, -1),
                                                                        expression_params=exp,
                                                                        full_pose=full_pose)
                full_pose_all.append(full_pose)
                verts_p_all.append(verts_p)
                landmarks2d_p_all.append(landmarks2d_p)
            # CAREFUL: FLAME head is scaled by 4 to fit unit sphere tightly
            # verts_p *= 4
            # landmarks3d_p *= 4
            # landmarks2d_p *= 4
            for i in range(n_view):
                verts_p_all[i] *= 4
                landmarks2d_p_all[i] *= 4

            # perspective projection
            # Global rotation is handled in FLAME, set camera rotation matrix to identity
            ident = torch.eye(3).float().cuda().unsqueeze(0).expand(num_img, -1, -1)
            w2c_p_all = []
            if GLOBAL_POSE:
                # w2c_p = torch.cat([ident, translation_p.unsqueeze(0).expand(num_img, -1).unsqueeze(2)], dim=2)
                for translation_p in translation_p_all:
                    w2c_p_all.append(torch.cat([ident, translation_p.unsqueeze(0).expand(num_img, -1).unsqueeze(2)], dim=2))
            else:
                # w2c_p = torch.cat([ident, translation_p.unsqueeze(2)], dim=2)
                for translation_p in translation_p_all:
                    w2c_p_all.append(torch.cat([ident, translation_p.unsqueeze(2)], dim=2))

            # trans_landmarks2d = projection(landmarks2d_p, cam_intrinsics, w2c_p)
            trans_landmarks2d_all = []
            for i in range(n_view):
                trans_landmarks2d_all.append(projection(landmarks2d_p_all[i], cam_intrinsics_all[i], w2c_p_all[i]))
            ## landmark loss
            landmark_loss2_all = []
            for i in range(n_view):
                landmark_loss2 = lossfunc.l2_distance(trans_landmarks2d_all[i][:, :len_landmark, :2], landmark_all[i][:, :len_landmark])
                landmark_loss2_all.append(landmark_loss2)
            # total_loss = landmark_loss2 + torch.mean(torch.square(shape)) * 1e-2 + torch.mean(torch.square(exp)) * 1e-2
            total_loss = sum(landmark_loss2_all) / n_view + torch.mean(torch.square(shape)) * 1e-2 + torch.mean(torch.square(exp)) * 1e-2 
            total_loss += torch.mean(torch.square(exp[1:] - exp[:-1])) * 1e-1
            for i in range(n_view):
                total_loss += torch.mean(torch.square(pose_all[i][1:] - pose_all[i][:-1])) * 10 / n_view
            # if not GLOBAL_POSE:
            #     total_loss += torch.mean(torch.square(translation_p[1:] - translation_p[:-1])) * 10

            opt_p.zero_grad()
            total_loss.backward()
            opt_p.step()

            # visualize
            if k % 100 == 0:
                # print(translation_p_all)
                with torch.no_grad():
                    for i in range (n_view):
                        landmark = landmark_all[i]
                        landmark_loss2 = landmark_loss2_all[i]
                        trans_landmarks2d = trans_landmarks2d_all[i]
                        cam_intrinsics = cam_intrinsics_all[i]
                        intrinsics = intrinsics_all[i]
                        w2c_p = w2c_p_all[i]
                        full_pose = full_pose_all[i]
                        visualize_images = visualize_images_all[i]
                        savefolder = savefolders[i]
                        
                        verts_p = verts_p_all[i]
                    
                        loss_info = '----iter: {}, time: {}\n'.format(k,
                                                                    datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
                        loss_info = loss_info + f'landmark_loss: {landmark_loss2}'
                        print(loss_info)
                        trans_verts = projection(verts_p[::50], cam_intrinsics, w2c_p[::50])
                        # trans_landmarks2d_for_visual = projection(landmarks2d_p, cam_intrinsics, w2c_p)
                        shape_images = self.deca.render.render_shape(verts_p[::50], trans_verts)
                        visdict = {
                            'inputs': visualize_images,
                            'gt_landmarks2d': util.tensor_vis_landmarks(visualize_images, landmark[::50]),
                            'landmarks2d': util.tensor_vis_landmarks(visualize_images, trans_landmarks2d.detach()[::50]),
                            'shape_images': shape_images
                        }
                        cv2.imwrite(os.path.join(savefolder, 'optimize_vis.jpg'), self.deca.visualize(visdict))

                        # shape_images = self.deca.render.render_shape(verts_p, trans_verts)
                        # print(shape_images.shape)

                        save = True
                        if save:
                            save_intrinsics = [-1 * intrinsics[0] / size, intrinsics[1] / size, intrinsics[2] / size,
                                            intrinsics[3] / size]
                            dict = {}
                            frames = []
                            for i in range(num_img):
                                frames.append({'file_path': './image/' + name[i],
                                            'world_mat': w2c_p[i].detach().cpu().numpy().tolist(),
                                            'expression': exp[i].detach().cpu().numpy().tolist(),
                                            'pose': full_pose[i].detach().cpu().numpy().tolist(),
                                            'bbox': torch.stack(
                                                [torch.min(landmark[i, :, 0]), torch.min(landmark[i, :, 1]),
                                                    torch.max(landmark[i, :, 0]), torch.max(landmark[i, :, 1])],
                                                dim=0).detach().cpu().numpy().tolist(),
                                            'flame_keypoints': trans_landmarks2d[i, :,
                                                                :2].detach().cpu().numpy().tolist()
                                            })

                            dict['frames'] = frames
                            dict['intrinsics'] = save_intrinsics
                            dict['shape_params'] = shape[0].cpu().numpy().tolist()
                            with open(os.path.join(savefolder, save_name + '.json'), 'w') as fp:
                                json.dump(dict, fp)

    def run(self, deca_code_files, face_kpts_files, iris_files, savefolders, image_paths, json_path, intrinsics_all, size,
            save_name):
        n_view = len(deca_code_files)

        deca_code = json.load(open(deca_code_files[0], 'r'))
        face_kpts_all = []
        iris_kpts_all = []
        for face_kpts_file in face_kpts_files:
            face_kpts_all.append(json.load(open(face_kpts_file, 'r')))
        try:
            for iris_file in iris_files:
                iris_kpts_all.append(json.load(open(iris_file, 'r')))
        except:
            # iris_kpts = None
            iris_kpts_all = None
            print("Not using Iris keypoint")
        visualize_images = []
        shape = []
        exps = []
        landmarks = []
        poses = []
        name = []
        num_img = len(deca_code)
        # init list: shape=(n_view, num_img)
        landmarks_all = [[] for _ in range(n_view)]
        visualize_images_all = [[] for _ in range(n_view)]
        # ffmpeg extracted frames, index starts with 1
        for k in range(1, num_img + 1):
            shape.append(torch.tensor(deca_code[str(k)]['shape']).float().cuda())
            exps.append(torch.tensor(deca_code[str(k)]['exp']).float().cuda())
            poses.append(torch.tensor(deca_code[str(k)]['pose']).float().cuda())
            name.append(str(k))
            landmark_all = []
            for face_kpt in face_kpts_all:
                landmark_all.append(np.array(face_kpt['{}.png'.format(str(k))]).astype(np.float32))
            # landmark = np.array(face_kpts['{}.png'.format(str(k))]).astype(np.float32)
            if iris_kpts_all[0] is not None:
                iris_all = []
                for index, iris_kpts in enumerate(iris_kpts_all):
                    iris_all.append(np.array(iris_kpts['{}.png'.format(str(k))]).astype(np.float32).reshape(2, 2))
                    # iris = np.array(iris_kpts['{}.png'.format(str(k))]).astype(np.float32).reshape(2, 2)
                    landmark_all[index] = np.concatenate([landmark_all[index], iris_all[index][[1,0], :]], 0)
            # landmark = landmark / size * 2 - 1
            # landmarks.append(torch.tensor(landmark).float().cuda())
            for index, landmark in enumerate(landmark_all):
                landmark = landmark / size * 2 - 1
                landmarks_all[index].append(torch.tensor(landmark).float().cuda())

            if k % 50 == 1:
                for index, image_path in enumerate(image_paths):
                    image = cv2.imread(image_path + '/{}.png'.format(str(k))).astype(np.float32) / 255.
                    image = image[:, :, [2, 1, 0]].transpose(2, 0, 1)
                    visualize_images_all[index].append(torch.from_numpy(image[None, :, :, :]).cuda())

        shape = torch.cat(shape, dim=0)
        if json_path is None:
            shape = torch.mean(shape, dim=0).unsqueeze(0)
        else:
            shape = torch.tensor(json.load(open(json_path, 'r'))['shape_params']).float().cuda().unsqueeze(0)
        exps = torch.cat(exps, dim=0)
        # landmarks = torch.stack(landmarks, dim=0)
        for index, landmarks in enumerate(landmarks_all):
            landmarks_all[index] = torch.stack(landmarks, dim=0)

        poses = torch.cat(poses, dim=0)
        
        for index, visualize_images in enumerate(visualize_images_all):
            visualize_images_all[index] = torch.cat(visualize_images, dim=0)
        # optimize
        self.optimize(shape, exps, landmarks_all, poses, name, visualize_images_all, savefolders, intrinsics_all, json_path, size,
                      save_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--path', type=str, help='Path to images and deca and landmark jsons')
    parser.add_argument('--shape_from', type=str, default='.', help="Use shape parameter from this video if given.")
    parser.add_argument('--save_name', type=str, default='flame_params', help='Name for json')
    parser.add_argument('--fx', type=float, default=1500)
    parser.add_argument('--fy', type=float, default=1500)
    parser.add_argument('--cx', type=float, default=256)
    parser.add_argument('--cy', type=float, default=256)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--paths', type=str)
    parser.add_argument('--camera_idx', type=str, default='0')
    
    args = parser.parse_args()
    model = Optimizer()

    paths = args.paths.split(',')
    camera_idx = args.camera_idx.split(',')
    # image_path = os.path.join(args.path, 'image')
    image_paths = []
    for path in paths:
        image_paths.append(os.path.join(path, 'image'))
        
    if args.shape_from == '.':
        args.shape_from = None
        json_path = None
    else:
        json_path = os.path.join(args.shape_from, args.save_name + '.json')
    
    for path in paths:
        print("Optimizing: {}".format(path))
    
    json_file = "/bufferhdd/zhanglibo/project/IMavatar/data/camera_parameters.json"
    with open(json_file) as f:
        camera_parameters = json.load(f)
    
    intrinsics_all = []
    for index in camera_idx:
        K = camera_parameters[index]['K']
        intrinsics_all.append([K[0][0], K[1][1], K[0][2] + 80, K[1][2]])
        # intrinsics_all.append([args.fx, args.fy, args.cx, args.cy])
        
    # change
    # intrinsics = [args.fx, args.fy, args.cx, args.cy]
    deca_code_files = []
    face_kpts_files = []
    iris_files = []
    for path in paths:
        deca_code_files.append(os.path.join(path, 'code.json'))
        face_kpts_files.append(os.path.join(path, 'keypoint.json'))
        iris_files.append(os.path.join(path, 'iris.json'))
    model.run(deca_code_files=deca_code_files,
              face_kpts_files=face_kpts_files,
              iris_files=iris_files, savefolders=paths, image_paths=image_paths,
              json_path=json_path, intrinsics_all=intrinsics_all, size=args.size, save_name=args.save_name)

