import json
import os,sys
import os.path as osp
import numpy as np
from plyfile import PlyData
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from kaolin.ops.mesh import check_sign
from kaolin.metrics.trianglemesh import point_to_mesh_distance
import trimesh

def save_obj_mesh(mesh_path, verts):
    file = open(mesh_path, 'w') # 打开mesh路径

    for v in verts: # 记录verts
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
    # for f in faces: # 记录faces
    #     f_plus = f + 1
    #     file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close() #关闭
def barycentric_coordinates_of_projection(points, vertices):
    ''' https://github.com/MPI-IS/mesh/blob/master/mesh/geometry/barycentric_coordinates_of_projection.py
    '''
    """Given a point, gives projected coords of that point to a triangle
    in barycentric coordinates.
    See
        **Heidrich**, Computing the Barycentric Coordinates of a Projected Point, JGT 05
        at http://www.cs.ubc.ca/~heidrich/Papers/JGT.05.pdf
    
    :param p: point to project. [B, 3]
    :param v0: first vertex of triangles. [B, 3]
    :returns: barycentric coordinates of ``p``'s projection in triangle defined by ``q``, ``u``, ``v``
            vectorized so ``p``, ``q``, ``u``, ``v`` can all be ``3xN``
    """
    #(p, q, u, v)
    v0, v1, v2 = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    p = points

    q = v0
    u = v1 - v0
    v = v2 - v0
    n = torch.cross(u, v)
    s = torch.sum(n * n, dim=1)
    # If the triangle edges are collinear, cross-product is zero,
    # which makes "s" 0, which gives us divide by zero. So we
    # make the arbitrary choice to set s to epsv (=numpy.spacing(1)),
    # the closest thing to zero
    s[s == 0] = 1e-6
    oneOver4ASquared = 1.0 / s
    w = p - q
    b2 = torch.sum(torch.cross(u, w) * n, dim=1) * oneOver4ASquared
    b1 = torch.sum(torch.cross(w, v) * n, dim=1) * oneOver4ASquared
    weights = torch.stack((1 - b1 - b2, b1, b2), dim=-1)
    # check barycenric weights
    # p_n = v0*weights[:,0:1] + v1*weights[:,1:2] + v2*weights[:,2:3]
    return weights
    
def build_triangles(vertices, faces):

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device=vertices.device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, vertices.shape[-1]))

    return vertices[faces.long()]

def cal_lbs(verts, faces, points, T, T_star):
    # functions modified from ICON
    
    # verts [B, N_vert, 3]
    # faces [B, N_face, 3]
    # triangles [B, N_face, 3, 3]
    # points [B, N_point, 3]
    
    Bsize = points.shape[0]
    
    triangles = build_triangles(verts, faces)
    T_ = build_triangles(T, faces)
    T_star_ = build_triangles(T_star, faces)
    residues, pts_ind, _ = point_to_mesh_distance(points.float(), triangles)
    
    closest_triangles = torch.gather(
        triangles, 1, pts_ind[:, :, None, None].expand(-1, -1, 3, 3)).view(-1, 3, 3)
    closest_T = torch.gather(
        T_, 1, pts_ind[:, :, None, None].expand(-1, -1, 3, 16)).view(-1, 3, 16)
    closest_Tstar = torch.gather(
        T_star_, 1, pts_ind[:, :, None, None].expand(-1, -1, 3, 16)).view(-1, 3, 16)
    bary_weights = barycentric_coordinates_of_projection(
        points.view(-1, 3), closest_triangles)
    
    pts_T = (closest_T*bary_weights[:,
               :, None]).sum(1).unsqueeze(0)
    pts_Tstar = (closest_Tstar*bary_weights[:,
               :, None]).sum(1).unsqueeze(0)

    return pts_T.view(Bsize, -1, 16), pts_Tstar.view(Bsize, -1, 16)
    
def lbs(betas, pose, v_template, shapedirs, posedirs, J_regressor, parents,
        lbs_weights, transl, pose2rot=True, dtype=torch.float32):
    ''' Performs Linear Blend Skinning with the given shape and pose parameters

        Parameters
        ----------
        betas : torch.tensor BxNB
            The tensor of shape parameters
        pose : torch.tensor Bx(J + 1) * 3
            The pose parameters in axis-angle format
        v_template torch.tensor BxVx3
            The template mesh that will be deformed
        shapedirs : torch.tensor 1xNB
            The tensor of PCA shape displacements
        posedirs : torch.tensor Px(V * 3)
            The pose PCA coefficients
        J_regressor : torch.tensor JxV
            The regressor array that is used to calculate the joints from
            the position of the vertices
        parents: torch.tensor J
            The array that describes the kinematic tree for the model
        lbs_weights: torch.tensor N x V x (J + 1)
            The linear blend skinning weights that represent how much the
            rotation matrix of each part affects each vertex
        pose2rot: bool, optional
            Flag on whether to convert the input pose tensor to rotation
            matrices. The default value is True. If False, then the pose tensor
            should already contain rotation matrices and have a size of
            Bx(J + 1)x9
        dtype: torch.dtype, optional

        Returns
        -------
        verts: torch.tensor BxVx3
            The vertices of the mesh after applying the shape and pose
            displacements.
        joints: torch.tensor BxJx3
            The joints of the model
    '''
    
    batch_size = betas.shape[0]
    
    v_shaped = v_template + blend_shapes(betas, shapedirs)
    J = vertices2joints(J_regressor, v_shaped)
    
    # build star posed smpl vertices
    v_posed = v_shaped
    ident_posed = torch.eye(3, dtype = pose.dtype, device = J_regressor.device)
    hand_pose = ident_posed.view(1,1,3,3).repeat(batch_size,2,1,1)
    if pose2rot:
        rot_mats_posed = batch_rodrigues(pose[:,  :-6].contiguous().view(-1, 3), dtype=pose.dtype).view([batch_size, -1, 3, 3]).type(pose.dtype)
        rot_mats_posed = torch.cat([rot_mats_posed, hand_pose], 1).contiguous()
        pose_feature_posed = (rot_mats_posed[:, 1:, :, :] - ident_posed).view([batch_size, -1]).type(pose.dtype)
        pose_offsets_posed = torch.matmul(pose_feature_posed, posedirs.type(pose.dtype)).view(batch_size, -1, 3)
    else:
        pose_feature_posed = pose[:, 1:].view(batch_size, -1, 3, 3).type(pose.dtype) - ident_posed
        rot_mats_posed = pose.view(batch_size, -1, 3, 3).type(pose.dtype)
        pose_offsets_posed = torch.matmul(pose_feature_posed.view(batch_size, -1), posedirs).view(batch_size, -1, 3)

    J_transformed_posed, A_posed = batch_rigid_transform(rot_mats_posed, J, parents, dtype=pose.dtype)
    
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    num_joints = J_regressor.shape[0]
        
        
        
    T_posed = torch.matmul(W, A_posed.view(batch_size, num_joints, 16)).view(batch_size, -1, 4, 4)
    homogen_coord_posed = torch.ones([batch_size, v_posed.shape[1], 1], dtype = pose.dtype, device = J_regressor.device)
    v_posed_homo_posed = torch.cat([v_posed, homogen_coord_posed], dim=2)
    v_homo_posed = torch.matmul(T_posed, torch.unsqueeze(v_posed_homo_posed, dim=-1))
    verts_posed = v_homo_posed[:, :, :3, 0]
    
        
    return verts_posed, T_posed, A_posed


def deform_lbs(betas, T_posed, v_template, shapedirs, posedirs, J_regressor, parents,
        lbs_weights, transl, pose2rot=True, dtype=torch.float32):
    ''' Performs Linear Blend Skinning with the given shape and pose parameters

        Parameters
        ----------
        betas : torch.tensor BxNB
            The tensor of shape parameters
        pose : torch.tensor Bx(J + 1) * 3
            The pose parameters in axis-angle format
        v_template torch.tensor BxVx3
            The template mesh that will be deformed
        shapedirs : torch.tensor 1xNB
            The tensor of PCA shape displacements
        posedirs : torch.tensor Px(V * 3)
            The pose PCA coefficients
        J_regressor : torch.tensor JxV
            The regressor array that is used to calculate the joints from
            the position of the vertices
        parents: torch.tensor J
            The array that describes the kinematic tree for the model
        lbs_weights: torch.tensor N x V x (J + 1)
            The linear blend skinning weights that represent how much the
            rotation matrix of each part affects each vertex
        pose2rot: bool, optional
            Flag on whether to convert the input pose tensor to rotation
            matrices. The default value is True. If False, then the pose tensor
            should already contain rotation matrices and have a size of
            Bx(J + 1)x9
        dtype: torch.dtype, optional

        Returns
        -------
        verts: torch.tensor BxVx3
            The vertices of the mesh after applying the shape and pose
            displacements.
        joints: torch.tensor BxJx3
            The joints of the model
    '''
    

    batch_size = betas.shape[0]
    
    v_shaped = v_template + blend_shapes(betas, shapedirs)
    J = vertices2joints(J_regressor, v_shaped)
    v_posed = v_shaped
        
    homogen_coord_posed = torch.ones([batch_size, v_posed.shape[1], 1], dtype = v_posed.dtype, device = v_posed.device)
    v_posed_homo_posed = torch.cat([v_posed, homogen_coord_posed], dim=2)
    v_homo_posed = torch.matmul(T_posed, torch.unsqueeze(v_posed_homo_posed, dim=-1))
    verts_posed = v_homo_posed[:, :, :3, 0]
        
    return verts_posed
    

def inv(v_posed, v_tpose, faces, verts, T, T_star):

    batch_size = T.shape[0]
    
    v_ply = verts.unsqueeze(0).to(v_posed.device)
    T_points, T_star_points = cal_lbs(v_posed, faces.unsqueeze(0), v_ply.contiguous(), T.view(T.shape[0], T.shape[1], 16), T_star.view(T_star.shape[0], T_star.shape[1], 16))
    T_points = T_points.view(T.shape[0], -1, 4, 4)
    T_star_points = T_star_points.view(T.shape[0], -1, 4, 4)
    
    T_inv = torch.Tensor(np.linalg.inv(T_points.detach().cpu().numpy())).to(v_posed.device)

    homogen_coord_invT = torch.ones([batch_size, v_ply.shape[1], 1], dtype=torch.float32, device = v_posed.device)
    v_posed_homo_invT = torch.cat([v_ply, homogen_coord_invT], dim=2)


    v_homo_invT = torch.matmul(T_inv, torch.unsqueeze(v_posed_homo_invT, dim=-1))
    verts_invT = v_homo_invT[:, :, :3, 0]
    
    homogen_coord_star = torch.ones([batch_size, verts_invT.shape[1], 1], dtype=torch.float32, device = v_posed.device)
    v_posed_homo_star = torch.cat([verts_invT, homogen_coord_star], dim=2)
    v_homo_star = torch.matmul(T_star_points, torch.unsqueeze(v_posed_homo_star, dim=-1))
    verts_star = v_homo_star[:, :, :3, 0]
    
    return verts_star.squeeze(0)
    
def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat

def blend_shapes(betas, shape_disps):
    ''' Calculates the per vertex displacement due to the blend shapes


    Parameters
    ----------
    betas : torch.tensor Bx(num_betas)
        Blend shape coefficients
    shape_disps: torch.tensor Vx3x(num_betas)
        Blend shapes

    Returns
    -------
    torch.tensor BxVx3
        The per-vertex displacement due to shape deformation
    '''

    # Displacement[b, m, k] = sum_{l} betas[b, l] * shape_disps[m, k, l]
    # i.e. Multiply each shape displacement by its corresponding beta and
    # then sum them.
    blend_shape = torch.einsum('bl,mkl->bmk', [betas, shape_disps])
    return blend_shape
    
def vertices2joints(J_regressor, vertices):
    ''' Calculates the 3D joint locations from the vertices

    Parameters
    ----------
    J_regressor : torch.tensor JxV
        The regressor array that is used to calculate the joints from the
        position of the vertices
    vertices : torch.tensor BxVx3
        The tensor of mesh vertices

    Returns
    -------
    torch.tensor BxJx3
        The location of the joints
    '''

    return torch.einsum('bik,ji->bjk', [vertices, J_regressor])

def batch_rigid_transform(rot_mats, joints, parents, dtype=torch.float32):
    """
    Applies a batch of rigid transformations to the joints

    Parameters
    ----------
    rot_mats : torch.tensor BxNx3x3
        Tensor of rotation matrices
    joints : torch.tensor BxNx3
        Locations of joints
    parents : torch.tensor BxN
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32

    Returns
    -------
    posed_joints : torch.tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """

    joints = torch.unsqueeze(joints, dim=-1)

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    #print(rot_mats.shape, rel_joints.shape,)
    transforms_mat = transform_mat(
        rot_mats.contiguous().view(-1, 3, 3),
        rel_joints.contiguous().view(-1, 3, 1)).contiguous().view(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(transform_chain[parents[i]],
                                transforms_mat[:, i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = F.pad(joints, [0, 0, 0, 1])

    rel_transforms = transforms - F.pad(
        torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

    return posed_joints, rel_transforms
    
def transform_mat(R, t):
    ''' Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    '''
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]),
                      F.pad(t, [0, 0, 0, 1], value=1)], dim=2)

def make_rotate(rx, ry, rz):
    sinX = np.sin(rx)
    sinY = np.sin(ry)
    sinZ = np.sin(rz)

    cosX = np.cos(rx)
    cosY = np.cos(ry)
    cosZ = np.cos(rz)

    Rx = np.zeros((3,3))
    Rx[0, 0] = 1.0
    Rx[1, 1] = cosX
    Rx[1, 2] = -sinX
    Rx[2, 1] = sinX
    Rx[2, 2] = cosX

    Ry = np.zeros((3,3))
    Ry[0, 0] = cosY
    Ry[0, 2] = sinY
    Ry[1, 1] = 1.0
    Ry[2, 0] = -sinY
    Ry[2, 2] = cosY

    Rz = np.zeros((3,3))
    Rz[0, 0] = cosZ
    Rz[0, 1] = -sinZ
    Rz[1, 0] = sinZ
    Rz[1, 1] = cosZ
    Rz[2, 2] = 1.0

    R = np.matmul(np.matmul(Rz,Ry),Rx)
    return R

#cuda = torch.device('cuda:%d' % 0)
#
#pose_path = '/userhome/cs/yukang/multiview_smpl_fitting/EasyMocap/THuman2.0_mv_2/0005_mv_8_1_png/output/smpl/smpl/000000.json'
#with open(pose_path) as json_file:
#    pose_multi = json.load(json_file)
##R = np.matmul(make_rotate(math.radians(0), 0, 0), make_rotate(0, math.radians(100), 0))
##print(R.shape)
#pose_multi_param = torch.tensor(pose_multi[0]['poses']).to(device=cuda)
#poses = pose_multi_param
##print(pose_multi_param, 'pose_multi')
#transl_multi = torch.tensor(pose_multi[0]['Th']).to(device=cuda)
#transl = transl_multi
#
#global_orient_multi = torch.tensor(pose_multi[0]['Rh']).to(device=cuda)
#global_orient = global_orient_multi
##global_orient = torch.tensor(np.dot(global_orient, R)).to(device=cuda)
##print(global_orient.shape)
##exit(1)
#poses[0, :3] = global_orient[0, :]
#
#betas_multi = torch.tensor(pose_multi[0]['shapes']).to(device=cuda)
#betas = betas_multi
#
#        # print(poses.shape, 'poses')
#        # print(transl.shape, 'transl')
#        # print(global_orient.shape, 'global_orient')
#        # print(betas.shape, 'betas')
#
#        # print(self.v_template, self.v_template.shape, 'v_template')
#        # print(self.shapedirs, self.shapedirs.shape, 'shapedirs')
#        # print(self.posedirs, self.posedirs.shape, 'posedirs')
#        # print(self.J_regressor, self.J_regressor.shape, 'J_regressor')
#        # print(self.parents, self.parents.shape, 'parents')
#        # print(self.lbs_weights, self.lbs_weights.shape, 'lbs_weights')
#        # smpl_data = {}
#        # smpl_data['v_template'] = self.v_template.detach().cpu().numpy()
#        # smpl_data['shapedirs'] = self.shapedirs.cpu().numpy()
#        # smpl_data['posedirs'] = self.posedirs.cpu().numpy()
#        # smpl_data['J_regressor'] = self.J_regressor.cpu().numpy()
#        # smpl_data['parents'] = self.parents.cpu().numpy()
#        # smpl_data['lbs_weights'] = self.lbs_weights.cpu().numpy()
#        # smpl_data['dtype'] = str(self.dtype)
#        # smpl_lbs_data = np.array([0])
#        # smpl_lbs_data[0] = smpl_data
#file_path = './smpl_lbs_data.npy'
#        # np.save(file_path, smpl_data)
#
#smpl_lbs_data = np.load(file_path, allow_pickle=True).item()
##print(smpl_lbs_data['v_template'])
#v_template = torch.tensor(smpl_lbs_data['v_template']).to(device=cuda)
#shapedirs = torch.tensor(smpl_lbs_data['shapedirs']).to(device=cuda)
#posedirs = torch.tensor(smpl_lbs_data['posedirs']).to(device=cuda)
#J_regressor = torch.tensor(smpl_lbs_data['J_regressor']).to(device=cuda)
#parents = torch.tensor(smpl_lbs_data['parents']).to(device=cuda)
#lbs_weights = torch.tensor(smpl_lbs_data['lbs_weights']).to(device=cuda)
#dtype = smpl_lbs_data['dtype']
#vertices, joints = lbs(betas, poses, v_template,
#                        shapedirs, posedirs,
#                        J_regressor, parents,
#                        lbs_weights, transl, dtype=dtype)
