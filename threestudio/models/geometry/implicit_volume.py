from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.geometry.base import (
    BaseGeometry,
    BaseImplicitGeometry,
    contract_to_unisphere,
)
from threestudio.models.networks import get_encoding, get_mlp
from threestudio.utils.ops import get_activation
from threestudio.utils.typing import *

from smplx.body_models import SMPL
from .inv_deformation import lbs, inv, deform_lbs

from kaolin.ops.mesh import check_sign
from kaolin.metrics.trianglemesh import point_to_mesh_distance
import trimesh
# import tensorflow as tf

def save_obj_mesh(mesh_path, verts, faces):
    file = open(mesh_path, 'w') # 打开mesh路径

    for v in verts: # 记录verts
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
    for f in faces: # 记录faces
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
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

def cal_sdf(verts, faces, points):
    # functions modified from ICON
    
    # verts [B, N_vert, 3]
    # faces [B, N_face, 3]
    # triangles [B, N_face, 3, 3]
    # points [B, N_point, 3]
    
    Bsize = points.shape[0]
    
    triangles = build_triangles(verts, faces)
    residues, pts_ind, _ = point_to_mesh_distance(points, triangles)
    
    closest_triangles = torch.gather(
        triangles, 1, pts_ind[:, :, None, None].expand(-1, -1, 3, 3)).view(-1, 3, 3)
    residues = residues.to(device=points.device)
    pts_dist = torch.sqrt(residues) / torch.sqrt(torch.tensor(3))

    pts_signs = 2.0 * (check_sign(verts, faces[0], points).float() - 0.5)
    pts_sdf = (pts_dist * pts_signs).unsqueeze(-1)

    return pts_sdf.view(Bsize, -1, 1)


@threestudio.register("implicit-volume")
class ImplicitVolume(BaseImplicitGeometry):
    
    smpl = SMPL(model_path='./smpl_data', gender='NEUTRAL', batch_size=1)
    shapedirs = torch.Tensor(smpl.shapedirs).cuda()
    posedirs = torch.Tensor(smpl.posedirs).cuda()
    J_regressor = torch.Tensor(smpl.J_regressor).cuda()
    parents = torch.Tensor(smpl.parents).cuda()
    lbs_weights = torch.Tensor(smpl.lbs_weights).cuda()
    faces = torch.Tensor(np.asarray(smpl.faces).astype(np.int64)).cuda()
    faces = faces.to(torch.int64).cuda()
    vt = smpl.v_template
    center = vt.mean(axis=0)
    vt = vt - center
    scale = np.max(np.linalg.norm(vt, axis=1))
    vt = (vt / scale) * 0.5
    vt = vt.cuda()
    vt[:, 1] = vt[:, 1] + 0.6
    vt[:, 0] = vt[:, 0] + 0.5
    vt[:, 2] = vt[:, 2] + 0.5
    
    z_ = np.array([0, 1, 0])
    x_ = np.array([0, 0, 1])
    y_ = np.cross(z_, x_)
                
    std2mesh = np.stack([x_, y_, z_], axis=0).T
    mesh2std = torch.from_numpy(np.linalg.inv(std2mesh)).cuda().double() 
   
    smpl_param= np.load('./SMPL_poses/0004_smpl.pkl', allow_pickle=True)
    smpl_pose = torch.tensor(smpl_param['body_pose']).cuda()
    smpl_betas = torch.tensor(smpl_param['betas']).cuda()
    # smpl_betas[:, 0] = -3.0
    global_pose = torch.tensor(smpl_param['global_orient']).unsqueeze(0).cuda()
    smpl_trans = torch.tensor(smpl_param['transl']).cuda()
    smpl_pose = torch.cat([global_pose, smpl_pose], dim=1).reshape(1, -1).float().cuda()
    smpl_scale = torch.tensor(smpl_param['scale']).cuda()
    # smpl_pose[:] = 0.0
    # smpl_pose[0, 41] = -np.pi / 4.0
    # smpl_pose[0, 44] = np.pi / 4.0
    v_posed, T_posed, _ = lbs(smpl_betas, smpl_pose, vt, shapedirs, posedirs, J_regressor, parents, lbs_weights, smpl_trans)
        
    star_pose = smpl_pose.clone()
    star_pose[:] = 0.0
    star_pose[0, 41] = -np.pi / 4
    star_pose[0, 44] = np.pi / 4
    
    v_star, T_star, _ = lbs(smpl_betas, star_pose, vt, shapedirs, posedirs, J_regressor, parents, lbs_weights, smpl_trans)

    @dataclass
    class Config(BaseImplicitGeometry.Config):
        n_input_dims: int = 3
        n_feature_dims: int = 3
        density_activation: Optional[str] = "softplus"
        density_bias: Union[float, str] = "blob_magic3d"
        density_blob_scale: float = 10.0
        density_blob_std: float = 0.5
        pos_encoding_config: dict = field(
            default_factory=lambda: {
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.447269237440378,
            }
        )
        mlp_network_config: dict = field(
            default_factory=lambda: {
                "otype": "VanillaMLP",
                "activation": "ReLU",
                "output_activation": "none",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            }
        )
        normal_type: Optional[
            str
        ] = "finite_difference"  # in ['pred', 'finite_difference', 'finite_difference_laplacian']
        finite_difference_normal_eps: float = 0.01

        # automatically determine the threshold
        isosurface_threshold: Union[float, str] = 25.0

    cfg: Config

    def configure(self) -> None:
        super().configure()
        self.encoding = get_encoding(
            self.cfg.n_input_dims, self.cfg.pos_encoding_config
        )
        self.density_network = get_mlp(
            self.encoding.n_output_dims, 1, self.cfg.mlp_network_config
        )
        if self.cfg.n_feature_dims > 0:
            self.feature_network = get_mlp(
                self.encoding.n_output_dims,
                self.cfg.n_feature_dims,
                self.cfg.mlp_network_config,
            )
        # if self.cfg.normal_type == "pred":
        self.normal_network = get_mlp(
            self.encoding.n_output_dims, 3, self.cfg.mlp_network_config
        )


    def get_activated_density(
        self, points: Float[Tensor, "*N Di"], density: Float[Tensor, "*N 1"], density_guide: Float[Tensor, "*N 1"], idx: int = 0
    ) -> Tuple[Float[Tensor, "*N 1"], Float[Tensor, "*N 1"]]:
        density_bias: Union[float, Float[Tensor, "*N 1"]]
        if self.cfg.density_bias == "blob_dreamfusion":
            # pre-activation density bias
            density_bias = (
                self.cfg.density_blob_scale
                * torch.exp(
                    -0.5 * (points**2).sum(dim=-1) / self.cfg.density_blob_std**2
                )[..., None]
            )
        elif self.cfg.density_bias == "blob_magic3d":
            # pre-activation density bias
            density_bias = (
                self.cfg.density_blob_scale
                * (
                    1
                    - torch.sqrt((points**2).sum(dim=-1)) / self.cfg.density_blob_std
                )[..., None]
            )
        elif isinstance(self.cfg.density_bias, float):
            density_bias = self.cfg.density_bias
        else:
            raise ValueError(f"Unknown density bias {self.cfg.density_bias}")
        
        raw_density: Float[Tensor, "*N 1"] = density + density_bias

        raw_density = raw_density + density_guide
        density = get_activation(self.cfg.density_activation)(raw_density)
        density_normal = density

        return raw_density, density, density_normal
    
    def forward(
        self, points: Float[Tensor, "*N Di"], output_normal: bool = False, observ: bool = False, idx: int = 0
    ) -> Dict[str, Float[Tensor, "..."]]:
        # observ = True
        grad_enabled = torch.is_grad_enabled()
        points = torch.mm(self.mesh2std.inverse(), points.transpose(1, 0).double()).transpose(1, 0).float().contiguous()
        # if output_normal and self.cfg.normal_type == "analytic":
        torch.set_grad_enabled(True)
        points.requires_grad_(True)
        
        points = contract_to_unisphere(
            points, self.bbox, self.unbounded
        )  # points normalized to (0, 1)
        
        points_unscaled = points
        sdf_guide = cal_sdf(self.v_star.float(), self.faces.unsqueeze(0), points.unsqueeze(0)).squeeze(0).squeeze(-1)
        points = contract_to_unisphere(
            points_unscaled, self.bbox, self.unbounded
        )
        
        beta = 0.001
        alpha = 1.0 / beta
        variable = -1.0 * torch.abs(sdf_guide) * alpha
        sigma_guide = alpha * torch.sigmoid(variable)
        softplus_guide = torch.log(torch.exp(sigma_guide) - 1)
        density_guide = torch.clamp(softplus_guide, min=0.0)
        enc = self.encoding(points.view(-1, self.cfg.n_input_dims))
        density = self.density_network(enc).view(*points.shape[:-1], 1)
        
        raw_density, density, density_normal = self.get_activated_density(points_unscaled, density, density_guide.unsqueeze(-1), idx=idx)
        output = {
            "density": density,
        }
        if self.cfg.n_feature_dims > 0:
            features = self.feature_network(enc).view(
                *points.shape[:-1], self.cfg.n_feature_dims
            )
            output.update({"features": features})

        if output_normal:
            if (
                self.cfg.normal_type == "finite_difference"
                or self.cfg.normal_type == "finite_difference_laplacian"
            ):
                # TODO: use raw density
                eps = self.cfg.finite_difference_normal_eps
                if self.cfg.normal_type == "finite_difference_laplacian":
                    offsets: Float[Tensor, "6 3"] = torch.as_tensor(
                        [
                            [eps, 0.0, 0.0],
                            [-eps, 0.0, 0.0],
                            [0.0, eps, 0.0],
                            [0.0, -eps, 0.0],
                            [0.0, 0.0, eps],
                            [0.0, 0.0, -eps],
                        ]
                    ).to(points_unscaled)
                    points_offset: Float[Tensor, "... 6 3"] = (
                        points_unscaled[..., None, :] + offsets
                    ).clamp(-self.cfg.radius, self.cfg.radius)
                    density_offset: Float[Tensor, "... 6 1"] = self.forward_density(
                        points_offset
                    )
                    normal = (
                        -0.5
                        * (density_offset[..., 0::2, 0] - density_offset[..., 1::2, 0])
                        / eps
                    )
                else:
                    offsets: Float[Tensor, "3 3"] = torch.as_tensor(
                        [[eps, 0.0, 0.0], [0.0, eps, 0.0], [0.0, 0.0, eps]]
                    ).to(points_unscaled)
                    points_offset: Float[Tensor, "... 3 3"] = (
                        points_unscaled[..., None, :] + offsets
                    ).clamp(-self.cfg.radius, self.cfg.radius)
                    density_offset: Float[Tensor, "... 3 1"] = self.forward_density(
                        points_offset
                    )
                    normal = -(density_offset[..., 0::1, 0] - density) / eps
                normal = F.normalize(normal, dim=-1)
            elif self.cfg.normal_type == "pred":
                normal = self.normal_network(enc).view(*points.shape[:-1], 3)
                normal = F.normalize(normal, dim=-1)
            elif self.cfg.normal_type == "analytic":
                normal = -torch.autograd.grad(
                    density,
                    points_unscaled,
                    grad_outputs=torch.ones_like(density),
                    create_graph=True,
                )[0]
                normal = F.normalize(normal, dim=-1)
                if not grad_enabled:
                    normal = normal.detach()
                # normal = self.normal_network(enc).view(*points.shape[:-1], 3)
                # normal = F.normalize(normal, dim=-1)
            else:
                raise AttributeError(f"Unknown normal type {self.cfg.normal_type}")
            output.update({"normal": normal, "shading_normal": normal})
            # output.update({"normal_grad": normal_grad})

        torch.set_grad_enabled(grad_enabled)
        return output

    def forward_density(self, points: Float[Tensor, "*N Di"], observ: bool = False, idx: int = 0) -> Float[Tensor, "*N 1"]:
        points = torch.mm(self.mesh2std.inverse(), points.transpose(1, 0).double()).transpose(1, 0).float().contiguous()
        points_unscaled = points
        points = contract_to_unisphere(points_unscaled, self.bbox, self.unbounded)


        sdf_guide = cal_sdf(self.v_star.float(), self.faces.unsqueeze(0), points.unsqueeze(0)).squeeze(0).squeeze(-1)


        points = contract_to_unisphere(
            points_unscaled, self.bbox, self.unbounded
        )
        
        beta = 0.001
        alpha = 1.0 / beta
        variable = -1.0 * torch.abs(sdf_guide) * alpha
        sigma_guide = alpha * torch.sigmoid(variable)
        softplus_guide = torch.log(torch.exp(sigma_guide) - 1)
        density_guide = torch.clamp(softplus_guide, min=0.0)
        
        enc = self.encoding(points.view(-1, self.cfg.n_input_dims))
        density = self.density_network(enc).view(*points.shape[:-1], 1)
        raw_density, density, density_normal = self.get_activated_density(points_unscaled, density, density_guide.unsqueeze(-1), idx=idx)
        
        return density

    def forward_field(
        self, points: Float[Tensor, "*N Di"]
    ) -> Tuple[Float[Tensor, "*N 1"], Optional[Float[Tensor, "*N 3"]]]:
        if self.cfg.isosurface_deformable_grid:
            threestudio.warn(
                f"{self.__class__.__name__} does not support isosurface_deformable_grid. Ignoring."
            )
        density = self.forward_density(points)
        return density, None

    def forward_level(
        self, field: Float[Tensor, "*N 1"], threshold: float
    ) -> Float[Tensor, "*N 1"]:
        return -(field - threshold)

    def export(self, points: Float[Tensor, "*N Di"], **kwargs) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.cfg.n_feature_dims == 0:
            return out
        points_unscaled = points
        points = contract_to_unisphere(points_unscaled, self.bbox, self.unbounded)
        enc = self.encoding(points.reshape(-1, self.cfg.n_input_dims))
        features = self.feature_network(enc).view(
            *points.shape[:-1], self.cfg.n_feature_dims
        )
        out.update(
            {
                "features": features,
            }
        )
        return out

    @staticmethod
    @torch.no_grad()
    def create_from(
        other: BaseGeometry,
        cfg: Optional[Union[dict, DictConfig]] = None,
        copy_net: bool = True,
        **kwargs,
    ) -> "ImplicitVolume":
        if isinstance(other, ImplicitVolume):
            instance = ImplicitVolume(cfg, **kwargs)
            instance.encoding.load_state_dict(other.encoding.state_dict())
            instance.density_network.load_state_dict(other.density_network.state_dict())
            if copy_net:
                if (
                    instance.cfg.n_feature_dims > 0
                    and other.cfg.n_feature_dims == instance.cfg.n_feature_dims
                ):
                    instance.feature_network.load_state_dict(
                        other.feature_network.state_dict()
                    )
                if (
                    instance.cfg.normal_type == "pred"
                    and other.cfg.normal_type == "pred"
                ):
                    instance.normal_network.load_state_dict(
                        other.normal_network.state_dict()
                    )
            return instance
        else:
            raise TypeError(
                f"Cannot create {ImplicitVolume.__name__} from {other.__class__.__name__}"
            )
