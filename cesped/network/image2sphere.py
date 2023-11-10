#Modified from https://colab.research.google.com/github/dmklee/image2sphere/blob/main/model_walkthrough.ipynb
import os.path

import tempfile

import functools
import joblib
import warnings

import numpy as np
import torch
import torch.nn as nn
import torchvision
warnings.filterwarnings("ignore", module="e3nn", category=UserWarning)
warnings.filterwarnings("ignore", message="Grad strides do not match bucket view strides",  module="torch", category=UserWarning)

import e3nn
from e3nn import o3
import healpy as hp
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def s2_healpix_grid(hp_order, max_beta):
    """Returns healpix grid up to a max_beta
    """
    n_side = 2**hp_order
    # npix = hp.nside2npix(n_side)
    m = hp.query_disc(nside=n_side, vec=(0,0,1), radius=max_beta)
    beta, alpha = hp.pix2ang(n_side, m)
    alpha = torch.from_numpy(alpha)
    beta = torch.from_numpy(beta)
    return torch.stack((alpha, beta)).float()


class Image2SphereProjector(nn.Module):
  '''Define orthographic projection from image space to half of sphere, returning
  coefficients of spherical harmonics

  :fmap_shape: shape of incoming feature map (channels, height, width)
  :fdim_sphere: dimensionality of featuremap projected to sphere
  :lmax: maximum degree of harmonics
  :coverage: fraction of feature map that is projected onto sphere
  :sigma: stdev of gaussians used to sample points in image space
  :max_beta: maximum azimuth angle projected onto sphere (np.pi/2 corresponds to half sphere)
  :taper_beta: if less than max_beta, taper magnitude of projected features beyond this angle
  :hp_order: recursion level of healpy grid where points are projected
  :rand_fraction_points_to_project: number of grid points used to perform projection, acts like dropout regularizer
  '''
  def __init__(self,
               fmap_shape,
               sphere_fdim: int,
               lmax: int,
               coverage: float = 0.9,
               sigma: float = 0.2,
               max_beta: float = np.radians(90),
               taper_beta: float = np.radians(75),
               hp_order: int = 2,
               rand_fraction_points_to_project: float = 0.2,
              ):
    super().__init__()
    self.lmax = lmax

    # point-wise linear operation to convert to proper dimensionality if needed
    if fmap_shape[0] != sphere_fdim:
      self.conv1x1 = nn.Conv2d(fmap_shape[0], sphere_fdim, 1)
    else:
      self.conv1x1 = nn.Identity()

    # determine sampling locations for orthographic projection
    self.kernel_grid = s2_healpix_grid(max_beta=max_beta, hp_order=hp_order)
    self.xyz = o3.angles_to_xyz(*self.kernel_grid)

    # orthographic projection
    max_radius = torch.linalg.norm(self.xyz[:,[0,2]], dim=1).max()
    sample_x = coverage * self.xyz[:,2] / max_radius # range -1 to 1
    sample_y = coverage * self.xyz[:,0] / max_radius

    gridx, gridy = torch.meshgrid(2*[torch.linspace(-1, 1, fmap_shape[1])], indexing='ij')
    scale = 1 / np.sqrt(2 * np.pi * sigma**2)
    data = scale * torch.exp(-((gridx.unsqueeze(-1) - sample_x).pow(2) \
                                +(gridy.unsqueeze(-1) - sample_y).pow(2)) / (2*sigma**2) )
    data = data / data.sum((0,1), keepdims=True)

    # apply mask to taper magnitude near border if desired
    betas = self.kernel_grid[1]
    if taper_beta < max_beta:
        mask = ((betas - max_beta)/(taper_beta - max_beta)).clamp(max=1).view(1, 1, -1)
    else:
        mask = torch.ones_like(data)

    data = (mask * data).unsqueeze(0).unsqueeze(0).to(torch.float32)
    self.weight = nn.Parameter(data= data, requires_grad=True)

    self.n_pts = self.weight.shape[-1]
    self.ind = torch.arange(self.n_pts)
    self.n_subset = int(rand_fraction_points_to_project * self.n_pts) + 1

    self.register_buffer(
        "Y", o3.spherical_harmonics_alpha_beta(range(lmax+1), *self.kernel_grid, normalization='component')
    )

  def forward(self, x):
    '''
    :x: float tensor of shape (B, C, H, W)
    :return: feature vector of shape (B,P,C) where P is number of points on S2
    '''
    x = self.conv1x1(x)

    if self.n_subset is not None:
        self.ind = torch.randperm(self.n_pts)[:self.n_subset]

    x = (x.unsqueeze(-1) * self.weight[..., self.ind]).sum((2,3))
    x = torch.relu(x)
    x = torch.einsum('ni,xyn->xyi', self.Y[self.ind], x) / self.ind.shape[0]**0.5
    return x


def s2_irreps(lmax):
  return o3.Irreps([(1, (l, 1)) for l in range(lmax + 1)])

def so3_irreps(lmax):
  return o3.Irreps([(2 * l + 1, (l, 1)) for l in range(lmax + 1)])

def flat_wigner(lmax, alpha, beta, gamma):
  return torch.cat([
    (2 * l + 1) ** 0.5 * o3.wigner_D(l, alpha, beta, gamma).flatten(-2) for l in range(lmax + 1)
  ], dim=-1)


# # ORIGINAL IMPLEMENTATION
# def so3_near_identity_grid(max_beta=np.pi / 8, max_gamma=2 * np.pi, n_alpha=8, n_beta=3, n_gamma=None):
#   """Spatial grid over SO3 used to parametrize localized filter
#
#   :return: rings of rotations around the identity, all points (rotations) in
#            a ring are at the same distance from the identity
#            size of the kernel = n_alpha * n_beta * n_gamma
#   """
#   if n_gamma is None:
#       n_gamma = n_alpha
#   beta = torch.arange(1, n_beta + 1) * max_beta / n_beta
#   alpha = torch.linspace(0, 2 * np.pi, n_alpha)[:-1]
#   pre_gamma = torch.linspace(-max_gamma, max_gamma, n_gamma)
#   A, B, preC = torch.meshgrid(alpha, beta, pre_gamma, indexing="ij")
#   C = preC - A
#   A = A.flatten()
#   B = B.flatten()
#   C = C.flatten()
#   return torch.stack((A, B, C))


def so3_near_identity_grid(max_rads=np.pi / 12, n_angles=8):
    """Spatial grid over SO3 used to parametrize localized filter

    :return: a local grid of SO(3) points
           size of the kernel = n_alpha**3
    """

    angles_range = torch.linspace(-max_rads, max_rads, n_angles)
    grid = torch.cartesian_prod(angles_range, angles_range, angles_range)
    return grid.T

class S2Conv(nn.Module):
  '''S2 group convolution which outputs signal over SO(3) irreps

  :f_in: feature dimensionality of input signal
  :f_out: feature dimensionality of output signal
  :lmax: maximum degree of harmonics used to represent input and output signals
         technically, you can have different degrees for input and output, but
         we do not explore that in our work
  :kernel_grid: spatial locations over which the filter is defined (alphas, betas)
                we find that it is better to parametrize filter in spatial domain
                and project to harmonics at every forward pass.
  '''
  def __init__(self, f_in: int, f_out: int, lmax: int, kernel_grid: tuple):
    super().__init__()
    # filter weight parametrized over spatial grid on S2
    self.register_parameter(
      "w", torch.nn.Parameter(torch.randn(f_in, f_out, kernel_grid.shape[1]))
    )  # [f_in, f_out, n_s2_pts]

    s2Cache = joblib.Memory(location=os.path.join(tempfile.gettempdir(), "S2_cache.joblib"), verbose=0)
    def _compute_parameters(lmax, kernel_grid):
        spherical_harmonics = o3.spherical_harmonics_alpha_beta(range(lmax + 1),
                                        *kernel_grid, normalization="component") # [n_s2_pts, (2*lmax+1)**2]
        s2_ir =  s2_irreps(lmax)
        so3_ir = so3_irreps(lmax)
        return spherical_harmonics, s2_ir, so3_ir
    compute_parameters = s2Cache.cache(_compute_parameters)
    spherical_harmonics, s2_ir, so3_ir = compute_parameters(lmax, kernel_grid)

    # linear projection to convert filter weights to fourier domain
    self.register_buffer(
      "Y", spherical_harmonics)  # [n_s2_pts, (2*lmax+1)**2]

    # defines group convolution using appropriate irreps
    # note, we set internal_weights to False since we defined our own filter above
    self.lin = o3.Linear(s2_ir, so3_ir,
                         f_in=f_in, f_out=f_out, internal_weights=False)

  def forward(self, x):
    '''Perform S2 group convolution to produce signal over irreps of SO(3).
    First project filter into fourier domain then perform convolution

    :x: tensor of shape (B, f_in, (2*lmax+1)**2), signal over S2 irreps
    :return: tensor of shape (B, f_out, sum_l^L (2*l+1)**2)
    '''
    psi = torch.einsum("ni,xyn->xyi", self.Y, self.w) / self.Y.shape[0] ** 0.5
    return self.lin(x, weight=psi)

class SO3Conv(nn.Module):
  '''SO3 group convolution

  :f_in: feature dimensionality of input signal
  :f_out: feature dimensionality of output signal
  :lmax: maximum degree of harmonics used to represent input and output signals
         technically, you can have different degrees for input and output, but
         we do not explore that in our work
  :kernel_grid: spatial locations over which the filter is defined (alphas, betas, gammas)
                we find that it is better to parametrize filter in spatial domain
                and project to harmonics at every forward pass
  '''
  def __init__(self, f_in: int, f_out: int, lmax: int, kernel_grid: tuple):
    super().__init__()

    # filter weight parametrized over spatial grid on SO3
    self.register_parameter(
      "w", torch.nn.Parameter(torch.randn(f_in, f_out, kernel_grid.shape[1]))
    )  # [f_in, f_out, n_so3_pts]

    # wigner D matrices used to project spatial signal to irreps of SO(3)
    so3Cache = joblib.Memory(location=os.path.join(tempfile.gettempdir(), "SO3_cache.joblib"), verbose=0)
    def _compute_parameters(lmax, kernel_grid):
        f_wigner =  flat_wigner(lmax, *kernel_grid)  # [n_so3_pts, sum_l^L (2*l+1)**2]
        so3_ir = so3_irreps(lmax)
        return f_wigner, so3_ir
    compute_parameters = so3Cache.cache(_compute_parameters)
    f_wigner, so3_ir = compute_parameters(lmax, kernel_grid)
    self.register_buffer("D", f_wigner)

    # defines group convolution using appropriate irreps
    self.lin = o3.Linear(so3_ir, so3_ir,
                         f_in=f_in, f_out=f_out, internal_weights=False)

  def forward(self, x):
    '''Perform SO3 group convolution to produce signal over irreps of SO(3).
    First project filter into fourier domain then perform convolution

    :x: tensor of shape (B, f_in, sum_l^L (2*l+1)**2), signal over SO3 irreps
    :return: tensor of shape (B, f_out, sum_l^L (2*l+1)**2)
    '''
    psi = torch.einsum("ni,xyn->xyi", self.D, self.w) / self.D.shape[0] ** 0.5
    return self.lin(x, weight=psi)


def compute_trace(rotA, rotB):
    '''
    rotA, rotB are tensors of shape (*,3,3)
    returns Tr(rotA, rotB.T)
    '''
    prod = torch.matmul(rotA, rotB.transpose(-1, -2))
    trace = prod.diagonal(dim1=-1, dim2=-2).sum(-1)
    return trace


def rotation_error_rads(rotA, rotB):
    '''
    rotA, rotB are tensors of shape (*,3,3)
    returns rotation error in radians, tensor of shape (*)
    '''
    trace = compute_trace(rotA, rotB)
    return torch.arccos(torch.clamp((trace - 1) / 2, -1, 1))


def nearest_rotmat(src, target):
    '''return index of target that is nearest to each element in src
    uses negative trace of the dot product to avoid arccos operation
    :src: tensor of shape (B, 3, 3)
    :target: tensor of shape (*, 3, 3)
    '''
    trace = compute_trace(src.unsqueeze(1), target.unsqueeze(0)) #TODO: This could be precomputed in the dataloader

    return torch.max(trace, dim=1)[1]

def so3_healpix_grid(hp_order: int = 3):
    """Returns healpix grid over so3 of equally spaced rotations

    https://github.com/google-research/google-research/blob/4808a726f4b126ea38d49cdd152a6bb5d42efdf0/implicit_pdf/models.py#L272
    alpha: 0-2pi around Y
    beta: 0-pi around X
    gamma: 0-2pi around Y
    hp_order | num_points | bin width (deg)  | N inplane
    ----------------------------------------
         0    |         72 |    60           |
         1    |        576 |    30
         2    |       4608 |    15           | 24
         3    |      36864 |    7.5          | 48
         4    |     294912 |    3.75         | 96
         5    |    2359296 |    1.875

    :return: tensor of shape (3, npix)
    """
    n_side = 2 ** hp_order
    npix = hp.nside2npix(n_side)
    beta, alpha = hp.pix2ang(n_side, torch.arange(npix))
    beta = beta.float()
    alpha = alpha.float()
    gamma = torch.linspace(0, 2 * np.pi, 6 * n_side + 1)[:-1]

    alpha = alpha.repeat(len(gamma))
    beta = beta.repeat(len(gamma))
    gamma = torch.repeat_interleave(gamma, npix)
    result = torch.stack((alpha, beta, gamma)).float()
    return result

@functools.cache
def compute_symmetry_group_matrices(symmetry:str):
    return torch.stack([torch.FloatTensor(x) for x in R.create_group(symmetry.upper()).as_matrix()])

class I2S(nn.Module):
    '''
    Instantiate I2S-style network for predicting distributions over SO(3) from
    single image
    '''

    def __init__(self, imageEncoder, imageEncoderOutputShape,
                 symmetry:str,
                 lmax:int=6, s2_fdim:int=512, so3_fdim:int=16,
                 hp_order_projector:int=2,
                 hp_order_s2:int=2,
                 hp_order_so3:int=3,
                 so3_act_resolution:int=10, #TODO: what is the effect of resolution??
                 rand_fraction_points_to_project:float=0.2):
        """

        Args:
            imageEncoder:
            imageEncoderOutputShape:
            symmetry (str): The symmetry to be applied during training
            lmax:
            s2_fdim:
            so3_fdim:
            hp_order_projector:
            hp_order_s2:
            hp_order_so3:
            so3_act_resolution:
            rand_fraction_points_to_project:
        """
        super().__init__()
        self.encoder = imageEncoder
        self.symmetry = symmetry.upper()
        self.lmax = lmax
        self.s2_fdim = s2_fdim
        self.so3_fdim = so3_fdim
        self.hp_order_projector = hp_order_projector
        self.hp_order_s2 = hp_order_s2
        self.hp_order = hp_order_so3

        self.n_sphere_pixels = hp.order2npix(self.hp_order)

        self.projector = Image2SphereProjector(
            fmap_shape=imageEncoderOutputShape,
            sphere_fdim=s2_fdim,
            lmax=lmax,
            hp_order=hp_order_projector,
            rand_fraction_points_to_project = rand_fraction_points_to_project
        )

        # s2 filter has global support
        s2_kernel_grid = s2_healpix_grid(max_beta=np.inf, hp_order=self.hp_order_s2)
        self.s2_conv = S2Conv(s2_fdim, so3_fdim, lmax, s2_kernel_grid)

        self.so3_act = e3nn.nn.SO3Activation(lmax, lmax, act=torch.relu, resolution=so3_act_resolution)

        # locally supported so3 filter
        so3_kernel_grid = so3_near_identity_grid()
        self.so3_conv = SO3Conv(so3_fdim, 1, lmax, so3_kernel_grid)

        # define spatial grid used to convert output irreps into valid prob distribution
        #hp_order=2 which corresponds to ~5000 points, is
        # sufficient for training in real world images.  Using denser grids will slow down loss computation

        output_eulerRad_yxy = so3_healpix_grid(hp_order=self.hp_order)
        self.register_buffer("output_eulerRad_yxy", output_eulerRad_yxy)

        i2sCache = joblib.Memory(location=os.path.join(tempfile.gettempdir(), "I2S_cache.joblib"), verbose=0)
        def _compute_parameters(lmax, output_eulerRad_yxy):
            output_wigners = flat_wigner(lmax, *output_eulerRad_yxy).transpose(0, 1)
            output_rotmats = o3.angles_to_matrix(*output_eulerRad_yxy)
            return output_wigners, output_rotmats

        compute_parameters = i2sCache.cache(_compute_parameters)
        output_wigners, output_rotmats = compute_parameters(lmax, output_eulerRad_yxy)


        self.register_buffer(
            "output_wigners", output_wigners
        )
        output_rotmats = o3.angles_to_matrix(*output_eulerRad_yxy)
        self.register_buffer(
            "output_rotmats", output_rotmats
        )

        self.register_buffer("symmetryGroupMatrix", compute_symmetry_group_matrices(self.symmetry))

    def _forward(self, x):
        '''Returns so3 irreps

        :x: image, tensor of shape (B, c, L, L)
        '''
        x = self.encoder(x)
        x = self.projector(x)
        x = self.s2_conv(x)
        x = self.so3_act(x)
        x = self.so3_conv(x)
        return x

    def forward(self, img):

        '''Compute cross entropy loss using ground truth rotation, the correct label
        is the nearest rotation in the spatial grid to the ground truth rotation

        :img: float tensor of shape (B, c, L, L)
        '''
        x = self._forward(img)
        grid_signal = torch.matmul(x, self.output_wigners).squeeze(1)
        rotmats = self.output_rotmats
        with torch.no_grad():
            probs = nn.functional.softmax(grid_signal, dim=-1)
            maxprob, pred_id = probs.max(dim=1)
            pred_rotmat = rotmats[pred_id]

        return grid_signal, pred_rotmat, maxprob, probs

    def forward_topk(self, img, k):
        x = self._forward(img)
        grid_signal = torch.matmul(x, self.output_wigners).squeeze(1)
        rotmats = self.output_rotmats
        with torch.no_grad():
            probs = nn.functional.softmax(grid_signal, dim=-1)
            maxprob, pred_id = torch.topk(probs, k=k, dim=-1, largest=True)
            pred_rotmat = rotmats[pred_id]
        return grid_signal, pred_rotmat, maxprob, probs


    def nearest_rotmat(self, rotMat, toCompareRotMats=None):
        if toCompareRotMats is None:
            toCompareRotMats = self.output_rotmats
        return nearest_rotmat(rotMat, toCompareRotMats) #THIS IS COMPUTATIONALLY EXPENSIVE


    @classmethod
    def rotation_error_rads(cls, rotA, rotB):
        '''
        rotA, rotB are tensors of shape (*,3,3)
        returns rotation error in radians, tensor of shape (*)
        '''
        return rotation_error_rads(rotA, rotB)

    def forward_and_loss(self, img, gt_rot, per_img_weight=None):
        '''Compute cross entropy loss using ground truth rotation, the correct label
        is the nearest rotation in the spatial grid to the ground truth rotation

        :img: float tensor of shape (B, c, L, L)
        :gt_rotation: valid rotation matrices, tensor of shape (B, 3, 3)
        :per_img_weight: float tensor of shape (B,) with per_image_weight for loss calculation
        '''

        grid_signal, pred_rotmats, maxprob, probs = self.forward(img)

        if self.symmetry != "C1":
            n_groupElems = self.symmetryGroupMatrix.shape[0]
            #Perform symmetry expansion
            gtrotMats = self.symmetryGroupMatrix[None, ...] @ gt_rot[:, None, ...]
            rotMat_gtIds = self.nearest_rotmat(gtrotMats.view(-1, 3, 3)).view(grid_signal.shape[0], -1)
            target_he = torch.zeros_like(grid_signal)
            rows = torch.arange(grid_signal.shape[0]).view(-1, 1).repeat(1, n_groupElems)
            target_he[rows, rotMat_gtIds] = 1 / n_groupElems
            loss = nn.functional.cross_entropy(grid_signal, target_he, reduction="none", label_smoothing=0.1)

            with torch.no_grad():
                error_rads = rotation_error_rads(gtrotMats.view(-1,3,3),
                                                 torch.repeat_interleave(pred_rotmats, n_groupElems, dim=0))
                error_rads = error_rads.view(-1, n_groupElems)
                error_rads = error_rads.min(1).values

        else:
            # find nearest grid point to ground truth rotation matrix
            rot_id = self.nearest_rotmat(gt_rot)
            loss = nn.functional.cross_entropy(grid_signal, rot_id, reduction="none", label_smoothing=0.1)
            with torch.no_grad():
                error_rads = rotation_error_rads(gt_rot, pred_rotmats)

        if per_img_weight is not None:
            loss = loss * per_img_weight.squeeze(-1)
        loss = loss.mean()

        return loss, error_rads, pred_rotmats, maxprob, probs

    @torch.no_grad()
    def compute_probabilities(self, img, hp_order=None):
        '''Computes probability distribution over arbitrary spatial grid specified by
        wigners

        Our method can be trained on a sparser spatial resolution, but queried at a much denser
        resolution (up to hp_order=5)
        '''
        if hp_order is None:
            hp_order = self.hp_order
            output_eulerRad_yxy = so3_healpix_grid(hp_order=hp_order)
        else:
            output_eulerRad_yxy = self.output_eulerRad_yxy

        output_wigners = flat_wigner(self.lmax, *output_eulerRad_yxy).transpose(0, 1)
        output_rotmats = o3.angles_to_matrix(*output_eulerRad_yxy)

        x = self._forward(img)
        logits = torch.matmul(x, output_wigners).squeeze(1)
        probs = nn.Softmax(dim=1)(logits)

        return probs, output_rotmats

def plot_so3_distribution(probs: torch.Tensor,
                          rots: torch.Tensor,
                          gt_rotation=None,
                          fig=None,
                          ax=None,
                          display_threshold_probability=0.000005,
                          show_color_wheel: bool=True,
                          canonical_rotation=torch.eye(3),
                         ):
    '''
    Taken from https://github.com/google-research/google-research/blob/master/implicit_pdf/evaluation.py
    '''
    cmap = plt.cm.hsv

    def _show_single_marker(ax, rotation, marker, edgecolors=True, facecolors=False):
        alpha, beta, gamma = o3.matrix_to_angles(rotation)
        color = cmap(0.5 + gamma.repeat(2) / 2. / np.pi)[-1]
        ax.scatter(alpha, beta-np.pi/2, s=2000, edgecolors=color, facecolors='none', marker=marker, linewidth=5)
        ax.scatter(alpha, beta-np.pi/2, s=1500, edgecolors='k', facecolors='none', marker=marker, linewidth=2)
        ax.scatter(alpha, beta-np.pi/2, s=2500, edgecolors='k', facecolors='none', marker=marker, linewidth=2)

    if ax is None:
        fig = plt.figure(figsize=(8, 4), dpi=200)
        fig.subplots_adjust(0.01, 0.08, 0.90, 0.95)
        ax = fig.add_subplot(111, projection='mollweide')

    rots = rots @ canonical_rotation
    scatterpoint_scaling = 3e3
    alpha, beta, gamma = o3.matrix_to_angles(rots)

    # offset alpha and beta so different gammas are visible
    R = 0.02
    alpha += R * np.cos(gamma)
    beta += R * np.sin(gamma)

    which_to_display = (probs > display_threshold_probability)

    # Display the distribution
    ax.scatter(alpha[which_to_display],
               beta[which_to_display]-np.pi/2,
               s=scatterpoint_scaling * probs[which_to_display],
               c=cmap(0.5 + gamma[which_to_display] / 2. / np.pi))
    if gt_rotation is not None:
        if len(gt_rotation.shape) == 2:
            gt_rotation = gt_rotation.unsqueeze(0)
        gt_rotation = gt_rotation @ canonical_rotation
        _show_single_marker(ax, gt_rotation, 'o')
    ax.grid()
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    if show_color_wheel:
        # Add a color wheel showing the tilt angle to color conversion.
        ax = fig.add_axes([0.86, 0.17, 0.12, 0.12], projection='polar')
        theta = np.linspace(-3 * np.pi / 2, np.pi / 2, 200)
        radii = np.linspace(0.4, 0.5, 2)
        _, theta_grid = np.meshgrid(radii, theta)
        colormap_val = 0.5 + theta_grid / np.pi / 2.
        ax.pcolormesh(theta, radii, colormap_val.T, cmap=cmap)
        ax.set_yticklabels([])
        ax.set_xticklabels([r'90$\degree$', None,
                            r'180$\degree$', None,
                            r'270$\degree$', None,
                            r'0$\degree$'], fontsize=14)
        ax.spines['polar'].set_visible(False)
        plt.text(0.5, 0.5, 'Tilt', fontsize=14,
                 horizontalalignment='center',
                 verticalalignment='center', transform=ax.transAxes)

    plt.show()

def _test():
    model = torchvision.models.resnet50(weights=None)  # Better if pretrained
    model = nn.Sequential(*list(model.children())[:-2])

    b,c,l = 4,3,224
    imgs = torch.rand(b,c,l,l)

    model = I2S(model, model(imgs).shape[1:], symmetry="C1", lmax=6, s2_fdim=512, so3_fdim=16, hp_order_so3=2)


    from scipy.spatial.transform import Rotation
    gt_rot = torch.from_numpy(Rotation.random(b).as_matrix().astype(np.float32))
    grid_signal, pred_rotmat, maxprob, probs = model.forward(imgs)
    pred_rotmat = model.forward(imgs)
    print("out", grid_signal.shape)
    print("pred_rotmat", pred_rotmat)
    probs, output_rotmats = model.compute_probabilities(imgs)
    plot_so3_distribution(probs[0], output_rotmats, gt_rotation=gt_rot[0])

if __name__ == "__main__":
    _test()
    print("Done!")