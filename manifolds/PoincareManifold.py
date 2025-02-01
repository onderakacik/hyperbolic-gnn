import torch as th
import torch.nn as nn
from hyperbolic_module.PoincareDistance import PoincareDistance
from utils import clip_by_norm, th_dot, th_norm, th_atanh
from .Manifold import Manifold

class PoincareManifold(Manifold):
    def __init__(self, args, logger, EPS=1e-5, PROJ_EPS=1e-5):
        super(PoincareManifold, self).__init__(args, logger)
        self.EPS = EPS
        self.PROJ_EPS = PROJ_EPS
        self.tanh = nn.Tanh()

    def normalize(self, x):
        """Normalize vector x to ensure it lies within the Poincaré ball."""
        return clip_by_norm(x, (1. - self.PROJ_EPS))

    def init_embed(self, embed, irange=1e-2):
        """Initialize the embedding weights uniformly and normalize them for the Poincaré manifold."""
        embed.weight.data.uniform_(-irange, irange)
        embed.weight.data.copy_(self.normalize(embed.weight.data))

    def mob_add(self, u, v):
        """Perform Möbius addition of two vectors u and v in hyperbolic space."""
        v = v + self.EPS
        dot_uv = 2. * th_dot(u, v)
        norm_u_sq = th_dot(u, u)
        norm_v_sq = th_dot(v, v)
        denominator = 1. + dot_uv + norm_u_sq * norm_v_sq
        result = ((1. + dot_uv + norm_v_sq) / (denominator + self.EPS)) * u + \
                 ((1. - norm_u_sq) / (denominator + self.EPS)) * v
        return self.normalize(result)

    def distance(self, u, v):
        """Compute the distance between points u and v in the Poincaré manifold."""
        return PoincareDistance.apply(u, v, self.EPS)

    def lambda_x(self, x):
        """Compute the conformal factor at point x."""
        return 2. / (1 - th_dot(x, x))

    def log_map_zero(self, y):
        """Compute the logarithmic map at the origin for point y."""
        diff = y + self.EPS
        norm_diff = th_norm(diff)
        return (1. / (th_atanh(norm_diff, self.EPS) + self.EPS) / (norm_diff + self.EPS)) * diff

    def log_map_x(self, x, y):
        """Compute the logarithmic map from point x to point y."""
        diff = self.mob_add(-x, y) + self.EPS
        norm_diff = th_norm(diff)
        lam = self.lambda_x(x)
        return ((2. / lam) * (th_atanh(norm_diff, self.EPS) / (norm_diff + self.EPS))) * diff

    def metric_tensor(self, x, u, v):
        """Compute the metric tensor (inner product) between tangent vectors u and v at point x."""
        dot_uv = th_dot(u, v)
        lam = self.lambda_x(x)
        return (lam * lam) * dot_uv

    def exp_map_zero(self, v):
        """Compute the exponential map at the origin for vector v."""
        v = v + self.EPS
        norm_v = th_norm(v)
        result = (self.tanh(norm_v) / (norm_v + self.EPS)) * v
        return self.normalize(result)

    def exp_map_x(self, x, v):
        """Compute the exponential map from point x in the direction of vector v."""
        v = v + self.EPS
        norm_v = th_norm(v)
        second_term = (self.tanh(self.lambda_x(x) * norm_v / 2) / (norm_v + self.EPS)) * v
        return self.normalize(self.mob_add(x, second_term))

    def gyr(self, u, v, w):
        """Compute the gyration of vector w with respect to vectors u and v."""
        u_norm = th_dot(u, u)
        v_norm = th_dot(v, v)
        u_dot_w = th_dot(u, w)
        v_dot_w = th_dot(v, w)
        u_dot_v = th_dot(u, v)
        A = - u_dot_w * v_norm + v_dot_w + 2 * u_dot_v * v_dot_w
        B = - v_dot_w * u_norm - u_dot_w
        D = 1 + 2 * u_dot_v + u_norm * v_norm
        return w + 2 * (A * u + B * v) / (D + self.EPS)

    def parallel_transport(self, src, dst, v):
        """Parallel transport vector v from point src to point dst in the Poincaré manifold."""
        return self.lambda_x(src) / th.clamp(self.lambda_x(dst), min=self.EPS) * self.gyr(dst, -src, v)

    def rgrad(self, p, d_p):
        """Compute the Riemannian gradient from the Euclidean gradient in the Poincaré ball."""
        p_sqnorm = th.sum(p.data ** 2, dim=-1, keepdim=True)
        return d_p * (((1 - p_sqnorm) ** 2 / 4.0).expand_as(d_p))