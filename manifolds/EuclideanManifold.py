import torch as th
from utils import clip_by_norm, th_dot, clamp_min
from .Manifold import Manifold

class EuclideanManifold(Manifold):
    def __init__(self, args, max_norm=1, EPS=1e-8):
        super(EuclideanManifold, self).__init__(args)
        self.max_norm = max_norm
        self.EPS = EPS

    def init_embed(self, embed, irange=1e-3):
        """Initialize the embedding weights uniformly within a specified range and normalize them."""
        embed.weight.data.uniform_(-irange, irange)
        embed.weight.data.copy_(self.normalize(embed.weight.data))

    def distance(self, u, v):
        """Compute the Euclidean distance between two points u and v."""
        return th.sqrt(clamp_min(th.sum((u - v).pow(2), dim=1), self.EPS))

    def log_map_zero(self, y):
        """Return the logarithmic map at the origin, which is simply the point itself in Euclidean space."""
        return y

    def log_map_x(self, x, y):
        """Compute the logarithmic map from point x to point y."""
        return y - x

    def metric_tensor(self, x, u, v):
        """Compute the inner product (metric tensor) between tangent vectors u and v at point x."""
        return th_dot(u, v)

    def exp_map_zero(self, v):
        """Return the exponential map at the origin, which is the normalized vector v."""
        return self.normalize(v)

    def exp_map_x(self, x, v, approximate=False):
        """Compute the exponential map from point x in the direction of vector v."""
        return self.normalize(x + v)

    def parallel_transport(self, src, dst, v):
        """Parallel transport of vector v from point src to point dst, which is trivial in Euclidean space."""
        return v

    def rgrad(self, p, d_p):
        """Return the Riemannian gradient, which is the same as the Euclidean gradient in Euclidean space."""
        return d_p

    def normalize(self, w):
        """Normalize the vector w to ensure it does not exceed the maximum norm."""
        if self.max_norm:
            return clip_by_norm(w, self.max_norm)
        else:
            return w