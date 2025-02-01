import torch as th
from torch.autograd import Function
from utils import th_dot
from .Manifold import Manifold

_eps = 1e-10

class LorentzManifold(Manifold):
    def __init__(self, args, logger, eps=1e-3, norm_clip=1, max_norm=1e3):
        super(LorentzManifold, self).__init__(args, logger)
        self.eps = eps
        self.norm_clip = norm_clip
        self.max_norm = max_norm

    @staticmethod
    def ldot(u, v, keepdim=False):
        """Compute the Lorentzian scalar product between vectors u and v."""
        d = u.size(1) - 1
        uv = u * v
        uv = th.cat((-uv.narrow(1, 0, 1), uv.narrow(1, 1, d)), dim=1)
        return th.sum(uv, dim=1, keepdim=keepdim)

    def from_lorentz_to_poincare(self, x):
        """Convert a point x from Lorentzian coordinates to Poincaré coordinates."""
        d = x.size(-1) - 1
        return x.narrow(-1, 1, d) / (x.narrow(-1, 0, 1) + 1)

    def from_poincare_to_lorentz(self, x):
        """Convert a point x from Poincaré coordinates to Lorentzian coordinates."""
        x_norm_square = th_dot(x, x)
        return th.cat((1 + x_norm_square, 2 * x), dim=1) / (1 - x_norm_square + self.eps)

    def distance(self, u, v):
        """Compute the distance between points u and v in the Lorentzian manifold."""
        d = -LorentzDot.apply(u, v)
        return Acosh.apply(d, self.eps)

    def normalize(self, w):
        """Normalize vector w to ensure it lies on the hyperboloid."""
        d = w.size(-1) - 1
        narrowed = w.narrow(-1, 1, d)
        if self.max_norm:
            narrowed = th.renorm(narrowed.view(-1, d), 2, 0, self.max_norm)
        first = 1 + th.sum(th.pow(narrowed, 2), dim=-1, keepdim=True)
        first = th.sqrt(first)
        return th.cat((first, narrowed), dim=1)

    def init_embed(self, embed, irange=1e-2):
        """Initialize the embedding weights uniformly and normalize them for the Lorentz manifold."""
        embed.weight.data.uniform_(-irange, irange)
        embed.weight.data.copy_(self.normalize(embed.weight.data))

    def rgrad(self, p, d_p):
        """Compute the Riemannian gradient for the hyperboloid."""
        u = d_p
        x = p
        u.narrow(-1, 0, 1).mul_(-1)
        u.addcmul_(self.ldot(x, u, keepdim=True).expand_as(x), x)
        return d_p

    def exp_map_zero(self, v):
        """Compute the exponential map at the origin for vector v."""
        zeros = th.zeros_like(v)
        zeros[:, 0] = 1
        return self.exp_map_x(zeros, v)

    def exp_map_x(self, p, d_p, d_p_normalize=True, p_normalize=True):
        """Compute the exponential map from point p in the direction of vector d_p."""
        if d_p_normalize:
            d_p = self.normalize_tan(p, d_p)

        ldv = self.ldot(d_p, d_p, keepdim=True)
        nd_p = th.sqrt(th.clamp(ldv + self.eps, _eps))

        t = th.clamp(nd_p, max=self.norm_clip)
        newp = (th.cosh(t) * p) + (th.sinh(t) * d_p / nd_p)

        if p_normalize:
            newp = self.normalize(newp)
        return newp

    def normalize_tan(self, x_all, v_all):
        """Normalize tangent vector v_all at point x_all."""
        d = v_all.size(1) - 1
        x = x_all.narrow(1, 1, d)
        xv = th.sum(x * v_all.narrow(1, 1, d), dim=1, keepdim=True)
        tmp = 1 + th.sum(th.pow(x_all.narrow(1, 1, d), 2), dim=1, keepdim=True)
        tmp = th.sqrt(tmp)
        return th.cat((xv / tmp, v_all.narrow(1, 1, d)), dim=1)

    def log_map_zero(self, y, i=-1):
        """Compute the logarithmic map at the origin for point y."""
        zeros = th.zeros_like(y)
        zeros[:, 0] = 1
        return self.log_map_x(zeros, y)

    def log_map_x(self, x, y, normalize=False):
        """Compute the logarithmic map from point x to point y."""
        xy = self.ldot(x, y).unsqueeze(-1)
        tmp = th.sqrt(th.clamp(xy * xy - 1 + self.eps, _eps))
        v = Acosh.apply(-xy, self.eps) / tmp * th.addcmul(y, xy, x)
        if normalize:
            result = self.normalize_tan(x, v)
        else:
            result = v
        return result

    def parallel_transport(self, x, y, v):
        """Parallel transport vector v from point x to point y on the hyperboloid."""
        xy = self.ldot(x, y, keepdim=True).expand_as(x)
        vy = self.ldot(v, y, keepdim=True).expand_as(x)
        vnew = v + vy / (1 - xy) * (x + y)
        return vnew

    def metric_tensor(self, x, u, v):
        """Compute the metric tensor (inner product) between tangent vectors u and v at point x."""
        return self.ldot(u, v, keepdim=True)

class LorentzDot(Function):
    @staticmethod
    def forward(ctx, u, v):
        """Forward pass for the LorentzDot function."""
        ctx.save_for_backward(u, v)
        return LorentzManifold.ldot(u, v)

    @staticmethod
    def backward(ctx, g):
        """Backward pass for the LorentzDot function."""
        u, v = ctx.saved_tensors
        g = g.unsqueeze(-1).expand_as(u).clone()
        g.narrow(-1, 0, 1).mul_(-1)
        return g * v, g * u

class Acosh(Function):
    @staticmethod
    def forward(ctx, x, eps):
        """Forward pass for the Acosh function."""
        z = th.sqrt(th.clamp(x * x - 1 + eps, _eps))
        ctx.save_for_backward(z)
        ctx.eps = eps
        return th.log(x + z)

    @staticmethod
    def backward(ctx, g):
        """Backward pass for the Acosh function."""
        z, = ctx.saved_tensors
        z = th.clamp(z, min=ctx.eps)
        return g / z, None