from abc import ABC, abstractmethod

class Manifold(ABC):
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger

    @abstractmethod
    def init_embed(self, embed, irange):
        """Initialize an embedding by projecting it onto the manifold."""
        pass

    @abstractmethod
    def distance(self, u, v):
        """Compute the distance between u and v on the manifold."""
        pass

    @abstractmethod
    def log_map_zero(self, y, **kwargs):
        """Logarithmic map at zero (or the base point)"""
        pass

    @abstractmethod
    def log_map_x(self, x, y, **kwargs):
        """Logarithmic map at an arbitrary point x"""
        pass

    @abstractmethod
    def exp_map_zero(self, v, **kwargs):
        """Exponential map at zero (or the base point)"""
        pass

    @abstractmethod
    def exp_map_x(self, x, v, **kwargs):
        """Exponential map at an arbitrary point x"""
        pass

    @abstractmethod
    def parallel_transport(self, src, dst, v, **kwargs):
        """Transport a tangent vector v from src to dst"""
        pass

    @abstractmethod
    def rgrad(self, p, d_p, **kwargs):
        """Convert Euclidean gradients to Riemannian gradients at point p"""
        pass

    @abstractmethod
    def normalize(self, w, **kwargs):
        """Project a vector onto the manifold (or ensure it satisfies constraints)"""
        pass

    @abstractmethod
    def metric_tensor(self, x, u, v, **kwargs):
        """Compute the metric tensor (or inner product) at point x applied to tangent vectors u and v"""
        pass