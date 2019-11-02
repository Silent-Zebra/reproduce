import torch
from torch.optim.optimizer import Optimizer, required


def normalize_param(W):
    return W / W.norm(2).clamp(min=1e-12)


multiobj_optims = {}


def register_multiobj_optim(name):
    """Decorator to register a new optimizer."""

    def register_multiobj_optim_cls(cls):
        if name in multiobj_optims:
            raise ValueError(
                'Cannot register duplicate optimizer ({})'.format(name))
        if not issubclass(cls, MultiObjSGD):
            raise ValueError(
                'Optimizer ({}: {}) must extend FairseqOptimizer'.format(name, cls.__name__))
        if cls.__name__ in multiobj_optims.values():
            # We use the optimizer class name as a unique identifier in
            # checkpoints, so all optimizer must have unique class names.
            raise ValueError(
                'Cannot register optimizer with duplicate class name ({})'.format(cls.__name__))
        multiobj_optims[name] = cls
        return cls

    return register_multiobj_optim_cls


class MultiObjSGD(Optimizer):
    """
    This optimizer works like SGD excepts:

    1. it stores gradient from an auxiliary task with `.save_constraints()`
    2. it uses those auxiliary gradients using `.apply_constraints()` before applying the update
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, always_project=True, reverse=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, frozen=False)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                "Nesterov momentum requires a momentum and zero dampening")
        super(MultiObjSGD, self).__init__(params, defaults)
        self.always_project = always_project
        self.reverse = reverse

    def __setstate__(self, state):
        super(MultiObjSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)


    def save_constraints(self):
        """This saves the gradients wrt. the source task/language/domain/whatever that is used as a constraint"""

        for group in self.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                # skip frozen parameters (TODO: remove this)
                if getattr(param_state, "frozen", False):
                    continue
                # Actually save the gradient
                param_state["constraint"] = torch.zeros_like(p.data)
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state["constraint"].add_(d_p)

    def apply_constraint(self, g_p, c_p):
        """Manipulate the gradient g_p using the gradient from the source task c_p"""
        raise NotImplementedError()

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                param_state = self.state[p]
                # skip frozen parameters
                if getattr(param_state, "frozen", False):
                    print("Skipping parameter of size", p.dim())
                    continue
                if momentum != 0:
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.zeros_like(
                            p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                # Apply constraint
                if "constraint" in param_state:
                    d_p = self.apply_constraint(
                        d_p, param_state["constraint"])

                p.data.add_(-group["lr"], d_p)

        return loss


@register_multiobj_optim("single")
class SingleObjSGD(MultiObjSGD):
    """Same as SGD ("single" task)"""

    def apply_constraint(self, g_p, c_p):
        return g_p


@register_multiobj_optim("avg")
class AvgMultiObjSGD(MultiObjSGD):
    """Average the gradients"""

    def apply_constraint(self, g_p, c_p):
        avg_p = 0.5 * (c_p + g_p)
        return avg_p


@register_multiobj_optim("ortho")
class OrthoMultiObjSGD(MultiObjSGD):
    """Project the gradient g_p on the hyperplane orthogonal to c_p"""

    def apply_constraint(self, g_p, c_p):
        c_unit = c_p / (c_p.norm(2) + 1e-10)
        dot = (g_p * c_unit).sum()
        # Only project if the gradients have negative dot product
        if self.always_project or dot.data <= 0:
            return g_p - dot * c_unit
        else:
            return g_p


@register_multiobj_optim("cwise-ortho")
class CwiseOrthoMultiObjSGD(MultiObjSGD):
    """Orthogonal projection but at the level of scalar parameters"""

    def apply_constraint(self, g_p, c_p):
        mask = torch.nn.functional.relu(torch.sign(g_p * c_p))
        return mask * g_p


@register_multiobj_optim("cosine-weighted")
class CosineWeightedMultiObjSGD(MultiObjSGD):
    """Weight the update by the (rectified) cosine similarity between the two gradients.
    Update in the direction of c_p"""

    def apply_constraint(self, g_p, c_p):
        c_unit = c_p / (c_p.norm(2) + 1e-10)
        g_unit = g_p / (g_p.norm(2) + 1e-10)
        cosine = (g_unit * c_unit).sum()
        return torch.nn.functional.relu(cosine) * g_p


@register_multiobj_optim("cosine-weighted-sum")
class CosineWeightedSumMultiObjSGD(MultiObjSGD):
    """Weight the update by the (rectified) cosine similarity between the two gradients.
    Update in the direction of g_p + c_p (see https://arxiv.org/abs/1812.02224)"""

    def apply_constraint(self, g_p, c_p):
        c_unit = c_p / (c_p.norm(2) + 1e-10)
        g_unit = g_p / (g_p.norm(2) + 1e-10)
        cosine = (g_unit * c_unit).sum()
        return torch.nn.functional.relu(cosine) * 0.5 * (g_p + c_p)


@register_multiobj_optim("colinear")
class ColinearMultiObjSGD(MultiObjSGD):
    """Project g_p on the direction of c_p (when the 2 are colinear)"""

    def apply_constraint(self, g_p, c_p):
        c_unit = c_p / (c_p.norm(2) + 1e-10)
        dot = (c_unit * g_p).sum()
        return torch.nn.functional.relu(dot) * c_unit


@register_multiobj_optim("same-contrib")
class SameContribMultiObjSGD(MultiObjSGD):
    """Here the update is a vector d such that Loss_1(x + d) - Loss_1(x) = Loss_2(x + d) - Loss_2(x)"""
    

    def apply_constraint(self, g_p, c_p):
        diff = g_p - c_p
        diff_norm = diff.norm(2) + 1e-10
        diff_unit = diff / diff_norm
        dot = (g_p * diff_unit).sum()
        return g_p - dot * diff_unit


@register_multiobj_optim("avg-ortho")
class AvgOrthoMultiObjSGD(MultiObjSGD):
    """Project g_p on the orthogonal of c_p, and c_p on the orthogonal of g_p, then average"""

    def apply_constraint(self, g_p, c_p):
        g_norm = g_p.norm(2)+1e-10
        c_norm = c_p.norm(2)+1e-10
        dot = (g_p * c_p).sum()
        if self.always_project or dot.data <= 0:
            g_unit = g_p / g_norm
            c_unit = c_p / c_norm
            g_proj_c = g_p - (g_p * c_unit).sum() * c_unit
            c_proj_g = c_p - (c_p * g_unit).sum() * g_unit
            return 0.5 * (g_proj_c + c_proj_g)
        else:
            # If the two are somewhat aligned, no need to project
            return g_p


class FullMultiObjSGD(MultiObjSGD):
    """Same as multiobj but the gradient manipulations are now done on the full gradient (not "per matrix")"""

    def compute_dot_and_norms(self):
        dot_val = 0
        c_p_norm_squared = 0
        g_p_norm_squared = 0
        for group in self.param_groups:

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                param_state = self.state[p]
                # skip frozen parameters
                if getattr(param_state, "frozen", False):
                    continue
                c_p = param_state["constraint"]
                c_p_norm_squared += (c_p * c_p).sum().data
                g_p_norm_squared += (d_p * d_p).sum().data
                dot_val += (d_p * c_p).sum().data
        return dot_val, c_p_norm_squared, g_p_norm_squared

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        dot_val, c_p_norm_squared, g_p_norm_squared = self.compute_dot_and_norms()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                param_state = self.state[p]
                # skip frozen parameters
                if getattr(param_state, "frozen", False):
                    continue
                if momentum != 0:
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.zeros_like(
                            p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                if "constraint" in param_state:
                    d_p = self.apply_constraint(
                        d_p, param_state["constraint"], dot_val, c_p_norm_squared, g_p_norm_squared)

                p.data.add_(-group["lr"], d_p)

        return loss


@register_multiobj_optim("full-ortho")
class FullOrthoMultiObjSGD(FullMultiObjSGD):

    def apply_constraint(self, g_p, c_p, dot_val, c_p_norm_squared, g_p_norm_squared):
        if self.always_project or dot_val <= 0:
            return g_p - dot_val / c_p_norm_squared * c_p
        else:
            return g_p


@register_multiobj_optim("full-nullify")
class FullNullifyMultiObjSGD(FullMultiObjSGD):

    def apply_constraint(self, g_p, c_p, dot_val, c_p_norm_squared, g_p_norm_squared):
        if dot_val <= 0:
            return torch.zeros_like(g_p)
        else:
            return c_p


@register_multiobj_optim("full-cosine-weighted")
class FullCosineWeightedMultiObjSGD(FullMultiObjSGD):

    def apply_constraint(self, g_p, c_p, dot_val, c_p_norm_squared, g_p_norm_squared):
        c_unit = c_p / (torch.sqrt(c_p_norm_squared) + 1e-10)
        g_unit = g_p / (torch.sqrt(g_p_norm_squared) + 1e-10)
        cosine = (g_unit * c_unit).sum()
        return torch.nn.functional.relu(cosine) * g_p
