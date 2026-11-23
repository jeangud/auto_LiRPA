#########################################################################
##   This file is part of the auto_LiRPA library, a core part of the   ##
##   α,β-CROWN (alpha-beta-CROWN) neural network verifier developed    ##
##   by the α,β-CROWN Team                                             ##
##                                                                     ##
##   Copyright (C) 2020-2025 The α,β-CROWN Team                        ##
##   Team leaders:                                                     ##
##          Faculty:   Huan Zhang <huan@huan-zhang.com> (UIUC)         ##
##          Student:   Xiangru Zhong <xiangru4@illinois.edu> (UIUC)    ##
##                                                                     ##
##   See CONTRIBUTORS for all current and past developers in the team. ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
import json
import math
import os
import numpy as np
import torch
from .utils import logger, eyeC
from .patches import Patches, patches_to_matrix, is_shape_used
from .linear_bound import LinearBound

from .concretize_func import constraints_solving, sort_out_constr_batches, construct_constraints

class Perturbation:
    r"""
    Base class for a perturbation specification. Please see examples
    at `auto_LiRPA/perturbations.py`.

    Examples:

    * `PerturbationLpNorm`: Lp-norm (p>=1) perturbation.

    * `PerturbationL0Norm`: L0-norm perturbation.

    * `PerturbationSynonym`: Synonym substitution perturbation for NLP.
    """

    def __init__(self):
        pass

    def set_eps(self, eps):
        self.eps = eps

    def concretize(self, x, A, sign=-1, aux=None):
        r"""
        Concretize bounds according to the perturbation specification.

        Args:
            x (Tensor): Input before perturbation.

            A (Tensor) : A matrix from LiRPA computation.

            sign (-1 or +1): If -1, concretize for lower bound; if +1, concretize for upper bound.

            aux (object, optional): Auxilary information for concretization.

        Returns:
            bound (Tensor): concretized bound with the shape equal to the clean output.
        """
        raise NotImplementedError

    def init(self, x, aux=None, forward=False):
        r"""
        Initialize bounds before LiRPA computation.

        Args:
            x (Tensor): Input before perturbation.

            aux (object, optional): Auxilary information.

            forward (bool): It indicates whether forward mode LiRPA is involved.

        Returns:
            bound (LinearBound): Initialized bounds.

            center (Tensor): Center of perturbation. It can simply be `x`, or some other value.

            aux (object, optional): Auxilary information. Bound initialization may modify or add auxilary information.
        """

        raise NotImplementedError


class PerturbationL0Norm(Perturbation):
    """Perturbation constrained by the L_0 norm.

    Assuming input data is in the range of 0-1.
    """

    def __init__(self, eps, x_L=None, x_U=None, ratio=1.0):
        self.eps = eps
        self.x_U = x_U
        self.x_L = x_L
        self.ratio = ratio

    def concretize(self, x, A, sign=-1, aux=None):
        if A is None:
            return None

        eps = math.ceil(self.eps)
        x = x.reshape(x.shape[0], -1, 1)
        center = A.matmul(x)

        x = x.reshape(x.shape[0], 1, -1)

        original = A * x.expand(x.shape[0], A.shape[-2], x.shape[2])
        neg_mask = A < 0
        pos_mask = A >= 0

        if sign == 1:
            A_diff = torch.zeros_like(A)
            A_diff[pos_mask] = A[pos_mask] - original[pos_mask]# changes that one weight can contribute to the value
            A_diff[neg_mask] = - original[neg_mask]
        else:
            A_diff = torch.zeros_like(A)
            A_diff[pos_mask] = original[pos_mask]
            A_diff[neg_mask] = original[neg_mask] - A[neg_mask]

        # FIXME: this assumes the input pixel range is between 0 and 1!
        A_diff, _= torch.sort(A_diff, dim = 2, descending=True)

        bound = center + sign * A_diff[:, :, :eps].sum(dim = 2).unsqueeze(2) * self.ratio

        return bound.squeeze(2)

    def init(self, x, aux=None, forward=False):
        # For other norms, we pass in the BoundedTensor objects directly.
        x_L = x
        x_U = x
        if not forward:
            return LinearBound(None, None, None, None, x_L, x_U), x, None
        batch_size = x.shape[0]
        dim = x.reshape(batch_size, -1).shape[-1]
        eye = torch.eye(dim).to(x.device).unsqueeze(0).repeat(batch_size, 1, 1)
        lw = eye.reshape(batch_size, dim, *x.shape[1:])
        lb = torch.zeros_like(x).to(x.device)
        uw, ub = lw.clone(), lb.clone()
        return LinearBound(lw, lb, uw, ub, x_L, x_U), x, None

    def __repr__(self):
        return 'PerturbationLpNorm(norm=0, eps={})'.format(self.eps)


class PerturbationLpNorm(Perturbation):
    """Perturbation constrained by the L_p norm."""
    def __init__(self, eps=0, norm=np.inf, x_L=None, x_U=None, eps_min=0, 
                 constraints=None, rearrange_constraints=False, no_return_inf=False, timer=None):
        r"""
        Initialize a p-norm perturbation instance.
        There are two ways to initialize it:
            -- x_L, x_U: (Higher priority)
            -- eps     : (Lower priority)
        If use eps to initialize it, the centroid x (or x0 as in the member attribute) will be
            passed into `init` and `concretize` function.  
        For the shape notations such as 'B' or 'X', please check the shape declaration 
            at the beginning of concretize_func.py

        Args:
            eps (Tensor): The epsilon tensor, it represents the pertubation added to a BoundedTensor.
            norm (int or torch.inf): The p in p-norm perturbation.
            x_L (Tensor): Lower bound of input box, shape (B, *input_shape[1:]).
            x_U (Tensor): Upper bound of input box, shape (B, *input_shape[1:]).
            eps_min ()
            constraints (Tuple[Tensor, Tensor] or None): 
                A tuple `(A, b)` representing per-batch linear constraints.
                - `A`: shape (B, N_constr, X)
                - `b`: shape (B, N_constr)
            rearrange_constraints (bool): 
                Whether to rearrange constraints for better solver performance. Default: False.
            no_return_inf (bool): 
                If True, infeasible batches will be excluded from `active_indices`.
                Otherwise, infeasible batches are still marked active. Default: False.
                Please check `constraints_solving` and `sort_out_constr_batches` for more details.
            timer (Timer):
                A timer recording the concretization time.
        """        
        self.eps = eps
        self.x0 = None

        # For p = inf, pre-compute x0 and eps would accerlerate the concretize function.
        if norm == np.inf and x_L is not None and x_U is not None:
            self.eps = (x_U - x_L) / 2
            self.x0 = (x_U + x_L) / 2
        
        # x0_act and eps_act stands for x0 and eps matrix for batches with active constraints
        self.x0_act = None          # shape (batchsize, *X_shape)
        self.eps_act = None         # shape (batchsize, *X_shape)
        # x0_sparse_act and eps_sparse_act are the active sparse x0 and eps matrix when sparse perturbation is enabled.
        # Check init_sparse_linf to see how sparse x0, eps, x0_act, eps_act are created.
        self.x0_sparse_act = None   # shape (batchsize, *X_sparse_shape)
        self.eps_sparse_act = None  # shape (batchsize, *X_sparse_shape)

        self.eps_min = eps_min
        self.norm = norm
        self.dual_norm = 1 if (norm == np.inf) else (np.float64(1.0) / (1 - 1.0 / self.norm))
        self.x_L = x_L
        self.x_U = x_U
        self.sparse = False

        self.timer = timer
        self.aux_lb = None
        self.aux_ub = None

        self.rearrange_constraints = rearrange_constraints

        # constraints is a tuple containing both the coefficient matrix and bias term
        # of the constraints. The constraints would appear in the form of:
        #                           A_c * x + b_c <= 0
        # Coefficient matrix will be reshaped into (batchsize, # of constraints,
        # input_dim). Bias term will be reshaped into (batchsize, # of constraints)
        # also see in `constraints_solving` in constraints_solver.py

        # Pre-process the constraints.
        self.constraints, self.sorted_out_batches = sort_out_constr_batches(x_L, x_U, constraints, 
                                                                            rearrange_constraints=rearrange_constraints,
                                                                            no_return_inf=no_return_inf)
        # The indices of hidden neurons to apply constraints.
        self.objective_indices = None   # shape: (batchsize, num_of_neurons)
        if self.constraints is None or self.constraints[0].numel() == 0:
            self._constraints_enable = False
        else:
            self._constraints_enable = True
        self.no_return_inf = no_return_inf

        self._use_grad = False

    def get_input_bounds(self, x, A):
        if self.sparse:
            if self.x_L_sparse.shape[-1] == A.shape[-1]:
                x_L, x_U = self.x_L_sparse, self.x_U_sparse
                act_x0, act_eps = self.x0_sparse_act, self.eps_sparse_act
            else:
                # In backward mode, A is not sparse.
                x_L, x_U = self.x_L, self.x_U
                act_x0, act_eps = self.x0_act, self.eps_act
        else:
            x_L = x - self.eps if self.x_L is None else self.x_L
            x_U = x + self.eps if self.x_U is None else self.x_U
            act_x0, act_eps = self.x0_act, self.eps_act
        return x_L, x_U, act_x0, act_eps

    def get_constraints(self, A):
        if self.constraints is None:
            return None
        if self.sparse and self.x_L_sparse.shape[-1] == A.shape[-1]:
            return self.constraints_sparse
        else:
            return self.constraints

    def concretize_matrix(self, x, A, sign, constraints=None):
        # If A is an identity matrix, we will handle specially.
        if not isinstance(A, eyeC):
            # A has (Batch, spec, *input_size). For intermediate neurons, spec is *neuron_size.
            A = A.reshape(A.shape[0], A.shape[1], -1)

        if self.norm == np.inf:
            x_L, x_U, act_x0, act_eps = self.get_input_bounds(x, A)
            if constraints is None:
                constraints = self.get_constraints(A)
            # The original code for matrix concretize has been merged into `constraints_solving`.
            # Pick out auxiliary bound based on the sign.
            aux_bounds = self.aux_lb if sign == -1.0 else self.aux_ub
            results = constraints_solving(x_L, x_U, A, constraints, sign,
                                        sorted_out_batches=self.sorted_out_batches, objective_indices=self.objective_indices, 
                                        constraints_enable=self._constraints_enable, no_return_inf=self.no_return_inf,
                                        timer=self.timer, 
                                        aux_bounds=aux_bounds, act_x0=act_x0, act_eps=act_eps,
                                        use_grad=self._use_grad)
            
            if self.no_return_inf:
                # return: (bound, infeasible_bounds)
                bound = results[0]
                infeasible_bounds = results[1]
                self.add_infeasible_batches(infeasible_bounds)
            else:
                # return: bound
                bound = results
        else:
            x = x.reshape(x.shape[0], -1, 1)
            if not isinstance(A, eyeC):
                # Find the upper and lower bounds via dual norm.
                deviation = A.norm(self.dual_norm, -1) * self.eps
                bound = A.matmul(x) + sign * deviation.unsqueeze(-1)
            else:
                # A is an identity matrix. Its norm is all 1.
                bound = x + sign * self.eps
        bound = bound.squeeze(-1)
        return bound

    def concretize_patches(self, x, A, sign):
        if self.norm == np.inf:
            x_L, x_U, _, _,  = self.get_input_bounds(x, A)

            # Here we should not reshape
            # Find the uppwer and lower bound similarly to IBP.
            center = (x_U + x_L) / 2.0
            diff = (x_U - x_L) / 2.0

            if not A.identity == 1:
                bound = A.matmul(center)
                bound_diff = A.matmul(diff, patch_abs=True)
                if sign == 1:
                    bound += bound_diff
                elif sign == -1:
                    bound -= bound_diff
                else:
                    raise ValueError("Unsupported Sign")
            else:
                # A is an identity matrix. No need to do this matmul.
                bound = center + sign * diff
            return bound
        else:  # Lp norm
            input_shape = x.shape
            if not A.identity:
                # Find the upper and lower bounds via dual norm.
                # matrix has shape
                # (batch_size, out_c * out_h * out_w, input_c, input_h, input_w)
                # or (batch_size, unstable_size, input_c, input_h, input_w)
                matrix = patches_to_matrix(
                    A.patches, input_shape, A.stride, A.padding, A.output_shape,
                    A.unstable_idx)
                # Note that we should avoid reshape the matrix.
                # Due to padding, matrix cannot be reshaped without copying.
                deviation = matrix.norm(p=self.dual_norm, dim=(-3,-2,-1)) * self.eps
                # Bound has shape (batch, out_c * out_h * out_w) or (batch, unstable_size).
                bound = torch.einsum('bschw,bchw->bs', matrix, x) + sign * deviation
                if A.unstable_idx is None:
                    # Reshape to (batch, out_c, out_h, out_w).
                    bound = bound.view(matrix.size(0), A.patches.size(0),
                                       A.patches.size(2), A.patches.size(3))
            else:
                # A is an identity matrix. Its norm is all 1.
                bound = x + sign * self.eps
            return bound

    def concretize(self, x, A, sign=-1, constraints=None, aux=None):
        """Given an variable x and its bound matrix A, compute worst case bound according to Lp norm."""
        if A is None:
            return None
        if isinstance(A, eyeC) or isinstance(A, torch.Tensor):
            ret = self.concretize_matrix(x, A, sign, constraints)
        elif isinstance(A, Patches):
            ret = self.concretize_patches(x, A, sign)
        else:
            raise NotImplementedError()
        if ret.ndim > 2:
            ret = ret.reshape(A.shape[1], -1)
        return ret

    def init_sparse_linf(self, x, x_L, x_U):
        """ Sparse Linf perturbation where only a few dimensions are actually perturbed"""
        self.sparse = True
        batch_size = x_L.shape[0]
        perturbed = (x_U > x_L).int()
        logger.debug(f'Perturbed: {perturbed.sum()}')
        lb = ub = x_L * (1 - perturbed) # x_L=x_U holds when perturbed=0
        perturbed = perturbed.view(batch_size, -1)
        index = torch.cumsum(perturbed, dim=-1)
        dim = max(perturbed.view(batch_size, -1).sum(dim=-1).max(), 1)
        self.x_L_sparse = torch.zeros(batch_size, dim + 1).to(x_L)
        self.x_L_sparse.scatter_(dim=-1, index=index, src=(x_L - lb).view(batch_size, -1), reduce='add')
        self.x_U_sparse = torch.zeros(batch_size, dim + 1).to(x_U)
        self.x_U_sparse.scatter_(dim=-1, index=index, src=(x_U - ub).view(batch_size, -1), reduce='add')
        self.x_L_sparse, self.x_U_sparse = self.x_L_sparse[:, 1:], self.x_U_sparse[:, 1:]
        
        # --- create x0 and eps for Lp Norm
        self.x0_sparse = (self.x_L_sparse + self.x_U_sparse) / 2
        self.eps_sparse = (self.x_U_sparse - self.x_L_sparse) / 2
        if self.sorted_out_batches is not None:
            active_indices = self.sorted_out_batches["active_indices"]
            self.x0_sparse_act = self.x0_sparse[active_indices].unsqueeze(-1)
            self.eps_sparse_act = self.eps_sparse[active_indices].unsqueeze(-1)

        lw = torch.zeros(batch_size, dim + 1, perturbed.shape[-1], device=x.device)
        perturbed = perturbed.to(torch.get_default_dtype())
        lw.scatter_(dim=1, index=index.unsqueeze(1), src=perturbed.unsqueeze(1))
        lw = uw = lw[:, 1:, :].view(batch_size, dim, *x.shape[1:])
        print(f'Using Linf sparse perturbation. Perturbed dimensions: {dim}.')
        print(f'Avg perturbation: {(self.x_U_sparse - self.x_L_sparse).mean()}')

        # When sparse linf is enabled, the input x perturbation would change its shape
        # Hence, the shape of constraints_A should change accordingly.
        # But for the final layer, we still need the dense linf, and use the original (dense) constraints
        if self.constraints is not None:
            # constraints_A: (batchsize, n_constraints, x_dim)
            constraints_A, constraints_b = self.constraints
            # reversed_lw: (batchsize, x_dim, sparse_dim)
            reversed_lw = lw.reshape((batch_size, dim, -1)).transpose(1, 2)
            lb_act = lb
            # When pre-processing the constraints, we only kept the active ones.
            # Hence, reversed_lw and lb_act should also be re-collected.
            active_indices = self.sorted_out_batches["active_indices"]
            reversed_lw = reversed_lw[active_indices]
            lb_act = lb_act[active_indices]
            # reversed lw will sort out the sparse dimensions out of all x dimension
            new_constraints_A = constraints_A.bmm(reversed_lw)
            # Besides original constraint_b, should also include the a*x terms where x is not perturbed
            # new_constraints_b = constraints_b + torch.einsum("bcx, bx -> bc", constraints_A, lb_act.flatten(1))
            new_constraints_b = constraints_b
            self.constraints_sparse = (new_constraints_A, new_constraints_b)
        return LinearBound(
            lw, lb, uw, ub, x_L, x_U), x, None

    def init(self, x, aux=None, forward=False):
        self.sparse = False
        if self.norm == np.inf:
            x_L = x - self.eps if self.x_L is None else self.x_L
            x_U = x + self.eps if self.x_U is None else self.x_U
        else:
            if int(os.environ.get('AUTOLIRPA_L2_DEBUG', 0)) == 1:
                # FIXME Experimental code. Need to change the IBP code also.
                x_L = x - self.eps if self.x_L is None else self.x_L
                x_U = x + self.eps if self.x_U is None else self.x_U
            else:
                # FIXME This causes confusing lower bound and upper bound
                # For other norms, we pass in the BoundedTensor objects directly.
                x_L = x_U = x

        if self.x_L is not None and self.x_U is not None:
            self.x0 = (self.x_L + self.x_U) / 2
        else:
            self.x0 = x.data
        if self.sorted_out_batches is not None and self.sorted_out_batches.get("active_indices") is not None:
            active_indices = self.sorted_out_batches["active_indices"]
            self.x0_act = self.x0[active_indices].flatten(1).unsqueeze(-1)
            self.eps_act = self.eps[active_indices].flatten(1).unsqueeze(-1)

        if not forward:
            return LinearBound(
                None, None, None, None, x_L, x_U), x, None
        if (self.norm == np.inf and x_L.numel() > 1
                and (x_L == x_U).sum() > 0.5 * x_L.numel()):
            return self.init_sparse_linf(x, x_L, x_U)

        batch_size = x.shape[0]
        dim = x.reshape(batch_size, -1).shape[-1]
        lb = ub = torch.zeros_like(x)
        eye = torch.eye(dim).to(x).expand(batch_size, dim, dim)
        lw = uw = eye.reshape(batch_size, dim, *x.shape[1:])
        return LinearBound(
            lw, lb, uw, ub, x_L, x_U), x, None

    def add_infeasible_batches(self, infeasible_batches):
        r"""
        Synchronize the `infeasible_batches` tensor between the global graph and the local perturbation node.

        If the computation graph includes multiple perturbed inputs, the BoundedModule (entire network) maintains a global
        `infeasible_batches` tensor, while each perturbed input (root) keeps its own local copy.

        - Before concretization: copy the global tensor to the local one.
        - After concretization: propagate updates from the local tensor back to the global tensor.

        Args:
            infeasible_batches: A boolean vector with shape (batchsize, ). A True value indicates that a batch is infeasible
                                given its constraints.
        """
        if self.constraints is not None and infeasible_batches is not None and infeasible_batches.any():
            if self.sorted_out_batches["infeasible_batches"] is None:
                self.sorted_out_batches["infeasible_batches"] = infeasible_batches
            else:
                infeasible_batches = infeasible_batches | self.sorted_out_batches["infeasible_batches"]
                self.sorted_out_batches["infeasible_batches"] = infeasible_batches
            
            active_indices = self.sorted_out_batches["active_indices"]
            B_act = active_indices.numel()
            active_feasible_mask = (~infeasible_batches)[active_indices]
            if active_feasible_mask.sum() < B_act:
                self.sorted_out_batches["active_indices"] = active_indices[active_feasible_mask]
                self.x0_act = self.x0_act[active_feasible_mask]
                self.eps_act = self.eps_act[active_feasible_mask]
                constraints_A, constraints_b = self.constraints
                constraints_A = constraints_A[active_feasible_mask]
                constraints_b = constraints_b[active_feasible_mask]
                self.constraints = (constraints_A, constraints_b)

    def add_objective_indices(self, objective_indices):
        if self.constraints is not None:
            self.objective_indices = objective_indices

    @property
    def constraints_enable(self):
        '''
        Enable / Disable the constrained concretize mode, regardless whether constraints is None or not. 
        '''
        return self._constraints_enable
    
    @constraints_enable.setter
    def constraints_enable(self, enable: bool):
        self._constraints_enable = enable

    @constraints_enable.deleter
    def constraints_enable(self):
        del self._constraints_enable  

    @property
    def use_grad(self):
        '''
        Enable / Disable the constrained concretize with gradient. 
        '''
        return self._use_grad
    
    @use_grad.setter
    def use_grad(self, use_grad: bool):
        self._use_grad = use_grad

    @use_grad.deleter
    def use_grad(self):
        del self._use_grad  

    def add_aux_bounds(self, aux_lb, aux_ub):
        self.aux_lb = aux_lb
        self.aux_ub = aux_ub

    def clear_aux_bounds(self):
        self.aux_lb = None
        self.aux_ub = None

    def reset_constraints(self, constraints, decision_thresh):
        r"""
        Reset the constraints of this perturbation. Also will call `sort_out_constr_batches` to preprocess the constraints.
        Be sure not to reset with the same constraints input repeatedly.
        """
        # We have to enable the gradient computation for the constraints
        # when using constraints_solving within alpha crown.
        self.use_grad = True
        constraints = construct_constraints(constraints[0], constraints[1], decision_thresh, self.x_L.shape[0], self.x_L.flatten(1).shape[1])
        self.constraints, self.sorted_out_batches = sort_out_constr_batches(self.x_L, self.x_U, constraints, 
                                                                            rearrange_constraints=self.rearrange_constraints,
                                                                            no_return_inf=self.no_return_inf)

    def __repr__(self):
        if self.norm == np.inf:
            if self.x_L is None and self.x_U is None:
                return f'PerturbationLpNorm(norm=inf, eps={self.eps})'
            else:
                return f'PerturbationLpNorm(norm=inf, eps={self.eps}, x_L={self.x_L}, x_U={self.x_U})'
        else:
            return f'PerturbationLpNorm(norm={self.norm}, eps={self.eps})'


class PerturbationLinear(Perturbation):
    """
    Perturbation defined by a Linear transformation.
    args:
        lower_A: Lower bound matrix of shape (B, output_dim, input_dim)
        upper_A: Upper bound matrix of shape (B, output_dim, input_dim)
        lower_b: Lower bound bias of shape (B, output_dim)
        upper_b: Upper bound bias of shape (B, output_dim)
        input_lb: Input lower bound of shape (B, input_dim)
        input_ub: Input upper bound of shape (B, input_dim)
        x_L: Output lower bound of shape (B, output_dim)
        x_U: Output upper bound of shape (B, output_dim)

        x_L and x_U can be None, in which case they will be computed from the other parameters.    
    """
    def __init__(self, lower_A, upper_A, lower_b, upper_b, input_lb, input_ub, x_L=None, x_U=None):
        super(PerturbationLinear, self).__init__()
        self.lower_A = lower_A
        self.upper_A = upper_A
        self.lower_b = lower_b.unsqueeze(-1) if lower_b is not None else None
        self.upper_b = upper_b.unsqueeze(-1) if upper_b is not None else None
        self.input_lb = input_lb.unsqueeze(-1) if input_lb is not None else None
        self.input_ub = input_ub.unsqueeze(-1) if input_ub is not None else None
        if x_L is None or x_U is None:
            mid = (self.input_lb + self.input_ub) / 2
            diff = (self.input_ub - self.input_lb) / 2
            self.x_U = (self.upper_A @ mid + torch.abs(self.upper_A) @ diff + self.upper_b).squeeze(-1)
            self.x_L = (self.lower_A @ mid - torch.abs(self.lower_A) @ diff + self.lower_b).squeeze(-1)
        else:
            self.x_L = x_L
            self.x_U = x_U

    def concretize(self, x, A, sign=-1, aux=None):
        if A is None:
            return None
        else:
            A_pos = torch.clamp(A, min=0)
            A_neg = torch.clamp(A, max=0)

            center = (self.input_lb + self.input_ub) / 2
            diff = (self.input_ub - self.input_lb) / 2

            if sign == 1:
                composite_A = A_pos @ self.upper_A + A_neg @ self.lower_A
                composite_b = A_pos @ self.upper_b + A_neg @ self.lower_b
                bound = composite_A @ center + torch.abs(composite_A) @ diff + composite_b
            else:
                composite_A = A_pos @ self.lower_A + A_neg @ self.upper_A
                composite_b = A_pos @ self.lower_b + A_neg @ self.upper_b
                bound = composite_A @ center - torch.abs(composite_A) @ diff + composite_b
            return bound.squeeze(-1)

    def init(self, x, aux=None, forward=False):
        if not forward:
            return LinearBound(None, None, None, None, self.x_L, self.x_U), x, None
        else:
            raise NotImplementedError("Linear perturbation does not support forward mode.")


class PerturbationPWL(Perturbation):
    """Perturbation with 2-piece piecewise-linear bounds over a scalar param t.

    Each input dimension x_i is bounded by:
        L_i(t) <= x_i <= U_i(t),   t in [t_min, t_max]

    where L_i and U_i are 2-piece linear functions with per-dimension,
    per-bound breakpoints bp_lower_i and bp_upper_i (they may differ):

        For the lower bound (breakpoint bp_lower_i):
            Piece 0 (t in [t_min, bp_lower_i]):
                L_i(t) = lower_A[b,0,i]*t + lower_b[b,0,i]
            Piece 1 (t in [bp_lower_i, t_max]):
                L_i(t) = lower_A[b,1,i]*t + lower_b[b,1,i]

        For the upper bound (breakpoint bp_upper_i):
            Piece 0 (t in [t_min, bp_upper_i]):
                U_i(t) = upper_A[b,0,i]*t + upper_b[b,0,i]
            Piece 1 (t in [bp_upper_i, t_max]):
                U_i(t) = upper_A[b,1,i]*t + upper_b[b,1,i]

    Args:
        t_min (float): Global lower bound on parameter t.
        t_max (float): Global upper bound on parameter t.
        lower_breakpoints: (B, D) — per-dim breakpoint for the lower bound.
        upper_breakpoints: (B, D) — per-dim breakpoint for the upper bound.
        lower_A: (B, 2, D) — slopes for lower bound (2 pieces).
        lower_b: (B, 2, D) — intercepts for lower bound (2 pieces).
        upper_A: (B, 2, D) — slopes for upper bound (2 pieces).
        upper_b: (B, 2, D) — intercepts for upper bound (2 pieces).
        x_L, x_U: Optional pre-computed concrete bounds (B, D).
    """

    def __init__(self, t_min, t_max, *,
                 lower_breakpoints, upper_breakpoints,
                 lower_A, upper_A, lower_b, upper_b,
                 x_L=None, x_U=None):
        super(PerturbationPWL, self).__init__()

        bp_lower = lower_breakpoints
        bp_upper = upper_breakpoints

        # --- Validation ---
        if t_min >= t_max:
            raise ValueError(f"t_min ({t_min}) must be < t_max ({t_max})")
        if bp_lower.ndim != 2:
            raise ValueError(
                f"lower_breakpoints must be 2-D (B, D), "
                f"got shape {bp_lower.shape}")
        if bp_upper.shape != bp_lower.shape:
            raise ValueError(
                f"upper_breakpoints shape {bp_upper.shape} must match "
                f"lower_breakpoints shape {bp_lower.shape}")
        B, D = bp_lower.shape
        for name, tensor in [('lower_A', lower_A), ('lower_b', lower_b),
                              ('upper_A', upper_A), ('upper_b', upper_b)]:
            if tensor.shape != (B, 2, D):
                raise ValueError(
                    f"{name} must have shape (B={B}, 2, D={D}), "
                    f"got {tensor.shape}")
        if not (bp_lower >= t_min).all() or not (bp_lower <= t_max).all():
            raise ValueError("All lower breakpoints must be in [t_min, t_max]")
        if not (bp_upper >= t_min).all() or not (bp_upper <= t_max).all():
            raise ValueError("All upper breakpoints must be in [t_min, t_max]")

        L0_bp = lower_A[:, 0] * bp_lower + lower_b[:, 0]
        L1_bp = lower_A[:, 1] * bp_lower + lower_b[:, 1]
        U0_bp = upper_A[:, 0] * bp_upper + upper_b[:, 0]
        U1_bp = upper_A[:, 1] * bp_upper + upper_b[:, 1]

        if not torch.allclose(L0_bp, L1_bp, atol=1e-4):
            raise ValueError("PWL pieces for lower bound must be continuous at bp_lower.")
        if not torch.allclose(U0_bp, U1_bp, atol=1e-4):
            raise ValueError("PWL pieces for upper bound must be continuous at bp_upper.")

        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.bp_lower = bp_lower                # (B, D)
        self.bp_upper = bp_upper                # (B, D)
        self.pwl_lower_A = lower_A              # (B, 2, D)
        self.pwl_upper_A = upper_A              # (B, 2, D)
        self.pwl_lower_b = lower_b              # (B, 2, D)
        self.pwl_upper_b = upper_b              # (B, 2, D)
        self.D = D

        # --- Concrete bounds ---
        if x_L is None or x_U is None:
            x_L_c, x_U_c = self._compute_concrete_bounds()
            self.x_L = x_L if x_L is not None else x_L_c
            self.x_U = x_U if x_U is not None else x_U_c
        else:
            self.x_L = x_L
            self.x_U = x_U

        assert (self.x_L <= self.x_U + 1e-6).all(), (
            f"x_L > x_U, max violation: {(self.x_L - self.x_U).max().item()}")

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------
    def _eval_at_t(self, t):
        """Evaluate L_i(t) and U_i(t) for all dims.

        Args:
            t: scalar or (B,) tensor.
        Returns:
            L, U: each (B, D).
        """
        t_tensor = torch.as_tensor(t, dtype=self.bp_lower.dtype, device=self.bp_lower.device)
        if t_tensor.ndim == 0:
            t_tensor = t_tensor.expand(self.bp_lower.shape[0])  # (B,)

        # Independent masks: True where piece 0 applies (t <= bp_i)
        mask_L = (t_tensor.unsqueeze(-1) <= self.bp_lower)  # (B, D)
        mask_U = (t_tensor.unsqueeze(-1) <= self.bp_upper)  # (B, D)

        t_exp = t_tensor.unsqueeze(-1)  # (B, 1)
        L0 = self.pwl_lower_A[:, 0] * t_exp + self.pwl_lower_b[:, 0]  # (B,D)
        L1 = self.pwl_lower_A[:, 1] * t_exp + self.pwl_lower_b[:, 1]
        U0 = self.pwl_upper_A[:, 0] * t_exp + self.pwl_upper_b[:, 0]
        U1 = self.pwl_upper_A[:, 1] * t_exp + self.pwl_upper_b[:, 1]

        L = torch.where(mask_L, L0, L1)
        U = torch.where(mask_U, U0, U1)
        return L, U

    def _worst_at_breakpoints(self):
        """Lowest value of L_i and highest value of U_i at their breakpoints.

        The two pieces are only required to agree at the breakpoint up to a
        tolerance (see the constructor). Piece 0 gives the value at the
        breakpoint; piece 1 gives the limit from the right, which the bound
        approaches for t just above the breakpoint (if any such t is in
        the interval). Both count.

        Returns:
            L_bp, U_bp: each (B, D).
        """
        L_bp = self.pwl_lower_A[:, 0] * self.bp_lower + self.pwl_lower_b[:, 0]
        L_right = self.pwl_lower_A[:, 1] * self.bp_lower + self.pwl_lower_b[:, 1]
        U_bp = self.pwl_upper_A[:, 0] * self.bp_upper + self.pwl_upper_b[:, 0]
        U_right = self.pwl_upper_A[:, 1] * self.bp_upper + self.pwl_upper_b[:, 1]
        L_bp = torch.where(self.bp_lower < self.t_max, torch.min(L_bp, L_right), L_bp)
        U_bp = torch.where(self.bp_upper < self.t_max, torch.max(U_bp, U_right), U_bp)
        return L_bp, U_bp

    def _compute_concrete_bounds(self):
        """Concrete box bounds by evaluating at t_min, t_max, and each bp."""
        L_min, U_min = self._eval_at_t(self.t_min)
        L_max, U_max = self._eval_at_t(self.t_max)

        # For bp evaluation, each dim uses its own breakpoint value.
        L_bp, U_bp = self._worst_at_breakpoints()

        x_L = torch.min(torch.min(L_min, L_bp), L_max)
        x_U = torch.max(torch.max(U_min, U_bp), U_max)
        return x_L, x_U


    # -----------------------------------------------------------------
    # Concretize (backward mode)
    # -----------------------------------------------------------------
    def concretize(self, x, A, sign=-1, aux=None):
        """Concretize bounds given backward A matrix.

        For sign=-1 (lower): compute min_{t,x} A·x  s.t. L(t)<=x<=U(t)
        For sign=+1 (upper): compute max_{t,x} A·x  s.t. L(t)<=x<=U(t)

        The objective is piecewise-linear in t with kinks at the
        per-dimension breakpoints of the lower and of the upper bound. Its
        optimum is at one of the 2D+2 candidate points
        {t_min, bp_lower_1..D, bp_upper_1..D, t_max}. The objective is
        evaluated at all of them at once (see `_concretize_matrix`) and
        the best value is returned.

        Supports three A representations:
          - torch.Tensor (dense matrix, shape (B, H, D))
          - eyeC (identity matrix)
          - Patches (sparse convolutional representation)

        Args:
            x: center input tensor (B, *input_shape).
            A: backward coefficients — Tensor, eyeC, or Patches.
            sign: -1 for lower, +1 for upper.
        Returns:
            bound: (B, H) for dense/eyeC, or patches-shaped output.
        """
        if A is None:
            return None

        if isinstance(A, eyeC):
            return self._concretize_identity(sign)
        elif isinstance(A, Patches):
            ret = self._concretize_patches(x, A, sign)
            if ret.ndim > 2:
                ret = ret.reshape(A.shape[1], -1)
            return ret
        else:
            return self._concretize_matrix(A, sign)

    def _concretize_identity(self, sign):
        """Concretize when A is the identity (eyeC).

        With A = I the bound is simply min_t L(t) = x_L (for lower)
        or max_t U(t) = x_U (for upper), already pre-computed.
        """
        if sign == -1:
            return self.x_L.reshape(self.x_L.shape[0], -1)
        else:
            return self.x_U.reshape(self.x_U.shape[0], -1)

    def _concretize_matrix(self, A, sign):
        """Dense-matrix concretize path.

        For sign=-1 the objective for output h at parameter t is
            g_h(t) = sum_j [ A+_hj * L_j(t) + A-_hj * U_j(t) ]
        (L and U swapped for sign=+1). Each of the 2D terms is linear in t
        on either side of its own breakpoint:
            c * (a0 * t + b0)  for t <= bp,     c * (a1 * t + b1)  for t > bp.
        With the terms sorted by breakpoint, the terms past their
        breakpoint at a given t are a prefix of that order, so
            g_h(t) = t * (S0_h + P_h[k]) + (T0_h + Q_h[k]),
        where S0, T0 sum c*a0 and c*b0 over all terms, P and Q are the
        prefix sums of c*(a1-a0) and c*(b1-b0), and k is the number of
        breakpoints strictly below t. One sort and two prefix sums give g
        at every candidate t, instead of one pass over the D inputs per
        candidate. Sums are accumulated in double precision.

        The pieces of a bound may differ at its breakpoint by the tolerance
        the constructor allows, so g can jump there. Each candidate is
        therefore evaluated twice: with the terms whose breakpoint equals t
        still on piece 0 (the value at t), and with those terms on piece 1
        (the limit from the right, approached for t just above). Between
        consecutive breakpoints g is linear, so the best of these values is
        the infimum (supremum) of g over the interval.

        Args:
            A: (B, H, D) dense backward coefficients.
            sign: -1 for lower, +1 for upper.
        Returns:
            bound: (B, H).
        """
        B, H, D = A.shape
        if D == 1 and self.D > 1:
            # Forward mode concretization: A is the coefficient for the 1D scalar t
            A_pos = A.clamp(min=0)
            A_neg = A.clamp(max=0)
            if sign == -1:
                return (A_pos * self.t_min + A_neg * self.t_max).squeeze(-1)
            else:
                return (A_pos * self.t_max + A_neg * self.t_min).squeeze(-1)

        work = torch.float64
        bp = torch.cat([self.bp_lower, self.bp_upper], dim=1)       # (B, 2D)
        order = torch.argsort(bp, dim=1)
        bp_sorted = torch.gather(bp, 1, order)

        def _sorted_terms(lower, upper):
            return torch.gather(
                torch.cat([lower, upper], dim=1).to(work), 1, order)

        # Piece 0 of every term, and what changes past its breakpoint
        a0 = _sorted_terms(self.pwl_lower_A[:, 0], self.pwl_upper_A[:, 0])
        b0 = _sorted_terms(self.pwl_lower_b[:, 0], self.pwl_upper_b[:, 0])
        da = _sorted_terms(self.pwl_lower_A[:, 1], self.pwl_upper_A[:, 1]) - a0
        db = _sorted_terms(self.pwl_lower_b[:, 1], self.pwl_upper_b[:, 1]) - b0

        # Candidates, in the precision the bounds are stored in:
        # t_min, every breakpoint, t_max                     (B, 2D + 2)
        ends = bp.new_tensor([self.t_min, self.t_max]).expand(B, 2)
        cand = torch.cat([ends[:, :1], bp_sorted, ends[:, 1:]], dim=1)
        # Number of breakpoints strictly below each candidate: those terms
        # are on piece 1 (piece 0 applies for t <= bp, as in `_eval_at_t`)
        num_past = torch.searchsorted(bp_sorted, cand, right=False)
        # Limit from the right: the terms whose breakpoint equals the
        # candidate are past it too. There is nothing to the right of t_max.
        num_past_right = torch.where(
            cand < ends[:, 1:],
            torch.searchsorted(bp_sorted, cand, right=True), num_past)
        num_past = torch.cat([num_past, num_past_right], dim=1)      # (B, 4D+4)
        cand = torch.cat([cand, cand], dim=1).to(work).unsqueeze(1)  # (B, 1, 4D+4)

        best = A.new_empty(B, H)
        # Bound the size of the (B, rows, 2D) temporaries
        rows = max(1, int(2 ** 22 // max(1, B * 2 * D)))
        for start in range(0, H, rows):
            A_c = A[:, start:start + rows].to(work)                  # (B, h, D)
            A_pos = A_c.clamp(min=0)
            A_neg = A_c.clamp(max=0)
            # Coefficient of L_j(t), then of U_j(t)
            if sign == -1:
                coef = torch.cat([A_pos, A_neg], dim=2)              # (B, h, 2D)
            else:
                coef = torch.cat([A_neg, A_pos], dim=2)
            index = order.unsqueeze(1).expand(-1, coef.shape[1], -1)
            coef = torch.gather(coef, 2, index)

            slope0 = (coef * a0.unsqueeze(1)).sum(dim=-1, keepdim=True)
            const0 = (coef * b0.unsqueeze(1)).sum(dim=-1, keepdim=True)
            zero = coef.new_zeros(B, coef.shape[1], 1)
            slope = torch.cat(
                [zero, torch.cumsum(coef * da.unsqueeze(1), dim=-1)], dim=-1)
            const = torch.cat(
                [zero, torch.cumsum(coef * db.unsqueeze(1), dim=-1)], dim=-1)

            index = num_past.unsqueeze(1).expand(-1, coef.shape[1], -1)
            values = (cand * (slope0 + torch.gather(slope, 2, index))
                      + const0 + torch.gather(const, 2, index))      # (B, h, 4D+4)
            if sign == -1:
                best[:, start:start + rows] = values.min(dim=-1).values.to(A.dtype)
            else:
                best[:, start:start + rows] = values.max(dim=-1).values.to(A.dtype)

        return best

    def _concretize_patches(self, x, A, sign):
        """Concretize bounds when A is a Patches object.

        The patches are expanded to a dense (B, H, D) matrix (H outputs, or
        only the unstable ones for sparse patches) and handed to
        `_concretize_matrix`.

        Args:
            x: center input tensor (B, C, H, W).
            A: Patches object.
            sign: -1 for lower, +1 for upper.
        Returns:
            bound: (B, out_c, out_h, out_w) or (B, unstable_size).
        """
        if A.identity == 1:
            # Identity patches: A·x = x, so the bound is min_t L(t) or
            # max_t U(t) for every input, reached at an end or a breakpoint.
            x_L, x_U = self._compute_concrete_bounds()
            return (x_L if sign == -1 else x_U).view(x.shape)
        if is_shape_used(A.output_padding):
            # `Patches.to_matrix` does not support output padding
            return self._concretize_patches_loop(x, A, sign)

        # Expand a limited number of outputs at a time: the dense matrix has
        # D entries per output.
        patches = A.patches
        sparse = A.unstable_idx is not None
        if sparse:
            rows_per_slice = 1
        else:
            # (out_c, B, out_h, out_w, ...): one slice is one output channel
            rows_per_slice = patches.shape[2] * patches.shape[3]
        step = max(1, int(2 ** 26 // max(1, self.D * rows_per_slice)))
        bounds = []
        for start in range(0, patches.shape[0], step):
            stop = start + step
            matrix = patches_to_matrix(
                patches[start:stop], x.shape, A.stride, A.padding,
                A.output_shape,
                tuple(idx[start:stop] for idx in A.unstable_idx) if sparse else None,
                A.inserted_zeros)              # (B, rows, C, H_in, W_in)
            B, rows = matrix.shape[:2]
            bounds.append(
                self._concretize_matrix(matrix.reshape(B, rows, -1), sign))
        bound = torch.cat(bounds, dim=1)
        if not sparse:
            bound = bound.view(
                bound.shape[0], patches.shape[0], patches.shape[2], patches.shape[3])
        return bound

    def _concretize_patches_loop(self, x, A, sign):
        """Reference for `_concretize_patches`: one pass per candidate t.

        Slow (2D+2 `Patches.matmul` pairs); kept to test the vectorized
        path against, and for patches that cannot be expanded to a matrix.

        Uses the center/diff decomposition:
            min_x A·x  s.t. L<=x<=U  =  A·center - |A|·diff
        where center = (L+U)/2, diff = (U-L)/2.

        This avoids materializing the full dense A matrix by using
        Patches.matmul() which operates via efficient unfolding.

        Args:
            x: center input tensor (B, C, H, W).
            A: Patches object.
            sign: -1 for lower, +1 for upper.
        Returns:
            bound: (B, out_c, out_h, out_w) or (B, unstable_size).
        """
        spatial_shape = x.shape  # (B, C, H, W)

        def _bound_at_t_patches(L, U):
            """Compute bound at a candidate t using patches matmul."""
            # L, U are flat (B, D). Reshape to spatial dims.
            L_spatial = L.view(spatial_shape)
            U_spatial = U.view(spatial_shape)

            center = (U_spatial + L_spatial) / 2.0
            diff = (U_spatial - L_spatial) / 2.0

            if A.identity == 1:
                # Identity patches: A·x = x
                return center + sign * diff
            else:
                bound = A.matmul(center)
                bound_diff = A.matmul(diff, patch_abs=True)
                return bound + sign * bound_diff

        # Evaluate at t_min and t_max
        L_min, U_min = self._eval_at_t(self.t_min)
        L_max, U_max = self._eval_at_t(self.t_max)
        best = _bound_at_t_patches(L_min, U_min)
        cand = _bound_at_t_patches(L_max, U_max)
        if sign == -1:
            best = torch.min(best, cand)
        else:
            best = torch.max(best, cand)

        # Evaluate at each per-dimension breakpoint (both lower and upper).
        for d in range(self.D):
            for bp_tensor in (self.bp_lower, self.bp_upper):
                t_d = bp_tensor[:, d]  # (B,)
                L_d, U_d = self._eval_at_t(t_d)  # (B, D)
                cand = _bound_at_t_patches(L_d, U_d)
                if sign == -1:
                    best = torch.min(best, cand)
                else:
                    best = torch.max(best, cand)

        return best

    # -----------------------------------------------------------------
    # Init (for both backward and forward mode)
    # -----------------------------------------------------------------
    def init(self, x, aux=None, forward=False):
        if not forward:
            return LinearBound(
                None, None, None, None, self.x_L, self.x_U), x, None

        # Forward mode: the perturbation variable is the scalar t.
        # dim_in = 1.  We compute a linear relaxation of the PWL envelope:
        #   lw * t + lb <= x_i <= uw * t + ub   for all t in [t_min, t_max]
        #
        # For the LOWER bound on x_i (which is L_i(t), a 2-piece PWL):
        #   lw_i, lb_i = tightest linear under-approximation of L_i(t).
        #   This is the chord connecting (t_min, L_i(t_min)) to (t_max, L_i(t_max)).
        #   (Works because the chord of a PWL function lower-bounds it if
        #    the function is concave, and we take the min of chord and pieces.)
        #
        # For the UPPER bound on x_i (U_i(t)):
        #   uw_i, ub_i = tightest linear over-approximation of U_i(t).

        B, D = self.bp_lower.shape
        t0, t1 = self.t_min, self.t_max
        dt = t1 - t0

        # Evaluate L and U at endpoints and breakpoint
        L_at_t0 = self.pwl_lower_A[:, 0] * t0 + self.pwl_lower_b[:, 0]  # (B,D)
        L_at_t1 = self.pwl_lower_A[:, 1] * t1 + self.pwl_lower_b[:, 1]
        U_at_t0 = self.pwl_upper_A[:, 0] * t0 + self.pwl_upper_b[:, 0]
        U_at_t1 = self.pwl_upper_A[:, 1] * t1 + self.pwl_upper_b[:, 1]

        L_at_bp_lower, U_at_bp_upper = self._worst_at_breakpoints()

        # Chord slope and intercept for lower bound (linear under-approx)
        lw = ((L_at_t1 - L_at_t0) / dt).unsqueeze(1)  # (B, 1, D)
        lb = (L_at_t0 - lw.squeeze(1) * t0)            # (B, D)

        # Shift chord down if it exceeds L(bp_lower) to ensure it strictly lower bounds
        L_chord_at_bp = lw.squeeze(1) * self.bp_lower + lb
        L_error = torch.clamp(L_chord_at_bp - L_at_bp_lower, min=0.0)
        lb = lb - L_error

        # Chord for upper bound (linear over-approx)
        uw = ((U_at_t1 - U_at_t0) / dt).unsqueeze(1)  # (B, 1, D)
        ub = (U_at_t0 - uw.squeeze(1) * t0)            # (B, D)

        # Shift chord up if it is below U(bp_upper) to ensure it strictly upper bounds
        U_chord_at_bp = uw.squeeze(1) * self.bp_upper + ub
        U_error = torch.clamp(U_at_bp_upper - U_chord_at_bp, min=0.0)
        ub = ub + U_error

        # Parameter range as x_L, x_U for the 1-D perturbation variable
        t_L = torch.full((B, 1), t0, dtype=self.bp_lower.dtype, device=self.bp_lower.device)
        t_U = torch.full((B, 1), t1, dtype=self.bp_lower.dtype, device=self.bp_lower.device)

        return LinearBound(
            lw, lb, uw, ub, self.x_L, self.x_U,
            x_L=t_L, x_U=t_U), x, None

    def __repr__(self):
        return (f'PerturbationPWL(D={self.D}, '
                f't_range=[{self.t_min:.4f}, {self.t_max:.4f}])')


class PerturbationSynonym(Perturbation):
    def __init__(self, budget, eps=1.0, use_simple=False):
        super(PerturbationSynonym, self).__init__()
        self._load_synonyms()
        self.budget = budget
        self.eps = eps
        self.use_simple = use_simple
        self.model = None
        self.train = False

    def __repr__(self):
        return (f'perturbation(Synonym-based word substitution '
                f'budget={self.budget}, eps={self.eps})')

    def _load_synonyms(self, path='data/synonyms.json'):
        with open(path) as file:
            self.synonym = json.loads(file.read())
        logger.info('Synonym list loaded for {} words'.format(len(self.synonym)))

    def set_train(self, train):
        self.train = train

    def concretize(self, x, A, sign, aux):
        assert(self.model is not None)

        x_rep, mask, can_be_replaced = aux
        batch_size, length, dim_word = x.shape[0], x.shape[1], x.shape[2]
        dim_out = A.shape[1]
        max_num_cand = x_rep.shape[2]

        mask_rep = torch.tensor(can_be_replaced, dtype=torch.get_default_dtype(), device=A.device)

        num_pos = int(np.max(np.sum(can_be_replaced, axis=-1)))
        update_A = A.shape[-1] > num_pos * dim_word
        if update_A:
            bias = torch.bmm(A, (x * (1 - mask_rep).unsqueeze(-1)).reshape(batch_size, -1, 1)).squeeze(-1)
        else:
            bias = 0.
        A = A.reshape(batch_size, dim_out, -1, dim_word)

        A_new, x_new, x_rep_new, mask_new = [], [], [], []
        zeros_A = torch.zeros(dim_out, dim_word, device=A.device)
        zeros_w = torch.zeros(dim_word, device=A.device)
        zeros_rep = torch.zeros(max_num_cand, dim_word, device=A.device)
        zeros_mask = torch.zeros(max_num_cand, device=A.device)
        for t in range(batch_size):
            cnt = 0
            for i in range(0, length):
                if can_be_replaced[t][i]:
                    if update_A:
                        A_new.append(A[t, :, i, :])
                    x_new.append(x[t][i])
                    x_rep_new.append(x_rep[t][i])
                    mask_new.append(mask[t][i])
                    cnt += 1
            if update_A:
                A_new += [zeros_A] * (num_pos - cnt)
            x_new += [zeros_w] * (num_pos - cnt)
            x_rep_new += [zeros_rep] * (num_pos - cnt)
            mask_new += [zeros_mask] * (num_pos - cnt)
        if update_A:
            A = torch.cat(A_new).reshape(batch_size, num_pos, dim_out, dim_word).transpose(1, 2)
        x = torch.cat(x_new).reshape(batch_size, num_pos, dim_word)
        x_rep = torch.cat(x_rep_new).reshape(batch_size, num_pos, max_num_cand, dim_word)
        mask = torch.cat(mask_new).reshape(batch_size, num_pos, max_num_cand)
        length = num_pos

        A = A.reshape(batch_size, A.shape[1], length, -1).transpose(1, 2)
        x = x.reshape(batch_size, length, -1, 1)

        if sign == 1:
            cmp, init = torch.max, -1e30
        else:
            cmp, init = torch.min, 1e30

        init_tensor = torch.ones(batch_size, dim_out).to(x.device) * init
        dp = [[init_tensor] * (self.budget + 1) for i in range(0, length + 1)]
        dp[0][0] = torch.zeros(batch_size, dim_out).to(x.device)

        A = A.reshape(batch_size * length, A.shape[2], A.shape[3])
        Ax = torch.bmm(
            A,
            x.reshape(batch_size * length, x.shape[2], x.shape[3])
        ).reshape(batch_size, length, A.shape[1])

        Ax_rep = torch.bmm(
            A,
            x_rep.reshape(batch_size * length, max_num_cand, x.shape[2]).transpose(-1, -2)
        ).reshape(batch_size, length, A.shape[1], max_num_cand)
        Ax_rep = Ax_rep * mask.unsqueeze(2) + init * (1 - mask).unsqueeze(2)
        Ax_rep_bound = cmp(Ax_rep, dim=-1).values

        if self.use_simple and self.train:
            return torch.sum(cmp(Ax, Ax_rep_bound), dim=1) + bias

        for i in range(1, length + 1):
            dp[i][0] = dp[i - 1][0] + Ax[:, i - 1]
            for j in range(1, self.budget + 1):
                dp[i][j] = cmp(
                    dp[i - 1][j] + Ax[:, i - 1],
                    dp[i - 1][j - 1] + Ax_rep_bound[:, i - 1]
                )
        dp = torch.cat(dp[length], dim=0).reshape(self.budget + 1, batch_size, dim_out)

        return cmp(dp, dim=0).values + bias

    def init(self, x, aux=None, forward=False):
        tokens, batch = aux
        self.tokens = tokens # DEBUG
        assert(len(x.shape) == 3)
        batch_size, length, dim_word = x.shape[0], x.shape[1], x.shape[2]

        max_pos = 1
        can_be_replaced = np.zeros((batch_size, length), dtype=bool)

        self._build_substitution(batch)

        for t in range(batch_size):
            cnt = 0
            candidates = batch[t]['candidates']
            # for transformers
            if tokens[t][0] == '[CLS]':
                candidates = [[]] + candidates + [[]]
            for i in range(len(tokens[t])):
                if tokens[t][i] == '[UNK]' or \
                        len(candidates[i]) == 0 or tokens[t][i] != candidates[i][0]:
                    continue
                for w in candidates[i][1:]:
                    if w in self.model.vocab:
                        can_be_replaced[t][i] = True
                        cnt += 1
                        break
            max_pos = max(max_pos, cnt)

        dim = max_pos * dim_word
        if forward:
            eye = torch.eye(dim_word).to(x.device)
            lw = torch.zeros(batch_size, dim, length, dim_word).to(x.device)
            lb = torch.zeros_like(x).to(x.device)
        word_embeddings = self.model.word_embeddings.weight
        vocab = self.model.vocab
        x_rep = [[[] for i in range(length)] for t in range(batch_size)]
        max_num_cand = 1
        for t in range(batch_size):
            candidates = batch[t]['candidates']
            # for transformers
            if tokens[t][0] == '[CLS]':
                candidates = [[]] + candidates + [[]]
            cnt = 0
            for i in range(length):
                if can_be_replaced[t][i]:
                    word_embed = word_embeddings[vocab[tokens[t][i]]]
                    # positional embedding and token type embedding
                    other_embed = x[t, i] - word_embed
                    if forward:
                        lw[t, (cnt * dim_word):((cnt + 1) * dim_word), i, :] = eye
                        lb[t, i, :] = torch.zeros_like(word_embed)
                    for w in candidates[i][1:]:
                        if w in self.model.vocab:
                            x_rep[t][i].append(
                                word_embeddings[self.model.vocab[w]] + other_embed)
                    max_num_cand = max(max_num_cand, len(x_rep[t][i]))
                    cnt += 1
                else:
                    if forward:
                        lb[t, i, :] = x[t, i, :]
        if forward:
            uw, ub = lw, lb
        else:
            lw = lb = uw = ub = None
        zeros = torch.zeros(dim_word, device=x.device)

        x_rep_, mask = [], []
        for t in range(batch_size):
            for i in range(length):
                x_rep_ += x_rep[t][i] + [zeros] * (max_num_cand - len(x_rep[t][i]))
                mask += [1] * len(x_rep[t][i]) + [0] * (max_num_cand - len(x_rep[t][i]))
        x_rep_ = torch.cat(x_rep_).reshape(batch_size, length, max_num_cand, dim_word)
        mask = torch.tensor(mask, dtype=torch.get_default_dtype(), device=x.device)\
            .reshape(batch_size, length, max_num_cand)
        x_rep_ = x_rep_ * self.eps + x.unsqueeze(2) * (1 - self.eps)

        inf = 1e20
        lower = torch.min(mask.unsqueeze(-1) * x_rep_ + (1 - mask).unsqueeze(-1) * inf, dim=2).values
        upper = torch.max(mask.unsqueeze(-1) * x_rep_ + (1 - mask).unsqueeze(-1) * (-inf), dim=2).values
        lower = torch.min(lower, x)
        upper = torch.max(upper, x)

        return LinearBound(lw, lb, uw, ub, lower, upper), x, (x_rep_, mask, can_be_replaced)

    def _build_substitution(self, batch):
        for example in batch:
            if not 'candidates' in example or example['candidates'] is None:
                candidates = []
                tokens = example['sentence'].strip().lower().split(' ')
                for i in range(len(tokens)):
                    _cand = []
                    if tokens[i] in self.synonym:
                        for w in self.synonym[tokens[i]]:
                            if w in self.model.vocab:
                                _cand.append(w)
                    if len(_cand) > 0:
                        _cand = [tokens[i]] + _cand
                    candidates.append(_cand)
                example['candidates'] = candidates

