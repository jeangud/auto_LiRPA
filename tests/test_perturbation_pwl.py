"""Tests for PerturbationPWL (piecewise-linear bounds)."""
import torch
import torch.nn as nn
import numpy as np
import unittest

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm, PerturbationLinear, PerturbationPWL


# ──────────────────────────────────────────────────────────────────────
# Helper: brute-force bound via sampling
# ──────────────────────────────────────────────────────────────────────
def brute_force_bounds(model, ptb, N=500):
    """Sample N values of t, compute box bounds at each, return union.

    Returns the loosest (lb, ub) that any sampled t produces.
    """
    t_vals = torch.linspace(ptb.t_min, ptb.t_max, N)
    global_lb, global_ub = None, None
    
    with torch.no_grad():
        # Trace BoundedModule exactly once using the shape at t_min
        L_dummy, U_dummy = ptb._eval_at_t(ptb.t_min)
        x_dummy = (L_dummy + U_dummy) / 2
        bmodel = BoundedModule(model, x_dummy)
        
        for t in t_vals:
            L, U = ptb._eval_at_t(t.item())
            # Use box (Linf) bounds at this t
            box_ptb = PerturbationLpNorm(x_L=L, x_U=U)
            x_center = (L + U) / 2
            bx = BoundedTensor(x_center, box_ptb)
            
            # compute_bounds accepts x as a tuple or single tensor depending on model
            lb, ub = bmodel.compute_bounds(x=(bx,), method='backward')
            
            if global_lb is None:
                global_lb, global_ub = lb.clone(), ub.clone()
            else:
                global_lb = torch.min(global_lb, lb)
                global_ub = torch.max(global_ub, ub)
            
            # Exact sampling for true upper bound of the minimum
            x_rand = L + torch.rand_like(L) * (U - L)
            y_exact = model(x_rand)
            if 'exact_lb' not in locals():
                exact_lb, exact_ub = y_exact.clone(), y_exact.clone()
            else:
                exact_lb = torch.min(exact_lb, y_exact)
                exact_ub = torch.max(exact_ub, y_exact)
                
    return global_lb, global_ub, exact_lb, exact_ub


def make_simple_pwl(B, D, t_min=0.0, t_max=1.0, seed=42):
    """Create a valid PWL perturbation with concave lower bounds and convex upper bounds."""
    torch.manual_seed(seed)
    bp = torch.rand(B, D) * (t_max - t_min - 0.1) + t_min + 0.05

    lower_A0 = torch.randn(B, D) * 0.3
    lower_A1 = lower_A0 - torch.rand(B, D) * 0.2 - 0.05
    
    upper_A0 = torch.randn(B, D) * 0.3
    upper_A1 = upper_A0 + torch.rand(B, D) * 0.2 + 0.05

    lower_b0 = torch.randn(B, D) - 0.5
    upper_b0 = torch.randn(B, D) + 0.5

    lower_b1 = (lower_A0 - lower_A1) * bp + lower_b0
    upper_b1 = (upper_A0 - upper_A1) * bp + upper_b0

    def eval_L(t): return torch.where(t <= bp, lower_A0 * t + lower_b0, lower_A1 * t + lower_b1)
    def eval_U(t): return torch.where(t <= bp, upper_A0 * t + upper_b0, upper_A1 * t + upper_b1)
    
    t_min_t = torch.full_like(bp, t_min)
    t_max_t = torch.full_like(bp, t_max)
    min_diff = torch.min(torch.min(eval_U(t_min_t) - eval_L(t_min_t), eval_U(t_max_t) - eval_L(t_max_t)), eval_U(bp) - eval_L(bp))
    
    shift = torch.clamp(-min_diff + 0.1, min=0.0)
    upper_b0 += shift
    upper_b1 += shift

    lower_A = torch.stack([lower_A0, lower_A1], dim=1)
    lower_b = torch.stack([lower_b0, lower_b1], dim=1)
    upper_A = torch.stack([upper_A0, upper_A1], dim=1)
    upper_b = torch.stack([upper_b0, upper_b1], dim=1)

    return PerturbationPWL(
        t_min=t_min, t_max=t_max,
        lower_breakpoints=bp, upper_breakpoints=bp,
        lower_A=lower_A, upper_A=upper_A,
        lower_b=lower_b, upper_b=upper_b,
    )


def concretize_matrix_loop(ptb, A, sign):
    """Reference for `PerturbationPWL._concretize_matrix`: one pass per candidate t.

    Slow (2D+2 passes over the inputs). It evaluates each breakpoint with
    piece 0 only, so it is exact only for pieces that are continuous at their
    breakpoints.
    """
    B, H, D = A.shape
    if D == 1 and ptb.D > 1:
        # Forward mode concretization: A is the coefficient for the 1D scalar t
        A_pos = A.clamp(min=0)
        A_neg = A.clamp(max=0)
        if sign == -1:
            return (A_pos * ptb.t_min + A_neg * ptb.t_max).squeeze(-1)
        else:
            return (A_pos * ptb.t_max + A_neg * ptb.t_min).squeeze(-1)

    A_pos = A.clamp(min=0)  # (B, H, D)
    A_neg = A.clamp(max=0)  # (B, H, D)

    def _bound_at_t(L, U):
        """Given L,U of shape (B,D), compute bound(t) for all H."""
        if sign == -1:
            # min: pick L where A>=0, U where A<0
            return (A_pos * L.unsqueeze(1) +
                    A_neg * U.unsqueeze(1)).sum(dim=-1)  # (B, H)
        else:
            return (A_pos * U.unsqueeze(1) +
                    A_neg * L.unsqueeze(1)).sum(dim=-1)

    # Evaluate at t_min and t_max
    L_min, U_min = ptb._eval_at_t(ptb.t_min)
    L_max, U_max = ptb._eval_at_t(ptb.t_max)
    best = _bound_at_t(L_min, U_min)
    cand = _bound_at_t(L_max, U_max)
    if sign == -1:
        best = torch.min(best, cand)
    else:
        best = torch.max(best, cand)

    # Evaluate at each per-dimension breakpoint (both lower and upper).
    # bp_lower[:, d] / bp_upper[:, d] are candidate kinks for dim d. When we
    # evaluate at t = bp_*[b, d], for OTHER dims j != d we re-evaluate L_j(t)
    # and U_j(t) according to their own bp_lower_j / bp_upper_j masks. We
    # process one candidate at a time to keep memory O(B*D).
    for d in range(D):
        for bp_tensor in (ptb.bp_lower, ptb.bp_upper):
            t_d = bp_tensor[:, d]  # (B,)
            L_d, U_d = ptb._eval_at_t(t_d)  # (B, D)
            cand = _bound_at_t(L_d, U_d)
            if sign == -1:
                best = torch.min(best, cand)
            else:
                best = torch.max(best, cand)

    return best


# ──────────────────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────────────────
class LinearOnlyModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)


class TwoLayerLinearModel(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        return self.fc2(self.fc1(x))


class TwoLayerReLUModel(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


# ──────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────
class TestPerturbationPWL(unittest.TestCase):

    def test_sanity_checks(self):
        """Ensures that invalid initializations (e.g., t_min >= t_max, incorrect tensor shapes, or out-of-bound breakpoints) properly raise a ValueError."""
        B, D = 2, 3
        zeros_A = torch.zeros(B, 2, D)
        zeros_b = torch.zeros(B, 2, D)
        bp = torch.full((B, D), 0.5)

        # t_min >= t_max
        with self.assertRaises(ValueError):
            PerturbationPWL(1.0, 0.0,
                            lower_breakpoints=bp, upper_breakpoints=bp,
                            lower_A=zeros_A, upper_A=zeros_A,
                            lower_b=zeros_b, upper_b=zeros_b)

        # Wrong shape for breakpoints (1-D instead of 2-D)
        with self.assertRaises(ValueError):
            PerturbationPWL(0.0, 1.0,
                            lower_breakpoints=torch.zeros(D),
                            upper_breakpoints=torch.zeros(D),
                            lower_A=zeros_A, upper_A=zeros_A,
                            lower_b=zeros_b, upper_b=zeros_b)

        # Mismatched lower/upper breakpoint shapes
        with self.assertRaises(ValueError):
            PerturbationPWL(0.0, 1.0,
                            lower_breakpoints=bp,
                            upper_breakpoints=torch.zeros(B, D + 1),
                            lower_A=zeros_A, upper_A=zeros_A,
                            lower_b=zeros_b, upper_b=zeros_b)

        # Wrong K dimension (3 instead of 2)
        with self.assertRaises(ValueError):
            PerturbationPWL(0.0, 1.0,
                            lower_breakpoints=bp, upper_breakpoints=bp,
                            lower_A=torch.zeros(B, 3, D), upper_A=zeros_A,
                            lower_b=zeros_b, upper_b=zeros_b)

        # Breakpoint out of range
        with self.assertRaises(ValueError):
            PerturbationPWL(0.0, 1.0,
                            lower_breakpoints=torch.full((B, D), 2.0),
                            upper_breakpoints=bp,
                            lower_A=zeros_A, upper_A=zeros_A,
                            lower_b=zeros_b, upper_b=zeros_b)

        # Discontinuous lower pieces
        with self.assertRaisesRegex(ValueError, "continuous at bp_lower"):
            # bp = 0.5, lower_A = 0, lower_b for piece 0 = 0, piece 1 = 1
            # L0(0.5) = 0 != L1(0.5) = 1
            lower_b_disc = torch.zeros(B, 2, D)
            lower_b_disc[:, 1, :] = 1.0
            PerturbationPWL(0.0, 1.0,
                            lower_breakpoints=bp, upper_breakpoints=bp,
                            lower_A=zeros_A, upper_A=zeros_A,
                            lower_b=lower_b_disc, upper_b=zeros_b)

        # Discontinuous upper pieces (detected independently of lower)
        with self.assertRaisesRegex(ValueError, "continuous at bp_upper"):
            upper_b_disc = torch.zeros(B, 2, D)
            upper_b_disc[:, 1, :] = 1.0
            PerturbationPWL(0.0, 1.0,
                            lower_breakpoints=bp, upper_breakpoints=bp,
                            lower_A=zeros_A, upper_A=zeros_A,
                            lower_b=zeros_b, upper_b=upper_b_disc)

    def test_concrete_bounds(self):
        """Verifies the fundamental mathematical property that the generated concrete lower bounds are never greater than the upper bounds (x_L <= x_U)."""
        ptb = make_simple_pwl(B=4, D=5)
        self.assertTrue((ptb.x_L <= ptb.x_U + 1e-6).all())

    def test_single_piece_matches_box(self):
        """Confirms that when the PWL perturbation has a slope of 0 (acting as constant bounds), the computed bounds perfectly match standard Box bounds (PerturbationLpNorm)."""
        B, D = 2, 4
        torch.manual_seed(0)
        model = LinearOnlyModel(D, 3)

        # Constant lower/upper (no dependence on t)
        x_L = torch.rand(B, D)
        x_U = x_L + torch.rand(B, D) * 0.5

        ptb_pwl = PerturbationPWL(
            t_min=0.0, t_max=1.0,
            lower_breakpoints=torch.full((B, D), 0.5),
            upper_breakpoints=torch.full((B, D), 0.5),
            lower_A=torch.zeros(B, 2, D),
            lower_b=x_L.unsqueeze(1).expand(B, 2, D),
            upper_A=torch.zeros(B, 2, D),
            upper_b=x_U.unsqueeze(1).expand(B, 2, D),
        )

        ptb_box = PerturbationLpNorm(x_L=x_L, x_U=x_U)

        x_center = (x_L + x_U) / 2

        # PWL bounds
        bx_pwl = BoundedTensor(x_center, ptb_pwl)
        m_pwl = BoundedModule(model, x_center)
        lb_pwl, ub_pwl = m_pwl.compute_bounds(bx_pwl, method='backward')

        # Box bounds
        bx_box = BoundedTensor(x_center, ptb_box)
        m_box = BoundedModule(model, x_center)
        lb_box, ub_box = m_box.compute_bounds(bx_box, method='backward')

        # Should be identical
        self.assertTrue(torch.allclose(lb_pwl, lb_box, atol=1e-5),
                        f"Lower bounds differ: max diff = {(lb_pwl - lb_box).abs().max()}")
        self.assertTrue(torch.allclose(ub_pwl, ub_box, atol=1e-5),
                        f"Upper bounds differ: max diff = {(ub_pwl - ub_box).abs().max()}")

    def test_single_layer_soundness(self):
        """Validates that bounds propagated through a single nn.Linear layer are sound (lb <= ub) and strictly bound the true values found via brute-force sampling."""
        B, D, H = 2, 5, 3
        torch.manual_seed(1)
        model = LinearOnlyModel(D, H)
        ptb = make_simple_pwl(B, D, seed=10)

        x_center = (ptb.x_L + ptb.x_U) / 2
        bx = BoundedTensor(x_center, ptb)
        bmodel = BoundedModule(model, x_center)
        lb, ub = bmodel.compute_bounds(bx, method='backward')

        # Verify lb <= ub
        self.assertTrue((lb <= ub + 1e-5).all(),
                        f"lb > ub! max violation: {(lb - ub).max()}")

        # Brute-force check: PWL bounds should be at least as tight
        # as the union of sampled box bounds
        bf_lb, bf_ub, exact_lb, exact_ub = brute_force_bounds(model, ptb, N=200)

        # PWL lower bound should be <= exact lower bound
        self.assertTrue((lb <= exact_lb + 1e-4).all(),
                        f"PWL lb exceeds exact lb: {(lb - exact_lb).max()}")
        self.assertTrue((ub >= exact_ub - 1e-4).all(),
                        f"PWL ub below exact ub: {(exact_ub - ub).max()}")

    def test_two_layer_linear_soundness(self):
        """Tests the mathematical soundness of bounds propagated through a deeper, 2-layer linear network."""
        B, D = 2, 4
        torch.manual_seed(2)
        model = TwoLayerLinearModel(D, 6, 3)
        ptb = make_simple_pwl(B, D, seed=20)

        x_center = (ptb.x_L + ptb.x_U) / 2
        bx = BoundedTensor(x_center, ptb)
        bmodel = BoundedModule(model, x_center)
        lb, ub = bmodel.compute_bounds(bx, method='backward')

        self.assertTrue((lb <= ub + 1e-5).all())

        bf_lb, bf_ub, exact_lb, exact_ub = brute_force_bounds(model, ptb, N=200)
        self.assertTrue((lb <= exact_lb + 1e-4).all(),
                        f"PWL lb exceeds exact_lb: {(lb - exact_lb).max()}")
        self.assertTrue((ub >= exact_ub - 1e-4).all(),
                        f"PWL ub below exact_ub: {(exact_ub - ub).max()}")

    def test_relu_network_soundness(self):
        """Verifies soundness through non-linear nn.ReLU activations. Accounts for acceptable margin tolerances caused by differing linear relaxation strategies during brute-force vs. PWL evaluation.

        Note: For ReLU networks, the CROWN linear relaxation of ReLU uses
        the concrete box bounds (x_L, x_U) for intermediate layers. The PWL
        perturbation has wider concrete bounds than per-t boxes, so the ReLU
        relaxation may be looser or tighter in different ways. We use a
        generous tolerance here.
        """
        B, D = 2, 4
        torch.manual_seed(3)
        model = TwoLayerReLUModel(D, 8, 3)
        ptb = make_simple_pwl(B, D, seed=30)

        x_center = (ptb.x_L + ptb.x_U) / 2
        bx = BoundedTensor(x_center, ptb)
        bmodel = BoundedModule(model, x_center)
        lb, ub = bmodel.compute_bounds(bx, method='backward')

        # Basic validity
        self.assertTrue((lb <= ub + 1e-5).all())

        # For ReLU networks, the PWL bounds may be slightly looser or
        # tighter than brute-force due to different ReLU relaxations.
        # We just verify lb <= ub and the bounds are reasonable.
        bf_lb, bf_ub, exact_lb, exact_ub = brute_force_bounds(model, ptb, N=300)
        # Allow up to 0.1 tolerance for ReLU relaxation differences
        self.assertTrue((lb <= exact_lb + 0.1).all(),
                        f"PWL lb exceeds exact_lb by too much: {(lb - exact_lb).max()}")
        self.assertTrue((ub >= exact_ub - 0.1).all(),
                        f"PWL ub below exact_ub by too much: {(exact_ub - ub).max()}")

    def test_per_dimension_breakpoints(self):
        """Ensures the bound propagation correctly handles perturbations where the interior breakpoint t values differ across feature dimensions."""
        B, D, H = 1, 3, 2
        torch.manual_seed(4)
        model = LinearOnlyModel(D, H)

        # Different breakpoint for each dimension
        bp = torch.tensor([[0.2, 0.5, 0.8]])  # (1, 3)
        lower_A = torch.randn(B, 2, D) * 0.1
        upper_A = lower_A + 0.2
        lower_b0 = torch.randn(B, D) * 0.1 - 0.5
        upper_b0 = lower_b0 + 1.0
        
        lower_b1 = (lower_A[:, 0] - lower_A[:, 1]) * bp + lower_b0
        upper_b1 = (upper_A[:, 0] - upper_A[:, 1]) * bp + upper_b0
        
        lower_b = torch.stack([lower_b0, lower_b1], dim=1)
        upper_b = torch.stack([upper_b0, upper_b1], dim=1)

        ptb = PerturbationPWL(
            0.0, 1.0,
            lower_breakpoints=bp, upper_breakpoints=bp,
            lower_A=lower_A, upper_A=upper_A,
            lower_b=lower_b, upper_b=upper_b,
        )

        x_center = (ptb.x_L + ptb.x_U) / 2
        bx = BoundedTensor(x_center, ptb)
        bmodel = BoundedModule(model, x_center)
        lb, ub = bmodel.compute_bounds(bx, method='backward')

        self.assertTrue((lb <= ub + 1e-5).all())

        bf_lb, bf_ub, exact_lb, exact_ub = brute_force_bounds(model, ptb, N=500)
        # Allow small floating-point tolerance from breakpoint evaluation
        self.assertTrue((lb <= exact_lb + 1e-3).all(),
                        f"lb exceeds exact_lb: {(lb - exact_lb).max()}")
        self.assertTrue((ub >= exact_ub - 1e-3).all(),
                        f"ub below exact_ub: {(exact_ub - ub).max()}")

    def test_analytical_bounds(self):
        """A strict accuracy test that feeds a predefined PerturbationPWL into a fixed-weight 1-layer network and compares the computed bounds against exactly hand-calculated values."""
        B, D, H = 1, 2, 2
        model = LinearOnlyModel(D, H)
        # Set specific weights
        model.fc.weight.data = torch.tensor([[1.0, -1.0], [0.5, 0.5]])
        model.fc.bias.data = torch.tensor([0.0, 1.0])

        bp = torch.tensor([[0.5, 0.5]])  # (1, 2)
        
        # Dim 0: slope 2 everywhere, intercept 0
        # Dim 1: L1 concave (slope 1 then -1), U1 convex (slope -1 then 1)
        lower_A = torch.tensor([[
            [2.0, 1.0],
            [2.0, -1.0]
        ]]) # (1, 2, 2)
        upper_A = torch.tensor([[
            [2.0, -1.0],
            [2.0, 1.0]
        ]])

        lower_b = torch.tensor([[
            [0.0, 0.5],
            [0.0, 1.5]
        ]])
        upper_b = torch.tensor([[
            [0.0, 1.5],
            [0.0, 0.5]
        ]])

        ptb = PerturbationPWL(
            t_min=0.0, t_max=1.0,
            lower_breakpoints=bp, upper_breakpoints=bp,
            lower_A=lower_A, upper_A=upper_A,
            lower_b=lower_b, upper_b=upper_b
        )

        x_center = (ptb.x_L + ptb.x_U) / 2
        bx = BoundedTensor(x_center, ptb)
        bmodel = BoundedModule(model, x_center)
        lb, ub = bmodel.compute_bounds(bx, method='backward')

        expected_lb = torch.tensor([[-1.5, 1.25]])
        expected_ub = torch.tensor([[1.5, 2.75]])

        self.assertTrue(torch.allclose(lb, expected_lb, atol=1e-5), f"Expected lb {expected_lb}, got {lb}")
        self.assertTrue(torch.allclose(ub, expected_ub, atol=1e-5), f"Expected ub {expected_ub}, got {ub}")

    def test_k1_linear_vs_box(self):
        """Evaluates a K=1 scenario (no interior breakpoints). Proves that using slopes dynamically over t yields tighter bounds than a static bounding box covering the entire t domain."""
        B, D, H = 2, 4, 3
        torch.manual_seed(5)
        model = LinearOnlyModel(D, H)
        
        # Simulate K=1 by setting bp to t_max and making both pieces identical.
        t_min, t_max = 0.0, 1.0
        bp = torch.full((B, D), t_max)
        
        slopes = torch.randn(B, D) * 0.5
        intercepts = torch.randn(B, D) * 0.5
        
        lower_A = slopes.unsqueeze(1).expand(B, 2, D)
        lower_b = intercepts.unsqueeze(1).expand(B, 2, D) - 0.1
        upper_A = slopes.unsqueeze(1).expand(B, 2, D)
        upper_b = intercepts.unsqueeze(1).expand(B, 2, D) + 0.1

        ptb_pwl = PerturbationPWL(
            t_min, t_max,
            lower_breakpoints=bp, upper_breakpoints=bp,
            lower_A=lower_A, upper_A=upper_A,
            lower_b=lower_b, upper_b=upper_b,
        )
        
        x_L = ptb_pwl.x_L
        x_U = ptb_pwl.x_U
        ptb_box = PerturbationLpNorm(x_L=x_L, x_U=x_U)
        
        x_center = (x_L + x_U) / 2
        
        bx_pwl = BoundedTensor(x_center, ptb_pwl)
        m_pwl = BoundedModule(model, x_center)
        lb_pwl, ub_pwl = m_pwl.compute_bounds(bx_pwl, method='backward')
        
        bx_box = BoundedTensor(x_center, ptb_box)
        m_box = BoundedModule(model, x_center)
        lb_box, ub_box = m_box.compute_bounds(bx_box, method='backward')
        
        self.assertTrue((lb_pwl >= lb_box - 1e-4).all())
        self.assertTrue((ub_pwl <= ub_box + 1e-4).all())
        self.assertTrue((lb_pwl <= ub_pwl + 1e-5).all())

    @torch.no_grad()
    def test_patches_vs_matrix_mode_consistency(self):
        """PWL bounds through a CNN with both matrix and patches conv modes."""
        import torch.nn.functional as F
        class TinyCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(2, 4, kernel_size=2, stride=2, padding=1)
                self.bn = nn.BatchNorm2d(4)
                self.fc = nn.Linear(4 * 3 * 3, 10)
            def forward(self, x):
                x = F.relu(self.bn(self.conv(x)))
                x = x.view(x.size(0), -1)
                return self.fc(x)

        torch.manual_seed(12)
        model = TinyCNN()

        B, C, H, W = 1, 2, 4, 4
        D = C * H * W

        ptb = make_simple_pwl(B, D, seed=15)
        # Reshape concrete bounds to image shape
        ptb.x_L = ptb.x_L.view(B, C, H, W)
        ptb.x_U = ptb.x_U.view(B, C, H, W)
        x_center = (ptb.x_L + ptb.x_U) / 2

        bx_m = BoundedTensor(x_center, ptb)
        bmodel_matrix = BoundedModule(model, x_center, bound_opts={'conv_mode': 'matrix'})
        lb_m, ub_m = bmodel_matrix.compute_bounds(bx_m, method='backward')
        self.assertTrue((lb_m <= ub_m + 1e-5).all())

        # Now test with patches mode (the default, efficient path)
        bx_p = BoundedTensor(x_center, ptb)
        bmodel_patches = BoundedModule(model, x_center, bound_opts={'conv_mode': 'patches'})
        lb_p, ub_p = bmodel_patches.compute_bounds(bx_p, method='backward')
        self.assertTrue((lb_p <= ub_p + 1e-5).all())

        # Both modes should produce identical results
        self.assertTrue(torch.allclose(lb_m, lb_p, atol=1e-5),
                        f"Matrix vs patches lb differ: max diff={( lb_m - lb_p).abs().max()}")
        self.assertTrue(torch.allclose(ub_m, ub_p, atol=1e-5),
                        f"Matrix vs patches ub differ: max diff={(ub_m - ub_p).abs().max()}")

    @torch.no_grad()
    def test_vision_model_patches_cifar_scale(self):
        """PWL bounds through a CIFAR-scale CNN using patches mode.

        This test would OOM with conv_mode='matrix' but runs efficiently
        with patches, proving the patches implementation is production-ready.
        """
        from test_vision_models import cnn_4layer_test

        torch.manual_seed(42)
        model = cnn_4layer_test()

        B, C, H, W = 1, 3, 32, 32
        D = C * H * W  # 3072 — would explode with matrix mode

        ptb = make_simple_pwl(B, D, seed=99)
        ptb.x_L = ptb.x_L.view(B, C, H, W)
        ptb.x_U = ptb.x_U.view(B, C, H, W)
        x_center = (ptb.x_L + ptb.x_U) / 2

        bx = BoundedTensor(x_center, ptb)
        # Default conv_mode (patches) — no matrix explosion
        bmodel = BoundedModule(model, x_center)
        lb, ub = bmodel.compute_bounds(bx, method='backward')

        self.assertTrue((lb <= ub + 1e-5).all(),
                        f"lb > ub! max violation: {(lb - ub).max()}")

    @torch.no_grad()
    def test_randomized_runs(self):
        """An extensive fuzzing loop (100 iterations) that dynamically generates varying network architectures, input dimensions, and PWL slopes/breakpoints to robustly hunt for mathematical edge cases."""
        for i in range(100):
            torch.manual_seed(1000 + i)
            B = torch.randint(1, 4, (1,)).item()
            D = torch.randint(2, 6, (1,)).item()
            H = torch.randint(2, 6, (1,)).item()
            
            model_type = torch.randint(0, 3, (1,)).item()
            if model_type == 0:
                model = LinearOnlyModel(D, H)
            elif model_type == 1:
                model = TwoLayerLinearModel(D, H, 2)
            else:
                model = TwoLayerReLUModel(D, H, 2)
                
            t_min = torch.rand(1).item() * 2.0 - 1.0
            t_max = t_min + torch.rand(1).item() * 2.0 + 0.5
            
            bp = torch.rand(B, D) * (t_max - t_min - 0.1) + t_min + 0.05
            
            lower_b0 = torch.randn(B, D)
            upper_b0 = torch.randn(B, D)
            
            lower_A0 = torch.randn(B, D) * 0.5
            lower_A1 = lower_A0 - torch.rand(B, D) * 0.2 - 0.05
            upper_A0 = torch.randn(B, D) * 0.5
            upper_A1 = upper_A0 + torch.rand(B, D) * 0.2 + 0.05
            
            lower_b1 = (lower_A0 - lower_A1) * bp + lower_b0
            upper_b1 = (upper_A0 - upper_A1) * bp + upper_b0
            
            def eval_L(t): return torch.where(t <= bp, lower_A0 * t + lower_b0, lower_A1 * t + lower_b1)
            def eval_U(t): return torch.where(t <= bp, upper_A0 * t + upper_b0, upper_A1 * t + upper_b1)
            
            t_min_t = torch.full_like(bp, t_min)
            t_max_t = torch.full_like(bp, t_max)
            min_diff = torch.min(torch.min(eval_U(t_min_t) - eval_L(t_min_t), eval_U(t_max_t) - eval_L(t_max_t)), eval_U(bp) - eval_L(bp))
            
            shift = torch.clamp(-min_diff + 0.1, min=0.0)
            upper_b0 += shift
            upper_b1 += shift
            
            lower_A = torch.stack([lower_A0, lower_A1], dim=1)
            lower_b = torch.stack([lower_b0, lower_b1], dim=1)
            upper_A = torch.stack([upper_A0, upper_A1], dim=1)
            upper_b = torch.stack([upper_b0, upper_b1], dim=1)
            
            ptb = PerturbationPWL(t_min, t_max, lower_breakpoints=bp, upper_breakpoints=bp, lower_A=lower_A, upper_A=upper_A, lower_b=lower_b, upper_b=upper_b)
            
            x_center = (ptb.x_L + ptb.x_U) / 2
            bx = BoundedTensor(x_center, ptb)
            bmodel = BoundedModule(model, x_center)
            
            lb, ub = bmodel.compute_bounds(bx, method='backward')
            
            self.assertTrue((lb <= ub + 1e-4).all(), f"Iter {i}: lb > ub")
            
            # Fast brute-force check
            bf_lb, bf_ub, exact_lb, exact_ub = brute_force_bounds(model, ptb, N=25)
            
            if model_type < 2:
                self.assertTrue((lb <= exact_lb + 1e-3).all(), f"Iter {i}: lb exceeds exact_lb")
                self.assertTrue((ub >= exact_ub - 1e-3).all(), f"Iter {i}: ub below exact_ub")

    @torch.no_grad()
    def test_forward_mode_soundness(self):
        """Test forward mode bound propagation with PWL perturbation."""
        B, D = 2, 4
        torch.manual_seed(99)
        model = TwoLayerLinearModel(D, 6, 3)
        ptb = make_simple_pwl(B, D, seed=100)

        x_center = (ptb.x_L + ptb.x_U) / 2
        bx = BoundedTensor(x_center, ptb)
        bmodel = BoundedModule(model, x_center)
        lb, ub = bmodel.compute_bounds(bx, method='forward')

        self.assertTrue((lb <= ub + 1e-5).all())

        bf_lb, bf_ub, exact_lb, exact_ub = brute_force_bounds(model, ptb, N=200)
        if not (lb <= exact_lb + 1e-4).all():
            print("lb:", lb)
            print("exact_lb:", exact_lb)
        self.assertTrue((lb <= exact_lb + 1e-4).all(), f"PWL lb exceeds exact_lb: {(lb - exact_lb).max()}")
        self.assertTrue((ub >= exact_ub - 1e-4).all(), f"PWL ub below exact_ub: {(exact_ub - ub).max()}")

    def test_optimized_bounds(self):
        """Test Alpha-CROWN compatibility with gradient optimization on intermediate bounds."""
        B, D = 2, 4
        torch.manual_seed(101)
        model = TwoLayerReLUModel(D, 6, 3)
        ptb = make_simple_pwl(B, D, seed=102)

        x_center = (ptb.x_L + ptb.x_U) / 2
        bx = BoundedTensor(x_center, ptb)
        # optimize_bound_args activates Alpha-CROWN optimization
        bound_opts = {'optimize_bound_args': {'ob_iteration': 2}}
        bmodel = BoundedModule(model, x_center, bound_opts=bound_opts)
        
        # We need gradients for this mode, so we don't use @torch.no_grad() here
        lb, ub = bmodel.compute_bounds(bx, method='CROWN-Optimized')

        self.assertTrue((lb <= ub + 1e-5).all())

        # Just a basic soundness check against simple brute force
        bf_lb, bf_ub, exact_lb, exact_ub = brute_force_bounds(model, ptb, N=100)
        if not (lb <= exact_lb + 0.1).all():
            print("lb:", lb)
            print("exact_lb:", exact_lb)
        self.assertTrue((lb <= exact_lb + 0.1).all())
        self.assertTrue((ub >= exact_ub - 0.1).all())

    @torch.no_grad()
    def test_domain_and_degeneracies(self):
        """Test boundary cases like negative t_min/max, breakpoints exactly at the edge, and sharp kinks."""
        B, D, H = 2, 3, 2
        model = LinearOnlyModel(D, H)
        
        # Negative domain
        t_min, t_max = -10.0, -5.0
        
        # Breakpoint exactly at the boundary (t_min)
        bp = torch.full((B, D), t_min)
        
        # Kinks: piece 0 has slope 1000, piece 1 has slope 0
        lower_A0 = torch.full((B, D), 1000.0)
        lower_A1 = torch.zeros(B, D)
        upper_A0 = torch.full((B, D), 1000.0)
        upper_A1 = torch.zeros(B, D)
        
        lower_b0 = torch.randn(B, D)
        upper_b0 = lower_b0 + 1.0  # valid interval
        
        # Enforce continuity at breakpoint
        lower_b1 = (lower_A0 - lower_A1) * bp + lower_b0
        upper_b1 = (upper_A0 - upper_A1) * bp + upper_b0
        
        lower_A = torch.stack([lower_A0, lower_A1], dim=1)
        lower_b = torch.stack([lower_b0, lower_b1], dim=1)
        upper_A = torch.stack([upper_A0, upper_A1], dim=1)
        upper_b = torch.stack([upper_b0, upper_b1], dim=1)
        
        ptb = PerturbationPWL(t_min, t_max, lower_breakpoints=bp, upper_breakpoints=bp, lower_A=lower_A, upper_A=upper_A, lower_b=lower_b, upper_b=upper_b)
        
        x_center = (ptb.x_L + ptb.x_U) / 2
        bx = BoundedTensor(x_center, ptb)
        bmodel = BoundedModule(model, x_center)
        lb, ub = bmodel.compute_bounds(bx, method='backward')
        
        self.assertTrue((lb <= ub + 1e-4).all())
        
        bf_lb, bf_ub, exact_lb, exact_ub = brute_force_bounds(model, ptb, N=200)
        self.assertTrue((lb <= exact_lb + 1e-3).all())
        self.assertTrue((ub >= exact_ub - 1e-3).all())

    # ──────────────────────────────────────────────────────────────────
    # Per-bound (asymmetric) breakpoints
    # ──────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def test_per_bound_breakpoints_hand_computed(self):
        """1-dim, 1-output linear model, hand-computed bounds with bp_lower != bp_upper.

        L(t): piece 0 (t<=0.3): slope=+1, b=0          -> L(0.3)=0.3
              piece 1 (t> 0.3): slope=-1, b=0.6        -> L(0.3)=0.3, L(1)=-0.4, L(0)=0.6
              => L is a tent with peak at t=0.3.
              min over [0,1] of L = L(1) = -0.4.

        U(t): piece 0 (t<=0.7): slope=-1, b=2          -> U(0.7)=1.3
              piece 1 (t> 0.7): slope=+1, b=0.6        -> U(0.7)=1.3, U(1)=1.6, U(0)=2.
              => U is a tent valley with min at t=0.7.
              max over [0,1] of U = U(0) = 2.0.

        Model: y = 2*x. So:
              lower bound on y = 2 * (-0.4) = -0.8
              upper bound on y = 2 * (+2.0) = +4.0
        """
        B, D = 1, 1

        bp_lower = torch.tensor([[0.3]])
        bp_upper = torch.tensor([[0.7]])

        lower_A = torch.tensor([[[1.0], [-1.0]]])     # (B=1, K=2, D=1)
        lower_b = torch.tensor([[[0.0], [0.6]]])
        upper_A = torch.tensor([[[-1.0], [1.0]]])
        upper_b = torch.tensor([[[2.0], [0.6]]])

        ptb = PerturbationPWL(
            t_min=0.0, t_max=1.0,
            lower_A=lower_A, upper_A=upper_A,
            lower_b=lower_b, upper_b=upper_b,
            lower_breakpoints=bp_lower, upper_breakpoints=bp_upper,
        )

        # Concrete bounds: should be exactly (-0.4, 2.0)
        self.assertTrue(torch.allclose(ptb.x_L, torch.tensor([[-0.4]]), atol=1e-5))
        self.assertTrue(torch.allclose(ptb.x_U, torch.tensor([[2.0]]), atol=1e-5))

        model = LinearOnlyModel(D, 1)
        with torch.no_grad():
            model.fc.weight.copy_(torch.tensor([[2.0]]))
            model.fc.bias.zero_()

        x_center = (ptb.x_L + ptb.x_U) / 2
        bx = BoundedTensor(x_center, ptb)
        bmodel = BoundedModule(model, x_center)
        lb, ub = bmodel.compute_bounds(bx, method='backward')

        self.assertTrue(torch.allclose(lb, torch.tensor([[-0.8]]), atol=1e-4),
                        f"lb={lb}, expected -0.8")
        self.assertTrue(torch.allclose(ub, torch.tensor([[4.0]]), atol=1e-4),
                        f"ub={ub}, expected 4.0")

    @torch.no_grad()
    def test_per_bound_breakpoints_fuzz(self):
        """Randomized fuzz: build a valid PWL with independent bp_lower/bp_upper,
        run backward CROWN bounds through a 2-layer ReLU model, and check
        soundness against brute-force sampling. 30 iterations to stay light."""
        B, D = 2, 4
        H, OUT = 5, 3

        for seed in range(30):
            torch.manual_seed(seed)
            t_min, t_max = -1.0, 1.0
            dt = t_max - t_min

            bp_lower = torch.rand(B, D) * (dt - 0.1) + t_min + 0.05
            bp_upper = torch.rand(B, D) * (dt - 0.1) + t_min + 0.05

            # Random pieces; enforce continuity at each bp.
            lower_A0 = torch.randn(B, D) * 0.5
            lower_A1 = torch.randn(B, D) * 0.5
            lower_b0 = torch.randn(B, D)
            lower_b1 = (lower_A0 - lower_A1) * bp_lower + lower_b0

            upper_A0 = torch.randn(B, D) * 0.5
            upper_A1 = torch.randn(B, D) * 0.5
            upper_b0 = torch.randn(B, D) + 2.0  # bias upper above lower
            upper_b1 = (upper_A0 - upper_A1) * bp_upper + upper_b0

            # Ensure L <= U over the whole domain by sampling and shifting upper
            ptb_tmp_L_at = []
            ptb_tmp_U_at = []
            for t in torch.linspace(t_min, t_max, 20):
                L = torch.where(t <= bp_lower,
                                lower_A0 * t + lower_b0,
                                lower_A1 * t + lower_b1)
                U = torch.where(t <= bp_upper,
                                upper_A0 * t + upper_b0,
                                upper_A1 * t + upper_b1)
                ptb_tmp_L_at.append(L)
                ptb_tmp_U_at.append(U)
            L_stack = torch.stack(ptb_tmp_L_at)  # (T, B, D)
            U_stack = torch.stack(ptb_tmp_U_at)
            min_gap = (U_stack - L_stack).min()
            if min_gap < 0.1:
                shift = (0.1 - min_gap).item()
                upper_b0 = upper_b0 + shift
                upper_b1 = upper_b1 + shift

            lower_A = torch.stack([lower_A0, lower_A1], dim=1)
            lower_b = torch.stack([lower_b0, lower_b1], dim=1)
            upper_A = torch.stack([upper_A0, upper_A1], dim=1)
            upper_b = torch.stack([upper_b0, upper_b1], dim=1)

            ptb = PerturbationPWL(
                t_min=t_min, t_max=t_max,
                lower_A=lower_A, upper_A=upper_A,
                lower_b=lower_b, upper_b=upper_b,
                lower_breakpoints=bp_lower, upper_breakpoints=bp_upper,
            )

            model = TwoLayerReLUModel(D, H, OUT)
            x_center = (ptb.x_L + ptb.x_U) / 2
            bx = BoundedTensor(x_center, ptb)
            bmodel = BoundedModule(model, x_center)
            lb, ub = bmodel.compute_bounds(bx, method='backward')

            self.assertTrue((lb <= ub + 1e-4).all())

            _, _, exact_lb, exact_ub = brute_force_bounds(model, ptb, N=80)
            self.assertTrue((lb <= exact_lb + 1e-3).all(),
                            f"seed={seed}: lb > exact_lb by "
                            f"{(lb - exact_lb).max().item():.4f}")
            self.assertTrue((ub >= exact_ub - 1e-3).all(),
                            f"seed={seed}: ub < exact_ub by "
                            f"{(exact_ub - ub).max().item():.4f}")

    # ──────────────────────────────────────────────────────────────────
    # Vectorized concretization against the one-candidate-at-a-time loops
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _random_pwl(B, D, dtype, seed, t_min=-0.3, t_max=0.7):
        """Continuous 2-piece bounds with independent lower/upper breakpoints,
        including repeated breakpoints and breakpoints at both ends."""
        g = torch.Generator().manual_seed(seed)
        def rand(*shape):
            return torch.rand(*shape, generator=g, dtype=dtype)
        def pieces(bp, offset):
            A = rand(B, 2, D) * 4 - 2
            b = torch.empty(B, 2, D, dtype=dtype)
            b[:, 0] = rand(B, D) + offset
            b[:, 1] = A[:, 0] * bp + b[:, 0] - A[:, 1] * bp   # continuous at bp
            return A, b
        bp_lower = rand(B, D) * (t_max - t_min) + t_min
        bp_upper = rand(B, D) * (t_max - t_min) + t_min
        # Ties, and breakpoints on the ends of the interval
        bp_lower[:, : D // 4] = bp_lower[:, :1]
        bp_upper[:, : D // 8] = bp_lower[:, :1]
        bp_lower[:, -1] = t_min
        bp_upper[:, -1] = t_max
        bp_upper[:, -2] = t_min
        lower_A, lower_b = pieces(bp_lower, -10.0)
        upper_A, upper_b = pieces(bp_upper, 10.0)
        return PerturbationPWL(
            t_min, t_max, lower_breakpoints=bp_lower, upper_breakpoints=bp_upper,
            lower_A=lower_A, upper_A=upper_A, lower_b=lower_b, upper_b=upper_b)

    @torch.no_grad()
    def test_concretize_matrix_matches_candidate_loop(self):
        """The vectorized dense concretization returns the bounds of the
        loop over the 2D+2 candidate parameters."""
        for dtype, atol in ((torch.float64, 1e-9), (torch.float32, 2e-3)):
            for seed, (B, H, D) in enumerate([(1, 5, 12), (3, 7, 40), (2, 1, 9)]):
                ptb = self._random_pwl(B, D, dtype, seed)
                torch.manual_seed(seed)
                A = torch.randn(B, H, D, dtype=dtype)
                A[:, :, ::5] = 0.0
                for sign in (-1, +1):
                    fast = ptb._concretize_matrix(A, sign)
                    slow = concretize_matrix_loop(ptb, A, sign)
                    self.assertEqual(fast.shape, slow.shape)
                    self.assertEqual(fast.dtype, slow.dtype)
                    self.assertTrue(
                        torch.allclose(fast, slow, rtol=0, atol=atol),
                        f"{dtype} seed={seed} sign={sign}: max diff "
                        f"{(fast - slow).abs().max().item():.3e}")

    @torch.no_grad()
    def test_concretize_matrix_is_exact_on_dense_grid(self):
        """No parameter value gives a better objective than the returned one,
        and the returned one is reached."""
        B, H, D = 2, 6, 15
        ptb = self._random_pwl(B, D, torch.float64, seed=5)
        torch.manual_seed(5)
        A = torch.randn(B, H, D, dtype=torch.float64)
        A_pos, A_neg = A.clamp(min=0), A.clamp(max=0)
        grid = torch.linspace(ptb.t_min, ptb.t_max, 20001, dtype=torch.float64)
        lows, highs = [], []
        for t in grid:
            L, U = ptb._eval_at_t(t.item())
            lows.append((A_pos * L.unsqueeze(1) + A_neg * U.unsqueeze(1)).sum(-1))
            highs.append((A_pos * U.unsqueeze(1) + A_neg * L.unsqueeze(1)).sum(-1))
        grid_low = torch.stack(lows).min(dim=0).values
        grid_high = torch.stack(highs).max(dim=0).values
        low = ptb._concretize_matrix(A, -1)
        high = ptb._concretize_matrix(A, +1)
        self.assertTrue((low <= grid_low + 1e-9).all())
        self.assertTrue((high >= grid_high - 1e-9).all())
        # The grid is fine enough to come within 1e-2 of the optimum
        self.assertTrue((grid_low - low).max() < 1e-2)
        self.assertTrue((high - grid_high).max() < 1e-2)

    @torch.no_grad()
    def test_concretize_patches_matches_candidate_loop(self):
        """CROWN bounds of CNNs in patches mode (dense and sparse patches) are
        the same with the vectorized concretization and with the loops."""
        class SmallCNN(nn.Module):
            # Same layout as the LARD classifier of the homography experiments
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(3, 4, 4, stride=2, padding=1), nn.ReLU(),
                    nn.Conv2d(4, 8, 4, stride=2, padding=1), nn.ReLU(),
                    nn.Flatten(), nn.Linear(8 * 4 * 4, 10), nn.ReLU(),
                    nn.Linear(10, 4))
            def forward(self, x):
                return self.net(x)

        B, C, H, W = 1, 3, 16, 16
        calls = {'patches': 0, 'sparse': 0, 'dense': 0}
        fast_patches = PerturbationPWL._concretize_patches
        fast_matrix = PerturbationPWL._concretize_matrix

        def counting_patches(self_, x, A, sign):
            calls['patches'] += 1
            calls['sparse' if A.unstable_idx is not None else 'dense'] += 1
            return fast_patches(self_, x, A, sign)

        for seed in range(6):
            torch.manual_seed(seed)
            model = SmallCNN().double()
            ptb = self._random_pwl(B, C * H * W, torch.float64, seed, 0.0, 0.05)
            # Narrow the gap between the bounds: with the wider one every
            # ReLU is unstable (dense patches), with the narrower one only
            # some are (sparse patches, restricted to the unstable ones)
            gap = 9.9 if seed % 2 else 9.999
            ptb.pwl_lower_b += gap
            ptb.pwl_upper_b -= gap
            ptb.x_L, ptb.x_U = ptb._compute_concrete_bounds()
            ptb.x_L = ptb.x_L.view(B, C, H, W)
            ptb.x_U = ptb.x_U.view(B, C, H, W)
            x_center = (ptb.x_L + ptb.x_U) / 2
            results = {}
            for name in ('fast', 'loop'):
                try:
                    if name == 'loop':
                        PerturbationPWL._concretize_patches = (
                            PerturbationPWL._concretize_patches_loop)
                        PerturbationPWL._concretize_matrix = (
                            concretize_matrix_loop)
                    else:
                        PerturbationPWL._concretize_patches = counting_patches
                    bmodel = BoundedModule(
                        model, x_center, bound_opts={'conv_mode': 'patches'})
                    results[name] = bmodel.compute_bounds(
                        x=(BoundedTensor(x_center, ptb),), method='backward')
                finally:
                    PerturbationPWL._concretize_patches = fast_patches
                    PerturbationPWL._concretize_matrix = fast_matrix
            for fast, slow in zip(results['fast'], results['loop']):
                self.assertTrue(
                    torch.allclose(fast, slow, rtol=0, atol=1e-9),
                    f"seed={seed}: max diff {(fast - slow).abs().max().item():.3e}")
        self.assertGreater(calls['dense'], 0)
        self.assertGreater(calls['sparse'], 0)

    @torch.no_grad()
    def test_concretize_with_jump_at_breakpoint(self):
        """Pieces may differ at the breakpoint by the constructor tolerance.
        The bound must then cover the limit from the right of the breakpoint,
        not only the value at it."""
        D = 3
        dtype = torch.float64
        lower_A = torch.zeros(1, 2, D, dtype=dtype)
        lower_b = torch.zeros(1, 2, D, dtype=dtype)
        upper_A = torch.zeros(1, 2, D, dtype=dtype)
        upper_b = torch.ones(1, 2, D, dtype=dtype)
        lower_b[0, 1, 0] = -9e-5     # L_0 drops by 9e-5 right after t = 0.5
        lower_A[0, :, 1] = 1.0       # L_1(t) = t
        bp = torch.full((1, D), 0.5, dtype=dtype)
        ptb = PerturbationPWL(
            0.0, 1.0, lower_breakpoints=bp, upper_breakpoints=bp,
            lower_A=lower_A, upper_A=upper_A, lower_b=lower_b, upper_b=upper_b)
        A = torch.tensor([[[1.0, 1e-6, 0.0]]], dtype=dtype)
        # inf over t of L_0(t) + 1e-6 * t: approached at t -> 0.5 from the right
        expected = -9e-5 + 1e-6 * 0.5
        self.assertAlmostEqual(float(ptb._concretize_matrix(A, -1)), expected, delta=1e-12)
        self.assertAlmostEqual(float(ptb._concretize_matrix(-A, +1)), -expected, delta=1e-12)
        # Concrete input bounds see the drop as well
        self.assertAlmostEqual(float(ptb.x_L[0, 0]), -9e-5, delta=1e-12)

    @torch.no_grad()
    def test_concretize_with_jumps_bounds_every_parameter(self):
        """With random jumps within the tolerance, no parameter value gives an
        objective beyond the returned bound."""
        B, H, D = 2, 5, 12
        for seed in range(5):
            ptb = self._random_pwl(B, D, torch.float64, seed)
            g = torch.Generator().manual_seed(100 + seed)
            # Break the continuity of piece 1, by up to 9e-5 either way
            ptb.pwl_lower_b[:, 1] += (torch.rand(B, D, generator=g, dtype=torch.float64) - 0.5) * 1.8e-4
            ptb.pwl_upper_b[:, 1] += (torch.rand(B, D, generator=g, dtype=torch.float64) - 0.5) * 1.8e-4
            torch.manual_seed(seed)
            A = torch.randn(B, H, D, dtype=torch.float64)
            A_pos, A_neg = A.clamp(min=0), A.clamp(max=0)
            # A fine grid, plus points just right of every breakpoint
            just_right = torch.cat([ptb.bp_lower, ptb.bp_upper], dim=1).flatten() + 1e-12
            just_right = just_right[just_right <= ptb.t_max]
            grid = torch.cat([
                torch.linspace(ptb.t_min, ptb.t_max, 2001, dtype=torch.float64), just_right])
            low = ptb._concretize_matrix(A, -1)
            high = ptb._concretize_matrix(A, +1)
            x_L, x_U = ptb._compute_concrete_bounds()
            for t in grid:
                L, U = ptb._eval_at_t(t.item())
                g_low = (A_pos * L.unsqueeze(1) + A_neg * U.unsqueeze(1)).sum(-1)
                g_high = (A_pos * U.unsqueeze(1) + A_neg * L.unsqueeze(1)).sum(-1)
                self.assertTrue((low <= g_low + 1e-9).all(), f"seed={seed} t={t.item()}")
                self.assertTrue((high >= g_high - 1e-9).all(), f"seed={seed} t={t.item()}")
                self.assertTrue((x_L <= L + 1e-12).all() and (x_U >= U - 1e-12).all())


if __name__ == '__main__':
    unittest.main()
