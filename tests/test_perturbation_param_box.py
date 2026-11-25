"""Tests for PerturbationParamBox (two planes over a box of parameters)."""
import unittest

import torch
import torch.nn as nn

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.patches import Patches
from auto_LiRPA.perturbations import (
    PerturbationLinear, PerturbationLpNorm, PerturbationParamBox, PerturbationPWL)

try:
    from scipy.optimize import linprog
except ImportError:
    linprog = None


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def random_box(B, D, P=2, seed=0, dtype=torch.float64, gap=0.05, gap_slope=0.3,
               width=1.0, degenerate=None):
    """A valid perturbation with planes that are not parallel.

    The upper plane is the lower plane plus an affine gap whose minimum
    over the box is `gap` and whose slopes are at most `gap_slope`, so the
    two planes are closest at one corner. `width` scales the box.
    Returns the perturbation and a function giving (L, U) in double
    precision at parameters (N, P) for batch element b.
    """
    g = torch.Generator().manual_seed(seed)

    def rand(*shape):
        return torch.rand(*shape, generator=g, dtype=torch.float64)

    lo = rand(B, P) - 0.7
    hi = lo + width * (0.2 + rand(B, P))
    if degenerate is not None:
        hi[:, degenerate] = lo[:, degenerate]
    lower_A = rand(B, D, P) * 2 - 1
    lower_b = rand(B, D) * 2 - 1
    gap_A = (rand(B, D, P) * 2 - 1) * gap_slope
    gap_min = torch.minimum(gap_A * lo.unsqueeze(1), gap_A * hi.unsqueeze(1)).sum(-1)
    upper_A = lower_A + gap_A
    upper_b = lower_b + gap - gap_min
    ptb = PerturbationParamBox(
        lo.to(dtype), hi.to(dtype),
        lower_A=lower_A.to(dtype), lower_b=lower_b.to(dtype),
        upper_A=upper_A.to(dtype), upper_b=upper_b.to(dtype))

    def exact(b, params):
        L = params @ ptb.box_lower_A[b].double().T + ptb.box_lower_b[b].double()
        U = params @ ptb.box_upper_A[b].double().T + ptb.box_upper_b[b].double()
        return L, U

    return ptb, exact


def corners(ptb, b):
    """The 2^P corners of the box of batch element b: (2^P, P), double."""
    lo, hi = ptb.param_lower[b].double(), ptb.param_upper[b].double()
    return torch.cartesian_prod(
        *[torch.stack([l, h]) for l, h in zip(lo, hi)]).reshape(-1, len(lo))


def dense_params(ptb, b, n=41, extra=2000, seed=0):
    """A grid over the box of batch element b (its corners included), plus
    random points: (N, P) in double precision."""
    lo, hi = ptb.param_lower[b].double(), ptb.param_upper[b].double()
    axes = [torch.linspace(float(l), float(h), n, dtype=torch.float64)
            if h > l else lo.new_tensor([float(l)]) for l, h in zip(lo, hi)]
    grid = torch.cartesian_prod(*axes).reshape(-1, len(lo))
    g = torch.Generator().manual_seed(seed)
    rand = lo + torch.rand(extra, len(lo), generator=g, dtype=torch.float64) * (hi - lo)
    return torch.cat([grid, rand])


def min_over_inputs(A, L, U):
    """min over x in [L, U] of A x, for every parameter sample: (H, N)."""
    return A.clamp(min=0) @ L.T + A.clamp(max=0) @ U.T


def brute_force_range(A, exact, b, params):
    """Smallest and largest A x over the sampled parameters: (H,) each."""
    L, U = exact(b, params)
    return (min_over_inputs(A, L, U).min(1).values,
            -min_over_inputs(-A, L, U).min(1).values)


def outputs_on_samples(model, exact, b, params, shape, dtype, seed=0):
    """Network outputs for inputs on the lower planes, on the upper planes
    and at random heights between them: (3 N, out)."""
    g = torch.Generator().manual_seed(seed)
    L, U = exact(b, params)
    heights = [torch.zeros_like(L), torch.ones_like(L),
               torch.rand(L.shape, generator=g, dtype=torch.float64)]
    inputs = torch.cat([L + t * (U - L) for t in heights])
    with torch.no_grad():
        return model(inputs.to(dtype).view(-1, *shape))


# ──────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────
class TestPerturbationParamBox(unittest.TestCase):

    def test_sanity_checks(self):
        """Wrong shapes, a reversed box and planes that cross in the box raise a ValueError."""
        B, D, P = 2, 3, 2
        lo, hi = torch.zeros(B, P), torch.ones(B, P)
        A, b = torch.zeros(B, D, P), torch.zeros(B, D)
        kw = dict(lower_A=A, upper_A=A, lower_b=b, upper_b=b + 1)
        PerturbationParamBox(lo, hi, **kw)
        with self.assertRaises(ValueError):
            PerturbationParamBox(hi, lo, **kw)
        with self.assertRaises(ValueError):
            PerturbationParamBox(lo[:1], hi, **kw)
        with self.assertRaises(ValueError):
            PerturbationParamBox(lo, hi, **{**kw, 'lower_b': b[:, :2]})
        with self.assertRaises(ValueError):
            PerturbationParamBox(lo, hi, **{**kw, 'lower_A': A[..., 0]})
        with self.assertRaises(ValueError):
            PerturbationParamBox(lo, hi, **{**kw, 'upper_b': b - 1})
        # Upper plane above the lower one at three corners, below at (1, 1)
        crossing = torch.full((B, D, P), -0.6)
        with self.assertRaises(ValueError):
            PerturbationParamBox(lo, hi, **{**kw, 'upper_A': crossing})

    def test_hand_computed(self):
        """Two inputs that move together with the parameters.

        On k in [0, 1]^2:
            k1      <= x1 <= k1 + 0.1,
            k1 + k2 <= x2 <= k1 + k2 + 0.2.
        x1 - x2 is between -k2 - 0.2 and -k2 + 0.1: its range is
        [-1.2, 0.1]. The intervals x1 in [0, 1.1], x2 in [0, 2.2] alone
        would give [-2.2, 1.1].
        """
        ptb = PerturbationParamBox(
            torch.zeros(1, 2, dtype=torch.float64),
            torch.ones(1, 2, dtype=torch.float64),
            lower_A=torch.tensor([[[1.0, 0.0], [1.0, 1.0]]], dtype=torch.float64),
            upper_A=torch.tensor([[[1.0, 0.0], [1.0, 1.0]]], dtype=torch.float64),
            lower_b=torch.tensor([[0.0, 0.0]], dtype=torch.float64),
            upper_b=torch.tensor([[0.1, 0.2]], dtype=torch.float64))
        torch.testing.assert_close(
            ptb.x_L, torch.tensor([[0.0, 0.0]], dtype=torch.float64))
        torch.testing.assert_close(
            ptb.x_U, torch.tensor([[1.1, 2.2]], dtype=torch.float64))
        A = torch.tensor([[[1.0, -1.0]]], dtype=torch.float64)
        self.assertAlmostEqual(ptb.concretize(None, A, sign=-1).item(), -1.2, places=12)
        self.assertAlmostEqual(ptb.concretize(None, A, sign=+1).item(), 0.1, places=12)
        interval = PerturbationLpNorm(x_L=ptb.x_L, x_U=ptb.x_U)
        x = (ptb.x_L + ptb.x_U) / 2
        self.assertAlmostEqual(interval.concretize(x, A, sign=-1).item(), -2.2, places=12)
        self.assertAlmostEqual(interval.concretize(x, A, sign=+1).item(), 1.1, places=12)

    def test_concrete_bounds(self):
        """x_L and x_U are the exact range of every input over the box, and
        `eval_at` returns the planes at given parameters."""
        ptb, exact = random_box(2, 7, seed=1)
        for b in range(2):
            L, U = exact(b, dense_params(ptb, b))
            torch.testing.assert_close(ptb.x_L[b], L.min(0).values, atol=1e-12, rtol=0)
            torch.testing.assert_close(ptb.x_U[b], U.max(0).values, atol=1e-12, rtol=0)
        params = torch.stack([dense_params(ptb, b)[-1] for b in range(2)])
        L_at, U_at = ptb.eval_at(params)
        for b in range(2):
            L, U = exact(b, params[b:b + 1])
            torch.testing.assert_close(L_at[b], L[0], atol=1e-12, rtol=0)
            torch.testing.assert_close(U_at[b], U[0], atol=1e-12, rtol=0)

    def test_concretize_is_exact_on_dense_grid(self):
        """The bound equals the optimum over a dense sample of the box, and
        that optimum is at a corner."""
        B, D, H = 3, 9, 5
        ptb, exact = random_box(B, D, seed=3)
        A = torch.randn(B, H, D, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
        lower = ptb.concretize(None, A, sign=-1)
        upper = ptb.concretize(None, A, sign=+1)
        for b in range(B):
            dense_min, dense_max = brute_force_range(A[b], exact, b, dense_params(ptb, b))
            corner_min, corner_max = brute_force_range(A[b], exact, b, corners(ptb, b))
            torch.testing.assert_close(lower[b], dense_min, atol=1e-11, rtol=0)
            torch.testing.assert_close(upper[b], dense_max, atol=1e-11, rtol=0)
            torch.testing.assert_close(lower[b], corner_min, atol=1e-11, rtol=0)
            torch.testing.assert_close(upper[b], corner_max, atol=1e-11, rtol=0)

    @unittest.skipIf(linprog is None, "needs SciPy")
    def test_concretize_solves_the_linear_program(self):
        """The bound equals the optimum of the linear program over the
        parameters and the inputs, solved by an LP solver."""
        B, D, H = 2, 6, 4
        for P in (1, 2, 3):
            ptb, _ = random_box(B, D, P=P, seed=80 + P)
            A = torch.randn(B, H, D, dtype=torch.float64,
                            generator=torch.Generator().manual_seed(P))
            lower = ptb.concretize(None, A, sign=-1)
            upper = ptb.concretize(None, A, sign=+1)
            eye = torch.eye(D, dtype=torch.float64)
            for b in range(B):
                # Variables (k, x):  lower_A k + lower_b <= x <= upper_A k + upper_b
                A_ub = torch.cat([torch.cat([ptb.box_lower_A[b], -eye], dim=1),
                                  torch.cat([-ptb.box_upper_A[b], eye], dim=1)]).numpy()
                b_ub = torch.cat([-ptb.box_lower_b[b], ptb.box_upper_b[b]]).numpy()
                bounds = ([(float(l), float(h)) for l, h in
                           zip(ptb.param_lower[b], ptb.param_upper[b])]
                          + [(None, None)] * D)
                for h in range(H):
                    cost = torch.cat([torch.zeros(P, dtype=torch.float64), A[b, h]]).numpy()
                    lp_min = linprog(cost, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
                    lp_max = linprog(-cost, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
                    self.assertTrue(lp_min.success and lp_max.success)
                    self.assertAlmostEqual(lower[b, h].item(), lp_min.fun, places=8)
                    self.assertAlmostEqual(upper[b, h].item(), -lp_max.fun, places=8)

    def test_more_parameters(self):
        """The closed form does not depend on the number of parameters."""
        for P in (1, 3, 4):
            ptb, exact = random_box(1, 6, P=P, seed=60 + P)
            A = torch.randn(1, 8, 6, dtype=torch.float64,
                            generator=torch.Generator().manual_seed(P))
            params = dense_params(ptb, 0, n=9 if P > 1 else 400, extra=500)
            dense_min, dense_max = brute_force_range(A[0], exact, 0, params)
            torch.testing.assert_close(
                ptb.concretize(None, A, sign=-1)[0], dense_min, atol=1e-10, rtol=0)
            torch.testing.assert_close(
                ptb.concretize(None, A, sign=+1)[0], dense_max, atol=1e-10, rtol=0)

    def test_linear_model_is_exact(self):
        """Backward bounds of a linear network are its exact output range."""
        B, D = 2, 8
        torch.manual_seed(0)
        model = nn.Linear(D, 3).double()
        ptb, exact = random_box(B, D, seed=11)
        x = (ptb.x_L + ptb.x_U) / 2
        bounded = BoundedModule(model, x)
        lb, ub = bounded.compute_bounds(x=(BoundedTensor(x, ptb),), method='backward')
        W, bias = model.weight.detach(), model.bias.detach()
        for b in range(B):
            dense_min, dense_max = brute_force_range(W, exact, b, dense_params(ptb, b))
            torch.testing.assert_close(lb[b], dense_min + bias, atol=1e-10, rtol=0)
            torch.testing.assert_close(ub[b], dense_max + bias, atol=1e-10, rtol=0)

    def test_never_looser_than_the_interval(self):
        """On a linear network the bounds are inside those of the interval
        perturbation built from x_L and x_U, and strictly inside for
        planes with a slope."""
        B, D = 2, 8
        torch.manual_seed(1)
        model = nn.Linear(D, 3).double()
        ptb, _ = random_box(B, D, seed=12)
        x = (ptb.x_L + ptb.x_U) / 2
        bounded = BoundedModule(model, x)
        lb, ub = bounded.compute_bounds(x=(BoundedTensor(x, ptb),), method='backward')
        interval = PerturbationLpNorm(x_L=ptb.x_L, x_U=ptb.x_U)
        lb_i, ub_i = bounded.compute_bounds(
            x=(BoundedTensor(x, interval),), method='backward')
        self.assertTrue((lb > lb_i + 1e-3).all())
        self.assertTrue((ub < ub_i - 1e-3).all())

    def _check_network_soundness(self, model, shape, seed, dtype=torch.float64, **box):
        """Bounds of `model` hold for inputs sampled between the planes at
        a dense sample of parameters. Returns the perturbation, the centre
        input and the bounds."""
        B = 2
        D = 1
        for s in shape:
            D *= s
        ptb, exact = random_box(B, D, seed=seed, dtype=dtype, **box)
        x = ((ptb.x_L + ptb.x_U) / 2).view(B, *shape)
        model = model.to(dtype)
        bounded = BoundedModule(model, x[:1])
        lb, ub = bounded.compute_bounds(x=(BoundedTensor(x, ptb),), method='backward')
        tol = 1e-9 if dtype == torch.float64 else 1e-4
        for b in range(B):
            out = outputs_on_samples(
                model, exact, b, dense_params(ptb, b, n=15, extra=300), shape, dtype, seed)
            self.assertTrue((out >= lb[b] - tol).all(), (out - lb[b]).min())
            self.assertTrue((out <= ub[b] + tol).all(), (ub[b] - out).min())
        return ptb, x, lb, ub

    def test_relu_network_soundness(self):
        """ReLU network: no sampled output is outside the bounds."""
        for seed in (21, 22, 23):
            torch.manual_seed(seed)
            model = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 12),
                                  nn.ReLU(), nn.Linear(12, 3))
            self._check_network_soundness(model, (10,), seed=seed)

    def test_conv_network_soundness_and_patches_match_matrix(self):
        """Convolutional network on an image-shaped input: sound, and the
        same bounds with patches (dense and sparse) and with dense matrices."""
        seen = {'dense': 0, 'sparse': 0}
        concretize = PerturbationParamBox.concretize

        def counting_concretize(self_, x, A, sign=-1, aux=None):
            if isinstance(A, Patches):
                seen['sparse' if A.unstable_idx is not None else 'dense'] += 1
            return concretize(self_, x, A, sign, aux)

        # Wide planes: every ReLU is unstable (dense patches). Narrow ones:
        # only some are (sparse patches, restricted to the unstable ones)
        wide = dict(gap=0.05, gap_slope=0.3, width=1.0)
        narrow = dict(gap=1e-3, gap_slope=1e-2, width=0.05)
        for dtype, box in ((torch.float64, wide), (torch.float64, narrow),
                           (torch.float32, wide), (torch.float32, narrow)):
            torch.manual_seed(3)
            # Same layout as the LARD classifier of the homography experiments
            model = nn.Sequential(
                nn.Conv2d(3, 4, 4, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(4, 8, 4, stride=2, padding=1), nn.ReLU(),
                nn.Flatten(), nn.Linear(8 * 4 * 4, 10), nn.ReLU(),
                nn.Linear(10, 4))
            try:
                PerturbationParamBox.concretize = counting_concretize
                ptb, x, lb, ub = self._check_network_soundness(
                    model, (3, 16, 16), seed=31, dtype=dtype, **box)
            finally:
                PerturbationParamBox.concretize = concretize
            # Matrix mode builds its first coefficients in the default dtype
            default = torch.get_default_dtype()
            torch.set_default_dtype(dtype)
            try:
                bounded = BoundedModule(
                    model.to(dtype), x[:1], bound_opts={'conv_mode': 'matrix'})
                lb_m, ub_m = bounded.compute_bounds(
                    x=(BoundedTensor(x, ptb),), method='backward')
            finally:
                torch.set_default_dtype(default)
            atol = 1e-9 if dtype == torch.float64 else 1e-4
            torch.testing.assert_close(lb, lb_m, atol=atol, rtol=0)
            torch.testing.assert_close(ub, ub_m, atol=atol, rtol=0)
        self.assertGreater(seen['dense'], 0)
        self.assertGreater(seen['sparse'], 0)

    def test_batch_elements_are_independent(self):
        """Bounding several boxes as one batch gives the bounds of each box alone."""
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 2)).double()
        ptb, _ = random_box(3, 6, seed=40)
        x = (ptb.x_L + ptb.x_U) / 2
        bounded = BoundedModule(model, x[:1])
        lb, ub = bounded.compute_bounds(x=(BoundedTensor(x, ptb),), method='backward')
        for b in range(3):
            one = PerturbationParamBox(
                ptb.param_lower[b:b + 1], ptb.param_upper[b:b + 1],
                lower_A=ptb.box_lower_A[b:b + 1], lower_b=ptb.box_lower_b[b:b + 1],
                upper_A=ptb.box_upper_A[b:b + 1], upper_b=ptb.box_upper_b[b:b + 1])
            lb1, ub1 = bounded.compute_bounds(
                x=(BoundedTensor(x[b:b + 1], one),), method='backward')
            torch.testing.assert_close(lb[b:b + 1], lb1, atol=1e-10, rtol=0)
            torch.testing.assert_close(ub[b:b + 1], ub1, atol=1e-10, rtol=0)

    def _check_matches_pwl(self, ptb, axis):
        """Same bounds as the 1-D perturbation with two identical pieces
        built from the lines the planes draw along `axis` (the other
        parameters are fixed)."""
        torch.manual_seed(1)
        B, D, P = ptb.B, ptb.D, ptb.P
        model = nn.Sequential(nn.Linear(D, 14), nn.ReLU(), nn.Linear(14, 3)).double()
        x = (ptb.x_L + ptb.x_U) / 2
        bounded = BoundedModule(model, x[:1])
        lb, ub = bounded.compute_bounds(x=(BoundedTensor(x, ptb),), method='backward')
        others = [p for p in range(P) if p != axis]
        for b in range(B):
            t_min, t_max = float(ptb.param_lower[b, axis]), float(ptb.param_upper[b, axis])

            def line(A, bias):
                # (1, 2, D) slopes and intercepts along `axis`
                slopes = A[b, :, axis]
                intercepts = bias[b] + A[b][:, others] @ ptb.param_lower[b, others]
                return (slopes.expand(1, 2, D).contiguous(),
                        intercepts.expand(1, 2, D).contiguous())

            lA, lB = line(ptb.box_lower_A, ptb.box_lower_b)
            uA, uB = line(ptb.box_upper_A, ptb.box_upper_b)
            bp = torch.full((1, D), 0.5 * (t_min + t_max), dtype=torch.float64)
            pwl = PerturbationPWL(t_min, t_max, lower_breakpoints=bp, upper_breakpoints=bp,
                                  lower_A=lA, upper_A=uA, lower_b=lB, upper_b=uB)
            lb1, ub1 = bounded.compute_bounds(
                x=(BoundedTensor(x[b:b + 1], pwl),), method='backward')
            torch.testing.assert_close(lb[b:b + 1], lb1, atol=1e-10, rtol=0)
            torch.testing.assert_close(ub[b:b + 1], ub1, atol=1e-10, rtol=0)

    def test_one_parameter_matches_pwl_perturbation(self):
        """A box with one parameter is the 1-D perturbation with the same lines."""
        ptb, _ = random_box(2, 10, P=1, seed=50)
        self._check_matches_pwl(ptb, axis=0)

    def test_zero_width_axis_matches_pwl_perturbation(self):
        """A box of two parameters that is a segment is the 1-D perturbation along it."""
        ptb, _ = random_box(2, 10, seed=51, degenerate=1)
        self._check_matches_pwl(ptb, axis=0)
        ptb, _ = random_box(2, 10, seed=52, degenerate=0)
        self._check_matches_pwl(ptb, axis=1)

    def test_matches_linear_perturbation(self):
        """On a fully connected network the bounds are those of
        `PerturbationLinear` built from the same planes and box."""
        torch.manual_seed(2)
        B, D = 3, 10
        model = nn.Sequential(nn.Linear(D, 14), nn.ReLU(), nn.Linear(14, 12),
                              nn.ReLU(), nn.Linear(12, 3)).double()
        for P in (1, 2, 3):
            ptb, _ = random_box(B, D, P=P, seed=90 + P)
            linear = PerturbationLinear(
                ptb.box_lower_A, ptb.box_upper_A, ptb.box_lower_b, ptb.box_upper_b,
                ptb.param_lower, ptb.param_upper)
            x = (ptb.x_L + ptb.x_U) / 2
            bounded = BoundedModule(model, x[:1])
            lb, ub = bounded.compute_bounds(x=(BoundedTensor(x, ptb),), method='backward')
            lb1, ub1 = bounded.compute_bounds(x=(BoundedTensor(x, linear),), method='backward')
            torch.testing.assert_close(lb, lb1, atol=1e-10, rtol=0)
            torch.testing.assert_close(ub, ub1, atol=1e-10, rtol=0)

    def test_forward_mode_is_refused(self):
        """Forward mode raises instead of returning bounds."""
        ptb, _ = random_box(1, 4, seed=70)
        with self.assertRaises(NotImplementedError):
            ptb.init(None, forward=True)
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 2)).double()
        x = (ptb.x_L + ptb.x_U) / 2
        bounded = BoundedModule(model, x)
        with self.assertRaises(NotImplementedError):
            bounded.compute_bounds(x=(BoundedTensor(x, ptb),), method='forward')


if __name__ == '__main__':
    unittest.main()
