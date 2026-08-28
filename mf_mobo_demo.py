"""
Minimal reproducible demonstration of the methodology used in the paper:
"Adaptive Fidelity Selection in Multi-Objective Bayesian Optimisation
 for CFD-Based Engineering Design"

This is an educational/demo implementation on synthetic functions.  It is
not the production OpenFOAM code used to generate the paper results.

Key features reproduced:
  * one expensive CFD-like objective + one exact objective;
  * two fidelities (LF/HF);
  * recursive autoregressive GP: f_H = rho_AR f_L + delta;
  * augmented weighted Tchebycheff scalarisation;
  * closed-form EI for one Gaussian + one deterministic objective;
  * cost-aware joint design/fidelity selection: EI(x,m)/c_m;
  * custom LF numerical-quality gate and LF->HF escalation;
  * EHF budget termination;
  * optional Fixed-MF and HF-only baselines.

To keep the example lightweight, acquisition optimisation uses a seeded
random candidate set instead of CMA-ES.  Replacing the candidate search by
CMA-ES does not change the acquisition function or the budget logic.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import math
import numpy as np

# Paper uses GpyTORCH for the GP training and CMA-ES for acquisition optimisation.  This is a minimal demo using scikit-learn and a random candidate set for acquisition optimisation.
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
except ImportError as exc:
    raise ImportError(
        "This demo requires scikit-learn. Install with: pip install -r requirements.txt"
    ) from exc

try:
    from scipy.special import ndtr
except ImportError as exc:
    raise ImportError(
        "This demo requires scipy. Install with: pip install -r requirements.txt"
    ) from exc


# ---------------------------------------------------------------------------
# 1. Black-box simulator interface
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    x: np.ndarray
    fidelity: str
    f1: float
    f2: float
    diagnostics: Dict[str, float]


class DummyCFDSimulator:
    """
    Synthetic drop-in replacement for a CFD solver.

    Contract
    --------
    evaluate(x, fidelity) -> SimulationResult

    Input
    -----
    x : design vector in [0,1]^d
    fidelity : "LF" or "HF"

    Output
    ------
    f1 : CFD-like expensive objective (minimise)
    f2 : exact deterministic objective (minimise)
    diagnostics : values used by the numerical-quality gate

    The LF response is biased but correlated with HF.  The quality
    diagnostics are deliberately worse in part of the design space so that
    the LF->HF escalation mechanism can be demonstrated.
    """

    def __init__(self, dimension: int = 2):
        self.dimension = int(dimension)

    @staticmethod
    def exact_objective(x: np.ndarray) -> float:
        # Compactness-like objective.  It is known exactly and has zero
        # predictive uncertainty.
        return float(x[0])

    def _hf_objective(self, x: np.ndarray) -> float:
        # Smooth nonlinear target objective with a trade-off against f2=x[0].
        x = np.asarray(x, dtype=float)
        value = (
            0.10
            + (x[0] - 0.78) ** 2
            + 0.035 * np.sin(7.0 * x[0])
        )
        if len(x) > 1:
            value += 0.20 * np.sum((x[1:] - 0.45) ** 2)
            value += 0.015 * np.sin(5.0 * np.sum(x[1:]))
        return float(value)

    def _lf_objective(self, x: np.ndarray) -> float:
        # Cheaper biased approximation to HF.
        hf = self._hf_objective(x)
        bias = 0.035 + 0.055 * np.sin(2.5 * np.pi * x[0])
        if len(x) > 1:
            bias += 0.015 * np.mean(x[1:] - 0.5)
        return float(0.92 * hf + bias)

    def evaluate(self, x: np.ndarray, fidelity: str) -> SimulationResult:
        x = np.asarray(x, dtype=float)
        if x.ndim != 1 or len(x) != self.dimension:
            raise ValueError(f"x must be a 1-D vector of length {self.dimension}")
        if np.any(x < 0.0) or np.any(x > 1.0):
            raise ValueError("Dummy demo expects each design variable in [0,1].")
        fidelity = fidelity.upper()
        if fidelity not in {"LF", "HF"}:
            raise ValueError("fidelity must be 'LF' or 'HF'")

        f2 = self.exact_objective(x)
        if fidelity == "HF":
            f1 = self._hf_objective(x)
            diagnostics = {
                "kp_stability": 3.0e-4,
                "flux_imbalance": 2.0e-4,
            }
        else:
            f1 = self._lf_objective(x)

            # Custom CFD-like quality diagnostics.
            # A region near large x[0] and oscillatory wall-shape coordinates
            # is made deliberately difficult for the LF model.
            shape_term = 0.0 if len(x) == 1 else float(np.mean(np.abs(x[1:] - 0.5)))
            kp_stability = (
                7.0e-4
                + 2.2e-3 * max(0.0, x[0] - 0.55) / 0.45
                + 7.0e-4 * shape_term
            )
            flux_imbalance = (
                2.5e-4
                + 1.15e-3 * max(0.0, x[0] - 0.72) / 0.28
                + 2.0e-4 * shape_term
            )
            diagnostics = {
                "kp_stability": float(kp_stability),
                "flux_imbalance": float(flux_imbalance),
            }

        return SimulationResult(
            x=x.copy(),
            fidelity=fidelity,
            f1=float(f1),
            f2=float(f2),
            diagnostics=diagnostics,
        )


# ---------------------------------------------------------------------------
# 2. Quality gate
# ---------------------------------------------------------------------------

def quality_control(
    result: SimulationResult,
    kp_stability_limit: float = 2.0e-3,
    flux_imbalance_limit: float = 1.0e-3,
) -> Tuple[bool, Dict[str, bool]]:
    """
    Example numerical-quality gate.

    It mirrors the paper's logic:
        Kp stability <= 2e-3
        flux imbalance <= 1e-3

    HF is accepted directly in this demo.  For LF, failure of either test
    causes escalation to HF at the same design.
    """
    if result.fidelity == "HF":
        return True, {"kp_stability": True, "flux_imbalance": True}

    checks = {
        "kp_stability": result.diagnostics["kp_stability"] <= kp_stability_limit,
        "flux_imbalance": result.diagnostics["flux_imbalance"] <= flux_imbalance_limit,
    }
    return bool(all(checks.values())), checks


# ---------------------------------------------------------------------------
# 3. Recursive multi-fidelity GP
# ---------------------------------------------------------------------------

class RecursiveMFGP:
    """
    f_H(x) = rho_AR f_L(x) + delta(x)

    LF GP and discrepancy GP both use Matérn-5/2 covariance.  This demo uses
    scikit-learn's marginal-likelihood optimiser rather than the production
    training code used in the paper.
    """

    def __init__(self, dimension: int, random_state: int = 0):
        self.dimension = dimension
        self.random_state = random_state
        self.gp_lf = None
        self.gp_delta = None
        self.rho_ar = 1.0

    def _kernel(self):
        return (
            ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(
                length_scale=np.ones(self.dimension),
                length_scale_bounds=(1e-2, 10.0),
                nu=2.5,
            )
            + WhiteKernel(noise_level=1e-7, noise_level_bounds=(1e-10, 1e-3))
        )

    def fit(
        self,
        X_lf: np.ndarray,
        y_lf: np.ndarray,
        X_hf: np.ndarray,
        y_hf: np.ndarray,
    ) -> "RecursiveMFGP":
        if len(X_lf) < 2 or len(X_hf) < 2:
            raise ValueError("Need at least two trusted LF and two HF points.")

        self.gp_lf = GaussianProcessRegressor(
            kernel=self._kernel(),
            normalize_y=True,
            n_restarts_optimizer=1,
            random_state=self.random_state,
        )
        self.gp_lf.fit(X_lf, y_lf)

        mu_lf_at_hf = self.gp_lf.predict(X_hf)
        denom = float(mu_lf_at_hf @ mu_lf_at_hf) + 1e-12
        self.rho_ar = float((mu_lf_at_hf @ y_hf) / denom)

        residual = y_hf - self.rho_ar * mu_lf_at_hf
        self.gp_delta = GaussianProcessRegressor(
            kernel=self._kernel(),
            normalize_y=True,
            n_restarts_optimizer=1,
            random_state=self.random_state + 1,
        )
        self.gp_delta.fit(X_hf, residual)
        return self

    def predict(self, X: np.ndarray, fidelity: str) -> Tuple[np.ndarray, np.ndarray]:
        fidelity = fidelity.upper()
        mu_lf, sd_lf = self.gp_lf.predict(X, return_std=True)
        if fidelity == "LF":
            return mu_lf, np.maximum(sd_lf, 1e-12)

        mu_delta, sd_delta = self.gp_delta.predict(X, return_std=True)
        mu_hf = self.rho_ar * mu_lf + mu_delta
        var_hf = (self.rho_ar ** 2) * (sd_lf ** 2) + sd_delta ** 2
        return mu_hf, np.sqrt(np.maximum(var_hf, 1e-18))


# ---------------------------------------------------------------------------
# 4. Augmented Tchebycheff EI
# ---------------------------------------------------------------------------

def phi(z):
    z = np.asarray(z, dtype=float)
    return np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)


def augmented_tchebycheff(f1n, f2n, w1, rho_aug=0.05):
    w2 = 1.0 - w1
    a = w1 * f1n
    b = w2 * f2n
    return np.maximum(a, b) + rho_aug * (a + b)


def closed_form_ei_one_random_objective(
    mu_A: np.ndarray,
    sigma_A: np.ndarray,
    b: np.ndarray,
    g_star: float,
    rho_aug: float = 0.05,
) -> np.ndarray:
    """
    Closed-form EI for
        G = max(A,b) + rho_aug (A+b),
    where A is Gaussian and b is deterministic.

    Correct second-branch PDF term:
        +(1+rho_aug) sigma_A [phi(z2) - phi(zb)].
    """
    mu_A = np.asarray(mu_A, dtype=float)
    sigma_A = np.asarray(sigma_A, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.zeros(np.broadcast(mu_A, sigma_A, b).shape, dtype=float)

    deterministic = sigma_A <= 1e-12
    if np.any(deterministic):
        G = np.maximum(mu_A[deterministic], b[deterministic]) + rho_aug * (
            mu_A[deterministic] + b[deterministic]
        )
        out[deterministic] = np.maximum(g_star - G, 0.0)

    idx = ~deterministic
    if not np.any(idx):
        return out

    mu = mu_A[idx]
    sd = sigma_A[idx]
    bb = b[idx]

    # Branch 1: A <= b
    tau1 = (g_star - (1.0 + rho_aug) * bb) / rho_aug
    u1 = np.minimum(bb, tau1)
    z1 = (u1 - mu) / sd
    ei1 = (
        (g_star - (1.0 + rho_aug) * bb - rho_aug * mu) * ndtr(z1)
        + rho_aug * sd * phi(z1)
    )
    ei1 = np.maximum(ei1, 0.0)

    # Branch 2: A > b
    tau2 = (g_star - rho_aug * bb) / (1.0 + rho_aug)
    zb = (bb - mu) / sd
    z2 = (tau2 - mu) / sd
    active = tau2 > bb
    ei2 = np.zeros_like(mu)
    if np.any(active):
        P = ndtr(z2[active]) - ndtr(zb[active])
        ei2[active] = (
            (g_star - rho_aug * bb[active] - (1.0 + rho_aug) * mu[active]) * P
            + (1.0 + rho_aug)
            * sd[active]
            * (phi(z2[active]) - phi(zb[active]))
        )
        ei2[active] = np.maximum(ei2[active], 0.0)

    out[idx] = ei1 + ei2
    return out


# ---------------------------------------------------------------------------
# 5. Sequential optimiser and EHF accounting
# ---------------------------------------------------------------------------

@dataclass
class EvalRecord:
    iteration: int
    requested_fidelity: str
    actual_fidelity: str
    x: list
    f1: float
    f2: float
    quality_pass: bool
    escalated: bool
    acquisition: float
    cumulative_ehf: float


class DemoOptimizer:
    COST = {"LF": 1.0, "HF": 14.5}

    def __init__(
        self,
        simulator: DummyCFDSimulator,
        budget_ehf: float = 12.0,
        seed: int = 4,
        candidate_pool: int = 500,
        rho_aug: float = 0.05,
    ):
        self.sim = simulator
        self.d = simulator.dimension
        self.budget_ehf = float(budget_ehf)
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.candidate_pool = int(candidate_pool)
        self.rho_aug = rho_aug

        self.X_lf: List[np.ndarray] = []
        self.y_lf: List[float] = []
        self.X_hf: List[np.ndarray] = []
        self.y_hf: List[float] = []
        self.f2_hf: List[float] = []
        self.records: List[EvalRecord] = []
        self.cumulative_ehf = 0.0

        # A small shuffled Das-Dennis-like cycle for two objectives.
        weights = np.linspace(0.0, 1.0, 21)
        self.rng.shuffle(weights)
        self.weights = weights
        self.weight_idx = 0

    @classmethod
    def ehf_cost(cls, fidelity: str) -> float:
        return cls.COST[fidelity] / cls.COST["HF"]

    def _charge(self, fidelity: str):
        self.cumulative_ehf += self.ehf_cost(fidelity)

    def _fits(self, fidelity: str) -> bool:
        return self.cumulative_ehf + self.ehf_cost(fidelity) <= self.budget_ehf + 1e-12

    def initialise(self, n_lf: int = 6, n_hf: int = 3):
        """Nested initial design: all initial HF points are also LF points."""
        if n_hf > n_lf:
            raise ValueError("n_hf must be <= n_lf for nested initial design.")
        X = self.rng.random((n_lf, self.d))
        hf_ids = np.arange(n_hf)

        for i, x in enumerate(X):
            if not self._fits("LF"):
                break
            r_lf = self.sim.evaluate(x, "LF")
            passed, _ = quality_control(r_lf)
            self._charge("LF")
            # Initial nested design: for a failed LF, do not trust it.
            if passed:
                self.X_lf.append(x.copy())
                self.y_lf.append(r_lf.f1)

            if i in hf_ids and self._fits("HF"):
                r_hf = self.sim.evaluate(x, "HF")
                self._charge("HF")
                self.X_hf.append(x.copy())
                self.y_hf.append(r_hf.f1)
                self.f2_hf.append(r_hf.f2)

        if len(self.X_lf) < 2 or len(self.X_hf) < 2:
            raise RuntimeError("Initial design did not provide enough trusted LF/HF points.")

    def _normalisation(self):
        y = np.asarray(self.y_hf)
        lo = float(np.min(y))
        hi = float(np.max(y))
        if hi - lo < 1e-8:
            hi = lo + 1.0
        # Exact f2=x[0] is already in [0,1].
        return lo, hi

    def _g_star(self, w1: float, lo: float, hi: float) -> float:
        f1n = (np.asarray(self.y_hf) - lo) / (hi - lo)
        f2n = np.asarray(self.f2_hf)
        g = augmented_tchebycheff(f1n, f2n, w1, self.rho_aug)
        return float(np.min(g))

    def _select_adaptive(self, model: RecursiveMFGP):
        w1 = float(self.weights[self.weight_idx % len(self.weights)])
        self.weight_idx += 1
        w2 = 1.0 - w1
        lo, hi = self._normalisation()
        g_star = self._g_star(w1, lo, hi)

        Xcand = self.rng.random((self.candidate_pool, self.d))
        f2n = Xcand[:, 0]

        best = None
        for fidelity in ("LF", "HF"):
            if not self._fits(fidelity):
                continue
            mu, sd = model.predict(Xcand, fidelity)
            mu_n = (mu - lo) / (hi - lo)
            sd_n = sd / (hi - lo)
            mu_A = w1 * mu_n
            sigma_A = abs(w1) * sd_n
            b = w2 * f2n
            ei = closed_form_ei_one_random_objective(
                mu_A, sigma_A, b, g_star, self.rho_aug
            )
            alpha = ei / self.COST[fidelity]
            j = int(np.argmax(alpha))
            candidate = (float(alpha[j]), Xcand[j].copy(), fidelity)
            if best is None or candidate[0] > best[0]:
                best = candidate

        return best

    def _select_design_for_forced_fidelity(self, model: RecursiveMFGP, fidelity: str):
        # Same scalarised EI idea, but fidelity is externally fixed.
        w1 = float(self.weights[self.weight_idx % len(self.weights)])
        self.weight_idx += 1
        w2 = 1.0 - w1
        lo, hi = self._normalisation()
        g_star = self._g_star(w1, lo, hi)
        Xcand = self.rng.random((self.candidate_pool, self.d))
        mu, sd = model.predict(Xcand, fidelity)
        mu_n = (mu - lo) / (hi - lo)
        sd_n = sd / (hi - lo)
        ei = closed_form_ei_one_random_objective(
            w1 * mu_n, abs(w1) * sd_n, w2 * Xcand[:, 0], g_star, self.rho_aug
        )
        j = int(np.argmax(ei))
        return float(ei[j] / self.COST[fidelity]), Xcand[j].copy(), fidelity

    def _evaluate_selected(self, iteration: int, x, requested, acquisition):
        result = self.sim.evaluate(x, requested)
        self._charge(requested)
        passed, _ = quality_control(result)

        escalated = False
        actual = requested

        if requested == "LF" and not passed:
            # Failed LF is NOT inserted into the trusted LF archive.
            # The attempted LF cost remains charged.
            escalated = True
            actual = "HF"
            result = self.sim.evaluate(x, "HF")
            self._charge("HF")  # may cause slight realised overrun
            self.X_hf.append(np.asarray(x).copy())
            self.y_hf.append(result.f1)
            self.f2_hf.append(result.f2)
        elif requested == "LF":
            self.X_lf.append(np.asarray(x).copy())
            self.y_lf.append(result.f1)
        else:
            self.X_hf.append(np.asarray(x).copy())
            self.y_hf.append(result.f1)
            self.f2_hf.append(result.f2)

        self.records.append(
            EvalRecord(
                iteration=iteration,
                requested_fidelity=requested,
                actual_fidelity=actual,
                x=np.asarray(x).tolist(),
                f1=float(result.f1),
                f2=float(result.f2),
                quality_pass=bool(passed),
                escalated=bool(escalated),
                acquisition=float(acquisition),
                cumulative_ehf=float(self.cumulative_ehf),
            )
        )

    def run(self, strategy: str = "adaptive", max_iterations: int = 200):
        """
        strategy:
          adaptive : fidelity chosen by EI/c_m
          fixed    : repeating LF,LF,LF,HF schedule
          hf_only  : HF,HF,HF,...

        Termination follows the paper:
          continue while the next requested/admissible fidelity fits the
          remaining nominal EHF budget.

        A post-LF escalation is triggered only after the LF solve, so the
        realised EHF can exceed the nominal budget slightly.
        """
        strategy = strategy.lower()
        if strategy not in {"adaptive", "fixed", "hf_only"}:
            raise ValueError("strategy must be adaptive, fixed, or hf_only")

        fixed_schedule = ["LF", "LF", "LF", "HF"]
        fixed_idx = 0

        for it in range(max_iterations):
            if not (self._fits("LF") or self._fits("HF")):
                break

            model = RecursiveMFGP(self.d, self.seed + it).fit(
                np.asarray(self.X_lf),
                np.asarray(self.y_lf),
                np.asarray(self.X_hf),
                np.asarray(self.y_hf),
            )

            if strategy == "adaptive":
                selected = self._select_adaptive(model)
                if selected is None:
                    break
            elif strategy == "fixed":
                requested = fixed_schedule[fixed_idx % len(fixed_schedule)]
                fixed_idx += 1
                if not self._fits(requested):
                    break
                selected = self._select_design_for_forced_fidelity(model, requested)
            else:
                if not self._fits("HF"):
                    break
                selected = self._select_design_for_forced_fidelity(model, "HF")

            acq, x, requested = selected
            self._evaluate_selected(it, x, requested, acq)

        return self.records

    def summary(self):
        req_hf = sum(r.requested_fidelity == "HF" for r in self.records)
        act_hf = sum(r.actual_fidelity == "HF" for r in self.records)
        escalations = sum(r.escalated for r in self.records)
        n = max(len(self.records), 1)
        return {
            "iterations": len(self.records),
            "realised_ehf": self.cumulative_ehf,
            "requested_hf_fraction": req_hf / n,
            "actual_hf_fraction": act_hf / n,
            "escalations": escalations,
            "n_trusted_lf": len(self.X_lf),
            "n_hf": len(self.X_hf),
        }


def run_demo(strategy="adaptive", dimension=2, budget_ehf=12.0, seed=4):
    sim = DummyCFDSimulator(dimension=dimension)
    opt = DemoOptimizer(sim, budget_ehf=budget_ehf, seed=seed)
    opt.initialise(n_lf=6, n_hf=3)
    opt.run(strategy=strategy)
    return opt


if __name__ == "__main__":
    opt = run_demo(strategy="adaptive")
    print("Adaptive-MF demo summary")
    for k, v in opt.summary().items():
        print(f"{k:>24s}: {v}")
    print("\nLast five sequential records:")
    for r in opt.records[-5:]:
        print(asdict(r))
