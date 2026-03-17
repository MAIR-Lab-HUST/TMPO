from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


MODE_SPECS = [
    {"label": "A", "center": (2.5, 2.5), "weight": 0.30, "sigma": 0.45, "color": "#e76f51"},
    {"label": "B", "center": (-2.5, 2.5), "weight": 0.25, "sigma": 0.45, "color": "#2a9d8f"},
    {"label": "C", "center": (-2.5, -2.5), "weight": 0.20, "sigma": 0.45, "color": "#457b9d"},
    {"label": "D", "center": (2.5, -2.5), "weight": 0.15, "sigma": 0.45, "color": "#f4a261"},
    {"label": "E", "center": (0.0, 0.0), "weight": 0.10, "sigma": 0.70, "color": "#8d5a97"},
]


@dataclass
class ExperimentConfig:
    device: str
    seed: int
    grid_size: int
    grid_limit: float
    hidden_dim: int
    lr: float
    weight_decay: float
    train_steps: int
    beta: float
    eval_samples: int
    coverage_threshold: float
    log_every: int
    output_dir: str
    run_name: str


class ScoreMLP(nn.Module):
    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def ensure_output_dir(base_dir: str, run_name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(base_dir) / f"{run_name}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if device_name == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)


def build_grid(grid_size: int, grid_limit: float, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    axis = torch.linspace(-grid_limit, grid_limit, grid_size, device=device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
    return axis, xx, yy, grid


def gaussian_log_prob(points: torch.Tensor, center: Tuple[float, float], sigma: float) -> torch.Tensor:
    center_tensor = torch.tensor(center, device=points.device, dtype=points.dtype)
    diff = points - center_tensor
    dim = points.shape[-1]
    norm_sq = diff.pow(2).sum(dim=-1)
    log_z = dim * math.log(sigma) + 0.5 * dim * math.log(2.0 * math.pi)
    return -0.5 * norm_sq / (sigma ** 2) - log_z


def mixture_log_density(points: torch.Tensor) -> torch.Tensor:
    component_terms = []
    for spec in MODE_SPECS:
        term = math.log(spec["weight"]) + gaussian_log_prob(points, spec["center"], spec["sigma"])
        component_terms.append(term)
    return torch.logsumexp(torch.stack(component_terms, dim=0), dim=0)


def assign_modes(points: torch.Tensor) -> torch.Tensor:
    centers = torch.tensor([spec["center"] for spec in MODE_SPECS], device=points.device, dtype=points.dtype)
    dists = torch.cdist(points, centers)
    return torch.argmin(dists, dim=1)


def probabilities_to_mode_mass(points: torch.Tensor, probs: torch.Tensor) -> List[float]:
    assignments = assign_modes(points)
    masses = torch.zeros(len(MODE_SPECS), device=points.device, dtype=probs.dtype)
    for idx in range(len(MODE_SPECS)):
        masses[idx] = probs[assignments == idx].sum()
    masses = masses / masses.sum().clamp_min(1e-12)
    return masses.cpu().tolist()


def samples_to_mode_mass(samples: torch.Tensor) -> List[float]:
    assignments = assign_modes(samples)
    masses = torch.zeros(len(MODE_SPECS), device=samples.device, dtype=samples.dtype)
    for idx in range(len(MODE_SPECS)):
        masses[idx] = (assignments == idx).to(samples.dtype).sum()
    masses = masses / masses.sum().clamp_min(1e-12)
    return masses.cpu().tolist()


def discrete_kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    eps = 1e-12
    return torch.sum(p * (torch.log(p + eps) - torch.log(q + eps)))


def discrete_js(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    m = 0.5 * (p + q)
    return 0.5 * discrete_kl(p, m) + 0.5 * discrete_kl(q, m)


def sample_from_distribution(points: torch.Tensor, probs: torch.Tensor, num_samples: int) -> torch.Tensor:
    indices = torch.multinomial(probs, num_samples=num_samples, replacement=True)
    return points[indices]


def evaluate_distribution(
    points: torch.Tensor,
    probs: torch.Tensor,
    reward: torch.Tensor,
    true_probs: torch.Tensor,
    eval_samples: int,
    coverage_threshold: float,
) -> Dict[str, object]:
    samples = sample_from_distribution(points, probs, eval_samples)
    model_mode_mass = probabilities_to_mode_mass(points, probs)
    sample_mode_mass = samples_to_mode_mass(samples)
    target_mode_mass = probabilities_to_mode_mass(points, true_probs)
    coverage = float(sum(mass > coverage_threshold for mass in model_mode_mass))
    peak_prob = float(probs.max().item())

    return {
        "avg_reward": float((probs * reward).sum().item()),
        "kl_to_true": float(discrete_kl(probs, true_probs).item()),
        "js_to_true": float(discrete_js(probs, true_probs).item()),
        "mode_coverage": coverage,
        "peak_prob": peak_prob,
        "model_mode_mass": model_mode_mass,
        "sample_mode_mass": sample_mode_mass,
        "target_mode_mass": target_mode_mass,
        "samples": samples.detach().cpu().tolist(),
    }


def prepare_environment(cfg: ExperimentConfig) -> Dict[str, object]:
    device = resolve_device(cfg.device)
    set_seed(cfg.seed)
    axis, xx, yy, grid = build_grid(cfg.grid_size, cfg.grid_limit, device)
    reward = mixture_log_density(grid)
    true_log_probs = F.log_softmax(reward, dim=0)
    true_probs = true_log_probs.exp()

    return {
        "device": device,
        "axis": axis,
        "xx": xx,
        "yy": yy,
        "grid": grid,
        "reward": reward,
        "true_log_probs": true_log_probs,
        "true_probs": true_probs,
    }


def train_softmax_tb(cfg: ExperimentConfig) -> Dict[str, object]:
    env = prepare_environment(cfg)
    grid = env["grid"]
    reward = env["reward"]
    true_probs = env["true_probs"]
    target_log_probs = F.log_softmax(cfg.beta * reward, dim=0)
    device = env["device"]

    model = ScoreMLP(hidden_dim=cfg.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = {
        "step": [],
        "loss": [],
        "avg_reward": [],
        "kl_to_true": [],
        "js_to_true": [],
        "mode_coverage": [],
        "peak_prob": [],
        "grad_norm": [],
    }

    for step in range(1, cfg.train_steps + 1):
        scores = model(grid)
        log_probs = F.log_softmax(scores, dim=0)
        probs = log_probs.exp()
        loss = ((log_probs - target_log_probs) ** 2).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0).item())
        optimizer.step()

        if step % cfg.log_every == 0 or step == 1 or step == cfg.train_steps:
            metrics = evaluate_distribution(
                points=grid,
                probs=probs.detach(),
                reward=reward.detach(),
                true_probs=true_probs.detach(),
                eval_samples=cfg.eval_samples,
                coverage_threshold=cfg.coverage_threshold,
            )
            history["step"].append(step)
            history["loss"].append(float(loss.item()))
            history["avg_reward"].append(metrics["avg_reward"])
            history["kl_to_true"].append(metrics["kl_to_true"])
            history["js_to_true"].append(metrics["js_to_true"])
            history["mode_coverage"].append(metrics["mode_coverage"])
            history["peak_prob"].append(metrics["peak_prob"])
            history["grad_norm"].append(grad_norm)

    with torch.no_grad():
        final_scores = model(grid)
        final_log_probs = F.log_softmax(final_scores, dim=0)
        final_probs = final_log_probs.exp()
        final_metrics = evaluate_distribution(
            points=grid,
            probs=final_probs,
            reward=reward,
            true_probs=true_probs,
            eval_samples=cfg.eval_samples,
            coverage_threshold=cfg.coverage_threshold,
        )

    return {
        "method": "softmax_tb",
        "config": cfg.__dict__,
        "history": history,
        "grid": grid.detach().cpu().tolist(),
        "reward": reward.detach().cpu().tolist(),
        "true_probs": true_probs.detach().cpu().tolist(),
        "learned_probs": final_probs.detach().cpu().tolist(),
        "final_metrics": final_metrics,
    }


def train_reward_max(cfg: ExperimentConfig) -> Dict[str, object]:
    env = prepare_environment(cfg)
    grid = env["grid"]
    reward = env["reward"]
    true_probs = env["true_probs"]
    device = env["device"]

    model = ScoreMLP(hidden_dim=cfg.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = {
        "step": [],
        "loss": [],
        "avg_reward": [],
        "kl_to_true": [],
        "js_to_true": [],
        "mode_coverage": [],
        "peak_prob": [],
        "grad_norm": [],
    }

    for step in range(1, cfg.train_steps + 1):
        scores = model(grid)
        probs = F.softmax(scores, dim=0)
        loss = -(probs * reward).sum()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0).item())
        optimizer.step()

        if step % cfg.log_every == 0 or step == 1 or step == cfg.train_steps:
            metrics = evaluate_distribution(
                points=grid,
                probs=probs.detach(),
                reward=reward.detach(),
                true_probs=true_probs.detach(),
                eval_samples=cfg.eval_samples,
                coverage_threshold=cfg.coverage_threshold,
            )
            history["step"].append(step)
            history["loss"].append(float(loss.item()))
            history["avg_reward"].append(metrics["avg_reward"])
            history["kl_to_true"].append(metrics["kl_to_true"])
            history["js_to_true"].append(metrics["js_to_true"])
            history["mode_coverage"].append(metrics["mode_coverage"])
            history["peak_prob"].append(metrics["peak_prob"])
            history["grad_norm"].append(grad_norm)

    with torch.no_grad():
        final_scores = model(grid)
        final_probs = F.softmax(final_scores, dim=0)
        final_metrics = evaluate_distribution(
            points=grid,
            probs=final_probs,
            reward=reward,
            true_probs=true_probs,
            eval_samples=cfg.eval_samples,
            coverage_threshold=cfg.coverage_threshold,
        )

    return {
        "method": "reward_max",
        "config": cfg.__dict__,
        "history": history,
        "grid": grid.detach().cpu().tolist(),
        "reward": reward.detach().cpu().tolist(),
        "true_probs": true_probs.detach().cpu().tolist(),
        "learned_probs": final_probs.detach().cpu().tolist(),
        "final_metrics": final_metrics,
    }


def _grid_to_image(arr: List[float], grid_size: int) -> List[List[float]]:
    return [arr[idx * grid_size:(idx + 1) * grid_size] for idx in range(grid_size)]


def _load_plt():
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for plotting. Install it first, for example: pip install matplotlib"
        ) from exc
    return plt


def _scatter_modes(ax: Any) -> None:
    for spec in MODE_SPECS:
        ax.scatter(spec["center"][0], spec["center"][1], s=120, c=spec["color"], edgecolors="black", linewidths=0.8)
        ax.text(spec["center"][0] + 0.08, spec["center"][1] + 0.08, spec["label"], fontsize=10, weight="bold")


def _plot_density(ax: Any, image: List[List[float]], grid_limit: float, title: str, cmap: str = "magma") -> None:
    ax.imshow(
        image,
        origin="lower",
        extent=[-grid_limit, grid_limit, -grid_limit, grid_limit],
        cmap=cmap,
        aspect="equal",
    )
    _scatter_modes(ax)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def _plot_samples(ax: Any, samples: List[List[float]], grid_limit: float, title: str) -> None:
    samples_tensor = torch.tensor(samples, dtype=torch.float32)
    assignments = assign_modes(samples_tensor)
    for idx, spec in enumerate(MODE_SPECS):
        pts = samples_tensor[assignments == idx]
        if pts.numel() == 0:
            continue
        ax.scatter(pts[:, 0].tolist(), pts[:, 1].tolist(), s=6, alpha=0.35, c=spec["color"], label=spec["label"])
    _scatter_modes(ax)
    ax.set_xlim(-grid_limit, grid_limit)
    ax.set_ylim(-grid_limit, grid_limit)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def _plot_mode_bars(ax: Any, target_mass: List[float], learned_mass: List[float], title: str) -> None:
    x = list(range(len(MODE_SPECS)))
    labels = [spec["label"] for spec in MODE_SPECS]
    ax.bar([v - 0.18 for v in x], target_mass, width=0.36, label="Target", color="#264653")
    ax.bar([v + 0.18 for v in x], learned_mass, width=0.36, label="Learned", color="#e76f51")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, max(max(target_mass), max(learned_mass)) * 1.25 + 1e-6)
    ax.set_title(title)
    ax.set_ylabel("Probability Mass")
    ax.legend()


def plot_single_method(result: Dict[str, object], output_dir: Path) -> Path:
    plt = _load_plt()
    cfg = result["config"]
    grid_size = int(cfg["grid_size"])
    grid_limit = float(cfg["grid_limit"])
    reward_image = _grid_to_image(result["reward"], grid_size)
    true_image = _grid_to_image(result["true_probs"], grid_size)
    learned_image = _grid_to_image(result["learned_probs"], grid_size)
    history = result["history"]
    final_metrics = result["final_metrics"]
    samples = final_metrics["samples"]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    _plot_density(axes[0, 0], reward_image, grid_limit, "Reward Landscape", cmap="viridis")
    _plot_density(axes[0, 1], true_image, grid_limit, "True Target Distribution")
    _plot_density(axes[0, 2], learned_image, grid_limit, f"Learned Distribution ({result['method']})")
    _plot_samples(axes[1, 0], samples, grid_limit, f"Sample Rollout ({result['method']})")
    _plot_mode_bars(
        axes[1, 1],
        final_metrics["target_mode_mass"],
        final_metrics["sample_mode_mass"],
        "Mode Mass: Target vs Samples",
    )

    axes[1, 2].plot(history["step"], history["loss"], label="Loss", color="#d62828", linewidth=2)
    axes[1, 2].plot(history["step"], history["kl_to_true"], label="KL", color="#1d3557", linewidth=2)
    axes[1, 2].plot(history["step"], history["js_to_true"], label="JS", color="#2a9d8f", linewidth=2)
    axes[1, 2].plot(history["step"], history["avg_reward"], label="Avg Reward", color="#f4a261", linewidth=2)
    axes[1, 2].set_title("Training Curves")
    axes[1, 2].set_xlabel("Step")
    axes[1, 2].legend()

    summary_text = (
        f"method={result['method']}\n"
        f"KL={final_metrics['kl_to_true']:.4f}\n"
        f"JS={final_metrics['js_to_true']:.4f}\n"
        f"coverage={final_metrics['mode_coverage']:.1f}/{len(MODE_SPECS)}\n"
        f"avg_reward={final_metrics['avg_reward']:.4f}"
    )
    fig.text(0.84, 0.08, summary_text, fontsize=11, bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9})
    fig.suptitle(f"Softmax-TB Toy Experiment: {result['method']}", fontsize=16, weight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])

    save_path = output_dir / f"{result['method']}_overview.png"
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
    return save_path


def plot_comparison(tb_result: Dict[str, object], rm_result: Dict[str, object], output_dir: Path) -> Path:
    plt = _load_plt()
    cfg = tb_result["config"]
    grid_size = int(cfg["grid_size"])
    grid_limit = float(cfg["grid_limit"])

    reward_image = _grid_to_image(tb_result["reward"], grid_size)
    tb_image = _grid_to_image(tb_result["learned_probs"], grid_size)
    rm_image = _grid_to_image(rm_result["learned_probs"], grid_size)
    tb_samples = tb_result["final_metrics"]["samples"]
    rm_samples = rm_result["final_metrics"]["samples"]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))

    _plot_density(axes[0, 0], reward_image, grid_limit, "Target Reward Landscape", cmap="viridis")
    _plot_density(axes[0, 1], tb_image, grid_limit, "Learned Distribution: Softmax-TB")
    _plot_density(axes[0, 2], rm_image, grid_limit, "Learned Distribution: Reward-Max")

    _plot_samples(axes[1, 0], tb_samples, grid_limit, "Samples: Softmax-TB")
    _plot_samples(axes[1, 1], rm_samples, grid_limit, "Samples: Reward-Max")

    x = list(range(len(MODE_SPECS)))
    labels = [spec["label"] for spec in MODE_SPECS]
    target_mass = tb_result["final_metrics"]["target_mode_mass"]
    tb_mass = tb_result["final_metrics"]["sample_mode_mass"]
    rm_mass = rm_result["final_metrics"]["sample_mode_mass"]
    axes[1, 2].bar([v - 0.25 for v in x], target_mass, width=0.25, label="Target", color="#264653")
    axes[1, 2].bar(x, tb_mass, width=0.25, label="Softmax-TB", color="#e76f51")
    axes[1, 2].bar([v + 0.25 for v in x], rm_mass, width=0.25, label="Reward-Max", color="#457b9d")
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(labels)
    axes[1, 2].set_title("Mode Proportion Comparison")
    axes[1, 2].legend()

    axes[2, 0].plot(tb_result["history"]["step"], tb_result["history"]["loss"], label="TB Loss", color="#e76f51")
    axes[2, 0].plot(rm_result["history"]["step"], rm_result["history"]["loss"], label="Reward-Max Loss", color="#457b9d")
    axes[2, 0].set_title("Objective Curves")
    axes[2, 0].set_xlabel("Step")
    axes[2, 0].legend()

    axes[2, 1].plot(tb_result["history"]["step"], tb_result["history"]["kl_to_true"], label="TB KL", color="#e76f51")
    axes[2, 1].plot(rm_result["history"]["step"], rm_result["history"]["kl_to_true"], label="RM KL", color="#457b9d")
    axes[2, 1].plot(tb_result["history"]["step"], tb_result["history"]["js_to_true"], label="TB JS", color="#f4a261")
    axes[2, 1].plot(rm_result["history"]["step"], rm_result["history"]["js_to_true"], label="RM JS", color="#2a9d8f")
    axes[2, 1].set_title("Distribution Distance")
    axes[2, 1].set_xlabel("Step")
    axes[2, 1].legend()

    axes[2, 2].plot(tb_result["history"]["step"], tb_result["history"]["avg_reward"], label="TB Avg Reward", color="#e76f51")
    axes[2, 2].plot(rm_result["history"]["step"], rm_result["history"]["avg_reward"], label="RM Avg Reward", color="#457b9d")
    axes[2, 2].plot(tb_result["history"]["step"], tb_result["history"]["peak_prob"], label="TB Peak Prob", color="#f4a261")
    axes[2, 2].plot(rm_result["history"]["step"], rm_result["history"]["peak_prob"], label="RM Peak Prob", color="#2a9d8f")
    axes[2, 2].set_title("Reward vs Collapse Tendency")
    axes[2, 2].set_xlabel("Step")
    axes[2, 2].legend()

    fig.suptitle("Softmax-TB vs Reward-Max on Multi-Peak Toy Experiment", fontsize=16, weight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])

    save_path = output_dir / "comparison_overview.png"
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
    return save_path


def save_result_bundle(result: Dict[str, object], output_dir: Path) -> None:
    payload = {
        "method": result["method"],
        "config": result["config"],
        "final_metrics": {
            key: value
            for key, value in result["final_metrics"].items()
            if key != "samples"
        },
        "history": result["history"],
    }
    with (output_dir / f"{result['method']}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def namespace_to_config(args) -> ExperimentConfig:
    return ExperimentConfig(
        device=args.device,
        seed=args.seed,
        grid_size=args.grid_size,
        grid_limit=args.grid_limit,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        train_steps=args.train_steps,
        beta=getattr(args, "beta", 1.0),
        eval_samples=args.eval_samples,
        coverage_threshold=args.coverage_threshold,
        log_every=args.log_every,
        output_dir=args.output_dir,
        run_name=args.run_name,
    )
