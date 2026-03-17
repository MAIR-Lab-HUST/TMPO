import argparse
import json
from pathlib import Path

from common import (
    ensure_output_dir,
    namespace_to_config,
    plot_comparison,
    plot_single_method,
    save_result_bundle,
    train_reward_max,
    train_softmax_tb,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run both Softmax-TB and reward-max, then generate a side-by-side comparison.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"], help="Device to run on.")  ## 运行设备，auto 会优先尝试 mps/cuda
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")  ## 随机种子，保证两种方法的初始化可对齐
    parser.add_argument("--grid-size", type=int, default=121, help="Number of grid points per axis.")  ## 每个坐标轴离散成多少个网格点
    parser.add_argument("--grid-limit", type=float, default=4.0, help="Coordinate range is [-limit, limit].")  ## 二维平面的边界大小
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden width of the score MLP.")  ## MLP 隐藏层宽度
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")  ## 两种方法共用的学习率
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")  ## AdamW 的权重衰减
    parser.add_argument("--train-steps", type=int, default=1000, help="Number of optimization steps.")  ## 每种方法训练多少步
    parser.add_argument("--beta", type=float, default=1.0, help="Softmax-TB temperature.")  ## 只作用于 Softmax-TB 的 beta
    parser.add_argument("--eval-samples", type=int, default=10000, help="Number of rollout samples for visualization.")  ## 训练后从学到的分布里采样多少个点
    parser.add_argument("--coverage-threshold", type=float, default=0.02, help="Minimum mode mass to count as covered.")  ## 某个峰的概率质量超过多少才算被覆盖
    parser.add_argument("--log-every", type=int, default=10, help="Log metrics every N steps.")  ## 每多少步记录一次训练指标
    parser.add_argument("--output-dir", type=str, default="toy-experiment/outputs", help="Directory to save figures and metrics.")  ## 输出图和指标文件的目录
    parser.add_argument("--run-name", type=str, default="compare_softmax_tb_vs_reward_max", help="Run name used to create the output folder.")  ## 对比实验输出目录名
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = namespace_to_config(args)
    output_dir = ensure_output_dir(cfg.output_dir, cfg.run_name)

    tb_result = train_softmax_tb(cfg)
    rm_result = train_reward_max(cfg)

    save_result_bundle(tb_result, output_dir)
    save_result_bundle(rm_result, output_dir)
    tb_single = None
    rm_single = None
    compare_path = None
    try:
        tb_single = plot_single_method(tb_result, output_dir)
        rm_single = plot_single_method(rm_result, output_dir)
        compare_path = plot_comparison(tb_result, rm_result, output_dir)
    except ModuleNotFoundError as exc:
        print(f"[Compare] plotting skipped: {exc}")

    summary = {
        "softmax_tb": {
            key: value
            for key, value in tb_result["final_metrics"].items()
            if key != "samples"
        },
        "reward_max": {
            key: value
            for key, value in rm_result["final_metrics"].items()
            if key != "samples"
        },
    }
    with (Path(output_dir) / "comparison_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[Compare] output_dir={Path(output_dir)}")
    if tb_single is not None:
        print(f"[Compare] softmax_tb_figure={tb_single}")
    if rm_single is not None:
        print(f"[Compare] reward_max_figure={rm_single}")
    if compare_path is not None:
        print(f"[Compare] comparison_figure={compare_path}")
    print(
        "[Compare] "
        f"TB_KL={tb_result['final_metrics']['kl_to_true']:.6f} "
        f"RM_KL={rm_result['final_metrics']['kl_to_true']:.6f} "
        f"TB_JS={tb_result['final_metrics']['js_to_true']:.6f} "
        f"RM_JS={rm_result['final_metrics']['js_to_true']:.6f}"
    )


if __name__ == "__main__":
    main()
