import argparse
from pathlib import Path

from common import ensure_output_dir, namespace_to_config, plot_single_method, save_result_bundle, train_softmax_tb


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Softmax-TB toy experiment on a 2D Gaussian-mixture landscape.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"], help="Device to run on.")  ## 运行设备，auto 会优先尝试 mps/cuda
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")  ## 随机种子，保证结果可复现
    parser.add_argument("--grid-size", type=int, default=121, help="Number of grid points per axis.")  ## 每个坐标轴离散成多少个网格点
    parser.add_argument("--grid-limit", type=float, default=4.0, help="Coordinate range is [-limit, limit].")  ## 二维平面的边界大小
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden width of the score MLP.")  ## MLP 隐藏层宽度
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")  ## 优化器学习率
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")  ## AdamW 的权重衰减
    parser.add_argument("--train-steps", type=int, default=1000, help="Number of optimization steps.")  ## 训练总步数
    parser.add_argument("--beta", type=float, default=1.0, help="Softmax-TB temperature.")  ## Softmax-TB 里的 beta，控制目标分布尖锐程度
    parser.add_argument("--eval-samples", type=int, default=10000, help="Number of rollout samples for visualization.")  ## 训练后从学到的分布里采样多少个点
    parser.add_argument("--coverage-threshold", type=float, default=0.02, help="Minimum mode mass to count as covered.")  ## 某个峰的概率质量超过多少才算被覆盖
    parser.add_argument("--log-every", type=int, default=10, help="Log metrics every N steps.")  ## 每多少步记录一次训练指标
    parser.add_argument("--output-dir", type=str, default="toy-experiment/outputs", help="Directory to save figures and metrics.")  ## 输出图和指标文件的目录
    parser.add_argument("--run-name", type=str, default="softmax_tb", help="Run name used to create the output folder.")  ## 本次实验的名字，会写进输出目录名
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = namespace_to_config(args)
    output_dir = ensure_output_dir(cfg.output_dir, cfg.run_name)

    result = train_softmax_tb(cfg)
    save_result_bundle(result, output_dir)
    figure_path = None
    try:
        figure_path = plot_single_method(result, output_dir)
    except ModuleNotFoundError as exc:
        print(f"[Softmax-TB] plotting skipped: {exc}")

    final_metrics = result["final_metrics"]
    print(f"[Softmax-TB] output_dir={Path(output_dir)}")
    print(
        f"[Softmax-TB] KL={final_metrics['kl_to_true']:.6f} "
        f"JS={final_metrics['js_to_true']:.6f} "
        f"coverage={final_metrics['mode_coverage']:.1f}/{5} "
        f"avg_reward={final_metrics['avg_reward']:.6f}"
    )
    if figure_path is not None:
        print(f"[Softmax-TB] figure={figure_path}")


if __name__ == "__main__":
    main()
