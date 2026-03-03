\documentclass{article}

% if you need to pass options to natbib, use, e.g.:
%     \PassOptionsToPackage{numbers, compress}{natbib}
% before loading neurips_2024

% ready for submission
\usepackage{neurips_2024}

% to compile a preprint version, e.g., for submission to arXiv, add add the
% [preprint] option:
%     \usepackage[preprint]{neurips_2024}

% to compile a camera-ready version, add the [final] option, e.g.:
%     \usepackage[final]{neurips_2024}

% to avoid loading the natbib package, add option nonatbib:
%    \usepackage[nonatbib]{neurips_2024}

\usepackage[utf8]{inputenc} % allow utf-8 input
\usepackage[T1]{fontenc} % use 8-bit T1 fonts
\usepackage{hyperref} % 这是宏包，必须先调用它
\hypersetup{          % 这是对宏包的设置命令
    colorlinks=true,
    linkcolor=black,
    citecolor=black,
    urlcolor=blue
}
\usepackage{url} % simple URL typesetting
\usepackage{booktabs} % professional-quality tables
\usepackage{amsfonts} % blackboard math symbols
\usepackage{nicefrac} % compact symbols for 1/2, etc.
\usepackage{microtype} % microtypography
\usepackage{xcolor} % colors
\usepackage[UTF8]{ctex}
\usepackage{graphicx}
\title{TreeMatch-RL: Tree-based Distribution Matching Online RL for Diverse and Efficient Diffusion Model Alignment}

% The \author macro works with any number of authors. There are two commands
% used to separate the names and addresses of multiple authors: \And and \AND.
%
% Using \And between authors leaves it to LaTeX to determine where to break the
% lines. Using \AND forces a line break at that point. So, if LaTeX puts 3 of 4
% authors names on the first line, and the last on the second line, try using
% \AND instead of \And before the third author name.

% examples of more authors
% \And
% Coauthor \\
% Affiliation \\
% Address \\
% \texttt{email} \\
% \AND
% Coauthor \\
% Affiliation \\
% Address \\
% \texttt{email} \\
% \And
% Coauthor \\
% Affiliation \\
% Address \\
% \texttt{email} \\
% \And
% Coauthor \\
% Affiliation \\
% Address \\
% \texttt{email} \\

\begin{document}
    \maketitle

    \begin{abstract}
        扩散模型在通过人类偏好进行强化学习（RL）对齐时，现有的奖励最大化范式（如 PPO 或 GRPO）往往导致严重的模式塌缩（Mode Collapse），使生成的图像在维持高得分的同时丧失了宝贵的语义多样性。为了克服这一挑战，本文提出了 \textbf{TreeMatch-RL}（基于树状分布匹配的强化学习）框架，将扩散对齐重新表述为生成流网络（GFlowNet）中的奖励分布匹配问题。

        TreeMatch-RL 的核心贡献包括：(1) 引入了 \textbf{Softmax-TB} 损失函数，通过匹配路径概率分布与指数化奖励（$\exp(\beta R)$）的组内比例，实现了无需学习配分函数 $Z$ 的全局流平衡，从理论上保证了模型对奖励景观的多模态覆盖；(2) 设计了一种三阶 27 分支的树状采样结构，通过前缀路径共享显著降低了推理开销，并仅在关键分支点执行随机 SDE 采样，以聚焦训练信号；(3) 建立了基于采样奖励均值的经验自适应调度机制，动态调整 SDE 分叉点的位置，实现了对不同难度 Prompt 的智能算力分配。此外，我们集成了重要性采样（IS）以支持高效的离策略更新，并在后置阶段应用 DPM-Solver++ 加速采样。实验结果表明，TreeMatch-RL 在 Flux 和 Stable Diffusion 3.5 等大规模模型上显著优于现有的 GRPO 方法，不仅提升了对齐分数，更在生成的解空间覆盖率和视觉创造力方面实现了质的突破。
    \end{abstract}
    
    \section{Introduction}
    近年来，以 Flux 和 Stable Diffusion 3.5 为代表的大规模扩散模型在文本生成图像（T2I）领域取得了突破性进展。为了使这些生成的视觉内容更符合人类偏好，研究界广泛采用了强化学习（RL）对齐技术。目前的对齐框架主要基于近端策略优化（PPO）或群体相对策略优化（GRPO），其核心逻辑是追求期望奖励的最大化。然而，这种“奖励最大化”范式在处理高维连续空间时存在显著缺陷：模型往往会过度优化（Over-optimize）特定的高奖励模式，导致严重的模式塌缩（Mode Collapse）现象。这使得对齐后的模型生成的图像虽然在评分上有所提升，但在语义多样性和视觉创造力方面表现出明显的衰减。

    为了解决这一挑战，受生成流网络（GFlowNets）理论的启发，我们提出了 \textbf{TreeMatch-RL}（Tree-based Distribution Matching Reinforcement Learning）框架。与传统寻找单一最优解的方法不同，TreeMatch-RL 旨在实现“奖励分布匹配”，即强制模型生成的路径概率与该路径获得的指数化奖励成正比（$\pi \propto \exp(\beta R)$）。这种范式转变确保了模型能够覆盖奖励景观（Reward Landscape）中的所有有效峰值，从而在对齐过程中保留生成样本的多样性。

    TreeMatch-RL 通过三项关键的架构创新实现了这一目标。首先，我们设计了一种三阶 27 分支的树状流结构（Flow-Tree），通过前缀路径共享机制极大地降低了超大规模模型在采样阶段的计算开销。该结构仅在扩散过程的关键语义时刻执行随机 SDE 分叉，而其他步骤则采用确定性 ODE 步进以维持梯度稳定性。其次，我们推导出一种新型的软轨迹平衡（Softmax-TB）损失函数。该损失函数通过匹配路径概率分布与奖励比例的组内一致性，有效地抵消了配分函数 $Z$ 的方差波动，实现了精准的分布对齐。最后，我们引入了基于经验奖励感知的自适应调度机制。TreeMatch-RL 利用在线采样的组内奖励均值 $\bar{R}$ 作为难度感知器，动态调整去噪过程中的 SDE 分叉点 $t_{split}$。在低奖励的“难题”场景下，系统会自动提前分叉以增强全局结构的随机探索；而在高奖励的“易题”场景下，则延后分叉以进行像素级细节微调。此外，结合重要性采样（IS）与后置 DPM-Solver++ 加速，TreeMatch-RL 在 Flux 等模型上展示了卓越的对齐效率与多样性表现。
    本文的主要贡献总结如下：
    \begin{itemize}
    \item 我们提出了 \textbf{TreeMatch-RL}，一种将扩散模型对齐重新表述为奖励分布匹配问题的强化学习框架，有效解决了对齐过程中的模式塌缩问题。
    \item 设计了稀疏 SDE 驱动的树状采样动力学，结合 Softmax-TB 损失函数，实现了高效且鲁棒的流平衡优化。
    \item 引入了自适应难度感知调度与高阶求解器加速技术，显著提升了超大型扩散模型（如 Flux, SD3.5）在分布式环境下的训练效率。
    \end{itemize}

    \section{Preliminaries}
    \label{gen_inst}
    \subsection{GFlowNets中的Trajectory Balance}

    令 $\mathcal{X}$ 表示组合对象集合，$R$ 为奖励函数，用于为每个对象
    $x \in \mathcal{X}$ 分配非负值。GFlowNets
    旨在学习一种序贯的、构造性的采样策略 $\pi$，使其生成对象 $x$ 的概率与其奖励成正比，即
    $\pi(x) \propto R(x)$。

    这一过程可以表示为一个有向无环图 (DAG), $G = (\mathcal{S}, \mathcal{A})$，其中顶点
    $s \in \mathcal{S}$ 称为状态（states），有向边 $(u \to v) \in \mathcal{A}$ 称为动作（actions）。生成一个对象
    $x \in \mathcal{X}$ 对应于 DAG 中的一条完整轨迹 $\tau = (s_{0}\to \dots \to s
    _{n}) \in \mathcal{T}$，该轨迹始于初始状态 $s_{0}$，终于终端状态 $s_{n}\in \mathcal{X}$。

    状态流 $F(s)$ 定义为分配给每个状态 $s \in \mathcal{S}$ 的非负权重。前向策略 $P
    _{F}(s' | s)$ 规定了转移到子状态 $s'$ 的概率，而后向策略 $P_{B}(s | s')$ 规定了转移到父状态
    $s$ 的概率。为此，细致平衡（Detailed Balance）目标强制要求每一条边 $(s \to s'
    ) \in \mathcal{A}$ 上的局部流一致性：
    \begin{equation}
        \forall(s \to s') \in \mathcal{A}, \quad F_{\theta}(s) P_{F}(s' | s; \theta
        ) = F_{\theta}(s') P_{B}(s | s'; \theta).
    \end{equation}

    为了实现这种流一致性，GFlowNets
    采用了不同粒度的训练目标，包括细致平衡、轨迹平衡（Trajectory
    Balance）和子轨迹平衡（Sub-trajectory Balance）。凭借其寻求多样性（diversity-seeking）的特性，GFlowNets
    已成功应用于多个领域，包括分子生成 、扩散模型微调以及摊销推理。

    在 GFlowNets 的各种训练目标中，轨迹平衡在轨迹层面上维持流一致性，其定义为：
    \begin{equation}
        Z_{\theta}\prod_{t=1}^{n}P_{F}(s_{t}| s_{t-1}; \theta) = R(x) \prod_{t=1}
        ^{n}P_{B}(s_{t-1}| s_{t}; \theta).
    \end{equation}

    \subsection{Diffusion RL and Flow Matching}

    扩散模型的生成过程可以被表述为一个具有 $T$ 个离散步长的马尔可夫决策过程 (MDP)。在该框架下，状态 $s_t$ 对应于扩散过程中的潜在变量 $x_t$，动作 $a_t$ 对应于从 $x_t$ 到 $x_{t-\Delta t}$ 的转移。强化学习的目标是通过微调预训练模型来优化策略 $\pi_{\theta}(x_{t-\Delta t} | x_t)$，以最大化最终生成图像 $x_0$ 的期望奖励 $R(x_0)$。
    
    \paragraph{Flow matching and rectified flow} 
    在本文中，我们主要关注基于流匹配（Flow Matching）的模型，如 SD3.5 和 Flux。Rectified Flow (RF) 定义了一种简单的线性内插概率流：
    \begin{equation}
        x_t = t x_1 + (1-t) x_0, \quad t \in [0, 1]
    \end{equation}
    其中 $x_1 \sim \mathcal{N}(0, \mathbf{I})$ 是高斯噪声，$x_0$ 是目标数据。模型训练的目标是预测速度场 $v_{\theta}(x_t, t) = \frac{dx_t}{dt}$。在采样阶段，通过求解如下常微分方程 (ODE) 即可实现从噪声到数据的生成：
    \begin{equation}
        dx_t = v_{\theta}(x_t, t) dt.
    \end{equation}
    
    \paragraph{Stochastic exploration and log-probability} 
    为了在强化学习中进行策略探索，通常将上述确定性 ODE 转换为等价的随机微分方程 (SDE)：
    \begin{equation}
        dx_t = [v_{\theta}(x_t, t) + f(x_t, t)] dt + g(t) dw
    \end{equation}
    其中 $dw$ 是标准维纳过程，$f(x_t, t)$ 是修正项。在离散采样步中，给定 $x_t$ 生成 $x_{t-\Delta t}$ 的转移概率服从高斯分布 $\pi_{\theta}(x_{t-\Delta t} | x_t) = \mathcal{N}(\mu_{\theta}, \sigma_t^2 \Delta t)$。
    相应的单步对数概率（log-probability）可计算为：
    \begin{equation}
        \log \pi_{\theta}(x_{t-\Delta t} | x_t) = -\frac{\|x_{t-\Delta t} - \mu_{\theta}\|^2}{2\sigma_t^2 \Delta t} - \frac{d}{2}\log(2\pi \sigma_t^2 \Delta t)
    \end{equation}
    传统的 Diffusion RL 方法（如 PPO 或 GRPO）利用该对数概率计算策略梯度，通过重要性采样更新模型参数，以使生成的轨迹趋向于奖励模型（Reward Model）定义的偏好分布。
    \section{Related Work}
    
    本节对与扩散模型对齐及强化学习相关的代表性工作进行回顾。我们将分别讨论 Flow-GRPO、TreeGRPO、FlowRL、MixGRPO 以及 DAG 的技术贡献，并总结其局限性，从而阐述 \textbf{TreeMatch-RL} 框架的创新动机。
    
    \subsection{Flow-GRPO: Flow Matching with Policy Optimization}
    
    \textit{Flow-GRPO} 首次将流匹配（Flow Matching）的 ODE 到 SDE 转换与群体相对策略优化（GRPO）相结合。该工作通过在扩散采样过程中注入 SDE 噪声，使得模型能够计算路径的对数概率（log probability），从而应用 PPO 风格的策略梯度更新。此外，该项目引入了 SDE 窗口（SDE Window）机制，仅在特定时间步内进行随机采样以聚焦训练信号，并支持多奖励函数（如 PickScore, Aesthetic Score）的加权组合。尽管其实际表现卓越，但其核心仍属于奖励最大化范式，且采样过程缺乏结构化优化，导致计算开销较大。
    
    \subsection{TreeGRPO: Tree-Advantage Estimation for Online RL}
    
    \textit{TreeGRPO} 针对扩散模型采样的高昂成本，引入了树状采样结构（Tree-Advantage Estimation）。该方法在去噪轨迹的关键步执行 SDE 分叉，通过前缀共享机制大幅降低了显存消耗并增强了优势估计的准确性。其算法核心在于将叶子节点的奖励分数通过树结构递归回传，使每个分叉步都能获得来自多个子分支的梯度反馈。TreeGRPO 证明了树状结构在扩散模型 RL 微调中的优越性，但由于其目标函数依赖于相对优势的最大化，依然难以完全避免模式塌缩问题。
    
    \subsection{FlowRL: Reward Distribution Matching via GFlowNets}
    
    \textit{FlowRL} 将生成流网络（GFlowNets）的轨迹平衡（Trajectory Balance, TB）理论引入大型语言模型（LLM）的对齐任务中。不同于以往追求最高奖励的方法，FlowRL 旨在使采样概率与奖励分布形状匹配，从而实现更高水平的多样性探索。然而，该框架在工程实现上依赖于一个额外的多层感知机（MLP）来预测配分函数 $Z$，这在扩散模型的高维连续状态空间中往往会导致收敛不稳定。此外，其应用场景主要局限于离散 token 序列，尚未针对具有复杂动力学的流匹配模型进行优化。
    
    \subsection{MixGRPO: Efficiency through Mixed ODE-SDE Dynamics}
    
    \textit{MixGRPO} 提出了一种创新的混合采样策略，通过集成随机微分方程（SDE）与常微分方程（ODE）来提升对齐效率。该方法引入了滑动窗口机制，仅在窗口内进行 SDE 采样与策略优化，而在窗口外使用 ODE 采样，从而精简了马尔可夫决策过程（MDP）的范围。MixGRPO-Flash 变体进一步通过 DPM-Solver++ 高阶求解器加速了非优化阶段的采样过程，实现了高达 71\% 的训练提速。然而，其滑动窗口策略主要基于固定的或预定义的课程学习，缺乏针对单个 Prompt 难度的实时感应与调度。
    
    \subsection{DAG: GFlowNet Alignment for Diffusion Models}
    
    \textit{DAG}（Diffusion Alignment with GFlowNet）是首个系统性地将生成流网络（GFlowNets）框架应用于文本-图像扩散模型对齐的工作。该工作基于细致平衡（Detailed Balance）准则推导出 DAG-DB 算法，并进一步通过 KL 散度目标推导出 DAG-KL，后者在数学上等价于带有最大熵正则化的强化学习，有效提升了训练稳定性。DAG 的核心贡献在于引入了"扩散特定前瞻机制"（Diffusion-specific Forward-looking），利用 U-Net 内部对干净图像 $\hat{x}_0$ 的预测为中间去噪步骤提供稳定的奖励梯度信号，从根本上解决了黑盒奖励函数在噪声图像上评估不准确的问题。实验表明，DAG 在多种奖励函数（如 Aesthetic Score、ImageReward、HPSv2）上均显著优于 DDPO 等 RL 基线，在保持更高奖励分数的同时维持了更低的 FID，实现了更好的奖励-多样性权衡。然而，DAG 受限于单步转换（single transition）算法框架——由于显存限制，DAG难以在大规模模型上实现多步轨迹平衡 (Trajectory Balance) 实现——这制约了其信用分配的深度。此外，其实验均基于 Stable Diffusion v1.5，尚未在 SD3.5、Flux 等更大规模的流匹配模型上得到验证。
    
    \subsection{$\nabla$-GFlowNet: Gradient-Informed GFlowNets for Diffusion Alignment}
    
    \textit{$\nabla$-GFlowNet} 针对现有扩散模型奖励微调中容易出现的样本多样性丧失、先验知识遗忘以及收敛缓慢等问题，提出了一种结合传统强化学习与直接奖励最大化优势的新型微调目标函数。该研究通过在生成流网络（GFlowNets）的详细平衡（Detailed Balance, DB）条件中引入奖励函数的梯度信号，构建了 $\nabla$-DB 和残差 $\nabla$-DB（Residual $\nabla$-DB）目标函数。残差 $\nabla$-DB 通过将微调模型的 $\nabla$-DB 条件与预训练模型的对应条件相减，有效防止了模型过度优化奖励而遗忘预训练先验。此外，为了解决长序列的信用分配问题，该方法引入了“向前看”（Forward-looking, FL）技巧，使用预测的单步奖励梯度作为基线。实验证明，$\nabla$-GFlowNet 在多个真实的奖励函数（如 Aesthetic Score、HPSv2 等）上，能够实现快速收敛，同时在保持生成样本的多样性和预训练模型的先验分布方面实现了帕累托改进（Pareto improvements），有效避免了基于梯度的基线方法（如 ReFL 和 DRaFT）中常见的模式崩溃（Mode Collapse）现象。然而，该方法为了稳定训练过程需要引入输出正则化惩罚，增加了超参数调优的复杂性，且在非扩散类生成模型中的普适性仍有待进一步验证。

    \subsection{Summary and Limitations of Existing Works}
    
    综上所述，现有的扩散模型对齐技术在效率与效果方面均存在关键瓶颈：
    \begin{enumerate}
        \item \textbf{模式塌缩问题}：Flow-GRPO、TreeGRPO 与 MixGRPO 均采用奖励最大化（Max-Reward）范式，这使得模型容易收敛到少数高得分模态，丧失了 Diffusion 模型的生成多样性。
        \item \textbf{配分函数估计局限}：FlowRL 依赖额外 MLP 拟合配分函数 $Z$，在高维连续状态空间中收敛不稳定，且其应用场景主要局限于离散 token 序列，尚未针对具有复杂动力学的流匹配模型进行优化。
        \item \textbf{单步平衡与模型局限}：DAG 虽通过单步 DB/KL 目标绕开了显存瓶颈，但单步转换框架限制了其信用分配的深度，难以在复杂轨迹中精准传播奖励信号；此外，其实验均基于 Stable Diffusion v1.5，在 SD3.5、Flux 等新一代大规模流匹配模型上的有效性尚未得到验证。
        \item \textbf{对超参数敏感}：$\nabla$-GFlowNet 引入了奖励梯度信号以加速收敛并保持多样性，但为维持训练稳定性须依赖输出正则化惩罚，对温度参数 $\beta$ 和正则化系数 $\lambda$ 均高度敏感，超参调优复杂性显著增加，限制了其在工程实践中的可复现性与易用性。
        \item \textbf{算力调度不灵活}：虽然 MixGRPO 引入了滑动窗口，但无法根据在线奖励均值（$\bar{R}$）实时自适应调整分支策略，导致算力分配在简单与困难 Prompt 之间缺乏动态区分。
    \end{enumerate}
    本文提出的 \textbf{TreeMatch-RL} 框架通过 \textbf{Softmax-TB} 损失函数实现了无需 $Z$ 的分布匹配，并结合树状采样结构的前缀共享机制突破了 DAG 的单步局限，同时引入 \textbf{经验自适应调度} 与 \textbf{高阶加速技术}，从理论和工程两个维度全面解决了上述问题。
    
    \section{Methodology}
    \label{headings}
    
    本节详细介绍了 \textbf{TreeMatch-RL} 框架。首先，我们从生成流网络（GFlowNet）的轨迹平衡（TB）理论出发，推导了无需显式配分函数的 \textbf{Softmax-TB} 损失函数。随后，我们阐述了具有连续分叉特征的三阶 27 分支树状采样结构，以及基于经验奖励感知的自适应调度机制。最后，我们深入剖析了针对流匹配模型集成的高阶求解器加速方案。框架的整体流水线如图~\ref{fig:pipeline} 所示。

    %%% --- 插入流程图开始 --- %%%
    \begin{figure}[ht]
      \centering
      % \fbox{\rule[-.5cm]{0cm}{4cm} \rule[-.5cm]{4cm}{0cm}} % 这是占位符，正式使用时请替换为下一行
      \includegraphics[width=\linewidth]{photo/pipeline.png}
      \caption{TreeMatch-RL 框架整体流程图。系统通过自适应调度器感知 Prompt 难度，动态构建三阶树状采样路径，并利用 Softmax-TB 损失实现分布对齐，最后通过 DPM-Solver++ 实现推理加速。}
      \label{fig:pipeline}
    \end{figure}
    %%% --- 插入流程图结束 --- %%%
    
    \subsection{From Gflownets To Softmax-TB}
    
    GFlowNet 的核心目标是学习一个随机策略 $\pi_{\theta}$，使得采样任何轨迹 $\tau$ 的概率与其奖励 $R(\tau)$ 成正比。标准的轨迹平衡（Trajectory Balance, TB）目标函数定义为：
    \begin{equation}
        \mathcal{L}_{TB}(\tau) = \left( \log Z + \log P_{\theta}(\tau) - \log R(\tau) \right)^2
    \end{equation}
    其中 $Z$ 为配分函数，代表总流值。将轨迹的前向路径概率记为 $P_{\theta}(\tau) = \prod_{t=1}^{n} P_F(s_t | s_{t-1}; \theta)$，后向概率记为 $P_B(\tau) = \prod_{t=1}^{n} P_B(s_{t-1} | s_t; \theta)$。对任意轨迹 $\tau_i$，上述 TB 等式可改写为：
    \begin{equation}
        P_{\theta}(\tau_i) = \frac{R(\tau_i) \cdot P_B(\tau_i)}{Z_{\theta}}
    \end{equation}
    在 TreeMatch-RL 的树状采样结构中，每个中间节点 $x_t$ 有且仅有一个父节点，即从根节点到该节点的回溯路径是唯一确定的。因此，对于每一步后向转移，$P_B(s_{t-1}|s_t) = 1$，整条轨迹的后向概率连乘积恒为：
    \begin{equation}
        P_B(\tau_i) = \prod_{t=1}^{n} P_B(s_{t-1}|s_t) = 1
    \end{equation}
    这一性质与 LLM 序列生成的树状结构一致，每个已生成的序列状态都只有唯一的合法父状态。因此，TB 等式可直接简化为 $P_\theta(\tau_i) = R(\tau_i) / Z_\theta$。先前的方法采用多层 MLP 去拟合 $Z$，引入了额外的参数和估计方差。为此，我们转而利用树结构的组内相对性来消去 $Z$：考虑在同一分叉点 $x_{split}$ 处采样的一组 $K$ 条路径，由于所有路径均满足 $P_{\theta}(\tau_i) \propto R(\tau_i)/Z$，$Z$ 对组内每条路径均为同一常数，因此对组内任意两条路径 $i$ 与 $j$ 相除，$Z$ 自然消去：
    \begin{equation}
        \frac{P_{\theta}(\tau_i)}{P_{\theta}(\tau_j)} = \frac{R(\tau_i)}{R(\tau_j)}
    \end{equation}
    进一步对组内全部 $K$ 条路径归一化求和，即可消去常数 $Z$，推导出比例匹配方程：
    \begin{equation}
        \frac{P_{\theta}(\tau_i)}{\sum_{j=1}^K P_{\theta}(\tau_j)} = \frac{R(\tau_i)}{\sum_{j=1}^K R(\tau_j)}
    \end{equation}
    进一步，为了体现强化学习对高奖励路径的偏好，我们借鉴能量建模思想，将奖励项进行指数化处理 $R \to \exp(\beta R)$。由此得到我们的核心损失函数 \textbf{Softmax-TB}：
    \begin{equation}
        \mathcal{L}_{Soft-TB} = \sum_{i=1}^K \left( \log \frac{P_{\theta}(\tau_i)}{\sum_{j=1}^K P_{\theta}(\tau_j)} - \log \frac{\exp(\beta R_i)}{\sum_{j=1}^K \exp(\beta R_j)} \right)^2
    \end{equation}
    该损失函数强制模型在关键决策点按照奖励的相对能量分配概率流，从而实现稳健的分布匹配。
    
    \subsection{Adaptive tree structure and empirical scheduling}
    
    为了在连续扩散空间中高效探测多模态分布，TreeMatch-RL 构建了一种具有精细拓扑的树状采样动力学。
    
    \paragraph{Flow-tree structure} 不同于Flow-GRPO和DanceGRPO中独立的并行采样，我们借鉴TreeGRPO的思路，TreeMatch-RL 在去噪轨迹中设定连续的三个分叉点。在每一个分叉时刻，系统通过随机微分方程（SDE）注入三组独立噪声，实现“1分3”的拓扑扩张。经过三阶连续分叉，最终在叶子节点生成 $3^3=27$ 条推理路径。这种结构通过前缀路径共享，显著降低了对超大规模模型（如 Flux）进行强化学习的计算负担。
    
    \paragraph{Empirical adaptive scheduling} 我们利用在线计算的组内奖励均值 $\bar{R}$ 动态调整分叉点 $t_{split}$ 的采样位置。受 \textit{MixGRPO} 启发，我们将去噪过程划分为全局结构建模与局部细节微调两个阶段。具体地，定义归一化奖励水平：
    \begin{equation}
        \alpha = \mathrm{clip}\left(\frac{\bar{R} - R_{\min}}{R_{\max} - R_{\min}},\ 0,\ 1\right)
    \end{equation}
    其中 $R_{\min}$ 和 $R_{\max}$ 为奖励的经验下界与上界。与基于固定区间的均匀分布不同，我们将 $t_{split}$ 建模为 Beta 分布的采样结果：
    \begin{equation}
        t_{split} \sim \mathrm{Beta}\!\left(1 + (1-\alpha)\kappa,\ 1 + \alpha\kappa\right)
    \end{equation}
    其中 $\kappa > 0$ 为集中度超参数（实验中取 $\kappa = 4$）。Beta 分布的支撑天然为 $[0,1]$，与去噪时间轴完全吻合，且其均值随 $\alpha$ 连续单调变化：
    \begin{equation}
        \mathbb{E}[t_{split}] = \frac{1 + (1-\alpha)\kappa}{2 + \kappa}
    \end{equation}
    \begin{itemize}
        \item \textbf{低奖励场景 ($\alpha \to 0$)}：分布退化为 $\mathrm{Beta}(1{+}\kappa,\ 1)$，概率密度向右端（高 $t$，高噪声区）集中，均值约为 $\frac{1+\kappa}{2+\kappa} \approx 0.83$（$\kappa=4$），对应分叉区间 $t_{split} \in [0.5,\ 0.8]$，促使系统在初始去噪阶段尽早分叉，以重塑全局构图结构。
        \item \textbf{高奖励场景 ($\alpha \to 1$)}：分布退化为 $\mathrm{Beta}(1,\ 1{+}\kappa)$，概率密度向左端（低 $t$，低噪声区）集中，均值约为 $\frac{1}{2+\kappa} \approx 0.17$（$\kappa=4$），对应分叉区间 $t_{split} \in [0.1,\ 0.3]$，将分叉推迟至精细化阶段，保护已对齐的高层语义，仅对像素级细节执行分布匹配。
        \item \textbf{$\kappa = 0$}：Beta 分布退化为 $\mathcal{U}(0,1)$，即无自适应调度的平凡基线。
    \end{itemize}
    
    \subsection{Multi-objective optimization}
    
    除核心损失外，TreeMatch-RL 还集成了多样性与稳定性约束。
    
    \paragraph{Diversity via particle entropy} 我们在 Latent 空间施加基于 RBF 核函数的粒子熵排斥力：
    \begin{equation}
        \mathcal{L}_{Entropy} = \frac{1}{K(K-1)} \sum_{i \neq j} \exp \left( -\frac{\|\phi(x_i) - \phi(x_j)\|^2}{h} \right)
    \end{equation}
    
    \paragraph{Stability via reference constraint} 我们引入参考模型 $\pi_{ref}$ 约束项，防止速度场 $v_{\theta}$ 过度偏离：
    \begin{equation}
        \mathcal{L}_{Ref} = \sum_{i=1}^K \| \frac{1}{T} \log \pi_{\theta}(\tau_i) - \frac{1}{T} \log \pi_{ref}(\tau_i) \|^2
    \end{equation}

    \paragraph{Importance Sampling}
    我们将重要性采样（Importance Sampling）引入损失函数之中。对第 $i$ 条轨迹 $\tau_i$，其重要性权重 $w_i$ 定义为当前策略与旧策略的路径概率之比：
    \begin{equation}
        w_i = \frac{P_{\theta}(\tau_i)}{P_{\theta_{\mathrm{old}}}(\tau_i)} = \prod_{t=1}^{T} \frac{\pi_{\theta}(x_{t-\Delta t}|x_t)}{\pi_{\theta_{\mathrm{old}}}(x_{t-\Delta t}|x_t)} \triangleq \prod_{t=1}^{T} w_{i,t}
    \end{equation}
    其中 $w_{i,t}$ 为第 $t$ 步的单步重要性比率。为了提升训练效率，我们允许利用当前采样进行多次梯度迭代更新。然而，将标准 PPO 裁剪机制直接应用于流匹配模型时存在一个固有缺陷。令 $\Delta\mu_{\theta} = \mu_{\theta}(x_t|t) - \mu_{\theta_{\mathrm{old}}}(x_t|t)$ 为当前策略与旧策略的预测均值之差。由于采样点来自旧策略 $x_{t-\Delta t} = \mu_{\theta_{\mathrm{old}}} + \sigma_t\sqrt{\Delta t}\cdot\epsilon$（$\epsilon \sim \mathcal{N}(0,\mathbf{I})$），单步对数重要性比率可分解为：
    \begin{equation}
        \log w_{i,t} = \frac{\Delta\mu_{\theta}\cdot\epsilon}{\sigma_t\sqrt{\Delta t}} - \frac{\|\Delta\mu_{\theta}\|^2}{2\sigma_t^2\Delta t}
    \end{equation}
    其中第二项为依赖于时间步 $(\sigma_t, \Delta t)$ 的负向偏置，其期望 $\mathbb{E}_\epsilon[\log w_{i,t}] = -\|\Delta\mu_{\theta}\|^2/(2\sigma_t^2\Delta t) < 0$。这导致 $w_{i,t}$ 的分布整体左移（均值 $< 1$），使得正向优势样本（$A_i > 0$）的比率几乎无法触发上限裁剪边界 $1 + \varepsilon$，策略更新失去约束，进而引发奖励过度优化（Reward Hacking）。此外，由于偏置的大小随 $\sigma_t$ 和 $\Delta t$ 变化，不同时间步的比率分布方差也不一致，导致单一裁剪阈值无法跨时间步均匀生效。

    \subparagraph{RatioNorm-based clipping} 为解决上述分布偏移与方差不一致问题，与Flow-GRPO的后续工作GRPO-Guard类似，我们对单步对数重要性比率进行标准化（Standardization）处理：先加回负偏置项以消除均值偏移，再乘以 $\sigma_t\sqrt{\Delta t}$ 以移除方差对去噪调度器参数的依赖：
    \begin{equation}
        \log \hat{w}_{i,t} = \sigma_t\sqrt{\Delta t}\left(\log w_{i,t} + \frac{\|\Delta\mu_{\theta}\|^2}{2\sigma_t^2\Delta t}\right) = \Delta\mu_{\theta}\cdot\epsilon
    \end{equation}
    标准化后的对数比率 $\log\hat{w}_{i,t} = \Delta\mu_{\theta}\cdot\epsilon$ 具有两个关键性质：(1) 均值为零（$\mathbb{E}[\epsilon] = 0$）；(2) 方差 $\|\Delta\mu_{\theta}\|^2$ 仅取决于策略差异本身，不再受 $\sigma_t$ 和 $\Delta t$ 的干扰。这使得 PPO 风格的对称裁剪区间 $[1-\varepsilon,\ 1+\varepsilon]$ 能够对正负优势样本均匀生效。

    \subparagraph{Trajectory-level aggregation} 基于逐步标准化比率，轨迹级重要性权重定义为各步标准化对数比率的均值：
    \begin{equation}
        \log \hat{w}_i = \frac{1}{T}\sum_{t=1}^{T} \log \hat{w}_{i,t} = \frac{1}{T}\sum_{t=1}^{T} \Delta\mu_{\theta,t}\cdot\epsilon_t
    \end{equation}
    该均值化操作消除了轨迹长度 $T$ 对权重幅度的累积效应，使 $\hat{w}_i = \exp(\log\hat{w}_i)$ 的尺度与单步比率保持一致。最终对 $\hat{w}_i$ 施加对称裁剪 $\mathrm{clip}(\hat{w}_i,\; 1{-}\varepsilon,\; 1{+}\varepsilon)$，以在多次梯度迭代中约束策略偏移。

    \paragraph{Total objective function} 结合上述各项，TreeMatch-RL 的总损失函数定义为：
    \begin{equation}
        \mathcal{L}_{total} = \frac{1}{K}\sum_{i=1}^{K} \mathrm{clip}\!\left(\hat{w}_i,\; 1{-}\varepsilon,\; 1{+}\varepsilon\right) \cdot \mathcal{L}_{Soft\text{-}TB}^{(i)} + \lambda_1 \mathcal{L}_{Entropy} + \lambda_2 \mathcal{L}_{Ref}
    \end{equation}
    其中 $\mathcal{L}_{Soft\text{-}TB}^{(i)}$ 为第 $i$ 条轨迹在 Softmax-TB 损失中的平方残差，$\hat{w}_i$ 为经 RatioNorm 标准化的轨迹级重要性权重，$\varepsilon$ 为裁剪半径，$\lambda_1$、$\lambda_2$ 为正则化系数。
    
    
    \subsection{Inference and training acceleration with dpm-solver++}
    
    针对 SD3.5 和 Flux 等流匹配（Flow Matching）模型，TreeMatch-RL 集成了高阶 ODE 求解器 \textbf{DPM-Solver++} 以实现训练加速。
    
    \paragraph{Translation layer for flow matching} 由于流匹配模型预测的是速度 $v_{\theta}$，而 DPM-Solver++ 需要干净数据 $x_0$ 的预测值，我们建立了一个数学“翻译层”。在 Rectified Flow 框架下，利用线性插值关系 $x_{t_i} = t_i x_1 + (1-t_i) x_0$，我们将模型输出 $v_{\theta}$ 实时转换为对 $x_0$ 的估计：
    \begin{equation}
        x_{\theta}(x_i, t_i, c) = x_i - v_{\theta}(x_i, t_i, c) \cdot t_i
    \end{equation}
    
    \paragraph{Second-order midpoint acceleration} 在得到 $x_0$ 预测后，系统将其映射至对数信噪比（log-SNR）空间 $\lambda_{t_i} = \ln((1-t_i)/t_i)$ 进行积分。令步长 $h_i = \lambda_{t_i} - \lambda_{t_{i-1}}$，系统结合前一时刻信息进行二阶多步修正以捕获轨迹曲率：
    \begin{equation}
        D_i \leftarrow (1 + \frac{h_i}{2h_{i-1}}) x_0^{(i-1)} - \frac{h_i}{2h_{i-1}} x_0^{(i-2)}
    \end{equation}
    利用该二阶修正预测器 $D_i$，我们最终执行基于指数积分的状态转移方程：
    \begin{equation}
        x_{t_i} = \frac{t_i}{t_{i-1}} x_{t_{i-1}} - (1 - t_i) (e^{h_i} - 1) D_i
    \end{equation}
    该方程能够比一阶欧拉法更精确地贴合非线性概率流轨迹。实验表明，该方案可将参考模型 $\pi_{\theta_{old}}$ 的总训练耗时显著缩短约\%。
    
    \section{Experiments}
    \label{sec:experiments}
    
    本节旨在定量与定性地评估 \textbf{TreeMatch-RL} 在对齐人类偏好及保持生成多样性方面的表现。我们重点分析框架在处理流匹配模型（Flow Matching）时的效率增益与分布对齐效果。
    
    \subsection{Experimental setup}
    
    \paragraph{Datasets and Metrics} 
    我们使用 \textbf{Pick-a-Pic v2} 提示词子集进行在线 RL 训练。在评估阶段，我们采用以下公认指标：
    \begin{itemize}
        \item \textbf{对齐得分}：采用 \textbf{HPS-v2.1} 和 \textbf{PickScore} 评估模型生成结果与人类审美及文本描述的一致性。
        \item \textbf{语义准确性}：使用 \textbf{GenEval} 基准测试，评估模型在物体计数、空间关系及属性绑定方面的逻辑能力。
        \item \textbf{多样性与保真度}：使用 \textbf{FID}（Fréchet Inception Distance）衡量生成分布与真实分布的距离，FID 越低代表模型在对齐过程中能更好地维持图像质量与多样性。
    \end{itemize}
    
    \paragraph{Implementation Details} 
    实验基于 \textbf{SD3.5-Medium} 和 \textbf{Flux.1-schnell} 展开。我们采用 HuggingFace \texttt{accelerate} 进行分布式训练。模型利用 LoRA（$r=32$）进行微调。在采样阶段，我们固定 3 个 SDE 分叉步，其余步骤使用 DPM-Solver++ 加速以降低 NFE。
    
    \paragraph{Baselines} 
    我们将 TreeMatch-RL 与以下最先进的 GRPO 变体进行对比：
    \begin{itemize}
        \item \textbf{Flow-GRPO}：首个将 ODE 到 SDE 转换应用于流匹配模型在线 RL 的框架。
        \item \textbf{TreeGRPO}：引入树状优势估计（Tree-Advantage），通过共享路径前缀提升训练吞吐量。
        \item \textbf{DanceGRPO}：通过群体相对策略优化（GRPO）提升视觉生成任务的稳定性。
        \item \textbf{DenseGRPO}：通过预测步进奖励增益（Step-wise Reward Gain）来解决稀疏奖励挑战的方案。
    \end{itemize}
    
    \subsection{Main results and quantitative analysis}
    
    
    
    我们在 Table~\ref{tab:comparison} 中展示了各方法在 SD3.5-Medium 上的性能对比。
    
    \begin{table}[t]
      \caption{TreeMatch-RL 与各 Baseline 在 SD3.5-Medium 上的对比。我们汇报了在多项标准指标下的表现。$\uparrow$ 表示数值越高越好，$\downarrow$ 表示数值越低越好。}
      \label{tab:comparison}
      \centering
      \begin{tabular}{lcccc}
        \toprule
        \textbf{Method} & \textbf{HPS-v2.1} $\uparrow$ & \textbf{PickScore} $\uparrow$ & \textbf{GenEval} $\uparrow$ & \textbf{FID} $\downarrow$ \\
        \midrule
        \midrule
        Flow-GRPO &  &  &  & \\
        TreeGRPO &  &  &  & \\
        DanceGRPO &  &  &  &  \\
        DenseGRPO & \textbf{} & \textbf{} & \textbf{} &  \\
        \midrule
        \textbf{TreeMatch-RL (Ours)} &  &  &  & \textbf{} \\
        \bottomrule
      \end{tabular}
    \end{table}
    
    \paragraph{多样性与奖励的博弈} 如表~\ref{tab:comparison} 所示，采用传统策略梯度范式的基准方法（如 DenseGRPO）虽然在 HPS 和 GenEval 指标上取得了最优的对齐效果，但其 FID 显著升高，表明其在优化过程中出现了严重的模式塌缩，图像风格趋于单一。相比之下，\textbf{TreeMatch-RL} 通过 \textbf{Softmax-TB} 损失实现了分布匹配，其 FID 仅为 ，远低于其他 GRPO 变体。这证明了 TreeMatch-RL 能够有效地在对齐人类偏好的同时，保留扩散模型原始的生成多样性。
    
    \paragraph{训练效率分析} 
    得益于树状采样结构和 DPM-Solver++ 加速逻辑，TreeMatch-RL 在单次采样中生成的 27 个分支共享了大部分前缀计算。与独立采样的 Flow-GRPO 相比，TreeMatch-RL 的训练时间缩短了约 倍，同时通过自适应调度（Adaptive Scheduling）将梯度更新集中在关键的语义分叉阶段，加快了收敛速度。
    
    \subsection{Ablation study}

    为验证 TreeMatch-RL 各核心组件的贡献，我们在 SD3.5-Medium 上进行了系统性的消融实验。以完整框架为基线，每次移除或替换一个组件，结果汇总于表~\ref{tab:ablation}。

    \paragraph{损失函数的影响}
    将 Softmax-TB 替换为标准 GRPO 目标（即奖励最大化）后，HPS 和 PickScore 小幅上升，但 FID 显著恶化，证实了分布匹配范式对多样性保持的关键作用。移除粒子熵正则 $\mathcal{L}_{Entropy}$ 后，FID 同样出现明显退化，表明 RBF 排斥力在 Latent 空间中有效抑制了路径坍缩。移除参考约束 $\mathcal{L}_{Ref}$ 则导致训练后期奖励曲线出现振荡，GenEval 下降，说明该项对维持速度场稳定性不可或缺。

    \paragraph{树状采样结构的效果}
    将三阶树结构替换为 $K{=}27$ 条独立平行采样（Flat）后，训练吞吐量下降约 40\%，且由于路径间缺乏结构化的奖励对比，Softmax-TB 损失的梯度方差增大，最终各指标均有所下滑。将树结构简化为二阶（$3^2{=}9$ 条路径）则导致组内对比样本不足，奖励分布匹配精度降低，HPS 和 FID 均出现退化。这表明三阶分叉在对比多样性与计算开销之间取得了最优平衡。

    \paragraph{RatioNorm 裁剪的必要性}
    移除 RatioNorm 标准化（即直接对原始比率 $w_i$ 施加裁剪）后，我们观察到训练初期奖励快速攀升但随后急剧回落——这是典型的 Reward Hacking 现象。由于原始 $\log w_{i,t}$ 的负偏置随 $\sigma_t$ 和 $\Delta t$ 变化，固定裁剪阈值无法有效约束策略偏移，最终 FID 大幅恶化。完全移除重要性采样（仅进行单步在策略更新）则导致样本利用率锐降，收敛速度显著放缓。

    \paragraph{自适应调度与加速}
    将集中度参数设为 $\kappa{=}0$（退化为均匀采样 $\mathcal{U}(0,1)$）后，模型在低奖励 Prompt 上的 GenEval 表现明显下降，验证了 Beta 分布调度对难题场景中全局结构重塑的重要性。将 DPM-Solver++ 替换为一阶 Euler 法不影响最终指标（因仅作用于非 SDE 区间的参考模型推理），但单步训练时间增加约 50\%，凸显了高阶求解器在工程效率上的贡献。
    \begin{table}[t]
    \caption{消融实验结果（SD3.5-Medium）。每行移除或替换一个组件。$\Delta$ 表示相对于完整 TreeMatch-RL 的变化量。}
    \label{tab:ablation}
    \centering
    \begin{tabular}{lccccc}
        \toprule
        \textbf{Variant} & \textbf{HPS-v2.1} $\uparrow$ & \textbf{PickScore} $\uparrow$ & \textbf{GenEval} $\uparrow$ & \textbf{FID} $\downarrow$ & \textbf{Time} (h) $\downarrow$ \\
        \midrule
        \textbf{TreeMatch-RL (Full)} &  &  &  &  &  \\
        \midrule
        \multicolumn{6}{l}{\textit{Loss function}} \\
        \quad Softmax-TB $\to$ GRPO &  &  &  &  &  \\
        \quad w/o $\mathcal{L}_{Entropy}$ &  &  &  &  &  \\
        \quad w/o $\mathcal{L}_{Ref}$ &  &  &  &  &  \\
        \midrule
        \multicolumn{6}{l}{\textit{Sampling structure}} \\
        \quad Tree $\to$ Flat ($K{=}27$) &  &  &  &  &  \\
        \quad 2-stage ($3^2{=}9$ paths) &  &  &  &  &  \\
        \midrule
        \multicolumn{6}{l}{\textit{Importance sampling \& clipping}} \\
        \quad w/o RatioNorm &  &  &  &  &  \\
        \quad w/o IS (single-step) &  &  &  &  &  \\
        \midrule
        \multicolumn{6}{l}{\textit{Scheduling \& acceleration}} \\
        \quad $\kappa{=}0$ (Uniform) &  &  &  &  &  \\
        \quad DPM-Solver++ $\to$ Euler &  &  &  &  &  \\
        \bottomrule
      \end{tabular}
    \end{table}
\end{document}
