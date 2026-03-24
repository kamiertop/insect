# Install

1. 安装`uv`
2. 克隆项目: `git clone git@github.com:kamiertop/insect.git`
3. 安装依赖：`uv sync`
4. 使用脚本默认参数直接运行（可按需通过命令行覆盖）

## 目录规范

- `dataset/`：数据加载代码（`InsectDataset`、`build_dataloader`）
- `data/`：原始数据目录（如 `data/datasets/`），不再放训练代码
- `artifacts/data/`：生成后的 `label.csv`、`active_split.txt`
- `artifacts/data/splits/<split_id>/`：版本化切分结果（`split.csv`、`meta.json`）
- `artifacts/checkpoints/`：固定训练权重输出目录（`best.pt`、`last.pt`）
- `artifacts/eval/`：固定评估输出目录（`summary.csv`、`per_class.csv`、`confusion_matrix.csv`）

## 典型流程

1. 处理数据，生成标签文件：`uv run -m scripts.gen_label`
2. 生成切分索引并激活（版本化 `split_id`）：`uv run -m scripts.gen_index --order="直翅目" --split-id orthoptera_s42 --seed 42 --activate`
3. 运行一轮 smoke 测试：`uv run -m scripts.once`
4. 训练模型：`uv run -m scripts.train`
   - 命令行调参示例：`uv run -m scripts.train --epochs 30 --batch-size 32 --lr 0.0003 --fine-tune`
5. 评估模型：`uv run -m scripts.eval --checkpoint ./artifacts/checkpoints/best.pt`

默认评估结果输出到 `artifacts/eval/`（可通过 `--save-dir` 覆盖）。

> 说明：当前激活切分由 `artifacts/data/active_split.txt` 管理，可通过 `scripts.gen_index --activate` 更新。

