# Install

1. 安装`uv`
2. 克隆项目: `git clone git@github.com:kamiertop/insect.git`
3. 安装依赖：`uv sync`
4. 编辑配置文件： `config.toml`
5. 处理数据，生成标签文件：`uv run -m scripts.gen_label`
6. 生成`train.csv`, `val.csv`, `test.csv`：`uv run -m scripts.gen_index --order="直翅目"`
7. 运行一轮测试：`uv run -m scripts.once`
8. 训练模型：`uv run -m scripts.train --config ./config.toml` 