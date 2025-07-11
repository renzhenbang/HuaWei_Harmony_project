# 📘 代码说明文档（Search-R1 / DeepSeek-R1 智能体系统）

本项目基于 DeepSeek-R1 大语言模型，构建了一个具备检索增强推理（Retrieval-Augmented Generation, RAG）能力的深度研究智能体系统，适用于 FRAMES benchmark 场景。系统支持结构化训练、多源数据协同、格式对齐推理和 REST 服务化接口，具备良好的可扩展性与复现性。
## 前期准备工作

### 推理环境配置
```bash
conda create -n searchr1 python=3.9
conda activate searchr1
# install torch [or you can skip this step and let vllm to install the correct version for you]
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# install vllm
pip3 install vllm==0.6.3 # or you can install 0.5.4, 0.4.2 and 0.3.1

# verl
pip install -e .

# flash attention 2
pip3 install flash-attn --no-build-isolation
pip install wandb
```

### 检索环境配置
```bash
conda create -n retriever python=3.10
conda activate retriever

# we recommend installing torch with conda for faiss-gpu
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets pyserini

## install the gpu version faiss to guarantee efficient RL rollout
conda install -c pytorch -c nvidia faiss-gpu=1.8.0

## API function
pip install uvicorn fastapi
```
### 下载语料库
```bash
save_path=/home/renzhenbang/Search-R1/index_corpus
python scripts/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz
```
### 处理数据集
（1）NQ数据集
```bash
python /home/renzhenbang/Search-R1/scripts/data_process/nq_search.py
```
（2）ARC数据集
```bash
python /home/renzhenbang/Search-R1/scripts/data_process/arc.py
```
（3）SpatialVLM数据集
```bash
python /home/renzhenbang/Search-R1/scripts/data_process/SpatialVLM.py
```
（4）TReB数据集
```bash
python /home/renzhenbang/Search-R1/scripts/data_process/treb.py
```

## 初步推理评估
在无任何训练的条件下直接运行推理脚本，可评估原始 DeepSeek-R1 模型在 FRAMES 测试集上的表现：
```bash
python /home/renzhenbang/run_eval.py
```
## NQ 数据集上的 GRPO 微调
我们参考 Search-R1 的结构化训练范式，在 NQ 数据集上对模型进行微调，以提升其在结构化推理任务中的表现。
步骤如下：
启动检索服务（默认基于 Jina embedding + reranker 模型）：
```bash
bash /home/renzhenbang/Search-R1/retrieval_launch.sh
```
启动 GRPO 训练流程：
```bash
bash /home/renzhenbang/Search-R1/train_grpo.sh
```
训练权重将自动保存至："/home/renzhenbang/Search-R1/checkpoint"目录下
替换推理模型路径后再次运行推理：
```bash
python /home/renzhenbang/run_eval.py
```
## 统一格式后的评估流程
由于训练集（NQ）与评测集（FRAMES）在输入格式存在差异，我们采用格式对齐机制对 prompt 输入进行转换，增强模型对 slot/schema 的结构化泛化能力。
运行如下脚本：
```bash
python /home/renzhenbang/Search-R1/evaluate_with_search_r1.py
```
该流程将自动调用格式构造器、检索模块及推理模块，并记录输出用于后续评估。
## 多源数据联合训练流程
系统支持使用多个外部数据集（如 TReB、ARC、SpatialVLM）进行联合训练。训练脚本结构已预留统一训练流程，您只需：
配置好目标数据集路径（仿照 NQ 格式预处理）；
启动训练脚本：
```bash
bash /home/renzhenbang/Search-R1/train_grpo.sh
```
注意：多源训练策略（如采样权重、融合比例）目前使用默认设置，未来可拓展为自适应调度策略。
## 基于 Jina 的 REST 接口服务
为支持服务化部署与在线调用，我们构建了基于 Jina 的接口模块。
调用步骤：
进入接口目录并启动服务：
```bash
cd /home/renzhenbang/Search-R1/evaluate_jina
python app.py
```
使用 curl 或任何 REST 客户端访问：
```bash
curl -X POST 'http://localhost:8000/search_r1' \
     -H 'Content-Type: application/json' \
     -d '{"data":[{"text":"Who wrote The Three-Body Problem?"}]}'
```
系统将自动完成检索、对齐、推理等流程，并返回最终生成结果。
之后采用curl直接启动推理服务即可。
## 实验结果与性能总结
系统在以下步骤中取得明显性能提升：
| 阶段         | 使用外部数据 | 格式对齐 | 联合训练 | FRAMES 准确率 |
| ---------- | ------ | ---- | ---- | ---------- |
| 初始模型（无训练）  | ×      | ×    | ×    | 5.1%       |
| NQ结构化微调    | √      | ×    | ×    | 35.1%      |
| 格式对齐增强     | √      | √    | ×    | 53.8%      |
| 多源联合训练（最终） | √      | √    | √    | 70.1%      |
报告中进一步提出了“熵-对齐奖励机制”的设想以实现端到端强化学习优化，但本次提交的系统尚未实现该部分，仅保留了接口与框架设计以供未来拓展使用。
