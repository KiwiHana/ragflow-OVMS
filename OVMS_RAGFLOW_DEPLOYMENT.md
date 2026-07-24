# RAGFlow + OVMS 部署手册

Date: 2026-07-23

本文档基于Ubuntu 24.04（kernel大于等于6.17）实测整理，目标是完成：

- 启动 OVMS 模型服务（Embedding on GPU / Rerank on NPU / Chat on NPU）
- 让 RAGFlow 正确接入 OVMS（多端口）
- 验证 RAGFlow + OVMS 可用

## 0. Quick start

### 0.1 准备

目标：在一台全新 Ubuntu 机器上，让 RAGFlow 在 Web UI 中正确看到并使用 OVMS 的 Chat / Embedding / Rerank。

（1）基础环境检查

```bash
# Docker
sudo docker pull hello-world

# OpenVINO Python 环境（用于模型检查/导出）
python -m venv ~/openvino_env
source ~/openvino_env/bin/activate
python -m pip install --upgrade pip
pip install openvino
python -c "from openvino import Core; print(Core().available_devices)"
```

期望至少包含 `CPU`，若有 iGPU/NPU 则显示 `GPU` / `NPU`。如果没有GPU或者NPU，可以参考指南来安装：https://zhuanlan.zhihu.com/p/2032752671648658519

（2）代码与模型目录

```bash
cd ~
git clone https://github.com/KiwiHana/ragflow-OVMS.git ragflow
cp -r ~/ragflow/models ~/
cd ~/models
sudo chmod -R 777 run*.sh
```
下载Qwen3.6-35B-A3B-int4-ov到~/models，并
```bash
mv Qwen3.6-35B-A3B-int4-ov Qwen3.6-35B-A3B-ov
```
下载地址：https://huggingface.co/OpenVINO/Qwen3.6-35B-A3B-int4-ov ，或者 https://www.modelscope.cn/models/OpenVINO/Qwen3.6-35B-A3B-int4-ov

下载bge-large-zh-v1.5和bge-reranker-large到~/models/BAAI

下载地址： https://www.modelscope.cn/models/kiwicoco/bge-large-zh-v1.5/files

https://www.modelscope.cn/models/kiwicoco/bge-reranker-large/files

验证OVMS所需要的文件准备完毕。
```
~$ tree ~/models
~/models
├── BAAI
│   ├── bge-large-zh-v1.5
│   ├── bge-reranker-large
├── config.json
├── config-reranker.json
├── run-bge-large-zh-v1.5-docker.sh
├── run-bge-reranker-large-docker.sh
├── run-35b-docker.sh
├── Qwen3.6-35B-A3B-ov
```

### 0.2 启动三路 OVMS（先确认服务可用）

Qwen3.6-35B-A3B-ov运行在GPU。bge-large-zh-v1.5和bge-reranker-large运行在NPU。

```bash
cd ~/models && ./run-bge-large-zh-v1.5-docker.sh
cd ~/models && ./run-bge-reranker-large-docker.sh
cd ~/models && ./run-35b-docker.sh
```

### 0.3 启动 RAGFlow（再做容器网络检查）

```bash
cd ~/ragflow/docker
sudo docker compose -f docker-compose.yml up -d
sudo docker compose -f docker-compose.yml ps
```

在 RAGFlow 容器内验证访问宿主机 OVMS：

```bash
sudo docker exec docker-ragflow-cpu-1 sh -lc 'curl -sS --max-time 8 http://host.docker.internal:8000/v3/models/bge-large-zh-v1.5-int8-ov'
sudo docker exec docker-ragflow-cpu-1 sh -lc 'curl -sS --max-time 8 http://host.docker.internal:8001/v3/models/bge-reranker-large-int8-ov'
sudo docker exec docker-ragflow-cpu-1 sh -lc 'curl -sS --max-time 8 http://host.docker.internal:8002/v3/models'
```

然后在浏览器打开RAGFLOW Web： http://127.0.0.1:8080

说明：

- 在 Docker 内配置 provider 时，`base_url` 必须优先用 `host.docker.internal`；不要填 `127.0.0.1`，否则会指向容器自己而不是宿主机 OVMS。

### 0.4 在 RAGFlow UI 填写 OpenAI-API-Compatible（关键）

首次使用需要注册，登录后先点击右上角用户头像，左侧栏选Model providers，右侧搜索 `OpenAI-API-Compatible`，并分别添加 3 个实例，最后在set default models选择LLM，Embedding，Rerank模型。OVMS到此部署完毕，可以开始使用RAGFLOW里的功能了。

1. Chat 实例

- Provider: `OpenAI-API-Compatible`
- Instance name: `OVMS-Chat`
- Base URL: `http://host.docker.internal:8002/v3`
- API Key: `unused`
- Model name: `Qwen3.6-35B-A3B-ov`
- Model type: `chat`
<img width="331" height="335" alt="image" src="https://github.com/user-attachments/assets/8c815851-6aa3-4370-a8e7-7060c77d9c41" />

2. Embedding 实例

- Provider: `OpenAI-API-Compatible`
- Instance name: `OVMS-Embedding`
- Base URL: `http://host.docker.internal:8000/v3/embedding`
- API Key: `unused`
- Model name: `bge-large-zh-v1.5-int8-ov`
- Model type: `embedding`

<img width="331" height="335" alt="image" src="https://github.com/user-attachments/assets/0a25207c-ea6e-425f-872e-9b4ea35b2db4" />
<img width="218" height="266" alt="image" src="https://github.com/user-attachments/assets/4d1b05e6-351a-4af0-b3d1-044558f39f9f" />

3. Rerank 实例

- Provider: `OpenAI-API-Compatible`
- Instance name: `OVMS-Rerank`
- Base URL: `http://host.docker.internal:8001/v3`
- API Key: `unused`
- Model name: `bge-reranker-large-int8-ov`
- Model type: `rerank`

<img width="330" height="335" alt="image" src="https://github.com/user-attachments/assets/bc861ea7-8ab1-4430-939c-d2b602d2325e" />

<img width="771" height="377" alt="image" src="https://github.com/user-attachments/assets/928e74f5-c8b9-4083-8230-814784cce1e2" />

以上模型 ID 是使用三段格式：

- chat: `Qwen3.6-35B-A3B-ov@OVMS-Chat@OpenAI-API-Compatible`
- embedding: `bge-large-zh-v1.5-int8-ov@OVMS-Embedding@OpenAI-API-Compatible`
- rerank: `bge-reranker-large-int8-ov@OVMS-Rerank@OpenAI-API-Compatible`


### 0.5 查看GPU，NPU使用情况（Option）
查看GPU状态。
```
sudo apt install xpu-smi
sudo watch -n 1 xpu-smi stats -d 0
```
如果安装不成功，解决方案参考：https://dgpu-docs.intel.com/installation-guides/installing-packages-from-the-intel-ppa.html

查看NPU状态。
```
git clone https://github.com/DMontgomery40/intel-npu-top.git
cd intel-npu-top
python3 intel-npu-top.py
```

### 0.6 已部署机器如何停止/恢复 RAGFlow

适用场景：机器已开机、容器已在运行，需要临时停止服务或完整下线。

先查看当前状态：

```bash
cd ~/ragflow/docker
sudo docker compose -f docker-compose.yml ps
```

1) 停止整套 compose 服务（容器保留，数据不删）

```bash
cd ~/ragflow/docker
sudo docker compose -f docker-compose.yml stop
```

恢复：

```bash
cd ~/ragflow/docker
sudo docker compose -f docker-compose.yml up -d
```

2) 下线并删除容器（数据卷默认保留）

```bash
cd ~/ragflow/docker
sudo docker compose -f docker-compose.yml down
```

恢复：

```bash
cd ~/ragflow/docker
sudo docker compose -f docker-compose.yml up -d
```

3) 下线并删除容器 + 数据卷（高风险，会清空业务数据）

```bash
cd ~/ragflow/docker
sudo docker compose -f docker-compose.yml down -v
```

仅在“确认不需要保留知识库/配置数据”时使用该命令。

停止后快速核验：

```bash
sudo docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
curl -I http://127.0.0.1:8080 || true
```

期望：

- 只停应用时，`docker-ragflow-cpu-1` 不在运行，其它基础组件仍可运行；
- `http://127.0.0.1:8080` 无法访问（或连接被拒绝）。

## 1. 部署文件介绍

给出的实际启动方式是三路 OVMS 服务：

- Embedding: 8000
- Rerank: 8001
- Chat: 8002

当前机器上对应脚本：

- `~/models/run-bge-large-zh-v1.5-docker.sh`
- `~/models/run-bge-reranker-large-docker.sh`
- `~/models/run-35b-docker.sh`

当前模型配置文件：

- `~/models/config.json` -> `bge-large-zh-v1.5-int8-ov`
- `~/models/config-reranker.json` -> `bge-reranker-large-int8-ov`

注意：

- 这三路服务不是同一个端口，RAGFlow 不能把 chat/embedding/rerank 全部指向单一 `OVMS_OPENAI_BASE_URL`。
- 本仓库已改为按模型类型分别设置 base_url（见第 4 节）。


## 2. 启动 OVMS（三路）

### 2.1 启动 Embedding OVMS（8000）

```bash
cd ~/models
./run-bge-large-zh-v1.5-docker.sh
```

### 2.2 启动 Rerank OVMS（8001）

```bash
cd ~/models
./run-bge-reranker-large-docker.sh
```

### 2.3 启动 Chat OVMS（8002）

```bash
cd ~/models
./run-35b-docker.sh
```

说明：

- 以上脚本内部使用 `sudo docker run ...`，若提示密码，按终端提示输入即可。
- 脚本使用 `--rm`，容器停止后会自动删除。

## 3. 检查 OVMS 服务状态

### 3.1 Embedding 模型检查（8000）

```bash
curl http://127.0.0.1:8000/v3/models/bge-large-zh-v1.5-int8-ov
```

期望返回包含：

```json
{"id":"bge-large-zh-v1.5-int8-ov","object":"model","owned_by":"OVMS"}
```

### 3.2 Rerank 模型检查（8001）

```bash
curl http://127.0.0.1:8001/v3/models/bge-reranker-large-int8-ov
```

期望返回包含：

```json
{"id":"bge-reranker-large-int8-ov","object":"model","owned_by":"OVMS"}
```

### 3.3 Chat 模型检查（8002）

```bash
curl http://127.0.0.1:8002/v3/models
```

期望返回中包含：

- 模型名 `Qwen3.6-35B-A3B-ov`
- 可用状态 `AVAILABLE`

## 4. RAGFlow 侧 OVMS 配置（已在本仓库完成）

已修改文件：

- `~/ragflow/docker/.env`
- `~/ragflow/docker/service_conf.yaml.template`

### 4.1 docker/.env

关键变量：

```env
OVMS_OPENAI_BASE_URL=http://host.docker.internal:8002/v3
OVMS_CHAT_BASE_URL=http://host.docker.internal:8002/v3
OVMS_EMBEDDING_BASE_URL=http://host.docker.internal:8000/v3
OVMS_RERANK_BASE_URL=http://host.docker.internal:8001/v3
OVMS_API_KEY=unused
OVMS_CHAT_MODEL=Qwen3.6-35B-A3B-ov
OVMS_EMBEDDING_MODEL=bge-large-zh-v1.5-int8-ov
OVMS_RERANK_MODEL=bge-reranker-large-int8-ov
```

说明：

- RAGFlow 在 Docker 容器内运行时，`127.0.0.1` 指向容器自身，不是宿主机。
- 因此 OVMS base_url 必须使用 `host.docker.internal`（或宿主机实际 IP）。
- `127.0.0.1` 仅适用于“客户端与 OVMS 在同一网络命名空间”的场景。

### 4.2 service_conf.yaml.template

`user_default_llm.default_models` 已改成：

- `chat_model` 使用 `OVMS_CHAT_BASE_URL`
- `embedding_model` 使用 `OVMS_EMBEDDING_BASE_URL`
- `rerank_model` 使用 `OVMS_RERANK_BASE_URL`

这样可直接适配三端口 OVMS。
