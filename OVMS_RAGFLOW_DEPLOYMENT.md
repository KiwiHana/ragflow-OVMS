# RAGFlow + OVMS 部署手册

Date: 2026-07-21

本文档基于Ubuntu 24.04（kernel大于等于6.17）实测整理，目标是完成：

- 启动 OVMS 模型服务（Embedding / Rerank / Chat）
- 让 RAGFlow 正确接入 OVMS（多端口）
- 验证 RAGFlow + OVMS 可用

## 0. Quick start

### 0.1 准备工作

（1）提前安装GPU/NPU相关驱动，并验证OpenVINO环境。
```
python3 -m venv openvino_env
source openvino_env/bin/activate
python3 -m pip install --upgrade pip
pip install openvino
#在openvino_env环境中检查是否识别iGPU和NPU
python3 -c "from openvino import Core; print(Core().available_devices)"

['CPU','GPU','NPU']
```

（2）验证docker正常使用。
```
sudo docker pull hello-world
```

### 0.2 下载模型和代码文件

```
cd ~

git clone https://github.com/KiwiHana/ragflow-OVMS.git ragflow

cp -r ragflow/models ~/

cd ~/models
```

下载Qwen3.6-35B-A3B-int4-ov到~/models，并
```
mv Qwen3.6-35B-A3B-int4-ov Qwen3.6-35B-A3B-ov
```

下载地址：https://huggingface.co/OpenVINO/Qwen3.6-35B-A3B-int4-ov ，或者 https://www.modelscope.cn/models/OpenVINO/Qwen3.6-35B-A3B-int4-ov

下载bge-large-zh-v1.5和bge-reranker-large到~/models/BAAI

下载地址：
https://www.modelscope.cn/models/kiwicoco/bge-large-zh-v1.5/files

https://www.modelscope.cn/models/kiwicoco/bge-reranker-large/files

验证OVMS所需要的文件准备完毕。
```
~$ tree ~/models
~/models
├── BAAI
│   ├── bge-large-zh-v1.5
│   │   ├── config.json
│   │   ├── graph.pbtxt
│   │   ├── openvino_config.json
│   │   ├── openvino_model.bin
│   │   ├── openvino_model.xml
│   │   ├── openvino_tokenizer.bin
│   │   ├── openvino_tokenizer.xml
│   │   ├── tokenizer_config.json
│   │   └── tokenizer.json
│   ├── bge-reranker-large
│   │   ├── config.json
│   │   ├── graph.pbtxt
│   │   ├── openvino_config.json
│   │   ├── openvino_model.bin
│   │   ├── openvino_model.xml
│   │   ├── openvino_tokenizer.bin
│   │   ├── openvino_tokenizer.xml
│   │   ├── tokenizer_config.json
│   │   └── tokenizer.json
├── config.json
├── config-reranker.json
├── run-bge-large-zh-v1.5-docker.sh
├── run-bge-reranker-large-docker.sh
├── run-35b-docker.sh
├── Qwen3.6-35B-A3B-ov
│   ├── chat_template.jinja
│   ├── config.json
│   ├── generation_config.json
│   ├── openvino_detokenizer.bin
│   ├── openvino_detokenizer.xml
│   ├── openvino_language_model.bin
│   ├── openvino_language_model.xml
│   ├── openvino_text_embeddings_model.bin
│   ├── openvino_text_embeddings_model.xml
│   ├── openvino_tokenizer.bin
│   ├── openvino_tokenizer.xml
│   ├── openvino_vision_embeddings_merger_model.bin
│   ├── openvino_vision_embeddings_merger_model.xml
│   ├── openvino_vision_embeddings_model.bin
│   ├── openvino_vision_embeddings_model.xml
│   ├── openvino_vision_embeddings_pos_model.bin
│   ├── openvino_vision_embeddings_pos_model.xml
│   ├── preprocessor_config.json
│   ├── processor_config.json
│   ├── tokenizer_config.json
│   └── tokenizer.json
```

### 0.3. 一键执行清单

```bash
# 1) 启动 OVMS
cd ~/models && ./run-bge-large-zh-v1.5-docker.sh
cd ~/models && ./run-bge-reranker-large-docker.sh
cd ~/models && ./run-35b-docker.sh

# 2) 启动/更新 RAGFlow
cd ~/ragflow/docker && sudo docker compose -f docker-compose.yml up -d

# 3) 核验
curl http://127.0.0.1:8000/v3/models/bge-large-zh-v1.5-int8-ov
curl http://127.0.0.1:8001/v3/models/bge-reranker-large-int8-ov
curl http://127.0.0.1:8002/v3/models
curl -I http://127.0.0.1:8080
```

若以上 4 个检查均返回正常，即可在浏览器里打开http://127.0.0.1:8080 RAGFlow Web 中直接使用 OVMS 模型。

### 0.4 在RAGFlow Web配置Model Providers

（1）依次填写Qwen3.6-35B-A3B-ov，http://host.docker.internal:8002/v3，unused。并补充list models

<img width="330" height="332" alt="image" src="https://github.com/user-attachments/assets/e51f2f6d-2f6b-40c1-be03-65560dfb9fb1" />

<img width="221" height="268" alt="image" src="https://github.com/user-attachments/assets/8c276732-7635-4696-819c-bae177068926" />

（2）依次填写bge-large-zh-v1.5-int8-ov，http://host.docker.internal:8000/v3/embedding，unused。并补充list models

<img width="329" height="332" alt="image" src="https://github.com/user-attachments/assets/240bd568-f43d-41c7-980d-4c08c9ac8153" />

<img width="221" height="266" alt="image" src="https://github.com/user-attachments/assets/0870c6b1-9364-47c3-bbd4-c3c3d5d10fff" />

（3）依次填写bge-reranker-large-int8-ov，http://host.docker.internal:8001/v3，unused。并补充list models
<img width="330" height="335" alt="image" src="https://github.com/user-attachments/assets/bc861ea7-8ab1-4430-939c-d2b602d2325e" />

<img width="215" height="266" alt="image" src="https://github.com/user-attachments/assets/d6cd5d45-f3b9-4440-a1a4-55fd7e08e197" />


<img width="771" height="377" alt="image" src="https://github.com/user-attachments/assets/928e74f5-c8b9-4083-8230-814784cce1e2" />


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

## 1. 参考来源与当前结论

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

当前状态（2026-07-21 实测）：

- Embedding（8000）可用。
- Chat（8002）可用。
- Rerank（8001）在替换兼容导出并完成第 9 节验收后，已可正常使用。

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

说明：

- 只要返回体中包含 `results[index,relevance_score]`，即可认为当前 rerank 导出与 `/v3/rerank` 兼容。
- 若后续替换导出，请重新执行第 9.3 节冒烟验证后再切换生产。

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

## 5. 启动/重启 RAGFlow

在仓库目录执行：

```bash
cd ~/ragflow/docker
sudo docker compose -f docker-compose.yml up -d
```

检查状态：

```bash
cd ~/ragflow/docker
sudo docker compose -f docker-compose.yml ps
```

Web 入口：

- http://localhost:8080

## 6. 联调验证（建议顺序）

### 6.1 宿主机验证 OVMS API

```bash
curl http://127.0.0.1:8002/v3/models
curl http://127.0.0.1:8000/v3/models/bge-large-zh-v1.5-int8-ov
curl http://127.0.0.1:8001/v3/models/bge-reranker-large-int8-ov
```

### 6.2 容器内验证到宿主机网络

```bash
cd ~/ragflow/docker
sudo docker exec docker-ragflow-cpu-1 sh -lc 'curl -sS --max-time 8 http://host.docker.internal:8002/v3/models'
sudo docker exec docker-ragflow-cpu-1 sh -lc 'curl -sS --max-time 8 http://host.docker.internal:8000/v3/models/bge-large-zh-v1.5-int8-ov'
sudo docker exec docker-ragflow-cpu-1 sh -lc 'curl -sS --max-time 8 http://host.docker.internal:8001/v3/models/bge-reranker-large-int8-ov'
```

### 6.3 RAGFlow 日志确认

```bash
cd ~/ragflow/docker
sudo docker logs --tail 200 docker-ragflow-cpu-1
```

关注是否出现：

- `user_default_llm` 已加载
- 默认模型名为：
  - `Qwen3.6-35B-A3B-ov`
  - `bge-large-zh-v1.5-int8-ov`
  - `bge-reranker-large-int8-ov`

## 7. 常见问题

### 7.1 8000/8001/8002 端口无监听

原因：OVMS 三个容器未启动或已退出。

处理：按第 2 节重新执行三条脚本。

### 7.2 RAGFlow 已起但调用模型失败

优先检查：

1. `host.docker.internal` 在容器内是否可达
2. 三个 OVMS 端口是否都在监听
3. 模型 ID 是否与配置完全一致（大小写敏感）

### 7.3 在 UI 中误选 `OpenAI` provider

现象：

- 点击连接测试后，日志里连续出现 `OpenAI/gpt-*`、`text-embedding-*`、`tts-*` 等模型探测失败。

原因：

- 这是在测试 `OpenAI` provider，不是在测试 OVMS。
- `OpenAI` provider 会按内置 OpenAI 模型清单逐个校验，因此会出现大量与 OVMS 无关的失败信息。

正确做法：

1. 在 RAGFlow 模型设置页面选择 `OpenAI-API-Compatible`。
2. `base_url` 填写 OVMS 地址，例如：
  - Chat: `http://host.docker.internal:8002/v3`
  - Embedding: `http://host.docker.internal:8000/v3`
  - Rerank: `http://host.docker.internal:8001/v3`
3. 模型名必须填写为 OVMS 实际暴露的模型名：
  - `Qwen3.6-35B-A3B-ov`
  - `bge-large-zh-v1.5-int8-ov`
  - `bge-reranker-large-int8-ov`
4. 不要使用 `OpenAI` provider 去验证 OVMS。

### 7.4 docker 权限问题

若出现 `permission denied while trying to connect to the Docker daemon socket`，使用 `sudo docker ...` 执行。

### 7.5 `102 No model`（OpenAI-API-Compatible）

现象：

- 在 `OpenAI-API-Compatible` 添加 embedding/rerank 后报 `102 No model`。

常见根因：

1. 只创建了实例，但没有把对应模型写入该实例。
2. 租户默认模型 ID 只写了两段格式（`model@provider`），系统会把实例名当成 `default`，而当前实例名并不是 `default`。
3. 在当前多端口 OVMS 部署下，chat / embedding / rerank 走的是不同 `base_url`，不能把三类模型都压到同一个两段格式默认实例上。

正确格式：

- `model@instance@provider`

示例：

- Chat: `Qwen3.6-35B-A3B-ov@Qwen3.6-35B-A3B-ov@OpenAI-API-Compatible`
- Embedding: `bge-large-zh-v1.5-int8-ov@OVMS-Embedding@OpenAI-API-Compatible`
- Rerank: `bge-reranker-large-int8-ov@Qwen3.6-35B-A3B-ov@OpenAI-API-Compatible`

重要限制：

- `model@provider` 这种两段格式只适用于该 provider 下所有模型都能共用同一个实例、同一个 `base_url` 的场景。
- 当前 OVMS 部署是：
  - chat -> `8002`
  - embedding -> `8000`
  - rerank -> `8001`
- 而 RAGFlow 的实例级配置只有一个 `base_url`。因此在当前部署方式下，默认模型必须保留三段格式，不能改成两段格式。

排查命令：

```bash
cd ~/ragflow/docker
sudo docker exec docker-mysql-1 mysql -uroot -pinfini_rag_flow -Drag_flow -e "SELECT i.instance_name,p.provider_name,i.extra FROM tenant_model_instance i JOIN tenant_model_provider p ON i.provider_id=p.id WHERE p.provider_name='OpenAI-API-Compatible';"
sudo docker exec docker-mysql-1 mysql -uroot -pinfini_rag_flow -Drag_flow -e "SELECT tm.model_name,tm.model_type,p.provider_name,i.instance_name,tm.status FROM tenant_model tm JOIN tenant_model_provider p ON tm.provider_id=p.id JOIN tenant_model_instance i ON tm.instance_id=i.id WHERE p.provider_name='OpenAI-API-Compatible' ORDER BY i.instance_name,tm.model_type,tm.model_name;"
sudo docker exec docker-mysql-1 mysql -uroot -pinfini_rag_flow -Drag_flow -e "SELECT id,llm_id,embd_id,rerank_id FROM tenant;"
```

### 7.6 `rerank` 接口报错排查（`/v3/rerank`）

现象：

- 报 `404 Client Error: Not Found for url: http://host.docker.internal:8002/v3/rerank`。
- 或者在切到 8001 后，返回 `Port for tensor name logits was not found`。

原因：

1. 路由错误：`rerank_id` 绑定到了 chat 实例（`base_url=8002`），因此请求落到聊天服务，触发 404 或 Wrong endpoint。
2. 持久化配置残留：即使已修改 `tenant.rerank_id`，`search.search_config` 里仍可能保存旧 `rerank_id`，检索时继续覆盖为旧路由。
3. 模型导出不兼容：若返回 `logits not found`，说明当前导出与 `/v3/rerank` 计算器不匹配。

先修复路由（必须做）：

```bash
cd ~/ragflow/docker

# 1) 创建独立 rerank 实例并绑定到 8001
sudo docker exec docker-mysql-1 mysql -uroot -pinfini_rag_flow -Drag_flow -e "SET @provider_id := (SELECT id FROM tenant_model_provider WHERE provider_name='OpenAI-API-Compatible' LIMIT 1); SET @rerank_instance_id := 'f4c8b972865c11f1ad31fbcda4fd1fe5'; INSERT INTO tenant_model_instance (id,create_time,create_date,update_time,update_date,instance_name,provider_id,api_key,status,extra) SELECT @rerank_instance_id, UNIX_TIMESTAMP()*1000, NOW(), UNIX_TIMESTAMP()*1000, NOW(), 'OVMS-Rerank', @provider_id, 'unused', 'active', '{\"base_url\": \"http://host.docker.internal:8001/v3\", \"region\": \"default\"}' FROM DUAL WHERE NOT EXISTS (SELECT 1 FROM tenant_model_instance WHERE id=@rerank_instance_id OR (provider_id=@provider_id AND instance_name='OVMS-Rerank'));"

# 2) 绑定 rerank 模型到 OVMS-Rerank
sudo docker exec docker-mysql-1 mysql -uroot -pinfini_rag_flow -Drag_flow -e "UPDATE tenant_model tm JOIN tenant_model_provider p ON tm.provider_id=p.id SET tm.instance_id=(SELECT id FROM tenant_model_instance WHERE provider_id=(SELECT id FROM tenant_model_provider WHERE provider_name='OpenAI-API-Compatible' LIMIT 1) AND instance_name='OVMS-Rerank' LIMIT 1), tm.update_time=UNIX_TIMESTAMP()*1000, tm.update_date=NOW() WHERE p.provider_name='OpenAI-API-Compatible' AND tm.model_type='rerank' AND tm.model_name='bge-reranker-large-int8-ov';"

# 3) 更新 tenant/dialog 的 rerank_id 为三段格式
sudo docker exec docker-mysql-1 mysql -uroot -pinfini_rag_flow -Drag_flow -e "UPDATE tenant SET rerank_id='bge-reranker-large-int8-ov@OVMS-Rerank@OpenAI-API-Compatible', tenant_rerank_id=NULL WHERE rerank_id LIKE 'bge-reranker-large-int8-ov%@OpenAI-API-Compatible'; UPDATE dialog d JOIN tenant t ON d.tenant_id=t.id SET d.rerank_id=t.rerank_id, d.tenant_rerank_id=NULL WHERE d.tenant_id=t.id AND (d.rerank_id='' OR d.rerank_id LIKE 'bge-reranker-large-int8-ov%@OpenAI-API-Compatible');"

# 4) 清理 Search 配置中持久化的旧 rerank_id（关键）
sudo docker exec docker-mysql-1 mysql -uroot -pinfini_rag_flow -Drag_flow -e "UPDATE search SET search_config=REPLACE(search_config, 'bge-reranker-large-int8-ov@Qwen3.6-35B-A3B-ov@OpenAI-API-Compatible', 'bge-reranker-large-int8-ov@OVMS-Rerank@OpenAI-API-Compatible') WHERE search_config LIKE '%bge-reranker-large-int8-ov@Qwen3.6-35B-A3B-ov@OpenAI-API-Compatible%';"

# 5) 重启 RAGFlow
sudo docker restart docker-ragflow-cpu-1
```

核验命令：

```bash
cd ~/ragflow/docker
sudo docker exec docker-mysql-1 mysql -uroot -pinfini_rag_flow -Drag_flow -e "SELECT id,llm_id,embd_id,rerank_id FROM tenant;"
sudo docker exec docker-mysql-1 mysql -uroot -pinfini_rag_flow -Drag_flow -e "SELECT id,name,search_config FROM search WHERE tenant_id='c2fd45ae843d11f1a059698ef4239658' ORDER BY update_time DESC LIMIT 20;"
curl -sS -X POST http://127.0.0.1:8001/v3/rerank -H 'Content-Type: application/json' -d '{"model":"bge-reranker-large-int8-ov","query":"什么是RAG","documents":["RAG是检索增强生成","天气很好"]}'
```

边界说明：

- 路由问题可完全修复（不会再打到 8002）。
- 若仍报 `logits not found`，属于模型导出/图兼容问题，不是 RAGFlow 配置问题。
- 本机当前状态：已完成兼容导出替换并通过 `/v3/rerank` 验证，rerank 可正常使用。

建议：

1. 保持当前三端口部署：
  - 8000 embedding 可用
  - 8002 chat 可用
  - 8001 作为本地实验端口
2. 生产可用 rerank 请使用“明确支持 OpenAI-compatible `/rerank`”的模型服务。

### 7.7 先关闭 rerank，优先保障主链路（临时方案）

适用场景：

- 8000 embedding 与 8002 chat 已可用；
- 8001 rerank 暂时不兼容 `/v3/rerank`。

处理方式（当前推荐）：

1. 保留三端口服务在线。
2. 将租户默认 `rerank_id` 置空，只使用 chat + embedding。

执行命令：

```bash
cd ~/ragflow/docker
sudo docker exec docker-mysql-1 mysql -uroot -pinfini_rag_flow -Drag_flow -e "UPDATE tenant SET rerank_id='', tenant_rerank_id=NULL WHERE id='c2fd45ae843d11f1a059698ef4239658'; UPDATE dialog SET rerank_id='', tenant_rerank_id=NULL WHERE tenant_id='c2fd45ae843d11f1a059698ef4239658'; SELECT id,llm_id,embd_id,rerank_id FROM tenant WHERE id='c2fd45ae843d11f1a059698ef4239658';"
```

说明：

- 该操作不会影响 8000/8002 主链路。
- 待后续切换到可用的 OpenAI-compatible rerank 服务后，再恢复 `rerank_id` 即可。

适用更新：

- 若你已经完成第 9 节替换并验证通过，可不再执行本节临时关闭方案。

补充：若已部署“检索链路 rerank 失败自动降级”补丁，则在 rerank 报错时系统会自动回退到基础排序（向量+词项融合），避免 API 500；但这不代表 rerank 实际生效。

### 7.8 `Embedding request failed ... graph definition not found`（已实测修复）

现象：

- 在 8080 UI 重试解析后仍报：
  - `Fail to bind embedding model`
  - `Embedding request failed for OpenAI_APIEmbed`
  - `404 {'error': 'Mediapipe graph definition with requested name is not found'}`

根因（本机实测）：

1. `embd_id` 虽是 embedding 模型名，但挂到了 chat 实例（`base_url=8002`）。
2. 解析任务在入队时会把 `embd_id` 固化到任务载荷；旧任务即使后续改了租户/知识库配置，仍会继续使用旧 `embd_id`。

修复步骤：

1. 为 `OpenAI-API-Compatible` 创建独立 embedding 实例（示例名：`OVMS-Embedding`），`base_url` 指向 `http://host.docker.internal:8000/v3`。
2. 将 embedding 模型 `bge-large-zh-v1.5-int8-ov` 绑定到该实例。
3. 将租户默认 `embd_id` 与目标知识库 `embd_id` 更新为：
   - `bge-large-zh-v1.5-int8-ov@OVMS-Embedding@OpenAI-API-Compatible`
4. 重启 `docker-ragflow-cpu-1`，清理该文档旧失败任务并重置文档状态，然后在 UI 点击“重新解析”。

快速核验：

```bash
cd ~/ragflow/docker
sudo docker exec docker-mysql-1 mysql -uroot -pinfini_rag_flow -Drag_flow -e "SELECT id,llm_id,embd_id,rerank_id FROM tenant;"
sudo docker exec docker-mysql-1 mysql -uroot -pinfini_rag_flow -Drag_flow -e "SELECT id,tenant_id,name,embd_id FROM knowledgebase WHERE tenant_id='c2fd45ae843d11f1a059698ef4239658';"
sudo docker exec docker-mysql-1 mysql -uroot -pinfini_rag_flow -Drag_flow -e "SELECT tm.model_name,tm.model_type,i.instance_name,i.extra FROM tenant_model tm JOIN tenant_model_provider p ON tm.provider_id=p.id JOIN tenant_model_instance i ON tm.instance_id=i.id WHERE p.provider_name='OpenAI-API-Compatible' ORDER BY tm.model_type,tm.model_name;"
sudo docker exec docker-ragflow-cpu-1 sh -lc "python - <<'PY'
from api.db.services.document_service import DocumentService
print(DocumentService.get_embd_id('7753b69284b011f19e56ed5a2246c487'))
PY"
```

期望：

- `DocumentService.get_embd_id(...)` 输出 `bge-large-zh-v1.5-int8-ov@OVMS-Embedding@OpenAI-API-Compatible`。
- 重试后的新任务不再出现 `graph definition not found`，embedding 可正常绑定与解析。
