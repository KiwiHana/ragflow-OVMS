cd ~

sudo docker run --name ovms-qwen35b --user $(id -u):$(id -g) -d --device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1) --rm -p 8002:8002 -v $(pwd)/models:/models:rw openvino/model_server:latest-gpu --source_model Qwen3.6-35B-A3B-ov --model_repository_path models --task text_generation --rest_port 8002 --target_device GPU --cache_size 0
