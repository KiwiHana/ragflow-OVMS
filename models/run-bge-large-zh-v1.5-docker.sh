cd ~

sudo docker run -d --user $(id -u):$(id -g) --rm -p 8000:8000 --device /dev/accel --group-add=$(stat -c "%g" /dev/dri/render*  | head -1) -v $(pwd)/models:/models:ro openvino/model_server:latest-gpu --rest_port 8000 --config_path /models/config.json
