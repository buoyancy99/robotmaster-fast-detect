# Installation
Dowload, Install OpenVINO 2020.2 and finish all setup & GPU setup in document
I don't recommend putting source /opt/intel/openvino/bin/setupvars.sh in .bashrc


# Video demo
[Model demo on official dataset ](https://www.bilibili.com/video/BV17a4y1e71p/?share_source=copy_web&vd_source=8e3571ebd266813b63df3ed61de3e822)

[Model demo in the wild](https://www.bilibili.com/video/BV1ae411p7So/#reply243057960)

# Model Optimization
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model CenterNet/onnx/rm_centernet_r18d4c6.onnx --data_type FP16 --batch 1 --mean_values [123.675,116.28,103.53] --scale_values [58.395,57.12,57.375]

python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model CenterNet/onnx/rm_centernet_r18d2c6.onnx --data_type FP16 --batch 1 --mean_values [123.675,116.28,103.53] --scale_values [58.395,57.12,57.375]

# OpenVINO Inference (Intel NCS recommended)
source /opt/intel/openvino/bin/setupvars.sh
export PYTHONPATH=$PYTHONPATH:.
python3 Inference/openvino_inference

# Paper
https://arxiv.org/pdf/1904.07850.pdf

