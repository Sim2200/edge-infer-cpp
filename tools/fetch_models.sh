#!/usr/bin/env bash
# Puts the six ONNX models into models/<name>/model.onnx. Run from the repo root.
#
# Vision models: ResNet-50 v2 and MobileNetV3-Large, fp32 exports from torchvision and their int8
# static-quantized versions, produced by my sla-inference-gateway project (`make models` there, which
# exports with torch.onnx and quantizes with ONNX Runtime's static quantizer on ImageNet calibration
# images). Point GATEWAY at a checkout that has run it.
#
# DistilBERT: the public SST-2 fine-tune exported to ONNX on the Hugging Face hub, plus an int8
# version made here with ONNX Runtime dynamic quantization (weights int8, activations at run time).
set -euo pipefail
GATEWAY=${GATEWAY:-../sla-inference-gateway}
for m in resnet50_v2_fp32 resnet50_v2_int8 mobilenet_v3_large_fp32 mobilenet_v3_large_int8; do
  mkdir -p "models/$m"
  cp "$GATEWAY/models/artifacts/$m/model.onnx" "models/$m/model.onnx"
done

base=https://huggingface.co/optimum/distilbert-base-uncased-finetuned-sst-2-english/resolve/main
mkdir -p models/distilbert_sst2_fp32 models/distilbert_sst2_int8
curl -sSfL -o models/distilbert_sst2_fp32/model.onnx "$base/model.onnx"
curl -sSfL -o models/distilbert_sst2_fp32/tokenizer.json "$base/tokenizer.json"
cp models/distilbert_sst2_fp32/tokenizer.json models/distilbert_sst2_int8/tokenizer.json
docker run --rm -v "$PWD:/src" edge-infer-dev python3 -c "
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic('models/distilbert_sst2_fp32/model.onnx', 'models/distilbert_sst2_int8/model.onnx', weight_type=QuantType.QInt8)"
ls -la models/*/model.onnx
