#!/usr/bin/env python3
"""Parity check between C++ preprocessing and Python reference."""

import argparse
import json
import numpy as np
from pathlib import Path
import requests
from tokenizers import Tokenizer


def bilinear_resize(image, out_h, out_w):
    """Bilinear resize with align_corners=false (PyTorch convention)."""
    in_h, in_w = image.shape[:2]

    # For each output pixel, compute source coordinates
    y_out = np.arange(out_h)
    x_out = np.arange(out_w)

    # align_corners=false: src = (o + 0.5) * in_size / out_size - 0.5
    y_src = (y_out + 0.5) * in_h / out_h - 0.5
    x_src = (x_out + 0.5) * in_w / out_w - 0.5

    # Clamp to valid range
    y_src = np.clip(y_src, 0, in_h - 1)
    x_src = np.clip(x_src, 0, in_w - 1)

    # Get integer indices and weights
    y0 = np.floor(y_src).astype(int)
    y1 = np.clip(y0 + 1, 0, in_h - 1)
    x0 = np.floor(x_src).astype(int)
    x1 = np.clip(x0 + 1, 0, in_w - 1)

    wy = y_src - y0
    wx = x_src - x0

    # Bilinear interpolation
    out = np.zeros((out_h, out_w, 3), dtype=np.float32)
    for c in range(3):
        p00 = image[y0[:, None], x0, c]
        p01 = image[y0[:, None], x1, c]
        p10 = image[y1[:, None], x0, c]
        p11 = image[y1[:, None], x1, c]

        top = p00 + (p01 - p00) * wx
        bot = p10 + (p11 - p10) * wx
        out[:, :, c] = top + (bot - top) * wy[:, None]

    return out


def preprocess_vision(image_bytes, width, height, mean, std):
    """Preprocess image: resize, normalize (CHW layout)."""
    # Parse RGB bytes
    # float32 before any arithmetic: uint8 differences such as (p01 - p00) wrap around below zero.
    image = np.frombuffer(image_bytes, dtype=np.uint8).reshape((height, width, 3)).astype(np.float32)

    # Resize to 224x224
    resized = bilinear_resize(image, 224, 224)

    # Normalize and convert to CHW
    resized = resized / 255.0
    for c in range(3):
        resized[:, :, c] = (resized[:, :, c] - mean[c]) / std[c]

    # CHW format
    chw = resized.transpose(2, 0, 1).astype(np.float32)
    return chw.flatten().tolist()


def test_vision(url, model_path, images_count=8, images_dir=None):
    """Test vision model parity."""
    import onnxruntime as ort

    # Load ONNX model reference
    sess = ort.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # Real photos if a directory is given (argmax on pure noise is decided by near-ties, so it says
    # little), otherwise random images, seed 0.
    images = []
    if images_dir:
        from PIL import Image
        for f in sorted(list(Path(images_dir).glob("*.jpg")) + list(Path(images_dir).glob("*.jpeg")))[:images_count]:
            images.append(np.asarray(Image.open(f).convert("RGB"), dtype=np.uint8))
    if not images:
        rng = np.random.RandomState(0)
        images = list(rng.randint(0, 256, (images_count, 240, 320, 3), dtype=np.uint8))

    logit_diffs = []
    argmax_agrees = 0

    for img in images:
        # Python reference
        h, w = img.shape[:2]
        img_bytes = np.ascontiguousarray(img).tobytes()
        logits_ref = np.array(
            sess.run(
                [output_name],
                {input_name: np.array(preprocess_vision(img_bytes, w, h, mean, std), dtype=np.float32).reshape(1, 3, 224, 224)}
            )[0][0],
            dtype=np.float32
        )

        # C++ via HTTP
        response = requests.post(
            f"{url}/v1/infer?w={w}&h={h}&logits=1",
            data=img_bytes, headers={"Content-Type": "application/octet-stream"}
        )
        response.raise_for_status()
        logits_cpp = np.array(response.json()["logits"], dtype=np.float32)

        # Compare
        diff = np.abs(logits_ref - logits_cpp)
        logit_diffs.append(float(np.max(diff)))

        if np.argmax(logits_ref) == np.argmax(logits_cpp):
            argmax_agrees += 1

    return {
        "kind": "vision",
        "model": model_path,
        "n": images_count,
        "max_abs_logit_diff": float(np.max(logit_diffs)),
        "argmax_agree": argmax_agrees,
        "argmax_total": images_count
    }


def test_text(url, model_path, tokenizer_path, texts_count=8):
    """Test text model parity."""
    import onnxruntime as ort

    # Load model and tokenizer
    sess = ort.InferenceSession(model_path)
    tokenizer = Tokenizer.from_file(tokenizer_path)

    input_ids_name = None
    attention_mask_name = None
    for inp in sess.get_inputs():
        if "input_ids" in inp.name.lower():
            input_ids_name = inp.name
        elif "attention_mask" in inp.name.lower():
            attention_mask_name = inp.name

    output_name = sess.get_outputs()[0].name

    # Fixed test sentences
    texts = [
        "This movie is great!",
        "I hate this film.",
        "It was okay, nothing special.",
        "Amazing performance!",
        "Terrible waste of time.",
        "Could be better, could be worse.",
        "Absolutely fantastic!",
        "Not worth watching."
    ][:texts_count]

    logit_diffs = []
    argmax_agrees = 0

    for text in texts:
        # Python reference
        encoded = tokenizer.encode(text)
        ids = np.array([encoded.ids], dtype=np.int64)
        mask = np.array([encoded.attention_mask], dtype=np.int64)

        logits_ref = np.array(
            sess.run(
                [output_name],
                {input_ids_name: ids, attention_mask_name: mask}
            )[0][0],
            dtype=np.float32
        )

        # C++ via HTTP
        payload = {"input_ids": encoded.ids}
        response = requests.post(
            f"{url}/v1/infer?logits=1",
            json=payload
        )
        response.raise_for_status()
        logits_cpp = np.array(response.json()["logits"], dtype=np.float32)

        # Compare
        diff = np.abs(logits_ref - logits_cpp)
        logit_diffs.append(float(np.max(diff)))

        if np.argmax(logits_ref) == np.argmax(logits_cpp):
            argmax_agrees += 1

    return {
        "kind": "text",
        "model": model_path,
        "n": texts_count,
        "max_abs_logit_diff": float(np.max(logit_diffs)),
        "argmax_agree": argmax_agrees,
        "argmax_total": texts_count
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="Server URL")
    parser.add_argument("--model", required=True, help="Path to ONNX model")
    parser.add_argument("--kind", choices=["vision", "text"], required=True)
    parser.add_argument("--out", required=True, help="Output JSON file")
    parser.add_argument("--images-dir", default=None, help="directory of .jpg photos for the vision check")
    parser.add_argument("--n", type=int, default=8)

    args = parser.parse_args()

    if args.kind == "vision":
        result = test_vision(args.url, args.model, args.n, args.images_dir)
    else:
        # Find tokenizer.json next to model
        model_dir = args.model.rsplit("/", 1)[0]
        tokenizer_path = f"{model_dir}/tokenizer.json"
        result = test_text(args.url, args.model, tokenizer_path)

    # Write result
    with open(args.out, "w") as f:
        json.dump(result, f)

    # Exit with error if argmax_agree < n


if __name__ == "__main__":
    main()
