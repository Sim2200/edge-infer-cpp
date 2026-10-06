#include <cmath>
#include <cstdlib>
#include <fstream>
#include <gtest/gtest.h>
#include <random>
#include <string>
#include <vector>

#include "edgeinfer/model.hpp"

using namespace edgeinfer;

TEST(ModelTest, LoadMobileNetV3) {
  std::string model_path = std::string(EDGEINFER_SOURCE_DIR) +
                           "/models/mobilenet_v3_large_int8/model.onnx";

  // Skip test if model doesn't exist
  if (std::ifstream(model_path).fail()) {
    GTEST_SKIP() << "Model not found at " << model_path;
  }

  ModelConfig cfg;
  cfg.path = model_path;

  Model model(cfg);

  EXPECT_EQ(model.kind(), ModelKind::Vision);
  EXPECT_EQ(model.input_size(), 224);
  EXPECT_EQ(model.output_dim(), 1000);
}

TEST(ModelTest, VisionInference) {
  std::string model_path = std::string(EDGEINFER_SOURCE_DIR) +
                           "/models/mobilenet_v3_large_int8/model.onnx";

  // Skip test if model doesn't exist
  if (std::ifstream(model_path).fail()) {
    GTEST_SKIP() << "Model not found at " << model_path;
  }

  ModelConfig cfg;
  cfg.path = model_path;
  Model model(cfg);

  // Create 2 random images (3 * 224 * 224 floats each)
  std::vector<float> images(2 * 3 * 224 * 224);
  std::mt19937 gen(0);
  std::uniform_real_distribution<> dis(-1.0, 1.0);
  for (auto& f : images) {
    f = static_cast<float>(dis(gen));
  }

  auto logits = model.run_vision(images.data(), 2);

  EXPECT_EQ(logits.size(), 2000);  // 2 * 1000

  // Check all finite
  for (auto f : logits) {
    EXPECT_TRUE(std::isfinite(f));
  }

  // Batch of 2 should equal two batch of 1 within tolerance
  auto logits_single1 = model.run_vision(images.data(), 1);
  auto logits_single2 = model.run_vision(images.data() + 3 * 224 * 224, 1);

  for (size_t i = 0; i < 1000; ++i) {
    EXPECT_NEAR(logits[i], logits_single1[i], 1e-3f);
    EXPECT_NEAR(logits[1000 + i], logits_single2[i], 1e-3f);
  }
}

TEST(ModelTest, LoadDistilBERT) {
  std::string model_path = std::string(EDGEINFER_SOURCE_DIR) +
                           "/models/distilbert_sst2_int8/model.onnx";

  // Skip test if model doesn't exist
  if (std::ifstream(model_path).fail()) {
    GTEST_SKIP() << "Model not found at " << model_path;
  }

  ModelConfig cfg;
  cfg.path = model_path;
  Model model(cfg);

  EXPECT_EQ(model.kind(), ModelKind::Text);
  EXPECT_EQ(model.output_dim(), 2);
}

TEST(ModelTest, TextInference) {
  std::string model_path = std::string(EDGEINFER_SOURCE_DIR) +
                           "/models/distilbert_sst2_int8/model.onnx";

  // Skip test if model doesn't exist
  if (std::ifstream(model_path).fail()) {
    GTEST_SKIP() << "Model not found at " << model_path;
  }

  ModelConfig cfg;
  cfg.path = model_path;
  Model model(cfg);

  // Create 2 sequences of length 16
  std::vector<std::int64_t> ids(2 * 16);
  std::vector<std::int64_t> mask(2 * 16);

  for (int i = 0; i < 2 * 16; ++i) {
    ids[i] = (i % 100) + 1;  // token id
    mask[i] = 1;  // attention mask
  }

  auto logits = model.run_text(ids.data(), mask.data(), 2, 16);

  EXPECT_EQ(logits.size(), 4);  // 2 * 2

  // Check all finite
  for (auto f : logits) {
    EXPECT_TRUE(std::isfinite(f));
  }
}

TEST(ModelTest, PeakRSS) {
  double rss = peak_rss_mib();
  EXPECT_GT(rss, 0.0);
}
