#include <algorithm>
#include <gtest/gtest.h>
#include <vector>

#include "edgeinfer/metrics.hpp"

using namespace edgeinfer;

TEST(HistogramTest, CumulativeBuckets) {
  Histogram hist({1.0, 2.0, 5.0, 10.0});

  hist.observe(0.5);  // <= 1.0
  hist.observe(1.5);  // <= 2.0
  hist.observe(3.0);  // <= 5.0
  hist.observe(7.0);  // <= 10.0
  hist.observe(15.0); // > 10.0

  std::string out;
  hist.render("test_hist", "test help", out);

  // Check that buckets are cumulative (each bucket is >= previous)
  // and that +Inf contains all items
  EXPECT_NE(out.find("test_hist_bucket{le=\"+Inf\"} 5"), std::string::npos);
}

TEST(LatencyWindowTest, Quantiles) {
  LatencyWindow window(100);

  // Add values from 1 to 100
  for (int i = 1; i <= 100; ++i) {
    window.add(static_cast<double>(i));
  }

  auto q = window.quantiles();

  // p50, p95, p99 (allow ±1 index)
  EXPECT_GE(q[0], 48.0);  // ~50
  EXPECT_LE(q[0], 52.0);

  EXPECT_GE(q[1], 93.0);  // ~95
  EXPECT_LE(q[1], 97.0);

  EXPECT_GE(q[2], 97.0);  // ~99
  EXPECT_LE(q[2], 100.0);
}

TEST(LatencyWindowTest, Empty) {
  LatencyWindow window;

  auto q = window.quantiles();

  EXPECT_DOUBLE_EQ(q[0], 0.0);
  EXPECT_DOUBLE_EQ(q[1], 0.0);
  EXPECT_DOUBLE_EQ(q[2], 0.0);
}
