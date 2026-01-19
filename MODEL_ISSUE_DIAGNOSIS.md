# Model Issue Diagnosis

## Problem Identified ✅

The model **works correctly** on synthetic data (training format):
- ✅ Probabilities: 0-55% on synthetic data
- ✅ Can detect debris with thresholds 0.3-0.5

But **fails on real GeoTIFF images**:
- ❌ Probabilities: 0-2% on real images
- ❌ No detections even with very low thresholds

## Root Cause

**Domain Mismatch**: The model was trained on **synthetic data** with:
- Range: -0.06 to 0.36
- Mean: ~0.12
- Std: ~0.08
- Distribution: Random patterns with Gaussian noise

Real satellite images have:
- Different value ranges
- Different distributions
- Different statistics

## Fixes Applied

1. **Better Normalization**: Match training data statistics (mean=0.12, std=0.08)
2. **Adaptive Thresholding**: Use percentile-based thresholds for low-confidence cases
3. **Lower Min Area**: Reduced to 5 pixels to catch small detections

## Testing

Try uploading an image again. The normalization should help, but the fundamental issue is:
- **Model needs retraining on real data** OR
- **Better preprocessing pipeline** to match training distribution

## Next Steps

If still getting 0% confidence:
1. The model may need fine-tuning on real satellite imagery
2. Or we need to create a better preprocessing pipeline
3. Or use the advanced model (currently training) which may generalize better

---

**Status**: Backend updated with better normalization and adaptive thresholds. Test again!
