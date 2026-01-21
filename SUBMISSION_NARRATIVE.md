# OceanGuard AI: Protecting Our Oceans Through Satellite-Powered Deep Learning

## The Problem We're Solving

Every year, over 14 million tons of plastic enter our oceans, creating massive floating debris fields that devastate marine ecosystems, damage fisheries, and wash onto coastlines worldwide. Traditional monitoring methods—ships, aircraft, and buoys—are prohibitively expensive and can only cover a fraction of the world's waters. Research shows that current detection methods miss approximately 99% of marine debris, leaving cleanup operations blind to the scale and location of the problem.

Our solution, OceanGuard AI, directly addresses this gap by enabling scalable, low-cost detection of marine debris using freely available Sentinel-2 satellite imagery. The system benefits environmental agencies, cleanup organizations, coastal communities, and researchers who need actionable data to prioritize cleanup efforts and understand debris accumulation patterns before material reaches shorelines.

## Our AI Approach

OceanGuard AI employs a deep learning semantic segmentation model based on an Improved U-Net architecture, specifically designed for processing 11-band multispectral satellite imagery. We trained our model on the MARIDA (Marine Debris Archive) dataset, a comprehensive benchmark containing 63 Sentinel-2 scenes with manually labeled debris patches. The model processes 256x256 pixel patches, learning to distinguish debris from water, algae, ships, and other marine features across visible, near-infrared, and shortwave infrared spectral bands.

The AI component works through a multi-stage pipeline: first, Sentinel-2 GeoTIFF images are preprocessed and normalized using MARIDA-specific statistics to match training data distribution. The Improved U-Net model then performs pixel-level classification, generating probability maps indicating debris likelihood at each location. We use Focal Loss with alpha=0.75 and gamma=2.0 to handle the extreme class imbalance—debris pixels represent only about 11% of the data, while background (water, sky) dominates at 89%. Post-processing extracts connected components, filters by minimum area, and generates ranked hotspot lists with GPS coordinates, confidence scores, and area estimates.

## Tools and Technologies

Our development stack includes PyTorch for deep learning model training and inference, FastAPI for the backend REST API, and Next.js with React for the interactive web interface. We processed Sentinel-2 satellite imagery using Rasterio for geospatial data handling and NumPy/SciPy for image processing operations. The European Space Agency's Sentinel-2 data provides free, global coverage with 10-meter resolution and 5-day revisit cycles, making it ideal for continuous monitoring. During development, we used Cursor as a coding assistant to help with boilerplate code and debugging, though all architectural decisions, model design, and training strategies were developed through our own research and experimentation.

## Challenges and Solutions

We faced several significant challenges during development. The first was class imbalance—debris pixels are rare compared to background, causing the model to default to predicting "no debris" for everything. We addressed this through Focal Loss, which downweights easy examples and focuses learning on hard cases, combined with class weighting that gives debris 1.86x more importance during training.

A second major challenge was domain mismatch. Initially, we trained on synthetic data, but when testing on real satellite imagery, the model returned zero detections. We discovered the normalization strategy didn't match real data distributions. After analyzing the MARIDA dataset statistics, we implemented percentile-based normalization (using 2nd and 98th percentiles) combined with MARIDA-specific mean and standard deviation normalization. This required multiple iterations of testing and refinement.

Third, we struggled with over-segmentation—the model detected thousands of tiny false positive hotspots. We solved this through adaptive thresholding that adjusts based on image statistics, combined with minimum area filtering (10 pixels minimum) and morphological operations to merge nearby detections. We also implemented a sensitivity parameter that allows users to tune the detection threshold based on their needs.

## Innovation and Creativity

What makes our solution innovative is the combination of freely available satellite data with accessible deep learning technology to create a scalable monitoring system. Unlike proprietary solutions requiring expensive sensors or aircraft, OceanGuard AI can process any Sentinel-2 scene globally at no data cost. The web-based interface makes the technology accessible to non-technical users—environmental agencies can upload imagery and receive actionable hotspot reports without needing GIS expertise or machine learning knowledge.

We also innovated in our approach to the class imbalance problem. Rather than simply oversampling debris patches, we combined multiple techniques: Focal Loss for hard example mining, class weighting for balanced gradients, and adaptive thresholding that adjusts to each image's probability distribution. This multi-pronged approach allows the model to detect rare debris events while maintaining reasonable precision.

## Testing and Verification

We rigorously tested our model's accuracy through comprehensive evaluation on 200 test samples containing 13.1 million pixels. Our evaluation metrics show the model achieves 67.24% recall (successfully detecting two-thirds of debris pixels) and 18.87% precision (about one in five predicted debris pixels is actually debris). The F1 score of 29.47% reflects the trade-off between these metrics, while Intersection over Union (IoU) of 17.28% indicates segmentation quality.

We verified accuracy through multiple methods: quantitative evaluation on held-out test data, visual inspection of prediction heatmaps overlaid on original imagery, and testing on real-world satellite scenes from different geographic regions. The model processes images in under one second (31.49ms per batch), making it suitable for real-time applications. While precision could be improved, the high recall rate ensures we don't miss debris hotspots, which is critical for environmental monitoring where false negatives are more costly than false positives.

## Learning and Growth

Working on this project dramatically deepened our understanding of semantic segmentation, class imbalance in machine learning, and the practical challenges of deploying AI models in production. We learned that model architecture is only part of the solution—data preprocessing, loss function design, and post-processing are equally critical. The class imbalance problem taught us that standard cross-entropy loss fails when classes are highly imbalanced, leading us to research and implement Focal Loss, which fundamentally changed our model's performance.

We also gained appreciation for the importance of domain adaptation—a model trained on one data distribution may completely fail on another, even if the task is the same. This required us to carefully analyze data statistics, implement robust normalization, and test extensively on real-world data. The project also taught us about responsible AI use: our system is designed to assist human decision-makers, not replace them, and we provide confidence scores and adjustable sensitivity to ensure users understand the model's limitations.

## Project Insights

Beyond technical skills, this project reinforced the importance of iterative development and user-centered design. We started with a simple spectral anomaly detection approach, then evolved to deep learning as we understood the problem better. The web interface went through multiple iterations based on testing with sample images, ensuring the visualization clearly communicates detection results to users.

We also learned that environmental AI applications require careful consideration of false positives and false negatives. For marine debris detection, missing a hotspot (false negative) could mean debris reaches a coastline, while false positives waste cleanup resources. Our adaptive thresholding and sensitivity controls allow users to balance these trade-offs based on their specific needs and available resources.

## Looking Forward

OceanGuard AI represents a proof-of-concept that demonstrates the feasibility of using AI and free satellite data for large-scale environmental monitoring. While our current model shows promise with 67% recall, we recognize that precision improvements are needed for production deployment. Future work could include training on larger datasets, experimenting with transformer-based architectures like SegFormer, and implementing active learning to continuously improve the model with user feedback.

The project has been an incredible learning journey, combining computer vision, geospatial analysis, web development, and environmental science. We're excited to see how this technology could be deployed by environmental agencies worldwide to protect our oceans at a scale that was previously impossible.

---

**Word Count: 1,247 words**

**Primary Sources:**
- Kikaki, K., et al. (2022). MARIDA: A benchmark for Marine Debris detection from Sentinel-2 remote sensing data. PLoS ONE, 17(1), e0262247.
- European Space Agency. Sentinel-2 Mission. https://sentinel.esa.int/web/sentinel/missions/sentinel-2
- PyTorch Documentation: https://pytorch.org/docs/
- FastAPI Documentation: https://fastapi.tiangolo.com/

**Tools Used:**
- PyTorch (deep learning framework)
- FastAPI (backend API)
- Next.js/React (frontend)
- Rasterio (geospatial data processing)
- NumPy/SciPy (scientific computing)
- Cursor (coding assistant for development efficiency)
