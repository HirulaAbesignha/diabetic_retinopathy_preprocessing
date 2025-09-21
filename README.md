# Diabetic Retinopathy Detection - Data Preprocessing Pipeline

## Project Overview

This project implements a comprehensive data preprocessing pipeline for diabetic retinopathy detection using retinal fundus images. The pipeline is designed by a 6-member team, with each member contributing a specific preprocessing technique while collaborating on an integrated solution.

### Key Features
- Memory-optimized for Google Colab (12GB RAM + 15GB GPU)
- Handles 25,000+ retinal images efficiently
- Addresses severe class imbalance (75% No DR vs 2% Proliferative DR)
- Individual EDA visualizations for each preprocessing technique
- Production-ready integrated pipeline
- Comprehensive error handling and validation

### Medical Context
Diabetic retinopathy is a leading cause of blindness worldwide. Early detection through automated analysis of retinal images can prevent vision loss. This preprocessing pipeline prepares retinal fundus images for deep learning models by addressing data quality issues, class imbalance, and medical image enhancement requirements.

## Dataset Details

### Dataset Source
- **Name**: Diabetic Retinopathy Detection (Preprocessed)
- **Source**: HuggingFace Datasets (`youssefedweqd/Diabetic_Retinopathy_Detection_preprocessed2`)
- **Size**: ~25,300 retinal fundus images
- **Format**: RGB color images (various original sizes, resized to 224x224)
- **Labels**: 5-class classification for DR severity

### Class Distribution
```
Class 0 (No DR):          ~19,000 samples (75.0%)
Class 1 (Mild):           ~2,400 samples (9.5%)
Class 2 (Moderate):       ~2,400 samples (9.5%)
Class 3 (Severe):         ~873 samples (3.5%)
Class 4 (Proliferative):  ~708 samples (2.8%)
```

### Data Characteristics
- **Imbalance Ratio**: 26.8:1 (No DR vs Proliferative DR)
- **Image Quality**: Variable lighting and contrast conditions
- **Medical Features**: Blood vessels, optic disc, hemorrhages, exudates
- **Challenges**: Severe class imbalance, variable image quality, medical domain complexity

## Group Member Roles

### IT24100259: Data Loading & Missing Data Handling
**Responsibilities:**
- Load dataset using streaming optimization for memory efficiency
- Detect and analyze missing or corrupted image data
- Identify samples with missing labels
- Perform data integrity validation
- Create EDA visualization showing data completeness

**Key Contributions:**
- `load_dataset()` - Efficient data loading with fallback to simulated data
- `analyze_missing_data()` - Comprehensive missing data detection
- `visualize_member1_eda()` - Pie chart and bar chart analysis

**Technical Innovation:** Streaming data loading prevents memory overflow while maintaining dataset integrity checks.

### IT24100349: Outlier Detection & Removal
**Responsibilities:**
- Implement IQR-based statistical outlier detection
- Analyze image statistics (brightness, contrast, dimensions)
- Perform memory-efficient batch processing
- Filter extreme outliers that could harm model training
- Create EDA visualization with box plots and statistical boundaries

**Key Contributions:**
- `detect_outliers()` - Batch-wise outlier detection using IQR method
- `visualize_member2_eda()` - Box plots showing outlier boundaries and statistics

**Technical Innovation:** Memory-efficient batch processing with automatic cleanup prevents system overload.

### IT24610824: Class Balancing & Data Augmentation
**Responsibilities:**
- Analyze severe class imbalance (26.8:1 ratio)
- Design strategic balancing approach combining over/under-sampling
- Implement medical-aware data augmentation
- Create class-specific augmentation strategies
- Create EDA visualization for imbalance analysis and balancing strategy

**Key Contributions:**
- `analyze_class_distribution()` - Class imbalance quantification
- `create_balancing_strategy()` - Strategic balancing approach
- `apply_augmentation()` - Medical-aware image transformations

**Technical Innovation:** Class-aware augmentation with higher rates for severe DR cases while preserving medical image characteristics.

### IT24100264: Image Normalization & Enhancement
**Responsibilities:**
- Analyze pixel value distributions across the dataset
- Implement CLAHE enhancement for medical image quality
- Apply ImageNet normalization for transfer learning compatibility
- Perform comprehensive image preprocessing
- Create EDA visualization for pixel analysis and normalization effects

**Key Contributions:**
- `analyze_pixel_distributions()` - Dataset-wide pixel statistics
- `normalize_image()` - CLAHE enhancement + ImageNet normalization
- `visualize_member4_eda()` - Multi-panel pixel analysis

**Technical Innovation:** Two-stage normalization combining medical imaging standards (CLAHE) with deep learning requirements (ImageNet).

### IT24100348: Feature Engineering
**Responsibilities:**
- Extract domain-specific retinal features for medical interpretation
- Implement medical image analysis techniques
- Analyze feature correlations with DR severity
- Provide model interpretability through engineered features
- Create EDA visualization for feature analysis and correlations

**Key Contributions:**
- `extract_retinal_features()` - 5 medical features: brightness, contrast, red dominance, vessel density, texture
- `analyze_feature_correlations()` - Statistical relationships with DR severity
- `visualize_member5_eda()` - Comprehensive feature correlation analysis

**Technical Innovation:** Clinically-relevant feature extraction including vessel density analysis and hemorrhage detection indicators.

### IT24100255: Data Splitting & Scaling
**Responsibilities:**
- Create stratified train/validation/test splits preserving class proportions
- Compute global scaling parameters for consistent normalization
- Ensure no data leakage between splits
- Validate split quality and class balance preservation
- Create EDA visualization for split analysis and scaling parameters

**Key Contributions:**
- `prepare_data_splits()` - Stratified splitting (70%/15%/15%)
- `compute_scaling_parameters()` - Global dataset statistics
- `visualize_member6_eda()` - Split quality and scaling analysis

**Technical Innovation:** Two-stage stratified splitting ensures representative class distribution across all data splits.

## Technical Specifications

### System Requirements
- **RAM**: 12GB minimum (Google Colab compatible)
- **GPU**: 15GB VRAM recommended (optional for preprocessing)
- **Storage**: 2GB for dataset cache
- **Python**: 3.7+ with scientific computing libraries

### Memory Optimizations
- **Streaming Data Loading**: Prevents full dataset loading into memory
- **Batch Processing**: Processes samples in chunks of 20-50
- **Automatic Cleanup**: Garbage collection after each member's contribution
- **Sample Limiting**: Analyzes representative subsets (1000 samples)
- **Image Resizing**: Reduces to 224x224 for memory efficiency

### Dependencies
```python
# Core packages (auto-installed)
datasets>=2.0.0          # HuggingFace dataset loading
opencv-python>=4.5.0     # Image processing
psutil>=5.8.0           # Memory monitoring
GPUtil                  # GPU monitoring

# Pre-installed in Colab
tensorflow>=2.8.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.0.0
```

## How to Run the Code

### Google Colab (Recommended)

1. **Open Google Colab**
   ```
   https://colab.research.google.com/
   ```

2. **Create new notebook and paste the complete code**
   - Copy the entire preprocessing pipeline code
   - Paste into a single Colab cell

3. **Run the cell**
   ```python
   # The code will automatically:
   # 1. Install required packages
   # 2. Load the dataset (real or simulated fallback)
   # 3. Execute all 6 member contributions
   # 4. Generate EDA visualizations
   # 5. Validate integrated pipeline
   ```

4. **Expected execution time**: 8-12 minutes
5. **Expected memory usage**: 8-10GB RAM, 5-7GB GPU

### Local Environment

1. **Install dependencies**
   ```bash
   pip install datasets opencv-python psutil GPUtil tensorflow numpy pandas matplotlib seaborn scikit-learn
   ```

2. **Run the preprocessing pipeline**
   ```python
   # Save code as: diabetic_retinopathy_preprocessing.py
   python diabetic_retinopathy_preprocessing.py
   
   # Or run in Jupyter notebook
   results = run_complete_pipeline()
   ```

### Step-by-Step Execution (For Testing Individual Members)

```python
# Test individual member contributions
samples = load_dataset()

# IT24100259
missing_info = analyze_missing_data(samples)
visualize_member1_eda(missing_info)

# IT24100349
outliers_info = detect_outliers(samples)
visualize_member2_eda(outliers_info)

# IT24610824
class_counts, ratio = analyze_class_distribution(samples)
strategy = create_balancing_strategy(class_counts)
visualize_member3_eda(class_counts, strategy)

# IT24100264
pixel_stats = analyze_pixel_distributions(samples)
visualize_member4_eda(pixel_stats)

# IT24100348
features = extract_retinal_features(samples)
correlations = analyze_feature_correlations(features, samples)
visualize_member5_eda(features, correlations, samples)

# IT24100255
train, val, test = prepare_data_splits(samples)
scaling_params = compute_scaling_parameters(train)
visualize_member6_eda(train, val, test, scaling_params)

# Integrated pipeline
preprocessing_fn = create_integrated_pipeline(scaling_params)
validation_results = validate_pipeline(train, val, test, preprocessing_fn)
visualize_integrated_pipeline(validation_results)
```

## Expected Output

### Individual Member Visualizations
1. **IT24100259**: Data completeness pie chart + missing data bar chart
2. **IT24100349**: Box plots for outlier detection (4 metrics)
3. **IT24610824**: Class distribution analysis (4-panel visualization)
4. **IT24100264**: Pixel distribution analysis (4-panel visualization)
5. **IT24100348**: Feature correlation analysis (4-panel visualization)
6. **IT24100255**: Data splits and scaling analysis (4-panel visualization)

### Integrated Pipeline Visualization
- Pipeline success rates across train/val/test splits
- Processing time analysis
- Member contribution scores
- Data flow through pipeline stages

### Final Results
```python
# Access results after execution
results = run_complete_pipeline()

if results['success']:
    # Get preprocessing function
    preprocess_fn = results['preprocessing_function']
    
    # Get data splits
    train_data = results['train_samples']
    val_data = results['val_samples']
    test_data = results['test_samples']
    
    # Get extracted features
    features = results['features']
    correlations = results['correlations']
    
    # Get scaling parameters
    scaling_params = results['scaling_params']
```

### Performance Metrics
```
Expected Results:
✓ Dataset loaded: 1000 samples (or full 25K if available)
✓ Data completeness: 95-99%
✓ Outliers detected: 5-15% across metrics  
✓ Class imbalance: 26.8:1 → 2.0:1 (after balancing)
✓ Features extracted: 5 medical features
✓ Pipeline success rate: 95-98%
✓ Processing speed: 35-50ms per image
✓ Memory usage: 8-10GB RAM peak
```

## Troubleshooting

### Common Issues

1. **Memory Error**
   ```python
   # Reduce sample sizes
   SAMPLE_SIZE = 500  # Instead of 1000
   
   # Or process smaller batches
   batch_size = 10    # Instead of 20-50
   ```

2. **Dataset Loading Failed**
   ```python
   # Code automatically falls back to simulated data
   # Check console output for "Creating simulated dataset..."
   ```

3. **Visualization Issues**
   ```python
   # If plots don't display in Colab:
   import matplotlib.pyplot as plt
   plt.show()  # After each visualization function
   ```

4. **GPU Out of Memory**
   ```python
   # Disable GPU if not needed for preprocessing
   import os
   os.environ['CUDA_VISIBLE_DEVICES'] = ''
   ```

### Performance Tips
- Run during off-peak hours for better Colab performance
- Clear output regularly to save memory
- Monitor memory usage with built-in functions
- Use CPU-only mode if GPU memory is limited

## File Structure
```
diabetic_retinopathy_preprocessing/
├── README.md
├── data/
|   └── raw/
|   └── external/
├── cache/
|   └── IT24100255_data_splitting&scaling.ipynb
|   └── IT24100259_data_loading&missing_data.ipynb
|   └── IT24100264_image_normalization&preprocessing.ipynb
|   └── IT24100348_feature_engineering.ipynb
|   └── IT24100349_outlier_detection&removal.ipynb
|   └── IT24610824_class_balancing&augmentation.ipynb
├── pipeline.ipynb
├── results/
    └── eda_visualization/
    └── logs/
    └── outputs/

```

## Academic Context

This preprocessing pipeline demonstrates:
- **Individual Expertise**: Each member contributes specialized preprocessing knowledge
- **Team Collaboration**: Integrated pipeline shows effective teamwork
- **Medical AI Applications**: Domain-specific preprocessing for healthcare
- **Production Readiness**: Memory optimization and error handling
- **Comprehensive Analysis**: EDA visualizations support preprocessing decisions

## Members [2025 Y2 S1 MLB B3G2 07]

This preprocessing pipeline demonstrates:
- **IT24100348**: Zinthu N.
- **IT24100255**: Fernando C.M.
- **IT24100264**: Bandara N.W.C.D.
- **IT24100259**: Yapa R.P.S.
- **IT24610824**: Abesingha J.M.H.P.
- **IT24100349**: Pavitha M.

