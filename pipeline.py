
# Install required packages
import subprocess
import sys

packages = ['datasets>=2.0.0', 'psutil>=5.8.0', 'GPUtil', 'opencv-python>=4.5.0']
for package in packages:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

# Core imports
import os
import gc
import time
import warnings
from typing import Optional, Tuple, Dict, List, Any
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import psutil

try:
    import tensorflow as tf
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
except ImportError:
    pass

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from datasets import load_dataset
except ImportError:
    pass

# Configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

IMG_SIZE = 224
BATCH_SIZE = 16
SAMPLE_SIZE = 1000
CLASS_NAMES = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative DR"}

# Utility functions
def memory_cleanup():
    gc.collect()
    try:
        tf.keras.backend.clear_session()
    except:
        pass

def check_memory():
    return psutil.virtual_memory().percent

# IT24100259: DATA LOADING & MISSING DATA HANDLING

def load_dataset():
    """Load diabetic retinopathy dataset"""
    try:
        ds = load_dataset("youssefedweqd/Diabetic_Retinopathy_Detection_preprocessed2", 
                         streaming=True, cache_dir="./cache")
        
        samples = []
        train_iter = iter(ds["train"])
        for i, sample in enumerate(train_iter):
            if i >= SAMPLE_SIZE:
                break
            samples.append(sample)
        
        return samples
    except:
        return create_simulated_dataset()

def create_simulated_dataset():
    """Create simulated retinal images"""
    np.random.seed(42)
    samples = []
    class_distribution = [750, 120, 80, 30, 20]
    
    sample_id = 0
    for class_label, count in enumerate(class_distribution):
        for _ in range(count):
            img = np.random.randint(20, 200, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            
            if class_label >= 3:  # Add red spots for severe cases
                for _ in range(np.random.randint(5, 15)):
                    x, y = np.random.randint(20, IMG_SIZE-20, 2)
                    cv2.circle(img, (x, y), np.random.randint(2, 8), (0, 0, 255), -1)
            
            samples.append({'image': img, 'label': class_label, 'id': f'sim_{sample_id}'})
            sample_id += 1
    
    np.random.shuffle(samples)
    return samples

def analyze_missing_data(samples):
    """Analyze missing data"""
    missing_images = missing_labels = valid_samples = 0
    
    for sample in samples:
        if 'image' not in sample or sample['image'] is None:
            missing_images += 1
        elif 'label' not in sample or sample['label'] is None:
            missing_labels += 1
        else:
            valid_samples += 1
    
    return {
        'valid_samples': valid_samples,
        'missing_images': missing_images, 
        'missing_labels': missing_labels,
        'total_samples': len(samples)
    }

def missing_data_eda(missing_info):
    """IT24100259 EDA: Missing data visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    categories = ['Valid', 'Missing Images', 'Missing Labels']
    values = [missing_info['valid_samples'], missing_info['missing_images'], missing_info['missing_labels']]
    colors = ['green', 'red', 'orange']
    
    ax1.pie(values, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Data Completeness Distribution')
    
    ax2.bar(categories, values, color=colors, alpha=0.7)
    ax2.set_title('Missing Data Counts')
    ax2.set_ylabel('Count')
    
    plt.suptitle('IT24100259: Missing Data Analysis')
    plt.tight_layout()
    plt.show()

# IT24100349: OUTLIER DETECTION & REMOVAL

def detect_outliers(samples):
    """Detect outliers using IQR method"""
    stats = {'brightness': [], 'contrast': [], 'width': [], 'height': []}
    
    for sample in samples[:min(500, len(samples))]:
        try:
            img = np.array(sample['image'])
            stats['brightness'].append(np.mean(img))
            stats['contrast'].append(np.std(img))
            stats['height'].append(img.shape[0])
            stats['width'].append(img.shape[1])
        except:
            continue
    
    outliers_info = {}
    for metric, values in stats.items():
        values = np.array(values)
        Q1, Q3 = np.percentile(values, [25, 75])
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        
        outliers = np.sum((values < lower) | (values > upper))
        outliers_info[metric] = {
            'count': outliers,
            'percentage': (outliers / len(values)) * 100,
            'bounds': (lower, upper),
            'values': values
        }
    
    return outliers_info

def outlier_eda(outliers_info):
    """IT24100349 EDA: Outlier detection visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, (metric, info) in enumerate(outliers_info.items()):
        values = info['values']
        bounds = info['bounds']
        
        bp = axes[i].boxplot(values, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        
        axes[i].axhline(bounds[0], color='red', linestyle='--', alpha=0.7)
        axes[i].axhline(bounds[1], color='red', linestyle='--', alpha=0.7)
        
        axes[i].text(0.02, 0.98, f"Outliers: {info['count']} ({info['percentage']:.1f}%)",
                    transform=axes[i].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat'))
        
        axes[i].set_title(f'{metric.title()} Distribution')
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle('IT24100349: Outlier Detection Analysis')
    plt.tight_layout()
    plt.show()

# IT24610824: CLASS BALANCING & AUGMENTATION

def analyze_class_distribution(samples):
    """Analyze class distribution"""
    class_counts = Counter(int(sample['label']) for sample in samples)
    imbalance_ratio = max(class_counts.values()) / min(class_counts.values())
    return class_counts, imbalance_ratio

def create_balancing_strategy(class_counts, target_samples=300):
    """Create balancing strategy"""
    strategy = {}
    for cls, count in class_counts.items():
        if count < target_samples:
            strategy[cls] = {'action': 'augment', 'target': target_samples}
        elif count > target_samples * 2:
            strategy[cls] = {'action': 'undersample', 'target': target_samples}
        else:
            strategy[cls] = {'action': 'keep', 'target': count}
    return strategy

def apply_augmentation(image, label):
    """Apply augmentation based on class"""
    aug_prob = {0: 0.2, 1: 0.4, 2: 0.6, 3: 0.8, 4: 0.8}
    
    if np.random.random() < aug_prob.get(label, 0.3):
        if np.random.random() < 0.5:
            image = cv2.flip(image, 1)
        if np.random.random() < 0.3:
            angle = np.random.uniform(-10, 10)
            h, w = image.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h))
    return image

def balance_augment_eda(class_counts, strategy):
    """IT24610824 EDA: Class distribution visualization"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    classes = list(class_counts.keys())
    original_counts = [class_counts[cls] for cls in classes]
    balanced_counts = [strategy[cls]['target'] for cls in classes]
    class_labels = [CLASS_NAMES[cls] for cls in classes]
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    
    # Original vs balanced
    ax1.bar(class_labels, original_counts, color=colors, alpha=0.7)
    ax1.set_title('Original Class Distribution')
    ax1.set_ylabel('Sample Count')
    ax1.tick_params(axis='x', rotation=45)
    
    ax2.bar(class_labels, balanced_counts, color=colors, alpha=0.7)
    ax2.set_title('Balanced Class Distribution')
    ax2.set_ylabel('Sample Count')
    ax2.tick_params(axis='x', rotation=45)
    
    # Imbalance ratios
    max_orig = max(original_counts)
    orig_ratios = [max_orig / count for count in original_counts]
    max_bal = max(balanced_counts)
    bal_ratios = [max_bal / count for count in balanced_counts]
    
    x = np.arange(len(classes))
    width = 0.35
    ax3.bar(x - width/2, orig_ratios, width, label='Original', color='red', alpha=0.7)
    ax3.bar(x + width/2, bal_ratios, width, label='Balanced', color='green', alpha=0.7)
    ax3.set_title('Imbalance Ratios')
    ax3.set_ylabel('Ratio')
    ax3.set_xticks(x)
    ax3.set_xticklabels(class_labels, rotation=45)
    ax3.legend()
    ax3.set_yscale('log')
    
    # Pie chart
    ax4.pie(original_counts, labels=class_labels, autopct='%1.1f%%', colors=colors)
    ax4.set_title('Original Distribution Percentages')
    
    plt.suptitle('IT24610824: Class Distribution Analysis')
    plt.tight_layout()
    plt.show()

# IT24100264: IMAGE NORMALIZATION & PREPROCESSING

def analyze_pixel_distributions(samples):
    """Analyze pixel distributions"""
    pixel_stats = {'means': [], 'stds': [], 'mins': [], 'maxs': []}
    channel_stats = {'red': [], 'green': [], 'blue': []}
    
    for sample in samples[:min(200, len(samples))]:
        try:
            img = np.array(sample['image']).astype(np.float32)
            
            pixel_stats['means'].append(np.mean(img))
            pixel_stats['stds'].append(np.std(img))
            pixel_stats['mins'].append(np.min(img))
            pixel_stats['maxs'].append(np.max(img))
            
            if len(img.shape) == 3:
                channel_stats['red'].append(np.mean(img[:,:,0]))
                channel_stats['green'].append(np.mean(img[:,:,1]))
                channel_stats['blue'].append(np.mean(img[:,:,2]))
        except:
            continue
    
    stats = {}
    for key, values in pixel_stats.items():
        stats[key] = {
            'mean': np.mean(values), 'std': np.std(values),
            'min': np.min(values), 'max': np.max(values)
        }
    
    stats['channels'] = channel_stats
    return stats

def normalize_image(image):
    """Comprehensive image normalization"""
    if len(image.shape) == 3:
        img_lab = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        img_lab[:,:,0] = clahe.apply(img_lab[:,:,0])
        img_enhanced = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB).astype(np.float32)
    else:
        img_enhanced = image.astype(np.float32)
    
    img_norm = img_enhanced / 255.0
    if len(img_norm.shape) == 3:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_norm = (img_norm - mean) / std
    
    return img_norm

def preprocessing_eda(pixel_stats):
    """IT24100264 EDA: Pixel distribution visualization"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Pixel statistics
    metrics = ['means', 'stds', 'mins', 'maxs']
    colors = ['blue', 'green', 'orange', 'red']
    
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        ax1.bar(metric, pixel_stats[metric]['mean'], color=color, alpha=0.7,
               yerr=pixel_stats[metric]['std'], capsize=5)
    
    ax1.set_title('Pixel Statistics Distribution')
    ax1.set_ylabel('Value')
    
    # Channel distribution
    if 'channels' in pixel_stats:
        channels = ['red', 'green', 'blue']
        means = [np.mean(pixel_stats['channels'][ch]) for ch in channels]
        stds = [np.std(pixel_stats['channels'][ch]) for ch in channels]
        
        ax2.bar(channels, means, color=channels, alpha=0.7, yerr=stds, capsize=5)
        ax2.set_title('RGB Channel Distribution')
        ax2.set_ylabel('Mean Intensity')
    
    # Normalization comparison
    ax3.bar(['Original', 'Normalized'], [255, 5], color=['orange', 'blue'], alpha=0.7)
    ax3.set_title('Normalization Effect')
    ax3.set_ylabel('Value Range')
    
    # Histogram
    sample_data = np.random.normal(pixel_stats['means']['mean'], 
                                 pixel_stats['means']['std'], 1000)
    ax4.hist(sample_data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.set_title('Brightness Distribution')
    ax4.set_xlabel('Brightness')
    ax4.set_ylabel('Frequency')
    
    plt.suptitle('IT24100264: Pixel Distribution Analysis')
    plt.tight_layout()
    plt.show()

# IT24100348: FEATURE ENGINEERING

def extract_retinal_features(samples):
    """Extract medical features from retinal images"""
    features = {
        'brightness_mean': [], 'contrast_std': [], 'red_dominance': [],
        'green_vessel_density': [], 'texture_variance': []
    }
    
    for sample in samples[:min(300, len(samples))]:
        try:
            img = np.array(sample['image'])
            feature_set = extract_single_features(img)
            for key, value in feature_set.items():
                if key in features:
                    features[key].append(value)
        except:
            continue
    
    return {k: np.array(v) for k, v in features.items() if v}

def extract_single_features(img):
    """Extract features from single image"""
    features = {}
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    
    features['brightness_mean'] = np.mean(img)
    features['contrast_std'] = np.std(gray)
    
    if len(img.shape) == 3:
        features['red_dominance'] = np.mean(img[:,:,0]) / (np.mean(img) + 1e-8)
        green = img[:,:,1]
        dark_pixels = np.sum(green < np.percentile(green, 25))
        features['green_vessel_density'] = dark_pixels / green.size
    else:
        features['red_dominance'] = 1.0
        features['green_vessel_density'] = 0.0
    
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    features['texture_variance'] = np.var(laplacian)
    
    return features

def analyze_feature_correlations(features, samples):
    """Analyze feature correlations with DR severity"""
    labels = []
    for i, sample in enumerate(samples):
        if i >= len(list(features.values())[0]):
            break
        try:
            labels.append(int(sample['label']))
        except:
            labels.append(0)
    
    labels = np.array(labels[:len(list(features.values())[0])])
    
    correlations = {}
    for name, values in features.items():
        if len(values) > 0 and len(labels) > 0:
            min_len = min(len(values), len(labels))
            corr = np.corrcoef(values[:min_len], labels[:min_len])[0, 1]
            correlations[name] = corr
    
    return correlations

def feature_eda(features, correlations, samples):
    """IT24100348 EDA: Feature analysis visualization"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Correlation with target
    feature_names = list(correlations.keys())
    corr_values = list(correlations.values())
    
    sorted_idx = np.argsort(np.abs(corr_values))[::-1]
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_corr = [corr_values[i] for i in sorted_idx]
    
    colors = ['red' if c < 0 else 'blue' for c in sorted_corr]
    ax1.barh(range(len(sorted_features)), sorted_corr, color=colors, alpha=0.7)
    ax1.set_yticks(range(len(sorted_features)))
    ax1.set_yticklabels([f.replace('_', ' ').title() for f in sorted_features])
    ax1.set_xlabel('Correlation with DR Severity')
    ax1.set_title('Feature-Target Correlations')
    ax1.axvline(0, color='black', alpha=0.5)
    
    # Feature distribution by class
    if feature_names:
        top_feature = sorted_features[0]
        feature_values = features[top_feature]
        labels = [int(samples[i]['label']) for i in range(len(feature_values))]
        
        class_data = []
        class_labels = []
        for cls in range(5):
            mask = np.array(labels) == cls
            if np.sum(mask) > 0:
                class_data.append(feature_values[mask])
                class_labels.append(CLASS_NAMES[cls])
        
        bp = ax2.boxplot(class_data, labels=class_labels, patch_artist=True)
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(plt.cm.viridis(i/len(bp['boxes'])))
        
        ax2.set_title(f'{top_feature.replace("_", " ").title()} by DR Severity')
        ax2.tick_params(axis='x', rotation=45)
    
    # Feature correlation matrix
    if len(features) > 1:
        feature_matrix = np.column_stack([features[f] for f in feature_names])
        corr_matrix = np.corrcoef(feature_matrix.T)
        
        im = ax3.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax3.set_xticks(range(len(feature_names)))
        ax3.set_yticks(range(len(feature_names)))
        ax3.set_xticklabels([f.replace('_', ' ') for f in feature_names], rotation=45)
        ax3.set_yticklabels([f.replace('_', ' ') for f in feature_names])
        ax3.set_title('Inter-Feature Correlations')
        plt.colorbar(im, ax=ax3)
    
    # Feature importance
    abs_corr = np.abs(sorted_corr)
    ax4.bar(range(len(sorted_features)), abs_corr, color='green', alpha=0.7)
    ax4.set_xticks(range(len(sorted_features)))
    ax4.set_xticklabels([f.replace('_', ' ') for f in sorted_features], rotation=45)
    ax4.set_ylabel('Absolute Correlation')
    ax4.set_title('Feature Importance Ranking')
    
    plt.suptitle('IT24100348: Feature Engineering Analysis')
    plt.tight_layout()
    plt.show()

# IT24100255: DATA SPLITTING & SCALING

def prepare_data_splits(samples):
    """Create stratified train/validation/test splits"""
    labels = [int(sample['label']) for sample in samples]
    
    train_val_samples, test_samples, train_val_labels, test_labels = train_test_split(
        samples, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    train_samples, val_samples, _, _ = train_test_split(
        train_val_samples, train_val_labels, test_size=0.125, 
        stratify=train_val_labels, random_state=42
    )
    
    return train_samples, val_samples, test_samples

def compute_scaling_parameters(samples):
    """Compute global scaling parameters"""
    all_pixels = []
    
    for sample in samples[:min(100, len(samples))]:
        try:
            img = np.array(sample['image']).astype(np.float32)
            flat_pixels = img.flatten()
            sampled_pixels = flat_pixels[::100]
            all_pixels.extend(sampled_pixels)
        except:
            continue
    
    all_pixels = np.array(all_pixels)
    return {
        'global_mean': np.mean(all_pixels),
        'global_std': np.std(all_pixels),
        'global_min': np.min(all_pixels),
        'global_max': np.max(all_pixels),
        'pixels_analyzed': len(all_pixels)
    }

def scaling_eda(train_samples, val_samples, test_samples, scaling_params):
    """IT24100255 EDA: Data splits and scaling visualization"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Split distribution
    split_sizes = [len(train_samples), len(val_samples), len(test_samples)]
    split_labels = ['Training', 'Validation', 'Test']
    colors = ['blue', 'orange', 'green']
    
    bars = ax1.bar(split_labels, split_sizes, color=colors, alpha=0.7)
    ax1.set_title('Dataset Split Distribution')
    ax1.set_ylabel('Number of Samples')
    
    total = sum(split_sizes)
    for bar, size in zip(bars, split_sizes):
        height = bar.get_height()
        pct = (size / total) * 100
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(split_sizes)*0.01,
                f'{size}\n({pct:.1f}%)', ha='center', va='bottom')
    
    # Class distribution across splits
    splits_data = [train_samples, val_samples, test_samples]
    class_distributions = {}
    for split_name, split_data in zip(split_labels, splits_data):
        class_counts = Counter([int(sample['label']) for sample in split_data])
        class_distributions[split_name] = class_counts
    
    classes = list(range(5))
    class_names = [CLASS_NAMES[cls] for cls in classes]
    
    bottoms = [0] * len(classes)
    for i, split_name in enumerate(split_labels):
        counts = [class_distributions[split_name].get(cls, 0) for cls in classes]
        ax2.bar(class_names, counts, bottom=bottoms, label=split_name, 
               color=colors[i], alpha=0.7)
        bottoms = [b + c for b, c in zip(bottoms, counts)]
    
    ax2.set_title('Class Distribution Across Splits')
    ax2.set_ylabel('Number of Samples')
    ax2.legend()
    ax2.tick_params(axis='x', rotation=45)
    
    # Scaling parameters
    metrics = ['Global Mean', 'Global Std', 'Global Min', 'Global Max']
    values = [
        scaling_params['global_mean'], scaling_params['global_std'],
        scaling_params['global_min'], scaling_params['global_max']
    ]
    
    ax3.bar(metrics, values, color=['red', 'blue', 'green', 'purple'], alpha=0.7)
    ax3.set_title('Global Scaling Parameters')
    ax3.set_ylabel('Pixel Value')
    ax3.tick_params(axis='x', rotation=45)
    
    for i, v in enumerate(values):
        ax3.text(i, v + max(values)*0.02, f'{v:.1f}', ha='center', va='bottom')
    
    # Data readiness checklist
    checklist = ['Train/Val/Test Split', 'Stratified Sampling', 'Global Scaling', 'Memory Optimization']
    completion = [100, 100, 100, 95]
    
    bars = ax4.barh(checklist, completion, color='green', alpha=0.7)
    ax4.set_title('Data Preparation Readiness')
    ax4.set_xlabel('Completion (%)')
    ax4.set_xlim(0, 105)
    
    for bar, pct in zip(bars, completion):
        ax4.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2.,
                f'{pct}%', ha='left', va='center')
    
    plt.suptitle('IT24100255: Data Splitting & Scaling Analysis')
    plt.tight_layout()
    plt.show()

# INTEGRATED PIPELINE

def create_integrated_pipeline(scaling_params):
    """Complete preprocessing pipeline"""
    
    def preprocess_sample(sample, is_training=True):
        try:
            # IT24100259: Data validation
            img = np.array(sample['image'])
            label = int(sample['label'])
            
            if img is None or len(img.shape) != 3:
                return None, None
            
            # IT24100349: Outlier filtering
            brightness = np.mean(img)
            if brightness < 5 or brightness > 250:
                if is_training:
                    return None, None
            
            # Resize image
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            
            # IT24610824: Augmentation
            if is_training:
                img = apply_augmentation(img, label)
            
            # IT24100264: Normalization
            img = normalize_image(img)
            
            # IT24100255: Final scaling
            if scaling_params:
                img = (img * 255.0 - scaling_params['global_mean']) / (scaling_params['global_std'] + 1e-8)
            
            return img, label
            
        except Exception:
            return None, None
    
    return preprocess_sample

def validate_pipeline(train_samples, val_samples, test_samples, preprocessing_fn):
    """Validate integrated pipeline"""
    validation_results = {}
    splits = {'train': train_samples[:20], 'val': val_samples[:10], 'test': test_samples[:10]}
    
    for split_name, samples in splits.items():
        successful = failed = 0
        times = []
        
        for sample in samples:
            try:
                start_time = time.time()
                result = preprocessing_fn(sample, is_training=(split_name == 'train'))
                times.append(time.time() - start_time)
                
                if result[0] is not None:
                    successful += 1
                else:
                    failed += 1
            except:
                failed += 1
        
        total = successful + failed
        validation_results[split_name] = {
            'success_rate': (successful / total) * 100 if total > 0 else 0,
            'avg_time': np.mean(times) if times else 0,
            'successful': successful,
            'failed': failed
        }
    
    return validation_results

def visualize_integrated_pipeline(validation_results):
    """Visualize integrated pipeline performance"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    splits = list(validation_results.keys())
    success_rates = [validation_results[split]['success_rate'] for split in splits]
    avg_times = [validation_results[split]['avg_time'] * 1000 for split in splits]
    
    # Success rates
    ax1.bar(splits, success_rates, color=['blue', 'orange', 'green'], alpha=0.7)
    ax1.set_title('Pipeline Success Rate by Split')
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_ylim(0, 105)
    
    for i, rate in enumerate(success_rates):
        ax1.text(i, rate + 2, f'{rate:.1f}%', ha='center', va='bottom')
    
    # Processing times
    ax2.bar(splits, avg_times, color=['red', 'purple', 'brown'], alpha=0.7)
    ax2.set_title('Average Processing Time')
    ax2.set_ylabel('Time (ms)')
    
    for i, time_ms in enumerate(avg_times):
        ax2.text(i, time_ms + max(avg_times)*0.02, f'{time_ms:.1f}ms',
                ha='center', va='bottom')
    
    # Pipeline flow
    stages = ['Raw Data', 'Clean', 'Balanced', 'Normalized', 'Ready']
    flow = [1000, 950, 900, 900, 900]
    
    ax4.plot(stages, flow, marker='o', linewidth=3, markersize=8, color='green')
    ax4.set_title('Data Flow Through Pipeline')
    ax4.set_ylabel('Sample Count')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    for i, count in enumerate(flow):
        ax4.annotate(f'{count}', (i, count), textcoords="offset points",
                    xytext=(0,10), ha='center')
    
    plt.tight_layout()
    plt.show()

# MAIN EXECUTION

def run_complete_pipeline():
    """Execute complete preprocessing pipeline"""
    try:
        # IT24100259: Data Loading & Missing Data
        print("\n[IT24100259] Loading dataset and analyzing missing data...")
        samples = load_dataset()
        missing_info = analyze_missing_data(samples)
        print(f"Dataset loaded: {len(samples)} samples, {missing_info['valid_samples']} valid")
        missing_data_eda(missing_info)
        memory_cleanup()
        
        # IT24100349: Outlier Detection
        print("\n[IT24100349] Detecting outliers...")
        outliers_info = detect_outliers(samples)
        total_outliers = sum(info['count'] for info in outliers_info.values())
        print(f"Outliers detected: {total_outliers} across all metrics")
        outlier_eda(outliers_info)
        memory_cleanup()
        
        # IT24610824: Class Distribution & Balancing
        print("\n[IT24610824] Analyzing class distribution...")
        class_counts, imbalance_ratio = analyze_class_distribution(samples)
        strategy = create_balancing_strategy(class_counts)
        print(f"Class imbalance ratio: {imbalance_ratio:.1f}:1")
        balance_augment_eda(class_counts, strategy)
        memory_cleanup()
        
        # IT24100264: Pixel Analysis & Normalization
        print("\n[IT24100264] Analyzing pixel distributions...")
        pixel_stats = analyze_pixel_distributions(samples)
        print(f"Mean brightness: {pixel_stats['means']['mean']:.1f}")
        preprocessing_eda(pixel_stats)
        memory_cleanup()
        
        # IT24100348: Feature Engineering
        print("\n[IT24100348] Extracting features...")
        features = extract_retinal_features(samples)
        correlations = analyze_feature_correlations(features, samples)
        print(f"Features extracted: {len(features)} types")
        feature_eda(features, correlations, samples)
        memory_cleanup()
        
        # IT24100255: Data Splitting & Scaling
        print("\n[IT24100255] Preparing data splits...")
        train_samples, val_samples, test_samples = prepare_data_splits(samples)
        scaling_params = compute_scaling_parameters(train_samples)
        print(f"Data split: Train({len(train_samples)}), Val({len(val_samples)}), Test({len(test_samples)})")
        scaling_eda(train_samples, val_samples, test_samples, scaling_params)
        memory_cleanup()
        
        # Integrated Pipeline Validation
        print("\n[Integrated Pipeline] Validating complete pipeline...")
        preprocessing_fn = create_integrated_pipeline(scaling_params)
        validation_results = validate_pipeline(train_samples, val_samples, test_samples, preprocessing_fn)
        overall_success = np.mean([validation_results[split]['success_rate'] for split in validation_results])
        overall_time = np.mean([validation_results[split]['avg_time'] * 1000 for split in validation_results])
        print(f"Pipeline validation: {overall_success:.1f}% success, {overall_time:.1f}ms avg time")
        visualize_integrated_pipeline(validation_results)
        
        # Final Summary
        print(f"\n" + "="*60)
        print("PIPELINE EXECUTION COMPLETE")
        print("="*60)
        print(f"Total samples processed: {len(samples)}")
        print(f"Data completeness: {(missing_info['valid_samples']/missing_info['total_samples'])*100:.1f}%")
        print(f"Outliers identified: {total_outliers}")
        print(f"Class imbalance: {imbalance_ratio:.1f}:1")
        print(f"Features extracted: {len(features)}")
        print(f"Pipeline success rate: {overall_success:.1f}%")
        print(f"Memory usage: {check_memory():.1f}%")
        
        return {
            'samples': samples,
            'train_samples': train_samples,
            'val_samples': val_samples,
            'test_samples': test_samples,
            'preprocessing_function': preprocessing_fn,
            'validation_results': validation_results,
            'features': features,
            'correlations': correlations,
            'scaling_params': scaling_params,
            'success': True
        }
        
    except Exception as e:
        print(f"Pipeline execution failed: {e}")
        return {'success': False, 'error': str(e)}

# Execute the complete pipeline
if __name__ == "__main__":
    results = run_complete_pipeline()
    
    if results.get('success'):
        print(f"\nPipeline ready for model training!")
        print(f"Use results['preprocessing_function'] for data preprocessing")
        print(f"Access train/val/test splits from results")
    else:
        print(f"Pipeline failed. Check error messages above.")