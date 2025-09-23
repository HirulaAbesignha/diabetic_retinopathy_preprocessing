# IT24100348: FEATURE ENGINEERING

def extract_retinal_features(samples):
    #Extract medical features from retinal images
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
    #Extract features from single image
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
    #Analyze feature correlations with DR severity
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
    #IT24100348 EDA: Feature analysis visualization
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
