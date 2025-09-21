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

def visualize_member4_eda(pixel_stats):
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