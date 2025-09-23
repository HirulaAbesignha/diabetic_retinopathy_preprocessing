# IT24100349: OUTLIER DETECTION & REMOVAL

def detect_outliers(samples):
    #Detect outliers using IQR method
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
    #IT24100349 EDA: Outlier detection visualization
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