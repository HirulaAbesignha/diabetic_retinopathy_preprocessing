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

def visualize_member3_eda(class_counts, strategy):
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