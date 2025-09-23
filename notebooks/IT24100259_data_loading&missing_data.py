# IT24100259: DATA LOADING & MISSING DATA HANDLING

def load_dataset():
    #Load diabetic retinopathy dataset
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
    #Create simulated retinal images
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
    #Analyze missing data
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
    #IT24100259 EDA: Missing data visualization
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
