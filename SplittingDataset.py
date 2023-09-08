import os
import shutil
from sklearn.model_selection import train_test_split

source = 'your path to preprocessed files directory'
output = 'your path to preprocessed split directory'

# Defining ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Going through subdirectories
class_dirs = os.listdir(source)

for class_dir in class_dirs:
    class_path = os.path.join(source, class_dir)
    out_train_dir = os.path.join(output, 'train', class_dir)
    out_val_dir = os.path.join(output, 'validation', class_dir)
    out_test_dir = os.path.join(output, 'test', class_dir)
    os.makedirs(out_train_dir, exist_ok=True)
    os.makedirs(out_val_dir, exist_ok=True)
    os.makedirs(out_test_dir, exist_ok=True)

    images = os.listdir(class_path)
    train_set, test_val_set = train_test_split(images, test_size=(val_ratio + test_ratio), random_state=42)
    val_set, test_set = train_test_split(test_val_set, test_size=test_ratio / (val_ratio + test_ratio), random_state=42)

    for image in train_set:
        src = os.path.join(class_path, image)
        dst = os.path.join(out_train_dir, image)
        shutil.copy(src, dst)

    for image in val_set:
        src = os.path.join(class_path, image)
        dst = os.path.join(out_val_dir, image)
        shutil.copy(src, dst)

    for image in test_set:
        src = os.path.join(class_path, image)
        dst = os.path.join(out_test_dir, image)
        shutil.copy(src, dst)

print("Dataset is successfully split.")


