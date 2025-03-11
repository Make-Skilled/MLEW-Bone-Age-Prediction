import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import shutil
from PIL import Image
import tensorflow as tf

# Paths
DATASET_DIR = 'dataset'
TRAIN_CSV = os.path.join(DATASET_DIR, 'boneage-training-dataset.csv')
TEST_CSV = os.path.join(DATASET_DIR, 'boneage-test-dataset.csv')
TRAIN_IMG_DIR = os.path.join(DATASET_DIR, 'boneage-training-dataset', 'boneage-training-dataset')
TEST_IMG_DIR = os.path.join(DATASET_DIR, 'boneage-test-dataset', 'boneage-test-dataset')

# Create organized dataset directories
ORGANIZED_DATA_DIR = os.path.join(DATASET_DIR, 'organized')
TRAIN_DIR = os.path.join(ORGANIZED_DATA_DIR, 'train')
VAL_DIR = os.path.join(ORGANIZED_DATA_DIR, 'validation')
TEST_DIR = os.path.join(ORGANIZED_DATA_DIR, 'test')

def create_directories():
    """Create necessary directories for organized dataset"""
    for dir_path in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)

def verify_dataset():
    """Verify that all required dataset files and directories exist"""
    # Check CSV files
    if not os.path.exists(TRAIN_CSV):
        raise FileNotFoundError(f"Training CSV file not found: {TRAIN_CSV}")
    if not os.path.exists(TEST_CSV):
        raise FileNotFoundError(f"Test CSV file not found: {TEST_CSV}")
    
    # Check image directories
    if not os.path.exists(TRAIN_IMG_DIR):
        raise FileNotFoundError(f"Training image directory not found: {TRAIN_IMG_DIR}")
    if not os.path.exists(TEST_IMG_DIR):
        raise FileNotFoundError(f"Test image directory not found: {TEST_IMG_DIR}")
    
    # Print dataset info
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    print(f"\nDataset statistics:")
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")

def organize_dataset():
    """Organize dataset into train/validation/test splits"""
    # Verify dataset structure
    verify_dataset()
    
    # Read CSV files
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    
    # Rename test dataset columns to match training
    test_df = test_df.rename(columns={'Case ID': 'id'})
    test_df['male'] = test_df['Sex'].map({'M': True, 'F': False})
    
    # Create main directories
    for dir_path in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)
    
    # Split training data into train and validation
    train_data, val_data = train_test_split(train_df, test_size=0.2, random_state=42)
    
    # Function to copy and preprocess images
    def copy_images(df, target_dir, is_test=False):
        os.makedirs(target_dir, exist_ok=True)
        image_data = []
        processed_count = 0
        error_count = 0
        
        # Create a subdirectory for images
        images_dir = os.path.join(target_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)
        
        print(f"\nProcessing {len(df)} images for {os.path.basename(target_dir)}...")
        
        for idx, row in df.iterrows():
            img_id = str(row['id'])
            source_dir = TEST_IMG_DIR if is_test else TRAIN_IMG_DIR
            img_path = os.path.join(source_dir, f'{img_id}.png')
            
            if os.path.exists(img_path):
                try:
                    # Load and preprocess image
                    img = Image.open(img_path)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img = img.resize((224, 224))  # VGG16 input size
                    
                    # Save preprocessed image
                    target_path = os.path.join(images_dir, f'{img_id}.png')
                    img.save(target_path)
                    
                    # Store image data
                    if not is_test:  # Only for training and validation data
                        image_data.append({
                            'filename': f'images/{img_id}.png',  # Update path to include images subdirectory
                            'boneage': row['boneage'],
                            'male': row['male']
                        })
                    processed_count += 1
                    
                    # Print progress
                    if processed_count % 100 == 0:
                        print(f"Processed {processed_count}/{len(df)} images...")
                        
                except Exception as e:
                    print(f"Error processing image {img_id}: {str(e)}")
                    error_count += 1
                    continue
            else:
                print(f"Image not found: {img_path}")
                error_count += 1
        
        # Save image data to CSV
        if image_data:
            df = pd.DataFrame(image_data)
            csv_path = os.path.join(target_dir, 'data.csv')
            df.to_csv(csv_path, index=False)
            print(f"\nProcessing complete for {os.path.basename(target_dir)}:")
            print(f"- Successfully processed: {processed_count}")
            print(f"- Errors: {error_count}")
            print(f"- Saved {len(image_data)} records to {csv_path}")
        
        return processed_count, error_count
    
    # Copy and preprocess images
    print("\nOrganizing training images...")
    train_processed, train_errors = copy_images(train_data, TRAIN_DIR)
    
    print("\nOrganizing validation images...")
    val_processed, val_errors = copy_images(val_data, VAL_DIR)
    
    print("\nOrganizing test images...")
    test_processed, test_errors = copy_images(test_df, TEST_DIR, is_test=True)
    
    # Print summary
    print("\nDataset organization complete:")
    print(f"Training: {train_processed} processed, {train_errors} errors")
    print(f"Validation: {val_processed} processed, {val_errors} errors")
    print(f"Test: {test_processed} processed, {test_errors} errors")
    
    return train_data, val_data, test_df

class BoneAgeGenerator(tf.keras.utils.Sequence):
    def __init__(self, directory, batch_size=32, shuffle=True, augment=False):
        self.directory = directory
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        
        try:
            # Read image data
            csv_path = os.path.join(directory, 'data.csv')
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"data.csv not found in {directory}")
            
            self.data = pd.read_csv(csv_path)
            if len(self.data) == 0:
                raise ValueError(f"No data found in {csv_path}")
            
            self.image_filenames = self.data['filename'].values
            self.boneages = self.data['boneage'].values / 12.0  # Convert to years
            self.male = self.data['male'].values
            
            print(f"Loaded {len(self.data)} images from {directory}")
        except Exception as e:
            print(f"Error initializing generator for {directory}: {str(e)}")
            raise
        
        # Create augmentation generator if needed
        if self.augment:
            self.augmentor = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )
        
        self.indexes = np.arange(len(self.image_filenames))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / self.batch_size))
    
    def __getitem__(self, idx):
        # Get batch indexes
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        # Initialize batch arrays
        batch_x = np.zeros((len(batch_indexes), 224, 224, 3))
        batch_y = np.zeros(len(batch_indexes))
        
        # Load and preprocess images
        for i, idx in enumerate(batch_indexes):
            try:
                # Load image
                img_path = os.path.join(self.directory, self.image_filenames[idx])
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
                x = tf.keras.preprocessing.image.img_to_array(img)
                
                # Apply augmentation if needed
                if self.augment:
                    x = self.augmentor.random_transform(x)
                
                # Preprocess
                x = preprocess_input(x)
                
                # Store
                batch_x[i] = x
                batch_y[i] = self.boneages[idx]
            except Exception as e:
                print(f"Error processing image {self.image_filenames[idx]}: {str(e)}")
                continue
        
        return batch_x, batch_y
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

def create_model():
    """Create VGG16 model for bone age prediction"""
    # Create base VGG16 model
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1)(x)  # Single output for age prediction
    
    # Create model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    return model

def train_model():
    """Main training function"""
    print("Creating directories...")
    create_directories()
    
    print("Organizing dataset...")
    train_data, val_data, test_df = organize_dataset()
    
    print("Creating data generators...")
    train_generator = BoneAgeGenerator(TRAIN_DIR, batch_size=32, shuffle=True, augment=True)
    val_generator = BoneAgeGenerator(VAL_DIR, batch_size=32, shuffle=False)
    
    print("Creating model...")
    model = create_model()
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='mean_absolute_error',
        metrics=['mae']
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            'bone_age_weights.best.hdf5',
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=5,
            min_lr=1e-6
        )
    ]
    
    # Train model
    print("Training model...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=50,
        callbacks=callbacks
    )
    
    print("Training completed!")
    return model, history

if __name__ == '__main__':
    train_model() 