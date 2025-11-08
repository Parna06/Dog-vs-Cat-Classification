# 🐶🐱 Dog vs Cat Image Classification using Transfer Learning

## 📘 Overview
This project focuses on building a **deep learning model** to classify images of dogs and cats using **transfer learning**. 
The model leverages a pre-trained CNN (such as VGG16, ResNet50, or InceptionV3) for feature extraction and fine-tuning to achieve high accuracy on the Kaggle *Dogs vs Cats* dataset.

---

## 📦 Dataset
The dataset used is the **Dogs vs Cats dataset** from Kaggle, which contains:

- 25,000 labeled training images (12,500 dogs, 12,500 cats)
- Separate test images for evaluation

The dataset is extracted directly using the **Kaggle API** within the notebook.

```python
# Example snippet
!kaggle datasets download -d tongpython/cat-and-dog
```

---

## ⚙️ Preprocessing Steps
1. **Data Extraction & Organization**
   - Downloaded and unzipped using Kaggle API.
   - Images separated into training and validation folders.

2. **Image Augmentation**
   - Implemented using `ImageDataGenerator` from Keras to prevent overfitting.

3. **Resizing & Normalization**
   - Images resized (e.g., 224x224).
   - Pixel values normalized between 0 and 1.

```python
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
```

---

## 🧠 Model Architecture
- **Base Model:** Pre-trained CNN (e.g., `VGG16`, `InceptionV3`, or `ResNet50`)
- **Top Layers Added:**
  - Global Average Pooling
  - Dense layer with ReLU activation
  - Dropout for regularization
  - Final dense layer with sigmoid activation for binary classification

```python
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=output)
```

---

## 🏋️ Training
- Optimizer: `Adam`
- Loss: `binary_crossentropy`
- Metrics: `accuracy`
- Fine-tuned the top layers while freezing early convolutional layers.

```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(train_generator, validation_data=val_generator, epochs=10)
```

---

## 📊 Results & Evaluation
- Achieved **high validation accuracy (typically >95%)**
- Training and validation loss/accuracy curves plotted using Matplotlib.
- Model performance evaluated using confusion matrix and classification report.

```python
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
```

---

## 💾 Model Saving & Deployment
- Model saved in HDF5 format (`.h5`) for later inference.
- Can be loaded using Keras for further evaluation or deployment.

```python
model.save('dog_cat_classifier.h5')
```

---

## 🧪 Prediction Example
```python
img = image.load_img('sample.jpg', target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)
prediction = model.predict(img_array)

if prediction[0][0] > 0.5:
    print("It's a Dog!")
else:
    print("It's a Cat!")
```

---

## 📈 Key Takeaways
- Transfer learning dramatically reduces training time and improves accuracy.
- Data augmentation helps prevent overfitting on limited datasets.
- Model achieves robust performance suitable for real-world classification tasks.

---

## 🧩 Dependencies
- Python ≥ 3.8  
- TensorFlow / Keras  
- NumPy, Matplotlib, Seaborn  
- Kaggle API

Install dependencies:
```bash
pip install tensorflow keras numpy matplotlib seaborn kaggle
```

---

## 👨‍💻 Author
Developed as part of a deep learning project on **Transfer Learning for Image Classification**.

