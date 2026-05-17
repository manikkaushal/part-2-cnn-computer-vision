# Part 2 – CNN for Manufacturing Defect Detection

## What is this project about?

This project builds a CNN (Convolutional Neural Network) that looks at images of product surfaces and decides whether the surface is normal or has a defect. The dataset has four types of images — normal, scratch, dent, and stain. The model picks one of these four labels for every image it sees.

---

## How to Run

1. Install all required packages:
   ```
   pip install -r requirements.txt
   ```

2. Put the `part_2_cnn_computer_vision` folder in the same directory as the notebook.

3. Open `notebook.ipynb` in Jupyter and run all cells from top to bottom.

4. Results (accuracy/loss curves, confusion matrix, sample predictions) will be saved automatically in the `results/` and `sample_predictions/` folders.

---

## Repository Structure

```
part-2-cnn-computer-vision/
│
├── README.md
├── notebook.ipynb
├── requirements.txt
├── sample_predictions/
│   └── prediction_outputs.png
└── results/
    ├── accuracy_loss_curves.png
    └── confusion_matrix.png
```

---

## Task 1 – Problem Identification

**Problem type: Image Classification**

This is an image classification problem. We have images of product surfaces and each image belongs to one of four fixed categories — normal, scratch, dent, or stain. The model just needs to predict the correct category for each image.

Why image classification is the right choice here:

- Every image already has exactly one label attached to it
- We do not need to draw bounding boxes around the defect area (that would be object detection)
- We do not need to color individual pixels to mark defect regions (that would be semantic segmentation)
- We just want one answer per image — what type of defect is this, or is it normal

So a simple CNN that outputs probabilities for 4 classes is exactly what is needed here.

---

## Task 2 – Dataset Exploration

**Dataset summary:**

| Detail | Value |
|---|---|
| Total images | 480 |
| Number of classes | 4 |
| Images per class | 120 each |
| Image size (original) | 96 x 96 pixels |
| Color format | RGB (3 channels) |
| Dataset balance | Balanced — all 4 classes have exactly 120 images |

**Classes:**
- `normal` — product surface with no defects (120 images)
- `scratch` — surface has a scratch (120 images)
- `dent` — surface has a dent (120 images)
- `stain` — surface has a stain (120 images)

**Is the dataset balanced?**

Yes. Each class has exactly 120 images, so there is no imbalance. The model is not going to be biased toward any one class just because it has more examples.

**Sample images from each class** are shown in the notebook by loading one image from each class folder and plotting them side by side with their labels.

---

## Task 3 – Image Preprocessing

The following steps are applied to prepare the images for training:

**1. Resizing**
All images are resized from their original 96x96 size to 64x64 pixels. This keeps the input size fixed and also makes training a bit faster.

**2. Normalizing**
Pixel values originally range from 0 to 255. We divide all values by 255 so they fall in the range 0 to 1. This helps the model train more smoothly because the numbers are smaller and more consistent.

**3. Train/Test Split**
The data is split 80/20 — 80% for training and 20% for testing. So out of 480 images, roughly 384 are used for training and 96 for testing. A validation split is also taken from the training data during model training.

**4. Label Encoding**
The class names (normal, scratch, dent, stain) are converted to integers (0, 1, 2, 3) and then one-hot encoded so the model can work with them.

**5. Augmentation**
Since the dataset is small (only 480 images), we apply basic data augmentation during training to stop the model from just memorizing the training images. Augmentation used:
- Horizontal flip
- Small rotation

This creates slightly different versions of training images each time they are seen, which helps the model learn more general patterns.

---

## Task 4 – CNN Model

The model is built using TensorFlow and Keras.

**Model architecture:**

```
Input (64 x 64 x 3)
        ↓
Conv2D (32 filters, 3x3) + ReLU
        ↓
MaxPooling2D (2x2)
        ↓
Conv2D (64 filters, 3x3) + ReLU
        ↓
MaxPooling2D (2x2)
        ↓
Flatten
        ↓
Dense (128 units) + ReLU
        ↓
Output Dense (4 units) + Softmax
```

**What each layer does:**

- **Conv2D** — Scans the image with small filters to pick up patterns like edges, textures, and shapes
- **ReLU** — Activation function that removes negative values so the model can learn non-linear patterns
- **MaxPooling2D** — Shrinks the feature maps by keeping only the strongest signal in each region
- **Flatten** — Turns the 2D feature maps into a 1D list so the dense layers can process them
- **Dense (128)** — A fully connected hidden layer that combines all the features learned by the conv layers
- **Output Dense (4) + Softmax** — Produces 4 probability values (one per class); the class with the highest probability is the prediction

---

## Task 5 – Model Training and Evaluation

**Training setup:**
- Optimizer: Adam
- Loss function: Categorical Crossentropy
- Epochs: 15
- Batch size: 32

**What to look at in results:**

- `results/accuracy_loss_curves.png` — Shows how training and validation accuracy and loss changed over each epoch. If the training accuracy keeps going up but validation accuracy stops improving, the model is overfitting.

- `results/confusion_matrix.png` — Shows how many images from each class were correctly classified vs misclassified. For example, some scratch images might get confused with dent images since they can look similar.

- `sample_predictions/prediction_outputs.png` — Shows actual test images with the model's predicted label and the true label side by side.

**Expected observations:**
- The model should reach around 80–90% accuracy on the test set given the balanced and relatively clean dataset
- Scratch and dent classes may have slightly more confusion with each other compared to normal and stain, which tend to look more distinct

---

## Task 6 – CNN Concepts Explained Simply

### What is convolution?

Convolution is the main thing that makes CNNs work. It takes a small filter — say 3x3 pixels — and slides it across the entire image, one position at a time. At each position it multiplies the filter values with the image values underneath and adds them up. This produces a number that tells how strongly that pattern was present at that spot.

For example, one filter might learn to detect horizontal edges. Another might detect diagonal lines. Early layers in the CNN learn simple patterns like edges and corners. Deeper layers combine those into more complex patterns like textures or shapes.

### Why is pooling used?

After convolution, the feature maps are still quite large. Pooling shrinks them down by taking the maximum value (Max Pooling) from small regions, usually 2x2.

This helps because:
- It makes the model smaller and faster
- It makes the model less sensitive to the exact position of a feature — if a scratch is shifted slightly, the model still picks it up
- It reduces the amount of data passing through the network, which helps avoid overfitting

### Why is ReLU commonly used in CNNs?

ReLU stands for Rectified Linear Unit. It is a very simple function — it turns any negative value into 0 and leaves positive values as they are.

Without an activation function like this, stacking many layers would mathematically collapse into just one linear operation. That means the model cannot learn complex patterns no matter how deep it is. ReLU breaks this linearity.

ReLU is popular because it is simple, fast to compute, and works well in practice. It does not slow down training the way older activation functions like sigmoid or tanh can.

### Why are CNNs better than regular feed-forward networks for image data?

A regular fully connected (dense) network treats each pixel as a separate input and does not know that nearby pixels are related to each other. For a 64x64 RGB image, that is 12,288 inputs — and each one gets its own weight for every neuron. This creates a huge number of parameters, most of which are not useful.

CNNs are better because:
- They use small filters shared across the whole image, so the number of parameters stays manageable
- They understand that pixels near each other form meaningful local patterns (edges, textures)
- They build up from simple to complex features layer by layer, which matches how visual patterns actually work
- They are naturally shift-invariant — the same filter detects a scratch whether it is on the left or right side of the image

---

## Task 7 – Business Use Case

**Domain: Manufacturing / Quality Control**

In a real factory, products move along a conveyor belt and need to be checked for defects before they are packed and sent to customers. Doing this manually is slow, tiring, and inconsistent — workers miss things, especially during long shifts.

A CNN model like this one can be set up with a camera pointed at the production line. Every time a product passes, the camera takes a photo and the model classifies it as normal, scratch, dent, or stain — all in real time.

**What this enables in practice:**
- Products with defects can be automatically rejected or flagged before packaging
- The system works 24 hours a day without breaks or loss of attention
- Every single defect is logged with a timestamp, image, and label — which helps in tracking where and when defects happen
- Managers can spot patterns — for example, if dents spike at a certain time, it might point to a machine issue

**Where this is already being used:**
This kind of computer vision system is already in use in car manufacturing (checking body panels), electronics (checking circuit boards), food processing (checking for damaged packaging), and textile manufacturing (checking for fabric defects).

Even a basic model like this one can serve as a starting point. With more data, fine-tuned hyperparameters, or a deeper architecture, the accuracy and reliability can be improved significantly.

---

## Results Summary

After training for 15 epochs the model was evaluated on the held-out test set. The accuracy and loss curves are saved in `results/accuracy_loss_curves.png`. The confusion matrix is saved in `results/confusion_matrix.png`. Sample predictions comparing true vs predicted labels are in `sample_predictions/prediction_outputs.png`.
