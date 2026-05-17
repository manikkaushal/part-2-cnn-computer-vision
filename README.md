# Part 2 – CNN for Manufacturing Defect Detection

## What is this project about?

This project is about building a CNN (Convolutional Neural Network) that can look at images of product surfaces and figure out if there is any defect on them. The dataset has four types of images — normal, scratch, dent, and stain. So the model has to pick one of these four labels for every image it sees.

This is basically an image classification problem.

---

## Dataset Info

- Total images: 480
- Classes: normal, scratch, dent, stain
- Images per class: 120 each (so the dataset is balanced, no class has more images than another)
- Image size (original): 96 x 96 pixels, RGB

---

## How to Run

1. Install all the required packages by running:
   ```
   pip install -r requirements.txt
   ```

2. Put the `part_2_cnn_computer_vision` folder in the same place as the notebook.

3. Open `notebook.ipynb` in Jupyter and run all the cells from top to bottom.

4. The results (accuracy/loss curves, confusion matrix, sample predictions) will be saved automatically in the `results/` and `sample_predictions/` folders.

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

## Task 1 – Problem Type

This is an **image classification** problem. We have images and each image belongs to one of 4 fixed categories. The model just needs to predict the correct category for each image. We are not drawing boxes around defects (that would be object detection) or coloring individual pixels (that would be segmentation). We just want a label for the whole image.

---

## Task 6 – CNN Concepts Explained Simply

### What is convolution?

Convolution is basically sliding a small filter (like a 3x3 grid) over the image and doing a dot product at each position. This helps the model detect simple things like edges or corners in early layers and more complex shapes in deeper layers. Think of it like scanning the image with a magnifying glass that highlights specific patterns.

### Why is pooling used?

After convolution the feature maps are pretty big. Pooling shrinks them down by taking the max value (or average) in small regions. This makes the model faster and also makes it care less about the exact position of a feature — if a scratch is slightly shifted, the model still picks it up. It also helps with reducing overfitting to some extent.

### Why is ReLU used in CNNs?

ReLU just turns any negative number to zero and keeps positive numbers as they are. Without something like ReLU, stacking multiple layers would basically collapse into one big linear operation and the model wouldn't learn complex patterns. ReLU is simple, fast, and works pretty well in most cases. Other activations exist but ReLU is the default choice most people go with.

### Why are CNNs better than regular networks for images?

A regular fully connected (dense) network treats each pixel independently and doesn't know that nearby pixels are related. For a 96x96 image, a dense layer would have 27,648 inputs — that's a huge number of weights and most of them don't contribute much. CNNs use small filters that are shared across the whole image, so they learn patterns locally. They also understand that spatial structure matters — the position of a scratch relative to its surroundings tells you something. Regular networks can't do this as naturally.

---

## Task 7 – Business Use Case

**Domain: Manufacturing / Quality Control**

In a real factory, products move along a conveyor belt and need to be checked for defects before they are packed and shipped. Doing this manually is slow and workers get tired and miss things.

A CNN model like this one can be set up with a camera on the production line. Every time a product passes, the camera takes a photo and the model classifies it as normal, scratch, dent, or stain in real time.

Benefits:
- Faster than human inspection
- Works 24/7 without breaks
- Catches more defects consistently
- Can log every single defect with a timestamp and image

This kind of system is already being used in car manufacturing, electronics assembly, and food packaging. Even a basic model like ours can be a starting point for something more accurate with more data and better training.

---

## Results Summary

After training for 15 epochs, the model was evaluated on the test set. Accuracy and loss curves are saved in the `results/` folder. The confusion matrix shows how well the model did on each class. Some classes might be harder for the model to separate (like scratch vs dent since they can look similar visually).
