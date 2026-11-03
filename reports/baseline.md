# MobileNetV2 Baseline

I used an ImageNet-pretrained MobileNetV2 with a frozen base and a classification head: global average pooling, Dense(128, ReLU), Dropout(0.3), and Dense(1, sigmoid). Input images were resized to 224 × 224 pixels.

The dataset contained 1,800 synthetic images: 1,308 for training, 216 for validation, and 276 for testing. All versions of the same source document stayed in one split. I used seed 24500007, Adam with a learning rate of 0.001, binary cross-entropy, and a batch size of 32.

Training took 1,604.90 seconds (26.75 minutes) on CPU. Early stopping ended training after 37 epochs, and the best checkpoint was from epoch 32, with a validation loss of 0.006803.

Training and validation loss decreased overall, while both accuracy curves approached 100%. There was no clear sustained gap between the curves suggesting strong overfitting on this split.

Test accuracy was 99.64%, macro precision was 0.9964, macro recall was 0.9964, and macro F1 was 0.9964. The model correctly classified 275 out of 276 test images.

The confusion matrix was [[137, 1], [0, 138]], with rows representing true classes and columns representing predicted classes in the order bad, good. Bad-image recall was 99.28%, and no good images were rejected.

The only misclassified image was an INN document with an artificial glare defect. The model accepted it as good. The glare may have been too subtle for the model to detect, but this explanation is based only on visual inspection.

Average CPU latency was 135.49 ms per image after 20 warm-up predictions and over 100 timed predictions with batch size 1. Timing included resizing, preprocessing, inference, and the final decision, but excluded image decoding, disk access, model loading, and network operations.

The inference model was 10.28 MB and contained 2,422,081 parameters. It met my predefined accuracy, macro F1, bad-image recall, and model-size criteria, but it did not meet the latency target of less than 100 ms per image.

MobileNetV2 provides a strong baseline on this synthetic test set. I will compare it with the other models before making the final choice. Its performance on real photographs has not yet been tested.
