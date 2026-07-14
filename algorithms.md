# Algorithms — Real-time Sign Language Alphabet Detection

---

## Algorithm 1: Dataset Preparation

```
Algorithm: PREPARE_DATASET

Input:  V = {V_A, V_B, ..., V_Z}    // Set of 26 video files, one per letter
        target_fps = 12              // Frames to extract per second
        img_size = 224               // Target image dimensions
        max_frames = 500             // Max frames per video

Output: D = {D_A, D_B, ..., D_Z}    // Dataset folders with labeled images

BEGIN
    1.  FOR each video V_i in V (where i = A to Z) DO

    2.      CREATE directory D_i = "dataset/{i}/"

    3.      OPEN video file V_i
    4.      READ video_fps ← native FPS of V_i
    5.      COMPUTE frame_interval ← FLOOR(video_fps / target_fps)

    6.      SET frame_index ← 0
    7.      SET saved_count ← 0

    8.      WHILE frame exists in V_i AND saved_count < max_frames DO
    9.          READ frame from V_i
   10.          IF frame_index MOD frame_interval == 0 THEN
   11.              RESIZE frame to (img_size × img_size)
   12.              SAVE frame as "D_i/{i}_{saved_count}.jpg"
   13.              saved_count ← saved_count + 1
   14.          END IF
   15.          frame_index ← frame_index + 1
   16.      END WHILE

   17.      CLOSE video V_i

   18.  END FOR

   19.  RETURN D
END
```

---

## Algorithm 2: Model Construction (MobileNetV2 + Classification Head)

```
Algorithm: BUILD_MODEL

Input:  num_classes = 26             // Letters A–Z
        input_shape = (224, 224, 3)  // RGB image dimensions
        pretrained_weights = "ImageNet"

Output: model                       // Compiled Keras model

BEGIN
    // ── Step 1: Load pre-trained backbone ──
    1.  LOAD MobileNetV2 with ImageNet weights
    2.  REMOVE top classification layer (include_top = False)
    3.  SET all backbone layers as NON-TRAINABLE (freeze weights)

    // ── Step 2: Build classification head ──
    4.  DEFINE input layer: shape = (224, 224, 3)

    5.  APPLY MobileNetV2 preprocessing:
            pixel_values ← (pixel_values / 127.5) - 1.0
            // Normalizes [0, 255] → [-1, +1]

    6.  PASS preprocessed input through frozen MobileNetV2 backbone
            // Output shape: (7, 7, 1280)

    7.  APPLY GlobalAveragePooling2D:
            FOR each of the 1280 feature channels DO
                channel_output ← MEAN of all 7×7 spatial values
            END FOR
            // Output shape: (1280,)

    8.  APPLY Dropout(rate = 0.3):
            // During training: randomly zero out 30% of values
            // During inference: pass all values (scaled)

    9.  APPLY Dense(128 neurons, activation = ReLU):
            FOR each neuron j (j = 1 to 128) DO
                z_j ← SUM(w_ji × input_i) + bias_j    // for all i = 1 to 1280
                output_j ← MAX(0, z_j)                 // ReLU activation
            END FOR
            // Output shape: (128,)

   10.  APPLY Dropout(rate = 0.2)

   11.  APPLY Dense(26 neurons, activation = Softmax):
            FOR each class k (k = 1 to 26) DO
                z_k ← SUM(w_kj × input_j) + bias_k    // for all j = 1 to 128
            END FOR
            FOR each class k DO
                P(k) ← exp(z_k) / SUM(exp(z_m) for all m = 1 to 26)
            END FOR
            // Output: 26 probabilities that sum to 1.0

    // ── Step 3: Compile model ──
   12.  SET optimizer ← Adam(learning_rate = 0.001)
   13.  SET loss_function ← Categorical Cross-Entropy
   14.  SET metrics ← [Accuracy]
   15.  COMPILE model

   16.  RETURN model
END
```

---

## Algorithm 3: Model Training

```
Algorithm: TRAIN_MODEL

Input:  model               // From BUILD_MODEL
        D_train              // Training dataset (80% of images)
        D_val                // Validation dataset (20% of images)
        epochs = 20          // Maximum training epochs
        batch_size = 32      // Images per training step
        patience = 7         // Early stopping patience

Output: trained_model        // Model with optimized weights
        training_history     // Accuracy and loss per epoch

BEGIN
    // ── Step 1: Initialize tracking variables ──
    1.  SET best_val_accuracy ← 0
    2.  SET epochs_without_improvement ← 0
    3.  SET learning_rate ← 0.001

    // ── Step 2: Training loop ──
    4.  FOR epoch = 1 TO epochs DO

    5.      PRINT "Epoch {epoch}/{epochs}"

            // ── Step 2a: Process all training batches ──
    6.      SHUFFLE D_train
    7.      FOR each batch B of batch_size images from D_train DO

                // ── Data Augmentation (on-the-fly) ──
    8.          FOR each image I in batch B DO
    9.              RANDOMLY APPLY:
                        - Rotation by angle θ ∈ [-15°, +15°]
                        - Horizontal shift by Δx ∈ [-15%, +15%]
                        - Vertical shift by Δy ∈ [-15%, +15%]
                        - Zoom by factor z ∈ [0.85, 1.15]
                        - Brightness by factor b ∈ [0.80, 1.20]
                        - Shear by factor s ∈ [-0.1, +0.1]
                        // NOTE: No horizontal flip (signs are asymmetric)
   10.          END FOR

                // ── Forward Pass ──
   11.          predictions ← model.forward(B)
                    // Shape: (batch_size, 26)

                // ── Compute Loss ──
   12.          FOR each sample i in batch DO
                    loss_i ← -SUM( y_true[i][k] × log(predictions[i][k]) )
                                for k = 1 to 26
                    // y_true is one-hot: only the true class contributes
                END FOR
   13.          batch_loss ← MEAN(loss_i for all i)

                // ── Backward Pass (Backpropagation) ──
   14.          COMPUTE gradients:
                    ∂(batch_loss) / ∂(w) for each trainable weight w
                    // Only classification head weights (NOT backbone)

                // ── Update Weights (Adam Optimizer) ──
   15.          FOR each trainable weight w DO
                    m_w ← β1 × m_w + (1 - β1) × gradient_w      // momentum
                    v_w ← β2 × v_w + (1 - β2) × gradient_w²     // velocity
                    m̂_w ← m_w / (1 - β1^t)                       // bias correction
                    v̂_w ← v_w / (1 - β2^t)
                    w ← w - learning_rate × m̂_w / (√v̂_w + ε)
                END FOR

   16.      END FOR  // end batch loop

            // ── Step 2b: Evaluate on validation set ──
   17.      val_predictions ← model.forward(D_val)     // no augmentation
   18.      val_loss ← MEAN( CrossEntropy(y_true, val_predictions) )
   19.      val_accuracy ← COUNT(correct predictions) / COUNT(total samples)

            // ── Step 2c: Callbacks ──

            // ModelCheckpoint
   20.      IF val_accuracy > best_val_accuracy THEN
   21.          best_val_accuracy ← val_accuracy
   22.          SAVE model to "best_model.keras"
   23.          epochs_without_improvement ← 0
   24.      ELSE
   25.          epochs_without_improvement ← epochs_without_improvement + 1
   26.      END IF

            // ReduceLROnPlateau
   27.      IF val_loss has not decreased for 3 consecutive epochs THEN
   28.          learning_rate ← learning_rate × 0.5
   29.          learning_rate ← MAX(learning_rate, 1e-6)
   30.      END IF

            // EarlyStopping
   31.      IF epochs_without_improvement >= patience THEN
   32.          PRINT "Early stopping at epoch {epoch}"
   33.          LOAD best saved model weights
   34.          BREAK
   35.      END IF

   36.  END FOR  // end epoch loop

    // ── Step 3: Save final artifacts ──
   37.  SAVE model as "final_model.keras"
   38.  CONVERT model to TFLite format → "sign_language_model.tflite"
   39.  PLOT training_accuracy, val_accuracy, training_loss, val_loss curves
   40.  SAVE class index mapping as "class_indices.json"

   41.  RETURN trained_model, training_history
END
```

---

## Algorithm 4: Real-time Webcam Prediction

```
Algorithm: REALTIME_PREDICT

Input:  trained_model        // From TRAIN_MODEL
        camera_id = 0        // Webcam device index
        confidence_threshold = 0.6
        roi_size = 300       // Region of interest box size (pixels)
        predict_interval = 3 // Predict every Nth frame

Output: On-screen display with predicted letter and confidence

BEGIN
    // ── Step 1: Initialize ──
    1.  LOAD trained_model from "best_model.keras"
    2.  OPEN webcam with camera_id
    3.  SET frame_width ← 1280, frame_height ← 720

    4.  COMPUTE ROI coordinates:
            roi_x1 ← (frame_width / 2) - (roi_size / 2) + 50
            roi_y1 ← (frame_height / 2) - (roi_size / 2)
            roi_x2 ← roi_x1 + roi_size
            roi_y2 ← roi_y1 + roi_size

    5.  SET current_label ← "?"
    6.  SET current_confidence ← 0.0
    7.  SET frame_count ← 0

    // ── Step 2: Main detection loop ──
    8.  WHILE True DO

    9.      READ frame from webcam
   10.      IF frame is empty THEN BREAK

            // Mirror for natural interaction
   11.      frame ← FLIP frame horizontally

            // Extract Region of Interest
   12.      roi ← frame[roi_y1 : roi_y2, roi_x1 : roi_x2]
                // Shape: (300, 300, 3) in BGR

   13.      frame_count ← frame_count + 1

            // ── Step 2a: Run prediction (every Nth frame) ──
   14.      IF frame_count MOD predict_interval == 0 THEN

                // Preprocess ROI
   15.          img ← RESIZE roi to (224, 224)
   16.          img ← CONVERT BGR to RGB
   17.          img ← RESHAPE to (1, 224, 224, 3)     // add batch dimension
   18.          img ← CAST to float32

                // Forward pass
   19.          probabilities ← model.predict(img)
                    // Shape: (1, 26)

                // Get best prediction
   20.          predicted_index ← ARGMAX(probabilities)
   21.          current_confidence ← probabilities[predicted_index]
   22.          current_label ← LABELS[predicted_index]
                    // LABELS = ['A', 'B', ..., 'Z']

                // Optional: Text-to-Speech
   23.          IF current_confidence >= confidence_threshold THEN
   24.              IF TTS_enabled AND (label changed OR cooldown expired) THEN
   25.                  SPEAK current_label aloud
   26.              END IF
   27.          END IF

   28.      END IF

            // ── Step 2b: Draw UI overlay ──
   29.      IF current_confidence >= confidence_threshold THEN
   30.          SET box_color ← GREEN
   31.      ELSE
   32.          SET box_color ← ORANGE
   33.      END IF

   34.      DRAW rectangle on frame at (roi_x1, roi_y1, roi_x2, roi_y2)
                with color = box_color
   35.      DRAW corner accent lines at all 4 corners of ROI

   36.      IF current_confidence >= confidence_threshold THEN
   37.          DRAW large letter current_label (top-right corner)
   38.          DRAW confidence bar (filled to current_confidence %)
   39.      ELSE
   40.          DRAW text "Detecting..." (top-right corner)
   41.      END IF

   42.      COMPUTE fps ← 1.0 / (current_time - previous_time)
   43.      DRAW FPS counter (top-left)
   44.      DRAW "Press Q to quit" (top-left)

            // ── Step 2c: Display and check for quit ──
   45.      DISPLAY frame in window "Sign Language Detector"

   46.      IF key pressed == 'Q' THEN
   47.          BREAK
   48.      END IF

   49.  END WHILE

    // ── Step 3: Cleanup ──
   50.  RELEASE webcam
   51.  CLOSE all windows
END
```

---

## Summary of Computational Complexity

| Algorithm | Time Complexity | Space Complexity |
|-----------|----------------|-----------------|
| Dataset Preparation | O(V × F) where V = videos, F = frames per video | O(F × img_size²) |
| Model Construction | O(1) — one-time setup | O(P) where P = 3.4M parameters |
| Training (per epoch) | O(N × C) where N = samples, C = model forward/backward cost | O(B × 224² × 3) where B = batch_size |
| Real-time Prediction | O(C) per frame — single forward pass | O(224² × 3) per frame |

> Where C (model inference cost) ≈ 300M multiply-add operations for MobileNetV2.
