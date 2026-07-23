# Transformer-based Human Activity Recognition (HAR)

Owner: Vikash

A Python project for Human Activity Recognition using smartphone sensor data and a Transformer-based neural network, along with a live vision-based activity dashboard.

## Project Overview

This repository combines two HAR workflows:

1. Sensor-based HAR using the UCI HAR dataset and a PyTorch Transformer model.
2. Vision-based HAR using MediaPipe pose landmark estimation and heuristic activity classification in real time.

The project is designed for experimentation, training, evaluation, and interactive visualization of activity recognition systems.

## Features

- Transformer-based model for sequence classification
- UCI HAR dataset download and preprocessing pipeline
- Training script with logging and checkpointing
- Evaluation pipeline with metrics and confusion matrix output
- Interactive random-sample inference script
- Streamlit dashboard for sensor analysis
- Live webcam-based activity HUD with pose visualization

## Repository Structure

- `app.py` — Streamlit dashboard for sensor-based HAR prediction and signal visualization
- `data_processing.py` — dataset download, extraction, loading, and feature normalization
- `model.py` — HAR Transformer model definition
- `train.py` — training loop for the sensor model
- `evaluate.py` — evaluation metrics and confusion matrix generation
- `predict.py` — interactive inference with random samples
- `vision_app.py` — real-time webcam activity recognition UI
- `vision_classifier.py` — pose landmark angle-based heuristic activity classifier
- `requirements.txt` — Python package dependencies
- `Dockerfile` — container setup for deployment
- `packages.txt` — system-level packages for OpenCV/MediaPipe support

## Dataset

The sensor model uses the UCI Human Activity Recognition dataset from the UCI Machine Learning Repository:

- URL: https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip

The dataset is automatically downloaded and extracted by `data_processing.py` when needed.

## Model Summary

The sensor model uses:

- input sequence length: 128
- input features: 9 sensor streams
- sequence model: Transformer Encoder
- classification head: fully connected classifier
- output classes: 6 activities

Activities used in the pipeline:

1. Walking
2. Walking Upstairs
3. Walking Downstairs
4. Sitting
5. Standing
6. Laying

## Environment Setup

### Prerequisites

- Python 3.10+
- pip
- Optional: CUDA-enabled GPU for faster training

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Optional System Packages

For Linux environments, install the OS packages listed in `packages.txt` before running OpenCV/MediaPipe-based features.

## Running the Sensor HAR Pipeline

### 1. Train the Model

```bash
python train.py
```

This script:

- downloads and loads the data
- initializes the Transformer model
- trains for the configured number of epochs
- saves the best model to `best_model.pth`

### 2. Evaluate the Model

```bash
python evaluate.py
```

This command prints accuracy, precision, recall, and a classification report, and saves a confusion matrix image.

### 3. Run Interactive Prediction

```bash
python predict.py
```

This launches an interactive prompt for sampling random test inputs and observing predicted labels.

### 4. Open the Sensor Dashboard

```bash
streamlit run app.py
```

This opens a dashboard where you can inspect predictions, confidence scores, and signal plots for individual samples.

## Running the Vision-Based HAR App

```bash
streamlit run vision_app.py
```

This starts a live webcam activity recognition interface that:

- captures frames from a camera
- runs pose landmark detection via MediaPipe
- overlays pose skeletons
- predicts posture-related activity classes
- displays a confidence radar visualization

## Docker Usage

A Dockerfile is included for containerized deployment.

### Build the image

```bash
docker build -t transformer-har .
```

### Run the container

```bash
docker run -p 8501:8501 transformer-har
```

The container launches the vision app by default.

## Notes and Limitations

- The sensor workflow is the primary trained model pipeline.
- The vision workflow is heuristic-based and depends on pose estimation quality.
- The vision app uses a different activity class set from the sensor Transformer model.
- The project expects the dataset to be downloaded automatically when not present locally.
- Model performance depends on the quality of the dataset split and the training configuration.

## Suggested Improvements

- Use a proper train/validation/test split instead of reusing the test set for validation
- Add logging and experiment tracking
- Save training curves and metrics to a file
- Add support for a richer vision classifier trained on labeled image or pose data
- Improve code modularity and shared configuration handling

## License

This project is provided as-is for educational and research purposes.
