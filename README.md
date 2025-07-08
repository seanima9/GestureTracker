# Gesture Tracker 🫴

Gesture Tracker is a Python-based application that leverages computer vision and machine learning to track and recognize hand gestures in real-time. By using a camera, users can interact with their computers through intuitive hand movements, making it an innovative approach for system interaction.

## Features

- Real-time hand gesture recognition with the Mediapipe library.
- Control of the mouse cursor based on the index finger's position.
- Executes specific computer commands like mouse clicks and keyboard presses based on recognized gestures.
- Customizable gesture recognition model and action mappings.
- Suitable for a variety of applications including virtual reality, touchless interfaces, accessibility, presentations, and gaming.

## Installation

To set up the Gesture Tracker on your system, follow these steps:

1. Clone the repository:
```bash
git clone https://github.com/seanima9/GestureTracker
```
2. Navigate to the project directory:
```bash
cd gesture-tracker
```
3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

A pre-trained model is already included in the models directory. You can start using Gesture Tracker immediately by running stream.py:
```bash
python main.py
```

### Data Preparation

If you want to train your own model, follow these steps:

1. Place your raw video files in the `data/raw_videos` directory. Each video file should be named according to its corresponding gesture category.
2. Run `data_processing.py` to preprocess the videos, extract frames, detect hand landmarks, and save the processed data:
```bash
python data_processing.py
```

### Model Training

1. Open `model_training.ipynb` in Jupyter or any compatible environment.
2. Execute the notebook cells sequentially to load the processed data, define the model architecture, train the model, and evaluate its performance.
3. The trained model will be saved in the `models` directory.

### Real-time Gesture Recognition

1. Ensure a camera is connected to your system.
2. Run `main.py` to initiate real-time gesture recognition:
```bash
python main.py
```
3. A window displaying the camera feed with detected hand landmarks will appear.
4. Perform gestures in front of the camera to see the corresponding actions executed based on recognized gestures.
5. Press 'q' to exit the application.

## Configuration

- The `data_processing.py` script includes variables for specifying paths to raw videos, processed data, and split data directories.
- The `model_training.ipynb` notebook allows adjustments to the model architecture, training parameters, and hyperparameters.
- Modify the `stream.py` script to change gesture-action mappings according to your needs.
