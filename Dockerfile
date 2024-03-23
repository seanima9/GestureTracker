# Use the official TensorFlow GPU image as the base image
FROM tensorflow/tensorflow:latest-gpu-jupyter

WORKDIR /gesture_tracker

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt && apt-get update && apt-get install -y libgl1-mesa-glx

EXPOSE 8888

ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''", "--no-browser"]