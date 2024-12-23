# YOLO Dog Breed Detection Web App

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Clone the Repository](#clone-the-repository)
  - [Set Up the Environment](#set-up-the-environment)
  - [Install Dependencies](#install-dependencies)
- [Usage](#usage)
  - [Running the Application](#running-the-application)
  - [Interacting with the Web App](#interacting-with-the-web-app)
- [Metrics Handling](#metrics-handling)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

The **YOLO Dog Breed Detection Web App** is a powerful and user-friendly tool designed to detect and classify dog breeds within uploaded images using the YOLO (You Only Look Once) object detection framework. Leveraging a pre-trained YOLO model with 120 dog breed classes, this web application provides real-time detection results, including bounding boxes, confidence scores, and comprehensive validation metrics.

## Features

- **Real-Time Detection**: Upload an image and receive instant detection results with bounding boxes and confidence scores.
- **Comprehensive Metrics**: View both overall and per-class validation metrics to assess model performance.
- **User-Friendly Interface**: Built with Gradio for an intuitive and interactive user experience.
- **Scalable Architecture**: Designed to handle a large number of classes efficiently.
- **High Accuracy**: Utilizes a robust YOLO model trained on a diverse dog breed dataset.

## Dataset

https://universe.roboflow.com/iliescu-mihail-doirn/stanford-dogs-dataset-dog-breed/dataset/1

## Demo

![Web App Demo](https://github.com/yourusername/your-repo-name/blob/main/assets/web_app_demo.gif?raw=true)

*Screenshot demonstrating the YOLO Dog Breed Detection Web App in action.*

## Installation

### Prerequisites

- **Python 3.10.12** or higher
- **CUDA 12.1** (for GPU acceleration) *(Optional but recommended for faster inference)*
- **Git** (to clone the repository)

### Clone the Repository

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

### Set Up the Environment

It's recommended to use a virtual environment to manage dependencies.

#### Using `venv`:

```bash
python -m venv venv
```

#### Activate the Virtual Environment:

- **Windows:**

  ```bash
  venv\Scripts\activate
  ```

- **macOS and Linux:**

  ```bash
  source venv/bin/activate
  ```

### Install Dependencies

Ensure you have `pip` updated to the latest version:

```bash
pip install --upgrade pip
```

Install the required Python packages using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

**Note:** The `torch` package specified in `requirements.txt` is built with CUDA 12.1 support. If your system has a different CUDA version or if you're using a CPU-only setup, adjust the PyTorch installation accordingly. Visit the [PyTorch Official Installation Page](https://pytorch.org/get-started/locally/) for the appropriate command.

For example, for CPU-only support:

```bash
pip install torch==2.5.1 torchvision==0.15.2 torchaudio==2.5.1
```

## Usage

### Running the Application

Ensure that the `best.pt` YOLO model file and the `data.yaml` configuration file are present in the project directory.

Execute the Python script to launch the web application:

```bash
python app.py
```

*Replace `app.py` with the actual filename if different.*

### Interacting with the Web App

1. **Access the Web Interface:**

   After running the script, Gradio will provide a local URL (e.g., `http://127.0.0.1:7860/`). Open this URL in your web browser.

2. **Upload an Image:**

   - Click on the "Upload Image" button.
   - Select an image of a dog from your local machine.

3. **Run Inference:**

   - Click the "Run Inference" button.
   - The application will process the image, detect dog breeds, and display the results.

4. **View Results:**

   - **Annotated Image:** Displays the uploaded image with bounding boxes and labels indicating detected breeds and confidence scores.
   - **Detection Results:** A table listing detected classes with their confidence scores and bounding box coordinates.
   - **Validation Metrics:** A table showing overall and per-class metrics to evaluate model performance.

## Metrics Handling

The application displays both **overall** and **per-class** metrics to provide a comprehensive view of the model's performance.

- **Overall Metrics:**
  - **Precision:** The ratio of true positive detections to the total predicted positives.
  - **Recall:** The ratio of true positive detections to the actual positives.
  - **mAP50:** Mean Average Precision at 50% IoU threshold.
  - **mAP50-95:** Mean Average Precision across IoU thresholds from 50% to 95%.

- **Per-Class Metrics:**
  - Detailed metrics for each of the 120 dog breed classes, including Precision, Recall, mAP50, and mAP50-95.

**Note:** Metrics are precomputed and hardcoded into the application for demonstration purposes. For real-world applications, consider implementing dynamic metric computation.

## Project Structure

```
your-repo-name/
│
├── app.py                   # Main application script
├── best.pt                  # Pre-trained YOLO model
├── data.yaml                # YOLO data configuration file
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
```

## Troubleshooting

- **Model File Not Found:**

  Ensure that `best.pt` is located in the project directory or update the `model_path` variable in `app.py` to the correct location.

- **CUDA Errors:**

  If you encounter CUDA-related errors, verify that your system has the appropriate CUDA version installed. If not using a GPU, adjust the PyTorch installation to CPU-only as mentioned in the [Installation](#installation) section.

- **Dependencies Issues:**

  Ensure all dependencies are installed correctly. You can reinstall them using:

  ```bash
  pip install --force-reinstall -r requirements.txt
  ```

- **Gradio Not Launching:**

  Check if the required ports are free and not blocked by firewalls. You can specify a different port by modifying the `demo.launch()` line:

  ```python
  demo.launch(server_port=7861)
  ```

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the Repository**

2. **Create a Feature Branch**

   ```bash
   git checkout -b feature/YourFeatureName
   ```

3. **Commit Your Changes**

   ```bash
   git commit -m "Add your feature"
   ```

4. **Push to the Branch**

   ```bash
   git push origin feature/YourFeatureName
   ```

5. **Open a Pull Request**

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For any questions or suggestions, please contact:
- **Name:** Partha Pratim Ray
- **Email:** parthapratimray1986@gmail.com

---

## Hugging Face Space

https://huggingface.co/spaces/csepartha/yolo11_stanford_dog_detection
*Happy Detecting! 🐶🔍*
