# Fall Detection System

A computer vision-based system for detecting falls and distinguishing them from normal physical activities like push-ups. This project uses YOLO object detection, pose estimation, and activity recognition to identify potential fall incidents.

## Project Overview

This system monitors video feeds to detect people, analyze their body positions and movements, and classify activities. When a fall is detected, the system can provide alerts. The project uses YOLO11 for object detection, OpenCV for image processing, and deep learning for activity recognition.

The system is built with a FastAPI backend, allowing for easy integration with other services and front-end applications.

## Features

- Real-time person detection
- Pose estimation for skeletal tracking
- Activity classification (falls vs. exercise activities)
- Customizable alert thresholds
- Support for camera integration
- RESTful API endpoints for integration with other services
- Database storage for event logging and analysis

## Environment Setup

### Requirements

This project requires Python 3.12 and Anaconda or Miniconda to manage dependencies.

### Installation

#### Option 1: Using Anaconda (Recommended)

1. Clone the repository:

   ```
   git clone https://github.com/yourusername/fallsafei-server-app.git
   cd fallsafei-server-app
   ```

2. Create a conda environment:

   ```
   conda create -n cv_env python=3.12
   conda activate cv_env
   ```

3. Install required packages:
   ```
   conda install -c conda-forge opencv
   conda install pytorch torchvision -c pytorch
   conda install -c conda-forge matplotlib
   pip install ultralytics
   ```

#### Option 2: Using Python venv (Without Anaconda)

1. Clone the repository:

   ```
   git clone https://github.com/yourusername/fallsafei-server-app.git
   cd fallsafei-server-app
   ```

2. Create a virtual environment:

   ```
   python -m venv venv
   ```

3. Activate the virtual environment:

   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```

4. Install required packages:

   ```
   # Computer vision dependencies
   pip install opencv-python
   pip install torch torchvision
   pip install matplotlib
   pip install ultralytics
   pip install scikit-image

   # API and backend dependencies
   pip install fastapi>=0.68.0
   pip install uvicorn>=0.15.0
   pip install sqlalchemy
   pip install alembic
   pip install pydantic
   pip install python-dotenv
   pip install psycopg2-binary
   pip install pydantic-settings
   ```

Note: Some packages might have complex dependencies and could be more difficult to install with pip alone compared to Anaconda, especially on Windows systems.

## Project Structure

```
fallsafei-server-app/
├── alembic/                 # Database migration tools
├── app/                     # Application code
├── configs/                 # Configuration files
├── data/                    # Dataset storage
├── models/                  # Model weights
├── notebooks/               # Jupyter notebooks
├── outputs/                 # Generated outputs
├── src/                     # Source code
├── tests/                   # Test files
├── .env                     # Environment variables
├── alembic.ini              # Alembic configuration
└── README.md                # This file
```

## Usage

1. Activate the environment:

   - If using Anaconda:
     ```
     conda activate cv_env
     ```
   - If using venv:
     - Windows: `venv\Scripts\activate`
     - macOS/Linux: `source venv/bin/activate`

2. Run the API server:

   ```
   uvicorn app.main:app --reload
   ```

3. Access the API documentation:
   ```
   Open your browser and navigate to http://localhost:8000/docs
   ```

## Development

### Testing the YOLO Model

You can quickly test the YOLOv8 model with:

```python
from ultralytics import YOLO

# Load a pretrained model
model = YOLO('yolo11n.pt')

# Run inference on an image
results = model('path_to_your_image.jpg')

# Process results (see documentation for details)
```

### Adding Custom Training

To train the model on your own dataset:

1. Organize your data in the appropriate format
2. Update the configuration files
3. Run the training script

## License

[Your License Here]

## Contributing

[Contribution guidelines]

## Contact

[Your Contact Information]
