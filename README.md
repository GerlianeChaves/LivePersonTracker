# <p align="center">Live Person Tracker</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.6%2B-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/OpenCV-4.x-green?style=for-the-badge&logo=opencv" alt="OpenCV">
  <img src="https://img.shields.io/badge/YOLOv3-Detection-red?style=for-the-badge" alt="YOLOv3">
</p>

## Features

- Real-time detection via webcam.  
- Processing of pre-recorded videos.  
- Visual proximity identification ("Close" in red and "Safe" in green).  
- Real-time visualization interface in OpenCV.  

---

##  Requirements

- Python 3.6+
- Install required dependencies via pip:
  ```bash
  pip install opencv-python numpy
  ```
## YOLO Files

Make sure the following files are available in the `yolo_files` directory:

- **yolov3.weights**: Contains the pre-trained weights of the YOLOv3 model.
- **yolov3.cfg**: YOLOv3 configuration file, defining the neural network architecture.
- **coco.names**: Text file containing the names of the 80 classes that the YOLOv3 model is capable of detecting.

## Project Structure
  ```bash
LivePersonTracker/
│
├── yolo_files/
│   ├── yolov3.weights
│   ├── yolov3.cfg
│   ├── coco.names
│
├── midia/
│   ├── walkpeople.mp4
│
├── camera.py
├── LiveDetection.py
├── PersonTracker.py
├── README.md
└── VideoDetection.py
  ```

## How to Use

Clone the repository and navigate to the directory:

  ```bash
git clone https://github.com/GerlianeChaves/LivePersonTracker.git
cd LivePersonTracker
  ```
Activate the virtual environment:
  ```bash
source LivePersonTracker/bin/activate 
  ```

## Run the script:

To interact with the webcam:
  ```bash
python camera.py
  ```

To detect people in real time via the webcam:
For video:
  ```bash
python LiveDetection.py
  ```

To detect people in a video (make sure you have a video on `midia` and that the name is correct in the code):
For video:
  ```bash
python VideoDetection.py
  ```

To detect if people are close or far from the camera's point of view:
For video:
  ```bash
python PersonTracker.py
  ```

Press `q` to exit the preview.

## Results

📌 Video Detection 

<img src="img/Gravação-de-tela-de-2025-03-21-21-34-52.gif" width="550">

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)