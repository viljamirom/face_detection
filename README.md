# Face detection
Face detection for deep speaking avatar, part of Bsc thesis project.\
Example for this project and the used models can be found [here](https://github.com/mahehu/TUT-live-age-estimator).

# How to use
Dependencies: OpenCV - 4.5.1+ (Used in tests)
\
\
Requires a webcam for the video feed. Documentation of the videocapture can be found [here](https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html#details).
\
\
Output: Desired directory path for the detections. Saves the detections in text (.txt) file to the given directory. Creates bounding box coordinates of the biggest detected face in form of: min_x;min_y;max_x;max_y.
Note the coordinate system:
\
\
![Image](https://i.stack.imgur.com/t4AiI.png)

```
usage: detect_faces.py [-h] --output OUTPUT

Detect faces from video feed and returns bounding box of biggest detection

optional arguments:
  -h, --help       show this help message and exit
  --output OUTPUT  Path to the output directory

 ```

