# Face detection
Face detection for deep speaking avatar, part of Bsc thesis project.\
Example for this project and the used models can be found [here](https://github.com/mahehu/TUT-live-age-estimator).

# How to use
Dependencies: OpenCV - 4.5.1+ (Used in tests)
\
\
Input: Path to the image file (tested with jpg, png). All supported formats can be found in OpenCv - imread() - [documentation](https://docs.opencv.org/master/d4/da8/group__imgcodecs.html).
\
\
Output: Desired path for the text (.txt) file. Creates bounding box coordinates of the biggest detected face in form of: min_x;min_y;max_x;max_y. 
```
usage: detect_faces.py [-h] --input INPUT --output OUTPUT

Detect faces from image and returns bounding box of biggest detection

optional arguments:
  -h, --help       show this help message and exit
  --input INPUT    Path to the image file
  --output OUTPUT  Path to the output text file
 ```

