import cv2
import argparse
import os

def extension_check(file, extension, file_use):
    if not file.endswith(extension):
        raise argparse.ArgumentTypeError('{} file must be of type {}'.format(file_use, extension))
    return file

def main():
    parser = argparse.ArgumentParser(description='Detect faces from image and return bounding box of biggest detection')
    parser.add_argument('--input', required=True,
                        help='Path to the image file')
    parser.add_argument('--output', required=True,
                        help='Path to the output text file',
                        type=lambda f: extension_check(f, '.txt', 'Output'))

    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(f'"{args.input}" does not point to a file')

    frozen_graph = 'frozen_inference_graph.pb'
    text_graph = 'graph.pbtxt'
    net = cv2.dnn.readNetFromTensorflow(frozen_graph, text_graph)

    img = cv2.imread(args.input)
    net_input_width = 240
    net_input_height = 180

    blob = cv2.dnn.blobFromImage(img, size=(net_input_width, net_input_height), swapRB=True, crop=False)
    net.setInput(blob)
    netOut = net.forward()

    rows, cols = img.shape[0:2]
    max_area = 0
    default_target = [int(cols / 3), int(rows / 3), int(2 * cols / 3), int(2 * rows / 3)]   # desired default bounding box
    target_face = default_target

    for detection in netOut[0, 0, :, :]:
        score = float(detection[2])
        min_x = int(detection[3] * cols)
        min_y = int(detection[4] * rows)
        max_x = int(detection[5] * cols)
        max_y = int(detection[6] * rows)
        width = max_x - min_x
        height = max_y - min_y
        if score > 0.3 and width > 60:
            area = int(width*height)
            if area > max_area:
                max_area = area
                target_face = [min_x, min_y, max_x, max_y]
    with open(args.output, 'w') as file:
        file.write(";".join(str(x) for x in target_face))

if __name__ == '__main__':
    main()