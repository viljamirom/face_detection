import cv2
import argparse
import time
import os


def parse_output_file(path, format):
    name = "latest_face_detection"
    output_file = path+name+format
    return output_file


def check_output_path(path):
    if os.path.isfile(path):
        raise Exception("Path points to file and not to directory")
    if not path[-1] == '/':
        path += '/'
    if not os.path.exists(path):
        raise Exception("Could not find given directory")
    return path


def main():
    parser = argparse.ArgumentParser(description='Detect faces from video feed and returns bounding box of biggest detection')
    parser.add_argument('--output', required=True,
                        help='Path to the output directory')
    args = parser.parse_args()
    output_path = check_output_path(args.output)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not open camera")

    frozen_graph = 'frozen_inference_graph.pb'
    text_graph = 'graph.pbtxt'
    net = cv2.dnn.readNetFromTensorflow(frozen_graph, text_graph)
    net_input_width = 240
    net_input_height = 180

    while True:
        _, img = cap.read()
        #img = cv2.flip(img, 1)  # might need to mirror the image

        blob = cv2.dnn.blobFromImage(img, size=(net_input_width, net_input_height), swapRB=True, crop=False)
        net.setInput(blob)
        netOut = net.forward()

        rows, cols = img.shape[0:2]
        max_area = 0

        default_target = [float(1 / 3), float(1 / 3), float(2 / 3), float(2 / 3)]   # desired default bounding box
        target_face = default_target

        for detection in netOut[0, 0, :, :]:
            score = float(detection[2])
            min_x = float(detection[3])
            min_y = float(detection[4])
            max_x = float(detection[5])
            max_y = float(detection[6])
            width = int(max_x * cols - min_x * cols)
            height = int(max_y * rows - min_y * rows)
            if score > 0.3 and width > 60:
                area = int(width * height)
                if area > max_area:
                    max_area = area
                    target_face = [min_x, min_y, max_x, max_y]

        # video feed from detection
        cv2.rectangle(img, (int(target_face[0] * cols), int(target_face[1] * rows)), (int(target_face[2] * cols), int(target_face[3] * rows)), (255, 0, 0), 2)
        cv2.imshow("image", img)

        file_to_save = parse_output_file(output_path, ".txt")
        with open(file_to_save, 'w') as file:
            file.write(";".join(str(x) for x in target_face))
            file.close()

        key = cv2.waitKey(1)  # note that if video feed window is not active, program must be stopped manually
        if key & 0xFF == ord('q'):
            break
        time.sleep(1)


if __name__ == '__main__':
    main()
