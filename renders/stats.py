import csv
import os

import cv2

captures = ["000030000", "000070000", "000110000", "000150000", "000190000","000230000"]

MAX_ACCUM = 1920 * 1080 * 255

accum_matrix = []

for c in captures:
    video_accum = cv2.VideoCapture(f'step-{c}-accum.mp4')
    print(f'Analyzing {c}...')
    steps = []
    for frame_num in range(0, int(video_accum.get(cv2.CAP_PROP_FRAME_COUNT)), int(video_accum.get(cv2.CAP_PROP_FPS))):
        video_accum.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        _, frame_accum = video_accum.read()
        # frame is a numpy array, convert to PIL Image
        frame_accum = cv2.cvtColor(frame_accum, cv2.COLOR_BGR2GRAY)
        # print frame values
        sum = 0
        for row in frame_accum:
            for pixel in row:
                sum += pixel
        steps.append(sum/MAX_ACCUM)
    accum_matrix.append(steps)

isExist = os.path.exists("images")
if not isExist:
    os.mkdir("images")

frame_to_image = 24
for c in captures:
    video_accum = cv2.VideoCapture(f'step-{c}-accum.mp4')
    video = cv2.VideoCapture(f'step-{c}.mp4')
    video_accum.set(cv2.CAP_PROP_POS_FRAMES, frame_to_image)
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_to_image)
    _, frame = video.read()
    _, frame_accum = video_accum.read()
    cv2.imwrite(f"images/{c}frame{frame_to_image}.jpg", frame)
    cv2.imwrite(f"images/{c}frame{frame_to_image}-accum.jpg", frame_accum)

print(accum_matrix)
with open("metrics.csv", "w") as file:
    writer = csv.writer(file)
    writer.writerow(["File"] + [f"Frame {i + 1}" for i in range(0, len(accum_matrix[0]))])
    for i, steps in enumerate(accum_matrix):
        writer.writerow([captures[i]] + steps)
