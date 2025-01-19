#imports
import cv2
import numpy as np

#yolo files
net = cv2.dnn.readNet('yolo_files/yolov3.weights', 'yolo_files/yolov3.cfg')
layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

#init camera
video = 'midia/walkpeople.mp4'
cam = cv2.VideoCapture(video)
if not cam.isOpened():
    print("Erro ao abrir o arquivo de vídeo.")
    exit()

#resize
resize_width = 800
resize_height = 800

while True:
    ret, frame = cam.read()

    if not ret:
        break

    #resize frame
    frame = cv2.resize(frame, (resize_width, resize_height))

    #preparar imagem pro yolo
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    #processar detecção
    height, width, channels = frame.shape
    class_ids = []
    confidences = []
    boxes = []

    for out in outputs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5: #(50%)
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)

                w = int(detection[2]*width)
                h = int(detection[3]*height)

                #boxe
                x = int(center_x-w/2)
                y = int(center_y-h/2)

                boxes.append([x, y,w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    #eliminar caixa sobreposta
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    #boxes draw
    for i in indexes:
        x, y, w, h = boxes[i]
        area = w*h  

        proximity = 20000

        if area > proximity:
            color = (0, 0, 255)  # Vermelho
            label = "Collision Risk"
        else:
            color = (0, 255, 0)  # Verde
            label = "Safe"

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    #show video
    cv2.imshow('Person Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()



