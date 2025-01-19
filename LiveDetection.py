#imports
import cv2
import numpy as np

#yolo files
net = cv2.dnn.readNet('yolo_files/yolov3.weights', 'yolo_files/yolov3.cfg')
layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

#init camera
cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()

    if not ret:
        break

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
    for i in range(len(boxes)):
        for i in indexes:
            x, y, w, h = boxes[i]
            label = str(class_ids[i])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    #show video
    cv2.imshow('Person Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()



