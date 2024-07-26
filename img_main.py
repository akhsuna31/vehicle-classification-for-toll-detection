import cv2
import numpy as np
from ultralytics import YOLO
import imutils

# Load YOLOv8 nano model for vehicle detection
model = YOLO("yolov8n.pt")  # Ensure you have the YOLOv8 nano model file

# Function to detect vehicles
def detect_vehicles(image):
    results = model(image)  # Inference with YOLOv8 nano
    vehicles = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf.cpu().numpy())
            cls = int(box.cls.cpu().numpy())
            if cls in [2, 3, 5, 7, 8]:  # Car, motorcycle, bus, truck, bicycle
                vehicles.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1), cls, conf))
    return vehicles

# Function to detect license plate region
def detect_license_plate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)
    
    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    
    license_plate = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            license_plate = (x, y, w, h)
            break
    
    return license_plate

# Main function to process an image
def process_image(image_path):
    image = cv2.imread(image_path)
    vehicles = detect_vehicles(image)
    
    for (x, y, w, h, cls, conf) in vehicles:
        label = f"{model.names[cls]}: {conf:.2f}"
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        vehicle_img = image[y:y + h, x:x + w]
        license_plate = detect_license_plate(vehicle_img)
        if license_plate:
            (lp_x, lp_y, lp_w, lp_h) = license_plate
            cv2.rectangle(vehicle_img, (lp_x, lp_y), (lp_x + lp_w, lp_y + lp_h), (255, 0, 0), 2)
    
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
process_image("img2.jpg")
