import cv2
import numpy as np

def preprocess_image(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detect edges using Canny
    edges = cv2.Canny(blurred, 50, 150)
    
    return edges

def apply_morphological_operations(edges):
    # Apply closing to fill gaps in the edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    return closed

def detect_axles(image):
    edges = preprocess_image(image)
    closed = apply_morphological_operations(edges)
    
    # Find contours in the closed image
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    axle_contours = []
    for contour in contours:
        # Approximate the contour
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        
        # A heuristic to filter possible axles: Adjust the area and shape criteria as needed
        area = cv2.contourArea(contour)
        if len(approx) > 4 and 1000 < area < 10000:
            axle_contours.append(contour)
    
    # Draw contours on the original image
    cv2.drawContours(image, axle_contours, -1, (0, 255, 0), 2)
    
    axle_count = len(axle_contours)
    return image, axle_count

def process_image(image_path):
    image = cv2.imread(image_path)
    result, axle_count = detect_axles(image)
    
    # Display the axle count on the image
    cv2.putText(result, f'Axles Count: {axle_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    cv2.imshow('Detected Axles', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        result, axle_count = detect_axles(frame)
        
        # Display the axle count on the frame
        cv2.putText(result, f'Axles Count: {axle_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        cv2.imshow('Detected Axles', result)
        print(axle_count)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def process_live_camera():
    cap = cv2.VideoCapture(1)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        result, axle_count = detect_axles(frame)
        
        # Display the axle count on the frame
        cv2.putText(result, f'Axles Count: {axle_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        cv2.imshow('Detected Axles', result)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Example usage
#process_image('5.jpg')
process_video('out.mp4')
#process_live_camera()
