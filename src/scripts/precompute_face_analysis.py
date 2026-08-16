import os
import glob
import cv2
import json
import numpy as np
from deepface import DeepFace

def main():
    base_dir = r"c:\Users\Vitoria\Documents\GitHub\REPO_ON_WORKING\dataScience\computerVision\src\assets\images_face_analysis"
    annotated_dir = os.path.join(base_dir, "annotated")
    
    if not os.path.exists(annotated_dir):
        os.makedirs(annotated_dir)
        
    image_paths = glob.glob(os.path.join(base_dir, "*.jpg"))
    results_dict = {}
    
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        print(f"Processing {filename}...")
        
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        try:
            results = DeepFace.analyze(img, actions=['emotion', 'age', 'gender'], enforce_detection=True, detector_backend='mtcnn', silent=True)
            if not isinstance(results, list):
                results = [results]
                
            face_data = []
            annotated_img = img.copy()
            
            for i, face in enumerate(results):
                region = face.get('region', {})
                x = region.get('x', 0)
                y = region.get('y', 0)
                w = region.get('w', 0)
                h = region.get('h', 0)
                
                if w > 0 and h > 0:
                    cv2.rectangle(annotated_img, (x, y), (x+w, y+h), (0, 255, 255), 2)
                    emotion = face.get('dominant_emotion', 'Unknown')
                    age = face.get('age', '?')
                    gender_data = face.get('gender', {})
                    if isinstance(gender_data, dict):
                        gender = max(gender_data.items(), key=lambda item: item[1])[0] if gender_data else 'Unknown'
                    else:
                        gender = gender_data
                        
                    info_text = f"{gender}, {age} | {emotion}"
                    (text_w, text_h), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(annotated_img, (x, y - text_h - 10), (x + text_w, y), (0, 255, 255), -1)
                    cv2.putText(annotated_img, info_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    
                face_data.append({
                    "age": face.get('age'),
                    "gender": gender,
                    "dominant_emotion": face.get('dominant_emotion'),
                    "emotion_probs": face.get('emotion', {})
                })
                
            # Save annotated image
            out_path = os.path.join(annotated_dir, filename)
            cv2.imwrite(out_path, annotated_img)
            
            # Save data
            results_dict[filename] = face_data
            
        except ValueError:
            print(f"No face detected in {filename}")
            # Save original image as annotated if no face is found
            out_path = os.path.join(annotated_dir, filename)
            cv2.imwrite(out_path, img)
            results_dict[filename] = [] # Empty results
            
        except Exception as e:
            print(f"Failed to process {filename}: {e}")
            
    # Save JSON report
    with open(os.path.join(annotated_dir, "report.json"), "w") as f:
        json.dump(results_dict, f, indent=4)
        
    print("Done!")

if __name__ == "__main__":
    main()
