import numpy as np

class VisionHARClassifier:
    def __init__(self):
        # Activity names
        self.classes = ["Standing", "Sitting", "Squatting", "Walking", "Laying"]
        # To smooth predictions
        self.history = []
        self.history_len = 10

    def calculate_angle(self, a, b, c):
        """
        Calculate the angle between three points.
        a, b, c are [x, y, z] coordinates. b is the vertex.
        """
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle

    def classify(self, landmarks):
        """
        landmarks: MediaPipe Tasks pose landmarks (list of landmark objects).
        Returns predicted class, confidence, and probability dictionary.
        """
        if not landmarks:
            return "No Person", 0.0, {c: 0.0 for c in self.classes}

        try:
            # Coordinates for Left side
            l_shoulder = [landmarks[11].x, landmarks[11].y]
            l_hip = [landmarks[23].x, landmarks[23].y]
            l_knee = [landmarks[25].x, landmarks[25].y]
            l_ankle = [landmarks[27].x, landmarks[27].y]
            
            # Coordinates for Right side
            r_shoulder = [landmarks[12].x, landmarks[12].y]
            r_hip = [landmarks[24].x, landmarks[24].y]
            r_knee = [landmarks[26].x, landmarks[26].y]
            r_ankle = [landmarks[28].x, landmarks[28].y]

            # Calculate Angles
            l_knee_angle = self.calculate_angle(l_hip, l_knee, l_ankle)
            r_knee_angle = self.calculate_angle(r_hip, r_knee, r_ankle)
            l_hip_angle = self.calculate_angle(l_shoulder, l_hip, l_knee)
            r_hip_angle = self.calculate_angle(r_shoulder, r_hip, r_knee)

            # Heuristic Probabilities (simulated for UI)
            probs = {c: 0.05 for c in self.classes} # Baseline
            
            # Laying Logic
            if abs(l_shoulder[1] - l_hip[1]) < 0.1 and abs(l_hip[1] - l_knee[1]) < 0.1:
                probs["Laying"] = 0.9
            
            # Sitting Logic
            elif (90 < l_knee_angle < 130 or 90 < r_knee_angle < 130) and \
                 (80 < l_hip_angle < 120 or 80 < r_hip_angle < 120):
                probs["Sitting"] = 0.85
                
            # Squatting Logic
            elif (l_knee_angle < 80 or r_knee_angle < 80):
                probs["Squatting"] = 0.85
            
            # Default to Standing
            else:
                probs["Standing"] = 0.8

            # Normalize probabilities
            total = sum(probs.values())
            probs = {k: v/total for k, v in probs.items()}

            # Prediction is the one with max prob
            prediction = max(probs, key=probs.get)
            confidence = probs[prediction]

            # Smoothing
            self.history.append(prediction)
            if len(self.history) > self.history_len:
                self.history.pop(0)
            
            final_pred = max(set(self.history), key=self.history.count)
            
            return final_pred, confidence, probs

        except Exception as e:
            return f"Error: {str(e)}", 0.0, {c: 0.0 for c in self.classes}
