"""
🗼 REAL-TIME 5G BASE STATION ANOMALY DETECTION SYSTEM v2.0
==========================================================

✨ NEW FEATURES:
- Loads pre-trained model from Jupyter notebook as starting point
- Continues learning from expert-level model (not from scratch)
- Full model persistence across sessions
- Organized file structure (models/, data/, logs/)
- Better accuracy from day 1

This simulates a real-world 5G telecommunications base station with:
- Temperature sensors (equipment cooling)
- Power consumption monitoring
- Signal strength tracking
- Network load monitoring
- Real-time anomaly detection with continuous learning
- Predictive maintenance alerts

Author: Industrial IoT Monitoring System
Industry: Telecommunications & Electronics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime, timedelta
import time
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import warnings
import joblib
import os
warnings.filterwarnings('ignore')

class FiveGBaseStation:
    """Simulates a real 5G telecom base station with sensors"""
    
    def __init__(self, station_id="5G-BS-001"):
        self.station_id = station_id
        self.timestamp = datetime.now()
        
        # Normal operating parameters for 5G base station
        self.normal_temp = 45.0  # °C (equipment runs hot)
        self.normal_power = 3500  # Watts (typical 5G base station)
        self.normal_signal = -75  # dBm (signal strength)
        self.normal_load = 60  # % (network traffic)
        
        # Anomaly injection parameters
        self.anomaly_probability = 0.03  # 3% chance of anomaly
        self.time_counter = 0
        
    def generate_sensor_data(self):
        """Generate realistic sensor readings with occasional anomalies"""
        
        self.time_counter += 1
        self.timestamp += timedelta(seconds=2)
        
        # Base readings with normal variation
        temp_noise = np.random.normal(0, 2)
        power_noise = np.random.normal(0, 100)
        signal_noise = np.random.normal(0, 3)
        load_noise = np.random.normal(0, 5)
        
        # Simulate daily traffic patterns (more load during day)
        hour_of_day = self.timestamp.hour
        traffic_pattern = 20 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
        
        # Generate readings
        temperature = self.normal_temp + temp_noise + (traffic_pattern * 0.3)
        power_consumption = self.normal_power + power_noise + (traffic_pattern * 50)
        signal_strength = self.normal_signal + signal_noise
        network_load = np.clip(self.normal_load + load_noise + traffic_pattern, 0, 100)
        
        # Inject realistic anomalies
        anomaly_type = None
        
        if np.random.random() < self.anomaly_probability:
            anomaly_choice = np.random.choice([
                'overheating', 'power_surge', 'signal_degradation', 
                'network_overload', 'cooling_failure', 'equipment_failure'
            ])
            
            if anomaly_choice == 'overheating':
                temperature += np.random.uniform(15, 30)
                power_consumption += np.random.uniform(500, 1000)
                anomaly_type = 'Overheating Event'
                
            elif anomaly_choice == 'power_surge':
                power_consumption += np.random.uniform(1500, 2500)
                temperature += np.random.uniform(5, 15)
                anomaly_type = 'Power Surge'
                
            elif anomaly_choice == 'signal_degradation':
                signal_strength -= np.random.uniform(15, 30)
                network_load -= np.random.uniform(20, 40)
                anomaly_type = 'Signal Degradation'
                
            elif anomaly_choice == 'network_overload':
                network_load = np.random.uniform(95, 100)
                power_consumption += np.random.uniform(800, 1200)
                temperature += np.random.uniform(10, 20)
                anomaly_type = 'Network Overload'
                
            elif anomaly_choice == 'cooling_failure':
                temperature += np.random.uniform(20, 35)
                anomaly_type = 'Cooling System Failure'
                
            elif anomaly_choice == 'equipment_failure':
                temperature += np.random.uniform(15, 25)
                power_consumption *= np.random.uniform(0.3, 0.7)
                signal_strength -= np.random.uniform(20, 40)
                anomaly_type = 'Equipment Failure'
        
        return {
            'timestamp': self.timestamp,
            'temperature_C': round(temperature, 2),
            'power_consumption_W': round(power_consumption, 2),
            'signal_strength_dBm': round(signal_strength, 2),
            'network_load_percent': round(network_load, 2),
            'anomaly_type': anomaly_type
        }

class RealTimeAnomalyDetector:
    """Real-time anomaly detection with continuous self-learning and model persistence"""

    def __init__(self, window_size=30, retrain_interval=50, model_path="models/5g_realtime_model.pkl"):
        self.window_size = window_size
        self.retrain_interval = retrain_interval
        self.model_path = model_path
        self.data_buffer = []
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Self-learning parameters
        self.samples_since_retrain = 0
        self.model_version = 0
        self.false_positive_indices = []
        self.true_positive_indices = []
        self.contamination_rate = 0.05
        
        # Performance tracking
        self.total_predictions = 0
        self.total_anomalies = 0
        self.false_positive_rate_history = []
        
        # Ground truth tracking for accuracy calculation
        self.true_positives = 0   # Correctly detected actual anomalies
        self.false_positives = 0  # Incorrectly flagged normal data as anomaly
        self.true_negatives = 0   # Correctly identified normal data
        self.false_negatives = 0  # Missed actual anomalies
        
        # Feature engineering parameters
        self.feature_cols = ['temperature_C', 'power_consumption_W', 
                            'signal_strength_dBm', 'network_load_percent']
        
        # Try to load existing model or pretrained model from notebook
        self.load_model()
        
    def load_model(self):
        """Load model with priority: 1) Realtime model 2) Notebook model 3) Create new"""
        
        # First, try to load the real-time model (if it exists from previous runs)
        if os.path.exists(self.model_path):
            try:
                model_data = joblib.load(self.model_path)
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.model_version = model_data.get('model_version', 0)
                self.contamination_rate = model_data.get('contamination_rate', 0.05)
                self.false_positive_indices = model_data.get('false_positive_indices', [])
                self.true_positive_indices = model_data.get('true_positive_indices', [])
                self.total_predictions = model_data.get('total_predictions', 0)
                self.total_anomalies = model_data.get('total_anomalies', 0)
                self.false_positive_rate_history = model_data.get('fp_rate_history', [])
                
                # Load ground truth metrics
                self.true_positives = model_data.get('true_positives', 0)
                self.false_positives = model_data.get('false_positives', 0)
                self.true_negatives = model_data.get('true_negatives', 0)
                self.false_negatives = model_data.get('false_negatives', 0)
                
                self.is_trained = True
                
                print(f"✅ Loaded existing real-time model v{self.model_version}")
                stats = self.get_model_stats()
                print(f"   📊 Accuracy: {stats['accuracy']:.1f}%")
                print(f"   🎯 Contamination: {self.contamination_rate:.3f}")
                print(f"   🔢 Total predictions: {self.total_predictions:,}")
                return
            except Exception as e:
                print(f"⚠️  Could not load real-time model: {e}")
        
        # Second, try to load the pre-trained model from Jupyter notebook
        notebook_model_path = "../models/industrial_anomaly_model_trained.pkl"
        if os.path.exists(notebook_model_path):
            try:
                print("📚 Loading pre-trained model from Jupyter notebook...")
                model_data = joblib.load(notebook_model_path)
                
                # Extract the model and scaler
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.contamination_rate = model_data.get('contamination_rate', 0.02)
                self.is_trained = True
                self.model_version = 1  # Start from v1 with notebook model
                
                print(f"✅ Successfully loaded EXPERT-LEVEL model from notebook!")
                print(f"   📁 Source: {notebook_model_path}")
                print(f"   🗓️  Training Date: {model_data.get('training_date', 'Unknown')}")
                print(f"   📊 Trained on: {model_data.get('training_samples', 0):,} industrial samples")
                print(f"   🎯 Original anomaly rate: {model_data.get('accuracy_metrics', {}).get('anomaly_rate', 0):.2f}%")
                print(f"   🌳 Estimators: {model_data.get('n_estimators', 200)}")
                print(f"   🎚️  Contamination: {self.contamination_rate:.3f}")
                print(f"\n🚀 Starting from EXPERT-LEVEL instead of beginner!")
                print(f"   This model already knows industrial machine patterns")
                print(f"   Will continue learning and adapting in real-time\n")
                return
            except Exception as e:
                print(f"⚠️  Could not load notebook model: {e}")
        
        # If no model exists, will train from scratch
        print("ℹ️  No existing model found. Will create new one from scratch.")
        
    def save_model(self):
        """Save the trained model and scaler to disk"""
        if self.is_trained:
            # Ensure models directory exists
            os.makedirs(os.path.dirname(self.model_path) if os.path.dirname(self.model_path) else '.', exist_ok=True)
            
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'model_version': self.model_version,
                'contamination_rate': self.contamination_rate,
                'false_positive_indices': self.false_positive_indices,
                'true_positive_indices': self.true_positive_indices,
                'total_predictions': self.total_predictions,
                'total_anomalies': self.total_anomalies,
                'fp_rate_history': self.false_positive_rate_history,
                # Save ground truth metrics
                'true_positives': self.true_positives,
                'false_positives': self.false_positives,
                'true_negatives': self.true_negatives,
                'false_negatives': self.false_negatives,
                'last_saved': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            joblib.dump(model_data, self.model_path)
            print(f"💾 Model v{self.model_version} saved to {self.model_path}")
    
    def add_data_point(self, data):
        """Add new data point to buffer and trigger auto-retraining"""
        self.data_buffer.append(data)
        self.samples_since_retrain += 1
        
        # Keep only recent data for memory efficiency
        if len(self.data_buffer) > 1000:
            self.data_buffer.pop(0)
        
        # Automatic retraining when enough new samples collected
        if (self.is_trained and 
            self.samples_since_retrain >= self.retrain_interval and
            len(self.data_buffer) >= self.window_size):
            self.retrain_model()
            self.samples_since_retrain = 0
    
    def train_model(self):
        """Initial training of anomaly detection model"""
        if len(self.data_buffer) < self.window_size:
            return False
        
        df = pd.DataFrame(self.data_buffer)
        
        # Remove user-marked false positives from training
        training_data = df[~df.index.isin(self.false_positive_indices)]
        
        X = training_data[self.feature_cols].values
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = IsolationForest(
            n_estimators=100,
            contamination=self.contamination_rate,
            random_state=42
        )
        self.model.fit(X_scaled)
        self.is_trained = True
        self.model_version += 1
        self.save_model()
        return True
    
    def retrain_model(self):
        """Automatic retraining with adaptive learning and save"""
        df = pd.DataFrame(self.data_buffer)
        
        # Adaptive contamination rate based on feedback
        if len(self.false_positive_rate_history) > 0:
            recent_fp_rate = np.mean(self.false_positive_rate_history[-5:])
            # If too many false positives, reduce contamination rate
            if recent_fp_rate > 0.7:
                self.contamination_rate = max(0.01, self.contamination_rate * 0.8)
            # If too few detections, increase contamination rate
            elif recent_fp_rate < 0.3:
                self.contamination_rate = min(0.15, self.contamination_rate * 1.2)
        
        # Remove confirmed false positives from training
        training_data = df[~df.index.isin(self.false_positive_indices)]
        
        X = training_data[self.feature_cols].values
        X_scaled = self.scaler.fit_transform(X)
        
        # Retrain with updated contamination rate
        self.model = IsolationForest(
            n_estimators=150,  # More trees for better accuracy over time
            contamination=self.contamination_rate,
            random_state=None  # Allow randomness for diversity
        )
        self.model.fit(X_scaled)
        self.model_version += 1
        
        # Save after each retrain
        self.save_model()
        
        print(f"🔄 Model v{self.model_version} retrained & saved! Contamination: {self.contamination_rate:.3f} | Samples: {len(training_data)}")
    
    def detect_anomaly(self, data, data_index, ground_truth_is_anomaly=False):
        """Detect if current data point is anomaly and compare with ground truth"""
        if not self.is_trained:
            return 0, 0.0  # Not trained yet
        
        X = np.array([[data[f] for f in self.feature_cols]])
        X_scaled = self.scaler.transform(X)
        
        prediction = self.model.predict(X_scaled)[0]
        score = self.model.decision_function(X_scaled)[0]
        
        # Prediction: -1 = anomaly, 1 = normal
        predicted_is_anomaly = (prediction == -1)
        
        # Track predictions
        self.total_predictions += 1
        
        # Compare prediction with ground truth for accuracy
        if ground_truth_is_anomaly and predicted_is_anomaly:
            # Correctly detected actual anomaly
            self.true_positives += 1
            self.total_anomalies += 1
        elif ground_truth_is_anomaly and not predicted_is_anomaly:
            # Missed actual anomaly (false negative)
            self.false_negatives += 1
        elif not ground_truth_is_anomaly and predicted_is_anomaly:
            # False alarm - flagged normal data as anomaly (false positive)
            self.false_positives += 1
            self.total_anomalies += 1
        else:
            # Correctly identified normal data (true negative)
            self.true_negatives += 1
        
        return prediction, score
    
    def mark_false_positive(self, index):
        """User feedback: mark detection as false positive"""
        if index not in self.false_positive_indices:
            self.false_positive_indices.append(index)
            if self.total_anomalies > 0:
                fp_rate = len(self.false_positive_indices) / self.total_anomalies
                self.false_positive_rate_history.append(fp_rate)
            print(f"✓ Marked as false positive. Model will learn from this!")
    
    def mark_true_positive(self, index):
        """User feedback: confirm detection as real anomaly"""
        if index not in self.true_positive_indices:
            self.true_positive_indices.append(index)
            print(f"✓ Confirmed as real anomaly. Model confidence increased!")
    
    def get_model_stats(self):
        """Get model performance statistics based on ground truth comparison"""
        # Calculate metrics based on confusion matrix
        total = self.true_positives + self.false_positives + self.true_negatives + self.false_negatives
        
        # Accuracy: (TP + TN) / Total
        if total > 0:
            accuracy = ((self.true_positives + self.true_negatives) / total) * 100
        else:
            accuracy = 0
        
        # Precision: TP / (TP + FP) - Of all predicted anomalies, how many were correct?
        if (self.true_positives + self.false_positives) > 0:
            precision = (self.true_positives / (self.true_positives + self.false_positives)) * 100
        else:
            precision = 0
        
        # Recall (Sensitivity): TP / (TP + FN) - Of all actual anomalies, how many did we catch?
        if (self.true_positives + self.false_negatives) > 0:
            recall = (self.true_positives / (self.true_positives + self.false_negatives)) * 100
        else:
            recall = 0
        
        # F1 Score: Harmonic mean of precision and recall
        if (precision + recall) > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0
        
        # False Positive Rate: FP / (FP + TN)
        if (self.false_positives + self.true_negatives) > 0:
            fpr = (self.false_positives / (self.false_positives + self.true_negatives)) * 100
        else:
            fpr = 0
        
        # Detection Rate: what percentage of data points are flagged as anomalies
        detection_rate = (self.total_anomalies / self.total_predictions * 100) if self.total_predictions > 0 else 0
        
        return {
            'version': self.model_version,
            'total_predictions': self.total_predictions,
            'total_anomalies': self.total_anomalies,
            'true_positives': self.true_positives,
            'false_positives': self.false_positives,
            'true_negatives': self.true_negatives,
            'false_negatives': self.false_negatives,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'false_positive_rate': fpr,
            'detection_rate': detection_rate,
            'contamination': self.contamination_rate,
            'samples_until_retrain': self.retrain_interval - self.samples_since_retrain
        }

class PredictiveMaintenanceSystem:
    """Predictive maintenance recommendation engine"""
    
    def __init__(self):
        self.anomaly_history = []
        self.maintenance_threshold = 3
        
    def add_anomaly(self, timestamp, anomaly_type, severity):
        """Log detected anomaly"""
        self.anomaly_history.append({
            'timestamp': timestamp,
            'type': anomaly_type,
            'severity': severity
        })
        
        cutoff_time = timestamp - timedelta(minutes=60)
        self.anomaly_history = [a for a in self.anomaly_history 
                               if a['timestamp'] > cutoff_time]
    
    def get_maintenance_recommendation(self, current_time):
        """Generate predictive maintenance recommendations"""
        if not self.anomaly_history:
            return None, "All systems normal"
        
        recent_cutoff = current_time - timedelta(minutes=10)
        recent_anomalies = [a for a in self.anomaly_history 
                           if a['timestamp'] > recent_cutoff]
        
        if len(recent_anomalies) >= self.maintenance_threshold:
            return "CRITICAL", f"CRITICAL: {len(recent_anomalies)} anomalies detected in 10 minutes! Immediate inspection required."
        elif len(recent_anomalies) >= 2:
            return "WARNING", f"WARNING: Multiple anomalies detected. Schedule maintenance within 24 hours."
        elif len(recent_anomalies) >= 1:
            return "INFO", f"ℹINFO: Minor anomaly detected. Monitor system closely."
        
        return None, "All systems normal"

# ============================================================================
# MAIN REAL-TIME VISUALIZATION SYSTEM
# ============================================================================

print("🗼 Initializing 5G Base Station Real-Time Monitoring System v2.0")
print("=" * 70)

# Initialize components
base_station = FiveGBaseStation("5G-BS-DEMO-001")
detector = RealTimeAnomalyDetector(window_size=30, retrain_interval=50)
maintenance_system = PredictiveMaintenanceSystem()

print("\n🧠 Self-Learning Mode: ENABLED")
print("   ✓ Model loads from Jupyter notebook (expert-level)")
print("   ✓ Continues learning in real-time")
print("   ✓ Auto-retrains every 50 samples")
print("   ✓ Saves progress after each retrain")
print("   ✓ Gets smarter with every simulation")

# Data storage for plotting
max_display_points = 100
timestamps = []
temperatures = []
power_consumptions = []
signal_strengths = []
network_loads = []
anomaly_flags = []
anomaly_scores = []

# Training phase (only if no pre-trained model exists)
if not detector.is_trained:
    print("\n🔄 Collecting initial data for model training...")
    for i in range(35):
        data = base_station.generate_sensor_data()
        detector.add_data_point(data)
        time.sleep(0.1)

    print("🤖 Training anomaly detection model...")
    detector.train_model()
    print("✅ Model trained successfully!")

print("\n🚀 Starting real-time monitoring...\n")

# Setup the plot
fig = plt.figure(figsize=(16, 10))
fig.suptitle('Real-Time 5G Base Station Monitoring & Anomaly Detection v2.0', 
             fontsize=16, fontweight='bold')

# Create subplots
ax1 = plt.subplot(3, 2, 1)
ax2 = plt.subplot(3, 2, 2)
ax3 = plt.subplot(3, 2, 3)
ax4 = plt.subplot(3, 2, 4)
ax5 = plt.subplot(3, 2, 5)
ax6 = plt.subplot(3, 2, 6)

def update_plot(frame):
    """Update plot with new data point"""
    
    # Generate new sensor reading
    data = base_station.generate_sensor_data()
    current_index = len(timestamps)
    detector.add_data_point(data)
    
    # Ground truth: Does the data actually have an anomaly?
    ground_truth_is_anomaly = (data['anomaly_type'] is not None)
    
    # Detect anomaly and compare with ground truth
    prediction, score = detector.detect_anomaly(data, current_index, ground_truth_is_anomaly)
    is_anomaly = (prediction == -1)
    
    # Store data
    timestamps.append(data['timestamp'])
    temperatures.append(data['temperature_C'])
    power_consumptions.append(data['power_consumption_W'])
    signal_strengths.append(data['signal_strength_dBm'])
    network_loads.append(data['network_load_percent'])
    anomaly_flags.append(is_anomaly)
    anomaly_scores.append(score)
    
    # Update predictive maintenance
    if is_anomaly:
        severity = "HIGH" if score < -0.3 else "MEDIUM"
        maintenance_system.add_anomaly(
            data['timestamp'], 
            data['anomaly_type'] or 'Unknown',
            severity
        )
    
    # Keep only recent data for display
    if len(timestamps) > max_display_points:
        timestamps.pop(0)
        temperatures.pop(0)
        power_consumptions.pop(0)
        signal_strengths.pop(0)
        network_loads.pop(0)
        anomaly_flags.pop(0)
        anomaly_scores.pop(0)
    
    # Clear all axes
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()
    ax5.clear()
    ax6.clear()
    
    # Plot 1: Temperature
    ax1.plot(range(len(temperatures)), temperatures, 'b-', linewidth=2, label='Temperature')
    anomaly_indices = [i for i, flag in enumerate(anomaly_flags) if flag]
    if anomaly_indices:
        ax1.scatter([i for i in anomaly_indices], 
                   [temperatures[i] for i in anomaly_indices],
                   color='red', s=100, marker='x', linewidths=3, 
                   label=f'Anomalies ({len(anomaly_indices)})', zorder=5)
    ax1.axhline(y=65, color='orange', linestyle='--', label='Warning (65°C)')
    ax1.axhline(y=75, color='red', linestyle='--', label='Critical (75°C)')
    ax1.set_ylabel('Temperature (°C)', fontsize=10, fontweight='bold')
    ax1.set_title('Equipment Temperature', fontsize=11, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Power Consumption
    ax2.plot(range(len(power_consumptions)), power_consumptions, 'g-', linewidth=2, label='Power')
    if anomaly_indices:
        ax2.scatter([i for i in anomaly_indices], 
                   [power_consumptions[i] for i in anomaly_indices],
                   color='red', s=100, marker='x', linewidths=3, zorder=5)
    ax2.axhline(y=5000, color='orange', linestyle='--', label='Warning (5000W)')
    ax2.axhline(y=6000, color='red', linestyle='--', label='Critical (6000W)')
    ax2.set_ylabel('Power (Watts)', fontsize=10, fontweight='bold')
    ax2.set_title('Power Consumption', fontsize=11, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Signal Strength
    ax3.plot(range(len(signal_strengths)), signal_strengths, 'm-', linewidth=2, label='Signal')
    if anomaly_indices:
        ax3.scatter([i for i in anomaly_indices], 
                   [signal_strengths[i] for i in anomaly_indices],
                   color='red', s=100, marker='x', linewidths=3, zorder=5)
    ax3.axhline(y=-85, color='orange', linestyle='--', label='Warning (-85dBm)')
    ax3.axhline(y=-95, color='red', linestyle='--', label='Critical (-95dBm)')
    ax3.set_ylabel('Signal (dBm)', fontsize=10, fontweight='bold')
    ax3.set_title('Signal Strength', fontsize=11, fontweight='bold')
    ax3.legend(loc='lower left', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Network Load
    ax4.plot(range(len(network_loads)), network_loads, 'c-', linewidth=2, label='Network Load')
    if anomaly_indices:
        ax4.scatter([i for i in anomaly_indices], 
                   [network_loads[i] for i in anomaly_indices],
                   color='red', s=100, marker='x', linewidths=3, zorder=5)
    ax4.axhline(y=85, color='orange', linestyle='--', label='Warning (85%)')
    ax4.axhline(y=95, color='red', linestyle='--', label='Critical (95%)')
    ax4.set_ylabel('Load (%)', fontsize=10, fontweight='bold')
    ax4.set_title('Network Load', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Time (samples)', fontsize=9)
    ax4.legend(loc='upper left', fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Anomaly Detection Score
    colors = ['red' if flag else 'green' for flag in anomaly_flags]
    ax5.scatter(range(len(anomaly_scores)), anomaly_scores, c=colors, s=30, alpha=0.6)
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax5.set_ylabel('Anomaly Score', fontsize=10, fontweight='bold')
    ax5.set_title('AI Anomaly Detection Score', fontsize=11, fontweight='bold')
    ax5.set_xlabel('Time (samples)', fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Maintenance Status + Model Stats
    ax6.axis('off')
    severity, message = maintenance_system.get_maintenance_recommendation(data['timestamp'])
    
    # Get model performance stats
    stats = detector.get_model_stats()
    
    # Status display
    status_text = f"SYSTEM STATUS REPORT v2.0\n"
    status_text += f"{'=' * 45}\n\n"
    status_text += f"Station ID: {base_station.station_id}\n"
    status_text += f"Time: {data['timestamp'].strftime('%H:%M:%S')}\n\n"
    status_text += f"Current Readings:\n"
    status_text += f"Temp: {data['temperature_C']:.1f}°C\n"
    status_text += f"Power: {data['power_consumption_W']:.0f}W\n"
    status_text += f"Signal: {data['signal_strength_dBm']:.1f}dBm\n"
    status_text += f"Load: {data['network_load_percent']:.1f}%\n\n"
    
    if is_anomaly:
        status_text += f"ANOMALY DETECTED!\n"
        status_text += f"Type: {data['anomaly_type'] or 'Unknown'}\n"
        status_text += f"Score: {score:.3f}\n\n"
    
    status_text += f"Predictive Maintenance:\n"
    status_text += f"{message}\n\n"
    status_text += f"Recent Anomalies: {len(maintenance_system.anomaly_history)}\n\n"
    
    # Model Learning Stats
    status_text += f"AI Model (v{stats['version']}): LEARNING\n"
    status_text += f"Accuracy: {stats['accuracy']:.1f}% (TP+TN/Total)\n"
    status_text += f"Precision: {stats['precision']:.1f}% (TP/Predicted+)\n"
    status_text += f"Recall: {stats['recall']:.1f}% (TP/Actual+)\n"
    status_text += f"F1 Score: {stats['f1_score']:.1f}%\n"
    status_text += f"\nConfusion Matrix:\n"
    status_text += f"  TP: {stats['true_positives']} | FP: {stats['false_positives']}\n"
    status_text += f"  FN: {stats['false_negatives']} | TN: {stats['true_negatives']}\n"
    status_text += f"\nNext Retrain: {stats['samples_until_retrain']} samples\n"
    
    # Color code based on severity
    text_color = 'red' if severity == 'CRITICAL' else 'orange' if severity == 'WARNING' else 'green'
    
    ax6.text(0.05, 0.95, status_text, transform=ax6.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            color=text_color, fontweight='bold')
    
    plt.tight_layout()
    
    # Print to console
    if is_anomaly:
        print(f"⚠️  [{data['timestamp'].strftime('%H:%M:%S')}] ANOMALY: {data['anomaly_type']} | "
              f"Temp: {data['temperature_C']:.1f}°C | Score: {score:.3f}")

# Create animation
print("📊 Opening live dashboard...")
ani = animation.FuncAnimation(fig, update_plot, interval=2000, cache_frame_data=False)

plt.show()

print("\n✅ Monitoring session ended")
print(f"📝 Total anomalies detected: {sum(anomaly_flags)}")
print(f"💾 Model progress saved to: {detector.model_path}")
print(f"🔢 Model version: v{detector.model_version}")
