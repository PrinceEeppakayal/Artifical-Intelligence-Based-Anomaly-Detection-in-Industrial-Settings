"""
🗼 REAL-TIME 5G BASE STATION ANOMALY DETECTION SYSTEM
=====================================================

This simulates a real-world 5G telecommunications base station with:
- Temperature sensors (equipment cooling)
- Power consumption monitoring
- Signal strength tracking
- Network load monitoring
- Real-time anomaly detection
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
                temperature += np.random.uniform(15, 30)  # Sudden temperature spike
                power_consumption += np.random.uniform(500, 1000)
                anomaly_type = 'Overheating Event'
                
            elif anomaly_choice == 'power_surge':
                power_consumption += np.random.uniform(1500, 2500)  # Power spike
                temperature += np.random.uniform(5, 15)
                anomaly_type = 'Power Surge'
                
            elif anomaly_choice == 'signal_degradation':
                signal_strength -= np.random.uniform(15, 30)  # Signal drop
                network_load -= np.random.uniform(20, 40)  # Users disconnect
                anomaly_type = 'Signal Degradation'
                
            elif anomaly_choice == 'network_overload':
                network_load = np.random.uniform(95, 100)  # Network congestion
                power_consumption += np.random.uniform(800, 1200)
                temperature += np.random.uniform(10, 20)
                anomaly_type = 'Network Overload'
                
            elif anomaly_choice == 'cooling_failure':
                temperature += np.random.uniform(20, 35)  # Cooling system fails
                anomaly_type = 'Cooling System Failure'
                
            elif anomaly_choice == 'equipment_failure':
                # Multiple parameters go haywire
                temperature += np.random.uniform(15, 25)
                power_consumption *= np.random.uniform(0.3, 0.7)  # Power drops
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
    """Real-time anomaly detection with continuous self-learning"""
    
    def __init__(self, window_size=30, retrain_interval=50):
        self.window_size = window_size
        self.retrain_interval = retrain_interval  # Retrain every N samples
        self.data_buffer = []
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Self-learning parameters
        self.samples_since_retrain = 0
        self.model_version = 0
        self.false_positive_indices = []  # User-marked false positives
        self.true_positive_indices = []   # Confirmed anomalies
        self.contamination_rate = 0.05    # Adaptive contamination
        
        # Performance tracking
        self.total_predictions = 0
        self.total_anomalies = 0
        self.false_positive_rate_history = []
        
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
        features = ['temperature_C', 'power_consumption_W', 
                   'signal_strength_dBm', 'network_load_percent']
        
        # Remove user-marked false positives from training
        training_data = df[~df.index.isin(self.false_positive_indices)]
        
        X = training_data[features].values
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = IsolationForest(
            n_estimators=100,
            contamination=self.contamination_rate,
            random_state=42
        )
        self.model.fit(X_scaled)
        self.is_trained = True
        self.model_version += 1
        return True
    
    def retrain_model(self):
        """Automatic retraining with adaptive learning"""
        df = pd.DataFrame(self.data_buffer)
        features = ['temperature_C', 'power_consumption_W', 
                   'signal_strength_dBm', 'network_load_percent']
        
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
        
        X = training_data[features].values
        X_scaled = self.scaler.fit_transform(X)
        
        # Retrain with updated contamination rate
        self.model = IsolationForest(
            n_estimators=150,  # More trees for better accuracy over time
            contamination=self.contamination_rate,
            random_state=None  # Allow randomness for diversity
        )
        self.model.fit(X_scaled)
        self.model_version += 1
        
        print(f"🔄 Model v{self.model_version} retrained! Contamination: {self.contamination_rate:.3f} | Samples: {len(training_data)}")
    
    def detect_anomaly(self, data, data_index):
        """Detect if current data point is anomaly"""
        if not self.is_trained:
            return 0, 0.0  # Not trained yet
        
        features = ['temperature_C', 'power_consumption_W', 
                   'signal_strength_dBm', 'network_load_percent']
        
        X = np.array([[data[f] for f in features]])
        X_scaled = self.scaler.transform(X)
        
        prediction = self.model.predict(X_scaled)[0]
        score = self.model.decision_function(X_scaled)[0]
        
        # Track predictions
        self.total_predictions += 1
        if prediction == -1:
            self.total_anomalies += 1
        
        return prediction, score
    
    def mark_false_positive(self, index):
        """User feedback: mark detection as false positive"""
        if index not in self.false_positive_indices:
            self.false_positive_indices.append(index)
            # Update false positive rate
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
        """Get model performance statistics"""
        fp_rate = (len(self.false_positive_indices) / self.total_anomalies * 100) if self.total_anomalies > 0 else 0
        accuracy = ((self.total_predictions - len(self.false_positive_indices)) / self.total_predictions * 100) if self.total_predictions > 0 else 0
        
        return {
            'version': self.model_version,
            'total_predictions': self.total_predictions,
            'total_anomalies': self.total_anomalies,
            'false_positives': len(self.false_positive_indices),
            'true_positives': len(self.true_positive_indices),
            'false_positive_rate': fp_rate,
            'accuracy': accuracy,
            'contamination': self.contamination_rate,
            'samples_until_retrain': self.retrain_interval - self.samples_since_retrain
        }

class PredictiveMaintenanceSystem:
    """Predictive maintenance recommendation engine"""
    
    def __init__(self):
        self.anomaly_history = []
        self.maintenance_threshold = 3  # Trigger after 3 anomalies in 10 minutes
        
    def add_anomaly(self, timestamp, anomaly_type, severity):
        """Log detected anomaly"""
        self.anomaly_history.append({
            'timestamp': timestamp,
            'type': anomaly_type,
            'severity': severity
        })
        
        # Keep only last hour of history
        cutoff_time = timestamp - timedelta(minutes=60)
        self.anomaly_history = [a for a in self.anomaly_history 
                               if a['timestamp'] > cutoff_time]
    
    def get_maintenance_recommendation(self, current_time):
        """Generate predictive maintenance recommendations"""
        if not self.anomaly_history:
            return None, "✅ All systems normal"
        
        # Check recent anomalies (last 10 minutes)
        recent_cutoff = current_time - timedelta(minutes=10)
        recent_anomalies = [a for a in self.anomaly_history 
                           if a['timestamp'] > recent_cutoff]
        
        if len(recent_anomalies) >= self.maintenance_threshold:
            return "CRITICAL", f"🚨 CRITICAL: {len(recent_anomalies)} anomalies detected in 10 minutes! Immediate inspection required."
        
        elif len(recent_anomalies) >= 2:
            return "WARNING", f"⚠️ WARNING: Multiple anomalies detected. Schedule maintenance within 24 hours."
        
        elif len(recent_anomalies) >= 1:
            return "INFO", f"ℹ️ INFO: Minor anomaly detected. Monitor system closely."
        
        return None, "✅ All systems normal"

# ============================================================================
# MAIN REAL-TIME VISUALIZATION SYSTEM
# ============================================================================

print("🗼 Initializing 5G Base Station Real-Time Monitoring System...")
print("=" * 70)

# Initialize components
base_station = FiveGBaseStation("5G-BS-DEMO-001")
detector = RealTimeAnomalyDetector(window_size=30, retrain_interval=50)  # Retrain every 50 samples
maintenance_system = PredictiveMaintenanceSystem()

print("🧠 Self-Learning Mode: ENABLED")
print("   - Model will retrain automatically every 50 samples")
print("   - Contamination rate adapts based on detection accuracy")
print("   - Model gets smarter with each simulation run")

# Data storage for plotting
max_display_points = 100
timestamps = []
temperatures = []
power_consumptions = []
signal_strengths = []
network_loads = []
anomaly_flags = []
anomaly_scores = []

# Training phase
print("🔄 Collecting initial data for model training...")
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
fig.suptitle('🗼 Real-Time 5G Base Station Monitoring & Anomaly Detection', 
             fontsize=16, fontweight='bold')

# Create subplots
ax1 = plt.subplot(3, 2, 1)  # Temperature
ax2 = plt.subplot(3, 2, 2)  # Power Consumption
ax3 = plt.subplot(3, 2, 3)  # Signal Strength
ax4 = plt.subplot(3, 2, 4)  # Network Load
ax5 = plt.subplot(3, 2, 5)  # Anomaly Score
ax6 = plt.subplot(3, 2, 6)  # Maintenance Status

def update_plot(frame):
    """Update plot with new data point"""
    
    # Generate new sensor reading
    data = base_station.generate_sensor_data()
    current_index = len(timestamps)
    detector.add_data_point(data)
    
    # Detect anomaly
    prediction, score = detector.detect_anomaly(data, current_index)
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
    ax1.set_title('🌡️ Equipment Temperature', fontsize=11, fontweight='bold')
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
    ax2.set_title('⚡ Power Consumption', fontsize=11, fontweight='bold')
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
    ax3.set_title('📡 Signal Strength', fontsize=11, fontweight='bold')
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
    ax4.set_title('🌐 Network Load', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Time (samples)', fontsize=9)
    ax4.legend(loc='upper left', fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Anomaly Detection Score
    colors = ['red' if flag else 'green' for flag in anomaly_flags]
    ax5.scatter(range(len(anomaly_scores)), anomaly_scores, c=colors, s=30, alpha=0.6)
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax5.set_ylabel('Anomaly Score', fontsize=10, fontweight='bold')
    ax5.set_title('🎯 AI Anomaly Detection Score', fontsize=11, fontweight='bold')
    ax5.set_xlabel('Time (samples)', fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Maintenance Status + Model Stats
    ax6.axis('off')
    severity, message = maintenance_system.get_maintenance_recommendation(data['timestamp'])
    
    # Get model performance stats
    stats = detector.get_model_stats()
    
    # Status display
    status_text = f"📊 SYSTEM STATUS REPORT\n"
    status_text += f"{'=' * 45}\n\n"
    status_text += f"Station ID: {base_station.station_id}\n"
    status_text += f"Time: {data['timestamp'].strftime('%H:%M:%S')}\n\n"
    status_text += f"Current Readings:\n"
    status_text += f"  🌡️  Temp: {data['temperature_C']:.1f}°C\n"
    status_text += f"  ⚡ Power: {data['power_consumption_W']:.0f}W\n"
    status_text += f"  📡 Signal: {data['signal_strength_dBm']:.1f}dBm\n"
    status_text += f"  🌐 Load: {data['network_load_percent']:.1f}%\n\n"
    
    if is_anomaly:
        status_text += f"🚨 ANOMALY DETECTED!\n"
        status_text += f"Type: {data['anomaly_type'] or 'Unknown'}\n"
        status_text += f"Score: {score:.3f}\n\n"
    
    status_text += f"🔧 Predictive Maintenance:\n"
    status_text += f"{message}\n\n"
    status_text += f"Recent Anomalies: {len(maintenance_system.anomaly_history)}\n\n"
    
    # Model Learning Stats
    status_text += f"🧠 AI Model (v{stats['version']}): LEARNING\n"
    status_text += f"Accuracy: {stats['accuracy']:.1f}%\n"
    status_text += f"False Positive Rate: {stats['false_positive_rate']:.1f}%\n"
    status_text += f"Contamination: {stats['contamination']:.3f}\n"
    status_text += f"Next Retrain: {stats['samples_until_retrain']} samples\n"
    
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
print("📝 Total anomalies detected:", sum(anomaly_flags))
