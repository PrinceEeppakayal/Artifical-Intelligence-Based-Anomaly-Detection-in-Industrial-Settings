import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Industrial machine monitoring parameters
start_date = datetime(2025, 8, 1, 0, 0, 0)
end_date = datetime(2025, 8, 7, 23, 58, 0)  # 7 days of data
interval_minutes = 2  # Read every 2 minutes for industrial monitoring

# Generate timestamps every 2 minutes
timestamps = []
current_time = start_date
while current_time <= end_date:
    timestamps.append(current_time)
    current_time += timedelta(minutes=interval_minutes)

print(f"Generating {len(timestamps):,} data points for industrial machine monitoring")
print(f"Time range: {start_date} to {end_date}")
print(f"Sampling interval: {interval_minutes} minutes")

# Industrial machine temperature and humidity simulation
# Typical industrial machine operating ranges:
# Temperature: 35-55°C (95-131°F) - higher than room temp due to machinery heat
# Humidity: 30-70% - controlled environment but some variation

data = []

for i, ts in enumerate(timestamps):
    # Base temperature pattern (industrial machinery runs hotter)
    base_temp = 45.0  # Base operating temperature
    
    # Daily cycle (machines may run cooler at night, hotter during peak hours)
    hour_of_day = ts.hour
    daily_temp_variation = 5 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
    
    # Weekly pattern (less usage on weekends)
    day_of_week = ts.weekday()  # 0 = Monday, 6 = Sunday
    if day_of_week >= 5:  # Weekend
        weekly_temp_adjustment = -3.0
    else:
        weekly_temp_adjustment = 0.0
    
    # Random noise
    temp_noise = np.random.normal(0, 1.5)
    
    # Calculate temperature
    temperature = base_temp + daily_temp_variation + weekly_temp_adjustment + temp_noise
    
    # Base humidity pattern (industrial environments)
    base_humidity = 50.0
    
    # Humidity tends to be inversely related to temperature in industrial settings
    humidity_temp_effect = -0.5 * (temperature - base_temp)
    
    # Daily humidity variation
    daily_humidity_variation = 8 * np.sin(2 * np.pi * (hour_of_day - 12) / 24)
    
    # Random noise
    humidity_noise = np.random.normal(0, 3.0)
    
    # Calculate humidity
    humidity = base_humidity + humidity_temp_effect + daily_humidity_variation + humidity_noise
    
    # Add some industrial anomalies (equipment issues)
    # 1. Overheating events (0.5% chance)
    if np.random.random() < 0.005:
        temperature += np.random.uniform(15, 25)  # Sudden temperature spike
        humidity -= np.random.uniform(10, 20)     # Humidity drops during overheating
    
    # 2. Cooling system failure (0.3% chance)
    if np.random.random() < 0.003:
        temperature += np.random.uniform(8, 15)
        humidity += np.random.uniform(15, 25)     # High humidity when cooling fails
    
    # 3. Sensor malfunction (0.2% chance)
    if np.random.random() < 0.002:
        if np.random.random() < 0.5:
            temperature = np.random.uniform(0, 100)  # Sensor gives random readings
        else:
            humidity = np.random.uniform(0, 100)
    
    # 4. Maintenance shutdown periods (very low temp/humidity)
    if np.random.random() < 0.001:
        temperature = np.random.uniform(18, 25)   # Room temperature during shutdown
        humidity = np.random.uniform(35, 50)      # Normal room humidity
    
    # Ensure realistic bounds
    temperature = np.clip(temperature, 15, 80)  # Industrial range with safety limits
    humidity = np.clip(humidity, 10, 95)        # Realistic humidity range
    
    # Round to sensor precision (DHT11 has ±2°C, ±5% accuracy)
    temperature = round(temperature + np.random.uniform(-0.5, 0.5), 2)
    humidity = round(humidity + np.random.uniform(-1, 1), 2)
    
    data.append({
        'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
        'temperature_C': temperature,
        'humidity_%': humidity
    })

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
output_file = "c:/Users/Prince Eppakayal/OneDrive/Documents/College/ECEI/industrial_machine_data.csv"
df.to_csv(output_file, index=False)

print(f"\n✅ Industrial machine monitoring data generated!")
print(f"📁 Saved to: {output_file}")
print(f"📊 Total data points: {len(df):,}")
print(f"📅 Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
print(f"🌡️ Temperature range: {df['temperature_C'].min():.1f}°C to {df['temperature_C'].max():.1f}°C")
print(f"💧 Humidity range: {df['humidity_%'].min():.1f}% to {df['humidity_%'].max():.1f}%")

# Show first few rows
print(f"\nFirst 10 rows:")
print(df.head(10))

# Calculate approximate anomaly count
temp_anomalies = len(df[df['temperature_C'] > 65])  # High temperature anomalies
humidity_anomalies = len(df[df['humidity_%'] > 80])  # High humidity anomalies
low_temp_anomalies = len(df[df['temperature_C'] < 25])  # Maintenance/shutdown periods

print(f"\n🚨 Approximate anomalies injected:")
print(f"   High temperature events: ~{temp_anomalies}")
print(f"   High humidity events: ~{humidity_anomalies}")
print(f"   Low temperature events (maintenance): ~{low_temp_anomalies}")
print(f"   Total suspicious readings: ~{temp_anomalies + humidity_anomalies + low_temp_anomalies}")
