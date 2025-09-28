

import pandas as pd
import random

# Generate synthetic data
data = []
for _ in range(2000):  # Add 2000 rows
    hour = random.randint(0, 23)
    temp = random.randint(15, 35)  # Random temperature between 15 and 35
    holiday = random.randint(0, 1)  # Random holiday (0 or 1)
    weather_main = random.choice(['Clear', 'Rain', 'Clouds', 'Snow'])  # Random weather condition
    traffic_volume = random.randint(10, 100)  # Random traffic volume
    data.append({'hour': hour, 'temp': temp, 'holiday': holiday, 'weather_main': weather_main, 'traffic_volume': traffic_volume})

# Create DataFrame and save to CSV
df = pd.DataFrame(data)
df.to_csv("data/training_data.csv", index=False)

print("Synthetic data generated and saved to training_data.csv")


