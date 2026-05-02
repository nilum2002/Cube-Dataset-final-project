import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("/media/nilum/New_Volume/01.projects/Dataset-cubes/runs/train-15/results.csv")
# Remove spaces in column names
df.columns = df.columns.str.strip()

# Plot Box Loss
plt.figure(figsize=(10,6))
plt.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss')
plt.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss')

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Validation Box Loss")
plt.legend()
plt.grid(True)
plt.show()