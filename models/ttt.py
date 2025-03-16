import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# Load dataset
df = pd.read_csv('job_descriptions.csv')

# User input: Job Title
job_name = input("Enter Job Title: ").strip()

# Filter dataset for the selected job title
df_job = df[df['Job Title'].str.contains(job_name, case=False, na=False)].copy()

if df_job.empty:
    print("No data found for this job title.")
    exit()

# Convert Salary Ranges
def extract_salary(salary_range):
    """ Extract the average salary from a range (e.g., '₹50,000 - ₹70,000') """
    import re
    numbers = [int(s.replace(",", "").strip()) for s in re.findall(r'\d+', str(salary_range))]
    return sum(numbers) / len(numbers) if numbers else None

df_job['Average Salary'] = df_job['Salary Range'].apply(extract_salary)

# Remove NaN values in Salary
df_job = df_job.dropna(subset=['Average Salary'])

# Convert date column
df_job['Job Posting Date'] = pd.to_datetime(df_job['Job Posting Date'], errors='coerce')
df_job = df_job.dropna(subset=['Job Posting Date'])

# **NEW FIX: Resample Data to Monthly Mean**
df_job = df_job.resample('ME', on='Job Posting Date')[['Average Salary']].mean().reset_index()

# Format for Prophet
df_salary = df_job[['Job Posting Date', 'Average Salary']].copy()
df_salary.rename(columns={'Job Posting Date': 'ds', 'Average Salary': 'y'}, inplace=True)

# Ensure sufficient data points
if df_salary.shape[0] < 2:
    print(f"Not enough valid data for Prophet! Only {df_salary.shape[0]} records found.")
    exit()

# Train the Prophet model
model = Prophet(interval_width=0.80)  # **New Fix: Reduce Uncertainty Interval**
model.fit(df_salary)

# Predict future salary trends
future = model.make_future_dataframe(periods=12, freq='ME')  # Forecasting 12 months
forecast = model.predict(future)

# **NEW FIX: Reduce Scatter Density (Display Only Every 10th Data Point)**
scatter_sample = df_salary.iloc[::10, :]

# Plot results
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(scatter_sample['ds'], scatter_sample['y'], color='black', s=5, alpha=0.4, label="Observed data points")  
model.plot(forecast, ax=ax)

plt.title(f"Salary Forecast for {job_name} (Next 12 Months)")
plt.xlabel("Date")
plt.ylabel("Predicted Salary")
plt.legend()
plt.show()
