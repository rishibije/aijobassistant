# Import required libraries
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from prophet import Prophet

# Adzuna API credentials
APP_ID = 'ac64a021'
APP_KEY = '03e5b839a923b56521d3f94854a18cd1'
COUNTRY = 'us'

# User Input for Job Title
job_name = input("Enter Job Title: ").strip().lower()

# Fetch data from Adzuna API
def fetch_job_data(job_name):
    url = f"https://api.adzuna.com/v1/api/jobs/{COUNTRY}/search/1"
    params = {
        'app_id': APP_ID,
        'app_key': APP_KEY,
        'results_per_page': 50,
        'what': job_name
    }
    response = requests.get(url, params=params)
    data = response.json()
    return data.get('results', [])

# Load API data into a DataFrame
jobs = fetch_job_data(job_name)
if not jobs:
    print("No job listings found for the given job title.")
else:
    df = pd.DataFrame(jobs)
    print(f"Found {len(df)} job listings for '{job_name}'.")
    top_locations = df['location'].apply(lambda x: x.get('display_name') if isinstance(x, dict) else x)
    top_locations = top_locations.value_counts().nlargest(10)

    plt.figure(figsize=(10, 5))
    sns.barplot(x=top_locations.values, y=top_locations.index, palette="viridis")
    plt.xlabel("Number of Jobs")
    plt.ylabel("Location")
    plt.title(f"Top 10 Locations for {job_name}")
    plt.show()