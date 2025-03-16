import requests
import matplotlib.pyplot as plt
from collections import Counter

# Adzuna API Credentials
APP_ID = 'ac64a021'
APP_KEY = '03e5b839a923b56521d3f94854a18cd1'
COUNTRY = 'us'

# Get user input for job title
job_name = input("Enter Job Title: ").strip().lower()

# API Request URL
api_url = f"https://api.adzuna.com/v1/api/jobs/{COUNTRY}/search/1?app_id={APP_ID}&app_key={APP_KEY}&results_per_page=50&title_only={job_name}&content-type=application/json"

# Fetch data from API
response = requests.get(api_url)
data = response.json()

# Check if results are available
if "results" not in data or not data["results"]:
    print("No job data found.")
    exit()

# Extract company names from job listings
companies = [job["company"]["display_name"] for job in data["results"] if "company" in job]

# Count job postings per company
company_counts = Counter(companies)

# Get top 10 hiring companies
top_companies = company_counts.most_common(10)

# Extract company names and counts for plotting
company_names, job_counts = zip(*top_companies) if top_companies else ([], [])

# Plot bar chart
plt.figure(figsize=(10, 5))
plt.barh(company_names, job_counts, color="skyblue")
plt.xlabel("Number of Job Postings")
plt.ylabel("Company Name")
plt.title(f"Top Hiring Companies for {job_name.title()}")
plt.gca().invert_yaxis()  # Invert y-axis for better readability
plt.show()