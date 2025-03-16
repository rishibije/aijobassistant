import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("preprocessed_dataset.csv")  # Ensure correct column names

# Function to plot job competition as pie charts
def plot_job_competition_pie(job_title):
    # Filter dataset for the given job
    job_data = df[df['Job Title'].str.contains(job_title, case=False, na=False)]

    if job_data.empty:
        print(f"No data found for '{job_title}'. Try another job title.")
        return

    # Work Type Distribution
    work_type_counts = job_data['Work Type'].value_counts()
    plt.figure(figsize=(7, 7))
    plt.pie(work_type_counts, labels=work_type_counts.index, autopct='%1.1f%%', 
            colors=plt.cm.Paired.colors, startangle=140)
    plt.title(f"Work Type Distribution for {job_title}")
    plt.axis('equal')  # Ensures pie chart is a circle
    plt.show()

    # Company Size Distribution
    company_size_counts = job_data['Company Size'].value_counts()
    plt.figure(figsize=(7, 7))
    plt.pie(company_size_counts, labels=company_size_counts.index, autopct='%1.1f%%', 
            colors=plt.cm.Paired.colors, startangle=140)
    plt.title(f"Company Size Distribution for {job_title}")
    plt.axis('equal')  # Ensures pie chart is a circle
    plt.show()

# Example usage
plot_job_competition_pie("Data Scientist")