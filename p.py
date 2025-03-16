import pandas as pd
import spacy
from collections import Counter
import matplotlib.pyplot as plt
import re
from typing import List

class SkillRecommender:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        
        try:
            self.skills_df = pd.read_csv('very_large_skill_dataset.csv')
            self.skill_keywords = set(self.skills_df['skill_name'].str.lower())
        except FileNotFoundError:
            print("Warning: very_large_skill_dataset.csv not found!")
            self.skill_keywords = {'python', 'java', 'sql', 'aws', 'docker'}
        
        try:
            self.job_data = pd.read_csv('preprocessed_dataset.csv')
            print(f"Loaded {len(self.job_data)} job listings from preprocessed dataset")
            
            # Use the exact column names from the dataset
            if 'Job Title' in self.job_data.columns:
                self.title_column = 'Job Title'
            else:
                raise KeyError("No 'Job Title' column found in dataset")
                
            if 'Job Description' in self.job_data.columns:
                self.desc_column = 'Job Description'
            elif 'Cleaned_Job_Description' in self.job_data.columns:
                self.desc_column = 'Cleaned_Job_Description'
            else:
                raise KeyError("No job description column found in dataset")
                
        except FileNotFoundError:
            print("Error: preprocessed_dataset.csv not found!")
            self.job_data = pd.DataFrame(columns=['Job Title', 'Job Description'])
            self.title_column = 'Job Title'
            self.desc_column = 'Job Description'

    def clean_skill(self, skill: str) -> str:
        """Clean and normalize a skill string."""
        # Remove common prefixes/suffixes and normalize
        skill = skill.lower().strip()
        prefixes_to_remove = ['experience in ', 'knowledge of ', 'proficiency in ', 'expertise in ']
        for prefix in prefixes_to_remove:
            if skill.startswith(prefix):
                skill = skill[len(prefix):]
        
        # Remove parentheses and their contents
        skill = re.sub(r'\([^)]*\)', '', skill)
        
        # Remove special characters and extra spaces
        skill = re.sub(r'[^\w\s-]', ' ', skill)
        skill = ' '.join(skill.split())
        
        return skill

    def parse_skills_string(self, skills_str: str) -> List[str]:
        """Parse a skills string into individual skills."""
        if not skills_str or pd.isna(skills_str):
            return []
            
        # Split on common delimiters
        skills = []
        for skill in re.split(r'[,;|]|\sand\s', skills_str):
            cleaned_skill = self.clean_skill(skill)
            if cleaned_skill and len(cleaned_skill) >= 2:  # Avoid single characters
                skills.append(cleaned_skill)
                
        return skills

    def fetch_job_data(self, job_title: str):
        matching_jobs = self.job_data[
            self.job_data[self.title_column].str.contains(job_title, case=False, na=False)
        ]
        print(f"Found {len(matching_jobs)} matching job listings")
        
        return [{
            'title': row[self.title_column],
            'description': row[self.desc_column],
            'skills': str(row['skills']) if pd.notna(row['skills']) else ''
        } for _, row in matching_jobs.iterrows()]

    def extract_skills(self, text: str, skills_list: str = '') -> List[str]:
        skills = set()
        
        # Extract skills from the skills column
        if skills_list:
            parsed_skills = self.parse_skills_string(skills_list)
            skills.update(parsed_skills)
        
        # Extract skills from description
        desc_skills = self.parse_skills_string(text)
        skills.update(desc_skills)
        
        # Filter against known skills
        valid_skills = {skill for skill in skills if skill in self.skill_keywords}
        return list(valid_skills)

    def analyze_skills(self, job_title: str):
        job_listings = self.fetch_job_data(job_title)
        if not job_listings:
            print(f"No jobs found matching '{job_title}'")
            return []
            
        all_skills = []
        for job in job_listings:
            skills = self.extract_skills(job['description'], job['skills'])
            all_skills.extend(skills)
        
        if not all_skills:
            print("No skills found in the job listings")
            return []
            
        skill_counts = Counter(all_skills)
        return skill_counts.most_common(15)  # Limit to top 15 skills

    def visualize_skills(self, skill_analysis):
        if not skill_analysis:
            print("No skills to visualize")
            return
            
        skills, counts = zip(*skill_analysis)
        
        plt.figure(figsize=(12, 8))
        plt.barh(skills, counts, color='skyblue')
        plt.xlabel("Frequency")
        plt.ylabel("Skills")
        plt.title("Top Skills in Job Listings")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    recommender = SkillRecommender()
    while True:
        job_title = input("\nEnter Job Title (or 'quit' to exit): ").strip()
        if job_title.lower() == 'quit':
            break
            
        print(f"\nAnalyzing skills for: {job_title}")
        skill_analysis = recommender.analyze_skills(job_title)
        
        if skill_analysis:
            print("\nTop skills found:")
            for skill, count in skill_analysis:
                print(f"{skill}: {count} occurrences")
            
            recommender.visualize_skills(skill_analysis)
