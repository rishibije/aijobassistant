AI Job Assistant

📌 Overview

AI Job Assistant is an intelligent career guidance platform that helps users find jobs based on their skills and market trends. It offers resume analysis, job recommendations, market insights, and an AI-powered chatbot to assist users in their job search.

🚀 Features

1️⃣ Resume Analysis & Skill Extraction

Extracts key skills from resumes using NLP (Natural Language Processing).

Stores only extracted skills in the database for efficient job matching.

2️⃣ Job Recommendations

Uses stored skills to match users with relevant jobs from a dataset.

Provides real-time job suggestions without storing large recommendation data.

3️⃣ Market Analysis

Users input a job title to receive four insights:
✅ Salary Trends – Predicts salary growth based on historical data.
✅ Top Hiring Locations – Shows demand hotspots for a job.
✅ Top Hiring Companies – Lists major recruiters.
✅ Job Competition Analysis – Displays job availability by work type.

4️⃣ AI ChatBot (Google Gemini API)

Assists users with job recommendations, resume feedback, and market insights.

Explains why a job was recommended by analyzing skills vs. job requirements.

Answers general career queries dynamically.

🛠️ Technologies Used

✅ Backend: Python (Flask)✅ Frontend: HTML, CSS, JavaScript✅ Machine Learning: NLP for skill extraction, market trend analysis✅ APIs: Adzuna API (job market data), Google Gemini API (ChatBot)

🔄 Workflow

User Registration/Login

New users upload their resume (mandatory).

Extracted skills are stored in the database.

Job Recommendations

Uses stored skills + dataset to dynamically generate job matches.

Market Insights

Users input a job title → Get salary trends, hiring locations, top companies, and competition analysis.

ChatBot Assistance

Answers career questions and provides market insights using Google Gemini API.
