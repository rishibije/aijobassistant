from flask import Flask, request, render_template, redirect, url_for, flash, session, jsonify
import PyPDF2
import spacy
import os
import time
from werkzeug.utils import secure_filename
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re
from werkzeug.security import generate_password_hash, check_password_hash
import os
import pickle
import numpy as np
import MySQLdb.cursors
from flask import Flask, request, render_template, session, redirect, url_for
from werkzeug.utils import secure_filename
from sklearn.metrics.pairwise import cosine_similarity
import csv
from prophet import Prophet
import pandas as pd
import requests
from collections import Counter
from functools import lru_cache
import concurrent.futures
import google.generativeai as genai
from dotenv import load_dotenv
from enum import Enum
from dotenv import load_dotenv
from threading import Lock
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import json
import random
from openai import OpenAI
from enum import Enum

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# MySQL Configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'Parthavi@1204'  # Replace with your MySQL root password
app.config['MYSQL_DB'] = 'ai_job_assistant'
app.secret_key = 'your_secret_key_here'  # Required for session and flash messages

mysql = MySQL(app)

# Create uploads folder if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Keep only the base spaCy model
nlp = spacy.load("en_core_web_sm")

# Load the custom NER model for skills
try:
    # Load the skills NER model from current directory
    nlp_skills = spacy.load('datasetmodel')  # Use relative path
    print("✅ Skills NER model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading skills model: {e}")
    nlp_skills = nlp  # Fallback to base model

# Cache for DataFrames
df_cache = {}
df_cache_lock = Lock()

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load API key from .env
load_dotenv()
api_key = os.getenv("GOOGLE_GEMINI_API_KEY")
if not api_key:
    raise ValueError("Missing Google Gemini API Key in .env file")
genai.configure(api_key=api_key)

MODEL_NAME = "gemini-1.5-pro" 

# Predefined chatbot responses
CHATBOT_RESPONSES = {
    'greetings': [
        "Hello! How can I help you today?",
        "Hi there! What would you like to know about?",
        "Welcome! I'm here to help with your career questions."
    ],
    'farewell': [
        "Goodbye! Feel free to come back if you have more questions.",
        "Have a great day! Let me know if you need anything else.",
        "Bye! Don't hesitate to ask if you need more help."
    ],
    'default': [
        "I can help you with resume analysis, job recommendations, and market insights. What would you like to know?",
        "I'm not sure I understand. Would you like to know about job recommendations, market analysis, or resume parsing?",
        "Could you please rephrase that? I can help with career guidance, job market analysis, and skill assessment."
    ]
}

class QueryType(Enum):
    GREETING = "greeting"
    FAREWELL = "farewell"
    JOB_RECOMMENDATION = "job_recommendation"
    MARKET_ANALYSIS = "market_analysis"
    RESUME = "resume"
    GENERAL = "general"

def classify_query(message):
    message = message.lower()
    if any(word in message for word in ['hi', 'hello', 'hey']):
        return QueryType.GREETING
    if any(word in message for word in ['bye', 'goodbye', 'thank']):
        return QueryType.FAREWELL
    if ('why' in message and ('recommend' in message or 'suggestion' in message)) or \
       ('job' in message and 'recommendation' in message):
        return QueryType.JOB_RECOMMENDATION
    if any(word in message for word in ['market', 'trend', 'salary', 'companies', 'hiring']):
        return QueryType.MARKET_ANALYSIS
    if 'resume' in message or 'skill' in message:
        return QueryType.RESUME
    return QueryType.GENERAL

def get_gemini_response(message, user_context=None):
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        system_prompt = """
        You are an AI career assistant with expertise in:
        1. Resume writing and improvement
        2. Career guidance and professional development
        3. Job search strategies
        4. Interview preparation
        5. Industry trends and insights
        Keep responses concise, practical, and focused on career development.
        """
        if user_context:
            system_prompt += f"\n\nUser Context:\n{user_context}"
        response = model.generate_content(f"{system_prompt}\n\nUser Query: {message}")
        return response.text.strip() if response and hasattr(response, 'text') else "I'm having trouble processing your request. Please try again."
    except Exception as e:
        return f"Gemini API Error: {str(e)}"

def get_user_context(user_id):
    """Get user context for the chatbot."""
    try:
        # Get user skills
        user_skills = get_user_skills(user_id)
        
        # Get recent job recommendations
        recommended_jobs = recommend_jobs(user_skills)[:3] if user_skills else []
        
        context = []
        if user_skills:
            context.append(f"User's skills: {', '.join(user_skills[:5])}")
        
        if recommended_jobs:
            job_titles = [job['Job Title'] for job in recommended_jobs]
            context.append(f"Recent job recommendations: {', '.join(job_titles)}")
            
        return " | ".join(context) if context else None
        
    except Exception as e:
        print(f"Error getting user context: {str(e)}")
        return None


@lru_cache(maxsize=100)
def load_and_filter_dataset(job_title):
    """Cache-optimized function to load and filter dataset"""
    if 'job_descriptions' not in df_cache:
        with df_cache_lock:
            if 'job_descriptions' not in df_cache:
                df_cache['job_descriptions'] = pd.read_csv('job_descriptions.csv')
    
    df = df_cache['job_descriptions']
    return df[df['Job Title'].str.contains(job_title, case=False, na=False)].copy()

@lru_cache(maxsize=100)
def load_competition_dataset(job_title):
    """Cache-optimized function to load competition dataset"""
    if 'preprocessed_dataset' not in df_cache:
        with df_cache_lock:
            if 'preprocessed_dataset' not in df_cache:
                df_cache['preprocessed_dataset'] = pd.read_csv('preprocessed_dataset.csv')
    
    df = df_cache['preprocessed_dataset']
    return df[df['Job Title'].str.contains(job_title, case=False, na=False)].copy()

def load_keywords(file_path):
    """Load skills keywords from CSV file."""
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        return set(row[0] for row in reader)

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def csv_skills(text):
    """Extract skills using CSV keyword matching."""
    skills_keywords = load_keywords('newSkills.csv')
    skills = set()
    
    for keyword in skills_keywords:
        if keyword.lower() in text.lower():
            skills.add(keyword)
    
    return skills

def extract_skills_from_ner(text):
    """Extract skills using custom NER model."""
    doc = nlp_skills(text)
    skills = set()
    
    # Extract entities labeled as SKILL
    for ent in doc.ents:
        if ent.label_ == 'SKILL':
            skill = ent.text.strip().lower()
            if is_valid_skill(skill):
                skills.add(skill)
    
    return skills

def is_valid_skill(skill_text):
    """Validate if extracted text is a valid skill."""
    return len(skill_text) > 1 and not any(char.isdigit() for char in skill_text)

def extract_skills(text):
    """Extract skills using pattern matching and CSV."""
    # Get skills from pattern matching
    ner_skills = extract_skills_from_ner(text)
    
    # Get skills from CSV keywords
    csv_skills_set = csv_skills(text)
    
    # Combine both sets of skills
    all_skills = ner_skills.union(csv_skills_set)
    
    # Filter and validate skills
    valid_skills = {skill for skill in all_skills if is_valid_skill(skill)}
    
    return list(valid_skills)

# Define a set of common skills across multiple domains

@app.route('/')
def index():
    return render_template('login1.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM users WHERE email = %s', (email,))
        user = cursor.fetchone()
        
        if user and check_password_hash(user['password'], password):
            session['loggedin'] = True
            session['id'] = user['id']
            session['email'] = user['email']
            session['name'] = user['name']
            return redirect(url_for('check_user_status'))
        else:
            return render_template('login1.html', error='Invalid email/password')
    return render_template('login1.html')

@app.route('/register', methods=['POST'])
def register():
    msg = ''
    if request.method == 'POST':
        try:
            name = request.form['name']
            email = request.form['email']
            password = request.form['password']
            confirm_password = request.form['confirm_password']

            # Check if passwords match
            if password != confirm_password:
                flash('Passwords do not match!')
                return render_template('login1.html', msg='Passwords do not match!')

            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            
            # Check if account exists
            cursor.execute('SELECT * FROM users WHERE email = %s', (email,))
            account = cursor.fetchone()
            
            if account:
                flash('Account already exists!')
                return render_template('login1.html', msg='Account already exists!')
            elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
                flash('Invalid email address!')
                return render_template('login1.html', msg='Invalid email address!')
            elif not re.match(r'[A-Za-z0-9]+', name):
                flash('Name must contain only characters and numbers!')
                return render_template('login1.html', msg='Name must contain only characters and numbers!')
            else:
                # Hash the password
                hashed_password = generate_password_hash(password)
                
                # Insert new user into database
                cursor.execute('INSERT INTO users (name, email, password) VALUES (%s, %s, %s)',
                             (name, email, hashed_password))
                mysql.connection.commit()
                flash('Registration successful! Please login.')
                return redirect(url_for('index'))
        except Exception as e:
            flash(f'An error occurred: {str(e)}')
            return render_template('login1.html', msg=f'An error occurred: {str(e)}')

    return render_template('login1.html', msg=msg)

@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('id', None)
    session.pop('email', None)
    session.pop('name', None)
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'loggedin' in session:
        return render_template('index.html', name=session['name'])
    return redirect(url_for('index'))

@app.route('/analyze_resume', methods=['POST'])
def analyze_resume():
    if not 'loggedin' in session:
        return redirect(url_for('index'))
        
    if 'resume' not in request.files:
        return render_template('index.html', error='No file provided')
    
    file = request.files['resume']
    if file.filename == '':
        return render_template('index.html', error='No file selected')

    if file and file.filename.endswith('.pdf'):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            text = extract_text_from_pdf(filepath)
            skills = extract_skills(text)
            
            # Convert skills list to comma-separated string
            skills_string = ','.join(skills)

            # Check if user already has skills in database
            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            cursor.execute('SELECT id FROM user_skills WHERE user_id = %s', (session['id'],))
            existing_record = cursor.fetchone()

            if existing_record:
                # Update existing skills
                cursor.execute('UPDATE user_skills SET skills = %s WHERE user_id = %s',
                             (skills_string, session['id']))
            else:
                # Insert new skills
                cursor.execute('INSERT INTO user_skills (user_id, skills) VALUES (%s, %s)',
                             (session['id'], skills_string))
            flash('Your skills have been saved successfully!')

            mysql.connection.commit()
            cursor.close()

            os.remove(filepath)  # Clean up uploaded file
            
            # Always show results page with extracted skills
            return render_template('results.html', 
                                 skills=skills, 
                                 is_new_user=not existing_record,
                                 next_url=url_for('dashboard'))
        
        except Exception as e:
            return render_template('index.html', error=str(e))

    return render_template('index.html', error='Invalid file format')

@app.route('/recommend_jobs')
def job_recommendations():
    if 'loggedin' not in session:
        return redirect(url_for('index'))

    try:
        user_id = session['id']
        user_skills = get_user_skills(user_id)
        
        if not user_skills:
            return render_template('jobs.html', error="Please upload your resume first to get job recommendations.")
        
        recommended_jobs = recommend_jobs(user_skills)
        return render_template('jobs.html', jobs=recommended_jobs)
    except Exception as e:
        print(f"Error in job recommendations: {e}")
        return render_template('jobs.html', error="An error occurred while getting job recommendations.")

@app.route('/market')
def market():
    if 'loggedin' in session:
        return render_template('market.html')
    return redirect(url_for('index'))

@app.route('/chatbot')
def chatbot():
    if 'loggedin' in session:
        return render_template('chatbot.html')
    return redirect(url_for('index'))

@app.route('/check_user_status')
def check_user_status():
    if 'loggedin' not in session:
        return redirect(url_for('index'))
    
    try:
        # Check if user has uploaded resume before
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM user_skills WHERE user_id = %s', (session['id'],))
        has_resume = cursor.fetchone() is not None
        cursor.close()
        
        if has_resume:
            # Existing user with resume - go to dashboard
            return redirect(url_for('dashboard'))
        else:
            # New user - go to resume upload page
            return redirect(url_for('upload_resume'))
    except Exception as e:
        print(f"Error checking user status: {e}")
        return redirect(url_for('index'))

@app.route('/upload_resume')
def upload_resume():
    if 'loggedin' not in session:
        return redirect(url_for('index'))
    return render_template('upload_resume.html', name=session['name'])

# Load trained models and dataset
job_data_path = "models/job_data.pkl"
job_matrix_path = "models/job_matrix.pkl"
tfidf_vectorizer_path = "models/tfidf_vectorizer.pkl"

with open(job_data_path, 'rb') as f:
    job_data = pickle.load(f)

with open(job_matrix_path, 'rb') as f:
    job_matrix = pickle.load(f)

with open(tfidf_vectorizer_path, 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

# Function to fetch user skills from the database
def get_user_skills(user_id):
    try:
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute("SELECT skills FROM user_skills WHERE user_id = %s", (user_id,))
        result = cursor.fetchone()
        cursor.close()
        
        if result and result['skills']:
            return result['skills'].split(',')  # Convert skills from string to list
        return []
    except Exception as e:
        print(f"Error getting user skills: {e}")
        return []

# Job recommendation function
def recommend_jobs(user_skills):
    try:
        if not user_skills:
            return []

        # Convert user skills into TF-IDF features
        user_skills_text = " ".join(user_skills)  # Convert list to space-separated text
        user_skills_vector = tfidf_vectorizer.transform([user_skills_text])

        # Compute similarity with job dataset
        similarity_scores = cosine_similarity(user_skills_vector, job_matrix)

        # Get top recommended job indices
        top_indices = np.argsort(similarity_scores[0])[::-1][:5]  # Get top 5 matches

        # Extract recommended jobs
        recommendations = []
        for idx in top_indices:
            job = job_data.iloc[idx]  # Assuming job_data is a DataFrame
            recommendations.append({
                "Job Title": job["job_position"],
                "Job Description": job["original_description"],
                "Salary Range": job["salary_range"],
                "Job Type": job["job_type"]
            })
        
        return recommendations
    except Exception as e:
        print(f"Error recommending jobs: {e}")
        return []

@app.route('/apply_job', methods=['POST'])
def apply_job():
    if 'loggedin' not in session:
        return redirect(url_for('index'))
    
    job_title = request.form.get('job_title')
    # Here you can add logic to handle job applications
    flash(f'Successfully applied for {job_title}')
    return redirect(url_for('recommend_jobs'))

@app.route('/analyze_salary_trends', methods=['POST'])
def analyze_salary_trends():
    if 'loggedin' not in session:
        return jsonify({'error': 'Not logged in'}), 401
        
    try:
        job_title = request.form.get('jobTitle')
        if not job_title:
            return jsonify({'error': 'Job title is required'}), 400

        # Ensure safe filename by replacing special characters
        safe_job_title = ''.join(c if c.isalnum() or c == '_' else '_' for c in job_title)
        model_filename = f"models/salary_model_{safe_job_title}.pkl"

        # Check if the model exists
        if not os.path.exists(model_filename):
            return jsonify({'error': 'No trained model found for this job title'}), 404

        # Load trained Prophet model
        with open(model_filename, "rb") as f:
            model = pickle.load(f)

        # Generate future dates for prediction (12 months)
        future = model.make_future_dataframe(periods=12, freq='ME')
        forecast = model.predict(future)

        # Load and filter dataset to get historical salary data
        df = pd.read_csv('job_descriptions.csv')
        df_job = df[df['Job Title'].str.contains(job_title, case=False, na=False)].copy()

        if df_job.empty:
            return jsonify({'error': 'No data found for this job title'}), 404

        # Extract & clean salary data
        def extract_salary(salary_range):
            import re
            numbers = [int(s.replace(",", "").strip()) for s in re.findall(r'\d+', str(salary_range))]
            return sum(numbers) / len(numbers) if numbers else None

        df_job['Average Salary'] = df_job['Salary Range'].apply(extract_salary)
        df_job = df_job.dropna(subset=['Average Salary'])
        df_job['Job Posting Date'] = pd.to_datetime(df_job['Job Posting Date'], errors='coerce')
        df_job = df_job.dropna(subset=['Job Posting Date'])

        # Resample to monthly data
        df_job = df_job.resample('ME', on='Job Posting Date')[['Average Salary']].mean().reset_index()

        # Prepare historical data
        historical_data = df_job[['Job Posting Date', 'Average Salary']].rename(
            columns={'Job Posting Date': 'ds', 'Average Salary': 'y'}
        ).to_dict('records')

        # Prepare forecast data
        forecast_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict('records')

        return jsonify({
            'historical_data': historical_data,
            'forecast_data': forecast_data,
            'job_title': job_title
        })

    except Exception as e:
        print(f"Error in salary analysis: {str(e)}")
        return jsonify({'error': 'An error occurred during analysis'}), 500

# Add these constants near the top of the file
ADZUNA_APP_ID = 'ac64a021'
ADZUNA_APP_KEY = '03e5b839a923b56521d3f94854a18cd1'
ADZUNA_COUNTRY = 'us'

@app.route('/analyze_hiring_companies', methods=['POST'])
def analyze_hiring_companies():
    if 'loggedin' not in session:
        return jsonify({'error': 'Not logged in'}), 401
        
    try:
        job_title = request.form.get('jobTitle')
        if not job_title:
            return jsonify({'error': 'Job title is required'}), 400

        # Construct API URL
        api_url = f"https://api.adzuna.com/v1/api/jobs/{ADZUNA_COUNTRY}/search/1"
        params = {
            'app_id': ADZUNA_APP_ID,
            'app_key': ADZUNA_APP_KEY,
            'results_per_page': 50,
            'what': job_title.lower(),
            'content-type': 'application/json'
        }

        # Fetch data from Adzuna API
        response = requests.get(api_url, params=params)
        data = response.json()

        if "results" not in data or not data["results"]:
            return jsonify({'error': 'No job data found'}), 404

        # Extract companies and locations
        companies = []
        locations = []
        
        for job in data["results"]:
            if "company" in job and "display_name" in job["company"]:
                companies.append(job["company"]["display_name"])
            if "location" in job and "display_name" in job["location"]:
                locations.append(job["location"]["display_name"])
        
        # Count job postings per company and location
        company_counts = Counter(companies)
        location_counts = Counter(locations)
        
        # Get top 10 companies and locations
        top_companies = company_counts.most_common(10)
        top_locations = location_counts.most_common(10)
        
        # Prepare data for frontend
        response_data = {
            'companies': {
                'names': [company for company, _ in top_companies],
                'counts': [count for _, count in top_companies]
            },
            'locations': {
                'names': [location for location, _ in top_locations],
                'counts': [count for _, count in top_locations]
            },
            'job_title': job_title,
            'total_jobs': len(data["results"])
        }

        return jsonify(response_data)

    except requests.RequestException as e:
        print(f"API Error: {str(e)}")
        return jsonify({'error': 'Failed to fetch data from job API'}), 503
    except Exception as e:
        print(f"Error in company analysis: {str(e)}")
        return jsonify({'error': 'An error occurred during analysis'}), 500

@app.route('/analyze_job_competition', methods=['POST'])
def analyze_job_competition():
    if 'loggedin' not in session:
        return jsonify({'error': 'Not logged in'}), 401
        
    try:
        job_title = request.form.get('jobTitle')
        if not job_title:
            return jsonify({'error': 'Job title is required'}), 400

        # Load dataset
        df = pd.read_csv("preprocessed_dataset.csv")
        
        # Filter dataset for the given job
        job_data = df[df['Job Title'].str.contains(job_title, case=False, na=False)]

        if job_data.empty:
            return jsonify({'error': 'No data found for this job title'}), 404

        # Work Type Distribution
        work_type_counts = job_data['Work Type'].value_counts()
        work_type_data = {
            'labels': work_type_counts.index.tolist(),
            'values': work_type_counts.values.tolist()
        }

        # Company Size Distribution
        company_size_counts = job_data['Company Size'].value_counts()
        company_size_data = {
            'labels': company_size_counts.index.tolist(),
            'values': company_size_counts.values.tolist()
        }

        return jsonify({
            'work_type': work_type_data,
            'company_size': company_size_data,
            'job_title': job_title,
            'total_jobs': len(job_data)
        })

    except Exception as e:
        print(f"Error in job competition analysis: {str(e)}")
        return jsonify({'error': 'An error occurred during analysis'}), 500

@app.route('/analyze_job_market', methods=['POST'])
def analyze_job_market():
    if 'loggedin' not in session:
        return jsonify({'error': 'Not logged in'}), 401
        
    try:
        job_title = request.form.get('jobTitle')
        if not job_title:
            return jsonify({'error': 'Job title is required'}), 400

        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all analysis tasks
            salary_future = executor.submit(analyze_salary_data, job_title)
            competition_future = executor.submit(analyze_competition_data, job_title)
            industry_future = executor.submit(analyze_industry_data, job_title)

            # Get results as they complete
            salary_data = salary_future.result()
            competition_data = competition_future.result()
            industry_data = industry_future.result()

        return jsonify({
            'salary': salary_data,
            'competition': competition_data,
            'industry': industry_data,
            'job_title': job_title
        })

    except Exception as e:
        print(f"Error in market analysis: {str(e)}")
        return jsonify({'error': 'An error occurred during analysis'}), 500

def analyze_salary_data(job_title):
    try:
        # Use cached dataset loading
        df_job = load_and_filter_dataset(job_title)
        
        if df_job.empty:
            return {'error': 'No historical salary data found'}

        # Ensure safe filename by replacing special characters
        safe_job_title = ''.join(c if c.isalnum() or c == '_' else '_' for c in job_title)
        model_filename = f"models/salary_model_{safe_job_title}.pkl"

        if not os.path.exists(model_filename):
            return {'error': 'No salary data available for this job title'}

        # Load trained Prophet model
        with open(model_filename, "rb") as f:
            model = pickle.load(f)

        # Generate future dates for prediction (12 months)
        future = model.make_future_dataframe(periods=12, freq='ME')
        forecast = model.predict(future)

        # Optimize salary extraction with vectorized operations
        def extract_salary_vectorized(salary_range):
            if pd.isna(salary_range):
                return None
            numbers = pd.Series([int(s.replace(",", "").strip()) 
                               for s in re.findall(r'\d+', str(salary_range))])
            return numbers.mean() if not numbers.empty else None

        # Apply vectorized operations
        df_job['Average Salary'] = df_job['Salary Range'].apply(extract_salary_vectorized)
        df_job = df_job.dropna(subset=['Average Salary'])
        df_job['Job Posting Date'] = pd.to_datetime(df_job['Job Posting Date'], errors='coerce')
        df_job = df_job.dropna(subset=['Job Posting Date'])

        # Optimize resampling
        df_job = df_job.set_index('Job Posting Date')['Average Salary'].resample('ME').mean().reset_index()

        # Prepare data efficiently
        historical_data = df_job.rename(
            columns={'Job Posting Date': 'ds', 'Average Salary': 'y'}
        ).to_dict('records')

        forecast_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict('records')

        return {
            'historical_data': historical_data,
            'forecast_data': forecast_data
        }
    except Exception as e:
        return {'error': f'Error in salary analysis: {str(e)}'}

def analyze_competition_data(job_title):
    try:
        # Use cached dataset loading
        job_data = load_competition_dataset(job_title)

        if job_data.empty:
            return {'error': 'No competition data found'}

        # Optimize with vectorized operations
        work_type_data = job_data['Work Type'].value_counts()
        company_size_data = job_data['Company Size'].value_counts()

        return {
            'work_type': {
                'labels': work_type_data.index.tolist(),
                'values': work_type_data.values.tolist()
            },
            'company_size': {
                'labels': company_size_data.index.tolist(),
                'values': company_size_data.values.tolist()
            },
            'total_jobs': len(job_data)
        }
    except Exception as e:
        return {'error': f'Error in competition analysis: {str(e)}'}

def analyze_industry_data(job_title):
    try:
        # Optimize API request
        api_url = f"https://api.adzuna.com/v1/api/jobs/{ADZUNA_COUNTRY}/search/1"
        params = {
            'app_id': ADZUNA_APP_ID,
            'app_key': ADZUNA_APP_KEY,
            'results_per_page': 50,
            'what': job_title.lower(),
            'content-type': 'application/json'
        }

        # Use session for connection pooling
        with requests.Session() as session:
            response = session.get(api_url, params=params, timeout=5)
            data = response.json()

        if "results" not in data or not data["results"]:
            return {'error': 'No industry data found'}

        # Optimize data extraction with list comprehensions
        companies = [job["company"]["display_name"] 
                    for job in data["results"] 
                    if "company" in job and "display_name" in job["company"]]
        
        locations = [job["location"]["display_name"] 
                    for job in data["results"] 
                    if "location" in job and "display_name" in job["location"]]

        # Use Counter for efficient counting
        company_counts = Counter(companies).most_common(10)
        location_counts = Counter(locations).most_common(10)

        return {
            'companies': {
                'names': [company for company, _ in company_counts],
                'counts': [count for _, count in company_counts]
            },
            'locations': {
                'names': [location for location, _ in location_counts],
                'counts': [count for _, count in location_counts]
            },
            'total_jobs': len(data["results"])
        }
    except requests.RequestException as e:
        return {'error': f'API Error: {str(e)}'}
    except Exception as e:
        return {'error': f'Error in industry analysis: {str(e)}'}

def extract_job_title(text):
    patterns = [
        r'(?i)jobs?\s+(?:as\s+)?(?:an?\s+)?([a-zA-Z\s]+(?:developer|engineer|analyst|manager|designer|consultant|specialist|architect|administrator|technician))',
        r'(?i)(?:about|for)\s+(?:an?\s+)?([a-zA-Z\s]+(?:developer|engineer|analyst|manager|designer|consultant|specialist|architect|administrator|technician))',
        r'(?i)([a-zA-Z\s]+(?:developer|engineer|analyst|manager|designer|consultant|specialist|architect|administrator|technician))\s+(?:jobs?|positions?|roles?)'
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return None


def get_market_insights(job_title):
    """Get market insights for a specific job title."""
    try:
        # Get salary trends
        salary_data = analyze_salary_data(job_title)
        
        # Get industry data
        industry_data = analyze_industry_data(job_title)
        
        # Get competition data
        competition_data = analyze_competition_data(job_title)
        
        # Prepare insights summary
        insights = []
        
        if not salary_data.get('error'):
            avg_salary = sum(item['y'] for item in salary_data['historical_data']) / len(salary_data['historical_data'])
            insights.append(f"The average salary for {job_title} positions is ${avg_salary:,.2f}")
        
        if not industry_data.get('error'):
            top_company = industry_data['companies']['names'][0]
            top_location = industry_data['locations']['names'][0]
            total_jobs = industry_data['total_jobs']
            insights.append(f"Currently there are {total_jobs} job openings, with {top_company} being the top hiring company")
            insights.append(f"The most jobs are available in {top_location}")
        
        if not competition_data.get('error'):
            top_work_type = competition_data['work_type']['labels'][0]
            insights.append(f"The most common work type is {top_work_type}")
        
        return " | ".join(insights) if insights else "Sorry, I couldn't find detailed market insights for this role."
    except Exception as e:
        return f"I encountered an error while analyzing market data: {str(e)}"

def explain_job_recommendation(user_id, job_title):
    """Explain why a specific job was recommended."""
    try:
        # Get user's skills
        user_skills = get_user_skills(user_id)
        if not user_skills:
            return "I don't have your skills information. Please upload your resume first."
        
        # Find the job in recommendations
        recommendations = recommend_jobs(user_skills)
        matching_job = next((job for job in recommendations if job['Job Title'].lower() == job_title.lower()), None)
        
        if not matching_job:
            return "This job wasn't in your recommendations. Would you like me to analyze your skills for a better match?"
        
        # Find matching skills
        job_skills = extract_skills(matching_job['Job Description'])
        matching_skills = set(user_skills) & set(job_skills)
        
        if matching_skills:
            skills_text = ", ".join(list(matching_skills)[:3])
            return f"This job was recommended because your skills ({skills_text}) match the requirements. You have {len(matching_skills)} matching skills for this role."
        else:
            return "This job matches your overall profile, but I couldn't find specific skill matches. Would you like to see other recommendations?"
            
    except Exception as e:
        return f"I encountered an error while analyzing the job match: {str(e)}"

@app.route('/chatbot/message', methods=['POST'])
def chatbot_message():
    if 'loggedin' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        query_type = classify_query(user_message.lower())
        if query_type == QueryType.GREETING:
            return jsonify({'response': random.choice(CHATBOT_RESPONSES['greetings'])})
        elif query_type == QueryType.FAREWELL:
            return jsonify({'response': random.choice(CHATBOT_RESPONSES['farewell'])})
        user_context = get_user_context(session['id'])
        chatgpt_response = get_gemini_response(user_message, user_context)
        return jsonify({'response': chatgpt_response})
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)