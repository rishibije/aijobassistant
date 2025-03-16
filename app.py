from flask import Flask, request, render_template, redirect, url_for, flash, session
import PyPDF2
import spacy
import os
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

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# Define a set of common skills across multiple domains
SKILLS_SET = {
    # Programming Languages
    'python', 'javascript', 'java', 'c++', 'c#', 'typescript', 'go', 'ruby', 
    'swift', 'kotlin', 'rust', 'dart', 'r', 'scala', 'perl', 'bash',

    # Web Development
    'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask', 'spring', 
    'php', 'laravel', 'magento', 'wordpress', 'bootstrap', 'jquery', 'html', 'css', 
    'sass', 'less', 'tailwind', 'next.js', 'nuxt.js', 'svelte', 'astro',

    # Databases
    'sql', 'mysql', 'postgresql', 'mongodb', 'cassandra', 'firebase', 'redis', 
    'elasticsearch', 'sqlite', 'dynamodb', 'oracle', 'neo4j', 'couchdb',

    # Cloud & DevOps
    'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'ansible', 'jenkins', 
    'git', 'github', 'gitlab', 'bitbucket', 'helm', 'prometheus', 'grafana', 'ci/cd',

    # Mobile Development
    'android', 'ios', 'flutter', 'react native', 'swift', 'kotlin', 'dart', 'xamarin', 

    # Cybersecurity
    'penetration testing', 'ethical hacking', 'nmap', 'metasploit', 'kali linux', 
    'wireshark', 'burp suite', 'splunk', 'firewalls', 'siem', 'soc',

    # Data Science & Machine Learning
    'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy', 'matplotlib', 
    'seaborn', 'keras', 'opencv', 'hadoop', 'spark', 'mlflow', 'bigquery',

    # System Administration & Networking
    'linux', 'windows', 'macos', 'nginx', 'apache', 'bash scripting', 'powershell', 
    'networking', 'tcp/ip', 'dns', 'vpn', 'firewalls'

    # Finance & Accounting
    'financial analysis', 'budgeting', 'taxation', 'auditing', 'investment analysis', 
    'risk management', 'accounting', 'cost accounting', 'financial modeling', 'quickbooks',

    # Healthcare & Medical
    'nursing', 'medical coding', 'patient care', 'pharmacology', 'clinical research', 
    'dental hygiene', 'physical therapy', 'telemedicine', 'emergency medicine',

    # Sales & Marketing
    'seo', 'digital marketing', 'social media marketing', 'content writing', 
    'google ads', 'facebook ads', 'email marketing', 'branding', 'crm', 'copywriting',
    # Human Resources (HR)
    'recruitment', 'talent acquisition', 'payroll management', 'hr analytics', 
    'employee relations', 'organizational development', 'performance appraisal',

    # Education & Teaching
    'curriculum development', 'lesson planning', 'classroom management', 'special education', 
    'e-learning', 'online tutoring', 'edtech tools',

    # Legal & Compliance
    'corporate law', 'contract drafting', 'intellectual property', 'compliance', 
    'legal research', 'dispute resolution', 'case management',

    # Miscellaneous
    'public speaking', 'negotiation', 'problem solving', 'time management', 
    'teamwork', 'leadership', 'adaptability', 'critical thinking'

    # Engineering (Mechanical, Electrical, Civil)
    'cad', 'solidworks', 'autocad', 'hvac', 'thermodynamics', 'structural analysis', 
    'control systems', 'electrical wiring', 'robotics', 'circuit design', 'renewable energy',


    # Management & Business
    'project management', 'business strategy', 'supply chain management', 'lean manufacturing', 
    'scrum', 'agile', 'operations management', 'market research', 'customer relationship management',

}


def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def extract_skills(text):
    """Extract skills from text using spaCy and match with SKILLS_SET."""
    doc = nlp(text.lower())
    extracted_skills = set()

    # Extract single-word skills
    for token in doc:
        if token.text in SKILLS_SET:
            extracted_skills.add(token.text)

    # Extract multi-word skills using noun phrases
    for chunk in doc.noun_chunks:
        if chunk.text in SKILLS_SET:
            extracted_skills.add(chunk.text)

    return list(extracted_skills)  # Convert to list before returning

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
                flash('Your skills have been updated successfully!')
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

if __name__ == '__main__':
    app.run(debug=True, port=5000)
