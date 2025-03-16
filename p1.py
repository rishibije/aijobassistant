import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure the Gemini API key
genai.configure(api_key=os.getenv("GOOGLE_GEMINI_API_KEY"))

# Choose the correct available model
model_name = "gemini-1.5-pro-latest"  # Change if needed

# Initialize the model
model = genai.GenerativeModel(model_name=model_name)

# Example prompt
response = model.generate_content("Hello! How can you assist me with my career?")

# Print response
print(response.text)
