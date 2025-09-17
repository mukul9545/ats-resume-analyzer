from flask import Flask, render_template, request, flash, session, jsonify
from groq import Groq
import os
import PyPDF2 as pdf
from dotenv import load_dotenv
import json
import re
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import time
import hashlib

# Load environment variables
load_dotenv()

# Initialize Groq client
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-here')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global cache for responses
response_cache = {}

# Improved function to clean and validate JSON response
def clean_json_response(response):
    response = response.strip()
    
    # Remove any markdown code fences if present
    response = re.sub(r'^```json\s*', '', response, flags=re.MULTILINE)
    response = re.sub(r'\s*```$', '', response, flags=re.MULTILINE)
    response = re.sub(r'^```\s*', '', response, flags=re.MULTILINE)
    
    # Find the JSON object boundaries
    start_idx = response.find('{')
    if start_idx == -1:
        return response
    
    # Find the matching closing brace
    brace_count = 0
    end_idx = -1
    for i, char in enumerate(response[start_idx:], start_idx):
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                end_idx = i
                break
    
    if end_idx != -1:
        json_str = response[start_idx:end_idx + 1]
    else:
        json_str = response[start_idx:]
    
    # Clean up common JSON issues
    json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas before }
    json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas before ]
    json_str = re.sub(r'"\s*\n\s*"', '" "', json_str)  # Fix broken strings across lines
    
    return json_str

# Function to get Groq response with retry logic
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(Exception)
)
def get_groq_response(prompt, model="llama-3.3-70b-versatile"):
    cache_key = hashlib.sha256((prompt + model).encode()).hexdigest()
    if cache_key in response_cache:
        return response_cache[cache_key]
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert ATS system. You MUST respond with ONLY valid JSON. Do not include any text before or after the JSON object. Ensure all strings are properly escaped and all arrays/objects are properly closed with correct comma placement."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model=model,
            temperature=0.1,  # Reduced temperature for more consistent output
            max_tokens=8000,  # Increased token limit for 25 questions
            top_p=0.7,
            stream=False,
        )
        
        response_text = chat_completion.choices[0].message.content
        response_cache[cache_key] = response_text
        return response_text
        
    except Exception as e:
        print(f"Groq API Error: {str(e)}")
        if model != "llama-3.3-8b-instant":
            try:
                time.sleep(1)
                return get_groq_response(prompt, "llama-3.3-8b-instant")
            except Exception as fallback_error:
                return f"Error: {str(fallback_error)}"
        return f"Error: {str(e)}"
    
# Function to extract text from PDF
def input_pdf_text(uploaded_file):
    try:
        reader = pdf.PdfReader(uploaded_file)
        text = ""
        for page in range(len(reader.pages)):
            page_obj = reader.pages[page]
            text += str(page_obj.extract_text() or "")
        return text
    except Exception as e:
        return f"Error extracting PDF text: {str(e)}"

# Improved ATS prompt with 25 questions
def create_ats_prompt(resume_text, jd_text):
    resume_text = resume_text[:6000]
    jd_text = jd_text[:2500]

    input_prompt = f"""
You are an expert ATS system and senior technical recruiter. Analyze this resume against the job description and respond with ONLY valid JSON.

JOB DESCRIPTION:
{jd_text}

RESUME:
{resume_text}

You must respond with ONLY a valid JSON object in this EXACT format:

{{
    "JD_Match": "XX%",
    "Missing_Keywords": ["keyword1", "keyword2", "keyword3"],
    "Recommendations": [
        {{
            "Priority": "High",
            "Area": "Missing Skills", 
            "Recommendation": "Specific skill to develop based on JD requirements",
            "Action_Items": ["Actionable step 1", "Actionable step 2"],
            "Timeline": "Suggested timeframe",
            "Impact": "How this will improve JD match"
        }}
    ],
    "Profile_Summary": "Brief summary of candidate profile and fit for the role",
    "Detailed_Analysis": {{
        "Technical_Skills_Match": "Analysis of technical skills alignment",
        "Experience_Relevance": "Assessment of experience relevance",
        "Education_Assessment": "Education evaluation",
        "Key_Strengths": ["strength1", "strength2", "strength3"],
        "Improvement_Areas": ["area1", "area2", "area3"],
        "ATS_Compatibility_Score": "X/10"
    }},
    "Interview_Questions_Answers": [
        {{
            "Category": "Project Experience",
            "Question": "Tell me about a challenging project from your resume and how you overcame technical obstacles",
            "Answer": "Based on resume context, provide specific project details and problem-solving approach"
        }},
        {{
            "Category": "Technical Skills",
            "Question": "How would you apply [specific technology from resume] to solve [JD requirement]",
            "Answer": "Demonstrate understanding using candidate's background"
        }},
        {{
            "Category": "System Design",
            "Question": "How would you scale one of your projects to handle 10x more users",
            "Answer": "Provide architectural considerations based on candidate's experience"
        }},
        {{
            "Category": "Problem Solving",
            "Question": "Describe a time when you had to debug a complex issue in your project",
            "Answer": "Use specific examples from candidate's work"
        }},
        {{
            "Category": "Technical Deep Dive",
            "Question": "Explain the trade-offs in your technology choices for [specific project]",
            "Answer": "Show technical reasoning based on resume projects"
        }},
        {{
            "Category": "Behavioral",
            "Question": "How do you handle tight deadlines based on your project experience",
            "Answer": "Reference actual project timelines and management"
        }},
        {{
            "Category": "Code Quality",
            "Question": "How do you ensure code quality in your projects",
            "Answer": "Discuss practices mentioned or implied in resume"
        }},
        {{
            "Category": "Learning",
            "Question": "Tell me about a new technology you learned for one of your projects",
            "Answer": "Reference learning curve from resume projects"
        }},
        {{
            "Category": "Collaboration",
            "Question": "How did you work with team members in your projects",
            "Answer": "Use team-based project experience from resume"
        }},
        {{
            "Category": "Technical Challenge",
            "Question": "What was the most technically challenging aspect of [specific project]",
            "Answer": "Dive deep into technical complexity from resume"
        }},
        {{
            "Category": "Architecture",
            "Question": "Walk me through the architecture of your most complex project",
            "Answer": "Explain system design based on resume details"
        }},
        {{
            "Category": "Optimization",
            "Question": "How would you improve the performance of [specific project from resume]",
            "Answer": "Suggest optimizations based on project context"
        }},
        {{
            "Category": "Testing",
            "Question": "How did you test and validate your projects",
            "Answer": "Discuss testing approaches used or suitable for their projects"
        }},
        {{
            "Category": "Deployment",
            "Question": "How did you deploy and maintain your applications",
            "Answer": "Reference deployment experience or suitable approaches"
        }},
        {{
            "Category": "Database",
            "Question": "Explain your database design choices in [specific project]",
            "Answer": "Justify data modeling decisions from resume context"
        }},
        {{
            "Category": "Security",
            "Question": "How did you handle security considerations in your projects",
            "Answer": "Discuss security measures relevant to their work"
        }},
        {{
            "Category": "Version Control",
            "Question": "Describe your workflow for managing code changes in team projects",
            "Answer": "Explain Git/version control practices from experience"
        }},
        {{
            "Category": "API Design",
            "Question": "How would you design an API for [functionality from their project]",
            "Answer": "Apply API design principles to their project context"
        }},
        {{
            "Category": "Data Handling",
            "Question": "How did you handle data processing in [specific project]",
            "Answer": "Explain data pipeline or processing logic from resume"
        }},
        {{
            "Category": "Error Handling",
            "Question": "How do you handle errors and exceptions in your code",
            "Answer": "Discuss error handling strategies used in their projects"
        }},
        {{
            "Category": "Monitoring",
            "Question": "How would you monitor the health of your application in production",
            "Answer": "Suggest monitoring approaches suitable for their projects"
        }},
        {{
            "Category": "Scalability",
            "Question": "What bottlenecks might your project face at scale",
            "Answer": "Identify scaling challenges based on their architecture"
        }},
        {{
            "Category": "Technology Choice",
            "Question": "Why did you choose [specific technology] for your project",
            "Answer": "Justify technology decisions from resume context"
        }},
        {{
            "Category": "Future Improvements",
            "Question": "If you could rebuild [project name], what would you do differently",
            "Answer": "Reflect on lessons learned and improvements"
        }},
        {{
            "Category": "Industry Application",
            "Question": "How would you apply your project experience to this role",
            "Answer": "Connect resume experience to JD requirements"
        }}
    ]
}}

CRITICAL REQUIREMENTS:
- Generate EXACTLY 25 interview Q&A pairs
- At least 15 questions must directly reference specific projects, internships, or experiences from the candidate's resume
- Questions must sound realistic and conversational, like from a real technical interviewer
- Each answer must leverage resume details and context wherever possible
- Provide 5-7 detailed recommendations with specific action items, timelines, and impact analysis
- Recommendations must directly address gaps between resume and JD requirements
- Include both technical and soft skill recommendations based on JD needs
- Ensure proper JSON formatting with correct comma placement
- Keep individual responses concise but meaningful
- JD_Match percentage must be realistic based on actual skill/requirement alignment
- Missing_Keywords must be actual keywords from JD that are missing in resume
"""
    return input_prompt

# Alternative simplified analysis for fallback
def create_simple_ats_prompt(resume_text, jd_text):
    resume_text = resume_text[:4000]
    jd_text = jd_text[:2000]

    input_prompt = f"""
Analyze this resume against the job description. Respond with ONLY valid JSON:

JOB DESCRIPTION: {jd_text}
RESUME: {resume_text}

{{
    "JD_Match": "XX%",
    "Missing_Keywords": ["keyword1", "keyword2"],
    "Recommendations": ["rec1", "rec2"],
    "Profile_Summary": "Brief summary",
    "Key_Strengths": ["strength1", "strength2"],
    "Improvement_Areas": ["area1", "area2"]
}}
"""
    return input_prompt

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_resume():
    try:
        jd = request.form.get('job_description', '').strip()
        if 'resume_file' not in request.files:
            flash('No resume file uploaded', 'error')
            return render_template('index.html')
        
        file = request.files['resume_file']
        if file.filename == '':
            flash('No file selected', 'error')
            return render_template('index.html')
        if not jd:
            flash('Please provide a job description', 'error')
            return render_template('index.html')
        if not file.filename.lower().endswith('.pdf'):
            flash('Please upload a PDF file', 'error')
            return render_template('index.html')
        
        resume_text = input_pdf_text(file)
        if "Error" in resume_text:
            flash(resume_text, 'error')
            return render_template('index.html')
        
        # Try the full prompt first
        prompt = create_ats_prompt(resume_text, jd)
        response = get_groq_response(prompt, "llama-3.3-70b-versatile")
        
        # Check if response actually contains an error (not just the word "Error" in content)
        if response.startswith("Error:"):
            flash(f'API Error: {response}', 'error')
            return render_template('index.html')
        
        try:
            # Clean and parse JSON
            cleaned_response = clean_json_response(response)
            response_json = json.loads(cleaned_response)
            session['analysis_results'] = response_json
            return render_template('results.html', results=response_json)
            
        except json.JSONDecodeError as e:
            print(f"JSON Parse Error: {str(e)}")
            print(f"Raw response length: {len(response)}")
            print(f"Cleaned response: {cleaned_response[:500]}...")
            
            # Fallback to simpler prompt
            try:
                simple_prompt = create_simple_ats_prompt(resume_text, jd)
                simple_response = get_groq_response(simple_prompt, "llama-3.3-8b-instant")
                
                if "Error" not in simple_response and not simple_response.startswith("Error:"):
                    cleaned_simple = clean_json_response(simple_response)
                    simple_json = json.loads(cleaned_simple)
                    session['analysis_results'] = simple_json
                    flash('Analysis completed with simplified results due to formatting issues.', 'warning')
                    return render_template('results.html', results=simple_json)
                    
            except Exception as fallback_error:
                print(f"Fallback error: {str(fallback_error)}")
            
            flash(f'Failed to parse response. Please try again. Error: {str(e)}', 'error')
            return render_template('index.html')
    
    except Exception as e:
        flash(f'Unexpected error: {str(e)}', 'error')
        return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    try:
        if 'resume_file' not in request.files:
            return jsonify({'error': 'No resume file uploaded'}), 400
        
        file = request.files['resume_file']
        jd = request.form.get('job_description', '').strip()
        
        if file.filename == '' or not jd:
            return jsonify({'error': 'Missing job description or resume file'}), 400
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'Only PDF files are supported'}), 400
        
        resume_text = input_pdf_text(file)
        if "Error" in resume_text:
            return jsonify({'error': resume_text}), 500
        
        prompt = create_ats_prompt(resume_text, jd)
        response = get_groq_response(prompt, "llama-3.3-70b-versatile")
        if response.startswith("Error:"):
            return jsonify({'error': response}), 500
        
        try:
            cleaned_response = clean_json_response(response)
            response_json = json.loads(cleaned_response)
            return jsonify(response_json)
        except json.JSONDecodeError as e:
            # Try fallback
            try:
                simple_prompt = create_simple_ats_prompt(resume_text, jd)
                simple_response = get_groq_response(simple_prompt, "llama-3.3-8b-instant")
                cleaned_simple = clean_json_response(simple_response)
                simple_json = json.loads(cleaned_simple)
                return jsonify(simple_json)
            except:
                return jsonify({'error': f'Failed to parse response: {str(e)}', 'raw_response': response}), 500
                
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(413)
def too_large(e):
    flash('File too large. Please upload a file smaller than 16MB.', 'error')
    return render_template('index.html'), 413

if __name__ == '__main__':
    app.run(debug=True)