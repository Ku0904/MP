import os
import io
import json
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from docx import Document
from PIL import Image
import PyPDF2
from bs4 import BeautifulSoup

@dataclass
class StudyRole:
    name: str
    description: str
    expertise: List[str]
    teaching_style: str

class GeminiModel:
    def __init__(self, api_key: str, temperature: float = 0.7, max_tokens: int = 2048):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')  # Updated to use gemini-1.5-flash
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate_response(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                    top_p=0.8,  # Added for better response consistency
                    top_k=40    # Added for better response quality
                )
            )
            return response.text
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return ""

class StudyGroupManager:
    def __init__(self, model: GeminiModel):
        self.model = model
        self.study_roles: Dict[str, StudyRole] = {}

    def create_study_roles(self, subject: str, num_roles: int) -> Dict[str, StudyRole]:
        prompt = f"""System: You are a specialized study group creator. Your task is to create {num_roles} distinct study buddy roles for learning {subject}. Respond with ONLY valid JSON.

Required format:
[
    {{"name": "<role name>", 
      "description": "<brief role description>", 
      "expertise": ["<skill1>", "<skill2>"], 
      "teaching_style": "<teaching approach>"}}
]

Constraints:
- Each role must be unique and complementary
- Names should be descriptive and professional
- Expertise should list 2-3 key skills
- Teaching style should be specific and clear
- Response must be valid JSON only

Generate the roles now:"""
        
        try:
            response = self.model.generate_response(prompt)
            # Clean the response to ensure it only contains the JSON part
            json_str = response.strip()
            if not json_str.startswith('['):
                start = json_str.find('[')
                end = json_str.rfind(']') + 1
                if start != -1 and end != 0:
                    json_str = json_str[start:end]
                else:
                    raise ValueError("No valid JSON array found in response")
            
            roles_data = json.loads(json_str)
            self.study_roles.clear()
            for role_data in roles_data:
                role = StudyRole(
                    name=role_data['name'],
                    description=role_data['description'],
                    expertise=role_data['expertise'],
                    teaching_style=role_data['teaching_style']
                )
                self.study_roles[role.name] = role
            return self.study_roles
        except json.JSONDecodeError as e:
            st.error(f"Error parsing JSON response: {str(e)}")
            st.code(response)
            return {}
        except Exception as e:
            st.error(f"Error creating study roles: {str(e)}")
            return {}

    def create_study_plan(self, subject: str, context: str, difficulty: str) -> Dict:
        prompt = f"""System: You are a specialized study plan generator. Create a structured study plan for {subject} at {difficulty} level.
Context: {context}

Required JSON format:
{{
    "learning_objectives": [
        "<clear, measurable objective>"
    ],
    "subtasks": [
        {{
            "id": "<task1>",
            "title": "<clear task title>",
            "description": "<detailed description>",
            "dependencies": ["<prerequisite tasks>"],
            "estimated_duration": "<time in minutes>",
            "best_role": "<matching study buddy role>"
        }}
    ]
}}

Constraints:
- Learning objectives must be specific and measurable
- Each subtask must be clearly defined
- Dependencies should reference other task IDs
- Duration should be realistic (15-60 minutes per task)
- Best role should match one of the study buddy roles
- Response must be valid JSON only

Generate the study plan now:"""
        
        try:
            response = self.model.generate_response(prompt)
            json_str = response.strip()
            if not json_str.startswith('{'):
                start = json_str.find('{')
                end = json_str.rfind('}') + 1
                if start != -1 and end != 0:
                    json_str = json_str[start:end]
                else:
                    raise ValueError("No valid JSON object found in response")
            
            plan_data = json.loads(json_str)
            # Validate the structure
            if not all(key in plan_data for key in ['learning_objectives', 'subtasks']):
                raise ValueError("Missing required keys in study plan")
            return plan_data
        except json.JSONDecodeError as e:
            st.error(f"Error parsing JSON response: {str(e)}")
            st.code(response)
            return {"learning_objectives": [], "subtasks": []}
        except Exception as e:
            st.error(f"Error creating study plan: {str(e)}")
            return {"learning_objectives": [], "subtasks": []}

def read_uploaded_file(uploaded_file):
    """Process the uploaded file and return its content."""
    if not uploaded_file:
        return None

    try:
        file_content = None
        if uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
            file_content = " ".join(page.extract_text() for page in pdf_reader.pages)
        elif uploaded_file.type in ["image/jpeg", "image/png"]:
            file_content = f"[Image uploaded: {uploaded_file.name}]"
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(io.BytesIO(uploaded_file.read()))
            file_content = "\n".join(para.text for para in doc.paragraphs)
        elif uploaded_file.type == "text/plain":
            file_content = uploaded_file.getvalue().decode("utf-8")
        else:
            st.warning(f"Unsupported file format: {uploaded_file.type}")
            return None
        
        # Truncate content if it's too long
        max_length = 1000
        if len(file_content) > max_length:
            file_content = file_content[:max_length] + "... [content truncated]"
        
        return file_content
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="AI Study Buddy", page_icon="📚", layout="wide")
    st.title("📚 AI Study Buddy")

    # Initialize session state
    if 'study_roles' not in st.session_state:
        st.session_state.study_roles = {}
    if 'study_plan' not in st.session_state:
        st.session_state.study_plan = {}

    # Sidebar for API key and study session setup
    with st.sidebar:
        st.subheader("Setup")
        api_key = st.text_input("Google API Key", type="password", help="Enter your Google API key")
        api_key1 = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key")
        api_key2 = st.text_input("Ollama API Key", type="password", help="Enter your Ollama API key")
        uploaded_file = st.file_uploader("Upload Study Material", type=["txt", "pdf", "docx", "jpeg", "png"])
        subject = st.text_input("Study Subject", placeholder="e.g., Python Programming")
        difficulty = st.selectbox("Difficulty Level", ["Beginner", "Intermediate", "Advanced"])
        context = st.text_area("Study Goals", placeholder="What do you want to focus on?")
        num_roles = st.slider("Number of Study Buddies", min_value=2, max_value=5, value=3)
        start_button = st.button("Start Session")

    if start_button and api_key and subject and context:
        try:
            with st.spinner("Initializing study session..."):
                model = GeminiModel(api_key=api_key)
                manager = StudyGroupManager(model)

                if uploaded_file:
                    file_content = read_uploaded_file(uploaded_file)
                    if file_content:
                        context += f"\n\nUploaded Content:\n{file_content}"

                # Create and display study buddies
                with st.spinner("Creating your personalized study team..."):
                    st.subheader("🤝 Study Buddies")
                    study_roles = manager.create_study_roles(subject, num_roles)
                    if study_roles:
                        st.session_state.study_roles = study_roles
                        cols = st.columns(len(study_roles))
                        for idx, (_, role) in enumerate(study_roles.items()):
                            with cols[idx]:
                                st.markdown(f"### 👤 {role.name}")
                                st.markdown(f"**Role:** {role.description}")
                                st.markdown("**Skills:**")
                                for skill in role.expertise:
                                    st.markdown(f"- {skill}")
                                st.markdown(f"**Style:** {role.teaching_style}")

                # Create and display study plan
                with st.spinner("Generating your personalized study plan..."):
                    st.subheader("📖 Study Plan")
                    study_plan = manager.create_study_plan(subject, context, difficulty)
                    if study_plan:
                        st.session_state.study_plan = study_plan
                        
                        # Display learning objectives
                        st.markdown("### 🎯 Learning Objectives")
                        for idx, objective in enumerate(study_plan.get("learning_objectives", []), 1):
                            st.markdown(f"{idx}. {objective}")
                        
                        # Display subtasks with progress tracking
                        st.markdown("### 📝 Study Tasks")
                        for task in study_plan.get("subtasks", []):
                            with st.expander(f"📌 {task['title']} ({task['estimated_duration']})"):
                                st.markdown(f"**Description:** {task['description']}")
                                st.markdown(f"**Prerequisites:** {', '.join(task.get('dependencies', ['None']))}")
                                st.markdown(f"**Study Buddy:** {task['best_role']}")
                                # Add task completion checkbox
                                task_key = f"task_{task['id']}"
                                if task_key not in st.session_state:
                                    st.session_state[task_key] = False
                                st.checkbox("Mark as completed", key=task_key)

        except Exception as e:
            st.error(f"Session Error: {str(e)}")
            st.error("Please check your API key and try again.")

if __name__ == "__main__":
    main()
