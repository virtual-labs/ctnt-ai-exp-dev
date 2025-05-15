import streamlit as st
from Agents.CodingAgent import CodingAgent
from Agents.HumanReviewAgentForRequirement import HumanReviewAgentForRequirement
from Agents.ImplementationAgent import ImplementationAgent
from Agents.RequirementsAgent import RequirementsAgent
from Agents.DocumentationAgent import DocumentationAgent
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import subprocess
from pathlib import Path
import http.server
import socketserver
import threading
import webbrowser
import socket
from Agents.WebsiteDesignAgent import WebsiteDesignAgent  # Import the WebsiteAgent

# ---------------------------------------------------
# Configuration & Initialization
# ---------------------------------------------------

# Initialize the language model instance. This will be used by all agents.
def init_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-pro-exp-03-25",
        temperature=0.1,
        max_tokens=100000,
        google_api_key = os.getenv('GOOGLE_API_KEY')
    )

if "llm" not in st.session_state:
    st.session_state.llm = init_llm()

# Prepare session states for pipeline steps
if "requirements_output" not in st.session_state:
    st.session_state.requirements_output = ""
if "reviewed_requirements" not in st.session_state:
    st.session_state.reviewed_requirements = ""
if "implementation_output" not in st.session_state:
    st.session_state.implementation_output = ""
if "code_loop" not in st.session_state:
    st.session_state.code_loop = 0
if "coding_agent_output" not in st.session_state:
    st.session_state.coding_agent_output = ""
if "documentation_output" not in st.session_state:
    st.session_state.documentation_output = ""
if 'server_started' not in st.session_state:
    st.session_state.server_started = False
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

MAX_CODE_LOOP = 3

# ---------------------------------------------------
# Agent Functions
# ---------------------------------------------------

def generate_requirements():
    """Generate initial requirements from an uploaded PDF using the RequirementsAgent."""
    if st.session_state.uploaded_file is None:
        st.error("Please upload a PDF file first!")
        return ""
    
    req_agent = RequirementsAgent(str(st.session_state.uploaded_file))
    req_agent.set_llm(st.session_state.llm)
    req_agent.set_prompt_enhancer_llm(st.session_state.llm)
    req_agent.enhance_prompt()
    output = req_agent.get_output()
    return output

def review_requirements(user_review, base_text):
    """Let the human review the generated requirements.
       If no review is provided, the original output is kept.
    """
    if user_review.strip() == "":
        return base_text
    review_agent = HumanReviewAgentForRequirement(user_review, base_text)
    review_agent.set_llm(st.session_state.llm)
    review_agent.set_prompt_enhancer_llm(st.session_state.llm)
    review_agent.enhance_prompt()
    return review_agent.get_output()

def generate_implementation(requirements_text):
    """Generate implementation output from reviewed requirements."""
    impl_agent = ImplementationAgent(requirements_text)
    impl_agent.set_llm(st.session_state.llm)
    impl_agent.set_prompt_enhancer_llm(st.session_state.llm)
    return impl_agent.get_output()

def generate_code(impl_text, code_review):
    """Generate code via the CodingAgent given implementation output and code review feedback."""
    coding_agent = CodingAgent(impl_text, code_review)
    coding_agent.set_llm(st.session_state.llm)
    coding_agent.set_prompt_enhancer_llm(st.session_state.llm)
    coding_agent.enhance_prompt()
    return coding_agent.get_output()

def generate_documentation(code_text):
    """Generate documentation using the DocumentationAgent."""
    # Combine requirements, implementation, and code for richer context
    context = (
        f"Requirements:\n{st.session_state.reviewed_requirements}\n\n"
        f"Implementation Plan:\n{st.session_state.implementation_output}\n\n"
        f"Code:\n{code_text}"
    )
    doc_agent = DocumentationAgent(context)
    doc_agent.set_llm(st.session_state.llm)
    doc_agent.set_prompt_enhancer_llm(st.session_state.llm)
    doc_agent.enhance_prompt()
    return doc_agent.get_output()

def start_http_server(directory, port=8000):
    """Start a HTTP server in a separate thread"""
    if st.session_state.server_started:
        return True
        
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)
    
    try:
        # Test if port is available
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('', port))
        sock.close()
        
        # Create server
        handler = Handler
        httpd = socketserver.TCPServer(("", port), handler)
        
        # Start server in a separate thread
        server_thread = threading.Thread(target=httpd.serve_forever)
        server_thread.daemon = True  # Thread will close when main program exits
        server_thread.start()
        
        st.session_state.server_started = True
        return True
        
    except OSError as e:
        if e.errno in [98, 10048]:  # Port already in use (Linux: 98, Windows: 10048)
            st.session_state.server_started = True
            return True
        st.error(f"Server error: {str(e)}")
        return False

def save_and_serve_code(code_content):
    """Save code to file and start server if needed"""
    try:
        # Get the directory of the current file
        current_dir = Path(__file__).parent
        code_file_path = current_dir / "code.html"
        
        # Save the code to file
        with open(code_file_path, "w", encoding="utf-8") as f:
            f.write(code_content)
        
        # Start the server if not already running
        if start_http_server(current_dir):
            return f"http://localhost:8000/code.html"
        return None
            
    except Exception as e:
        st.error(f"Error saving code file: {str(e)}")
        return None

def generate_website(simulation_code, website_feedback=None, previous_website_code=None):
    """Generate a complete Virtual Lab Website using the WebsiteAgent."""
    website_agent = WebsiteDesignAgent(simulation_code, website_feedback, previous_website_code)
    website_agent.set_llm(st.session_state.llm)
    website_agent.set_prompt_enhancer_llm(st.session_state.llm)
    website_agent.enhance_prompt()
    return website_agent.get_output()

# ---------------------------------------------------
# Streamlit UI
# ---------------------------------------------------
st.title("Interactive Pipeline UI")

# Step 1: Generate Requirements
st.header("1. Requirements Generation")
uploaded_file = st.file_uploader("Upload your requirements PDF", type=['pdf'])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    temp_path = Path("temp_requirements.pdf")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    st.session_state.uploaded_file = temp_path
    
    if st.button("Generate Requirements"):
        with st.spinner("Generating requirements from uploaded PDF..."):
            # Modify the generate_requirements function call
            req_agent = RequirementsAgent(str(st.session_state.uploaded_file))
            req_agent.set_llm(st.session_state.llm)
            req_agent.set_prompt_enhancer_llm(st.session_state.llm)
            req_agent.enhance_prompt()
            st.session_state.requirements_output = req_agent.get_output()
        st.success("Requirements generated.")

if st.session_state.requirements_output:
    st.subheader("Requirements Output")
    st.text_area("Generated Requirements", st.session_state.requirements_output, height=200)

# Step 2: Human Review of Requirements
st.header("2. Human Review for Requirements")
review_text = st.text_area("Enter your review for the requirements (leave blank to use generated text)", height=100)
if st.button("Submit Requirements Review"):
    if st.session_state.requirements_output == "":
        st.warning("You must first generate the requirements!")
    else:
        reviewed = review_requirements(review_text, st.session_state.requirements_output)
        st.session_state.reviewed_requirements = reviewed
        st.success("Requirements reviewed.")
        st.text_area("Reviewed Requirements", reviewed, height=200)

# Step 3: Implementation Generation
st.header("3. Generate Implementation")
if st.button("Generate Implementation"):
    if st.session_state.reviewed_requirements == "":
        st.warning("You must review the requirements before generating implementation!")
    else:
        with st.spinner("Generating implementation..."):
            impl_output = generate_implementation(st.session_state.reviewed_requirements)
            st.session_state.implementation_output = impl_output
        st.success("Implementation generated.")
        st.text_area("Implementation Output", impl_output, height=200)

# Step 4: Iterative Code Generation with Review
st.header("4. Code Generation and Review")
if st.session_state.implementation_output == "":
    st.info("Generate implementation first to start code generation.")
else:
    st.write(f"Code Generation Iteration: {st.session_state.code_loop + 1} of {MAX_CODE_LOOP}")
    code_review_input = st.text_area("Enter your code review feedback (for the current iteration)", height=100)
    if st.button("Generate/Refine Code"):
        with st.spinner("Generating code..."):
            # Use implementation output for the first iteration, then use the previous code
            input_text = (
                st.session_state.implementation_output
                if st.session_state.code_loop == 0
                else st.session_state.coding_agent_output
            )
            new_code_output = generate_code(input_text, code_review_input)
            st.session_state.coding_agent_output = new_code_output
            st.session_state.code_loop += 1
            
            # Save code and get URL
            localhost_url = save_and_serve_code(new_code_output)
            if localhost_url:
                st.success("Code generated/refined.")
                st.code(new_code_output, language="html")
                
                # Add link to view in browser
                st.markdown(f"""
                ### View Live Preview
                Click here to view the code in your browser: [Open Preview]({localhost_url})
                
                Note: Preview server is running on port 8000
                """)
                
                # Optionally, open the browser automatically
                webbrowser.open(localhost_url)
    if st.session_state.code_loop >= MAX_CODE_LOOP:
        st.info("Reached maximum number of code generation iterations.")

# Step 5: Documentation Generation
st.header("5. Generate Documentation")
if st.session_state.coding_agent_output == "":
    st.info("Please complete at least one code generation iteration first.")
else:
    # Show current iteration status
    st.info(f"Using code from iteration {st.session_state.code_loop} of {MAX_CODE_LOOP}")
    
    if st.button("Generate Documentation"):
        with st.spinner("Generating documentation..."):
            doc_output = generate_documentation(st.session_state.coding_agent_output)
            st.session_state.documentation_output = doc_output
        st.success("Documentation generated.")
        st.markdown(doc_output)

# Step 6: Generate Complete Virtual Lab Website
st.header("6. Generate Complete Virtual Lab Website")
# Add website design feedback text area
website_feedback = st.text_area(
    "Enter your feedback for the website design (optional)",
    help="Provide any specific requirements for colors, layout, features, etc.",
    height=150
)

if st.button("Generate Virtual Lab Website"):
    with st.spinner("Generating complete website with all sections..."):
        # Use the coding agent output directly instead of reading from yoyo.html
        if not st.session_state.coding_agent_output:
            st.error("No code has been generated yet. Please complete the code generation step first.")
        else:
            # Get the simulation code directly from the coding agent's output
            simulation_code = st.session_state.coding_agent_output
            
            # Check if we have previous website code and feedback
            previous_website_code = st.session_state.get("website_output", None) if website_feedback else None
            
            # Generate the website using the WebsiteAgent
            website_output = generate_website(
                simulation_code, 
                website_feedback=website_feedback,
                previous_website_code=previous_website_code
            )
            st.session_state.website_output = website_output
            
            # Save website and get URL
            localhost_url = save_and_serve_code(website_output)
            if localhost_url:
                st.success("Virtual Lab Website generated!")
                
                # Show a preview of the first part of the HTML
                preview_length = 1000  # Show first 1000 chars
                if len(website_output) > preview_length:
                    preview = website_output[:preview_length] + "... (truncated)"
                else:
                    preview = website_output
                    
                st.code(preview, language="html")
                
                # Add link to view in browser
                st.markdown(f"""
                ### View Complete Virtual Lab
                Click here to view the complete virtual lab in your browser: [Open Virtual Lab]({localhost_url})
                
                Note: Preview server is running on port 8000
                """)
                
                # Optionally, open the browser automatically
                webbrowser.open(localhost_url)

# Optional: Reset pipeline
if st.button("Reset Pipeline"):
    for key in ["requirements_output", "reviewed_requirements", "implementation_output",
                "code_loop", "coding_agent_output", "documentation_output", "website_output", "uploaded_file"]:
        st.session_state[key] = "" if key != "code_loop" else 0
    
    # Clean up temporary file if it exists
    if Path("temp_requirements.pdf").exists():
        Path("temp_requirements.pdf").unlink()
    
    st.success("Pipeline reset.")
