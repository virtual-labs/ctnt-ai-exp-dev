# from Agents.BaseAgent import BaseAgent
from BaseAgent import BaseAgent
import json
import os
import markdown
import re

class WebsiteDesignAgent(BaseAgent):
    """
    Agent responsible for creating a complete webpage with multiple sections/tabs
    by combining simulation code with content from markdown and JSON files.
    """
    
    def __init__(self, simulation_code, aim_path=None, theory_path=None, procedure_path=None, 
                 objective_path=None, pretest_path=None, enhanced_css=False, left_tabs=False,
                 generate_procedure=False, generate_content=False, feedback="", 
                 aim_content=None, theory_content=None, objective_content=None, pretest_content=None):
        """
        Initialize the Website Design Agent
        
        Args:
            simulation_code (str): HTML/JS code for the simulation from CodingAgent
            aim_path (str): Path to aim.md file
            theory_path (str): Path to theory.md file  
            procedure_path (str): Path to procedure.md file
            objective_path (str): Path to objective.md file
            pretest_path (str): Path to pretest.json file
            enhanced_css (bool): Flag to indicate if enhanced CSS should be used
            left_tabs (bool): Flag to indicate if tabs should be on the left
            generate_procedure (bool): Flag to indicate if procedure should be generated
            generate_content (bool): Flag to indicate if all content should be generated from simulation
            feedback (str): User feedback for website customization
            aim_content (str/dict): Content for aim section, provided directly instead of from file
            theory_content (str/dict): Content for theory section, provided directly instead of from file
            objective_content (str/dict): Content for objective section, provided directly instead of from file
            pretest_content (list): Pretest questions provided directly instead of from file
        """
        # Define role and basic prompt for the BaseAgent constructor
        role = "Web Developer"
        basic_prompt = "Create a complete virtual lab website with professional styling, intuitive navigation, and comprehensive content organization."
        
        # Call the parent class constructor with required arguments
        super().__init__(role=role, basic_prompt=basic_prompt)
        
        self.simulation_code = simulation_code
        self.aim_path = aim_path
        self.theory_path = theory_path
        self.procedure_path = procedure_path
        self.objective_path = objective_path
        self.pretest_path = pretest_path
        self.enhanced_css = enhanced_css
        self.left_tabs = left_tabs
        self.generate_procedure = generate_procedure
        self.generate_content = generate_content
        self.custom_enhancement = None
        self.feedback = feedback
        self.llm = None
        self.prompt_enhancer_llm = None
        self.prompt_template = ""
        
        # Direct content (overrides file paths if provided)
        self.aim_content = aim_content
        self.theory_content = theory_content
        self.objective_content = objective_content
        self.pretest_content = pretest_content
        
        # Add field for previous website code
        self.previous_website_code = None
    
    def set_llm(self, llm):
        """Set the language model to be used by this agent"""
        self.llm = llm
        
    def set_prompt_enhancer_llm(self, llm):
        """Set the language model to enhance prompts"""
        self.prompt_enhancer_llm = llm
    
    def set_custom_enhancement(self, custom_text):
        """Set custom enhancement text for the prompt."""
        self.custom_enhancement = custom_text

    def set_previous_website_code(self, website_code):
        """
        Set the previously generated website code that should be modified
        based on new feedback rather than creating a new website from scratch.
        
        Args:
            website_code (str): The HTML code of the previously generated website
        """
        self.previous_website_code = website_code
        
        # If we have previous code, modify the prompt template to include it
        if self.prompt_template and self.previous_website_code:
            # Add the previous website code to the prompt template
            self.prompt_template += f"""
            
            # PREVIOUS WEBSITE CODE
            Below is the previously generated website code. When implementing the feedback,
            modify this existing code rather than creating a completely new website.
            Focus on making targeted changes based on the feedback while preserving the
            overall structure and content.
            
            ```html
            {self.previous_website_code}
            ```
            
            IMPORTANT: Your task is to MODIFY the PREVIOUS WEBSITE CODE above according to the USER FEEDBACK.
            Do NOT generate a completely new website. Instead, identify the specific changes needed
            based on the feedback and update the existing code accordingly.
            """
    
    def _read_content_file(self, file_path):
        """Read content from a file, if file exists."""
        if file_path and os.path.exists(file_path):
            with open(file_path, "r", encoding='utf-8') as f:
                return f.read()
        return ""

    def _read_pretest_json(self, file_path):
        """Read pretest questions from JSON file."""
        if file_path and os.path.exists(file_path):
            with open(file_path, "r", encoding='utf-8') as f:
                return json.load(f)
        return []

    def _extract_simulation_content(self):
        """
        Extract the relevant simulation code from the CodingAgent output.
        This handles both cases where the code might be a complete HTML file
        or just the simulation-specific JavaScript/HTML.
        """
        # If it's already just the simulation part, return as is
        if "<html" not in self.simulation_code.lower():
            return self.simulation_code
            
        # Try to extract just the simulation-specific part
        # Look for content in the body or a specific div
        body_match = re.search(r'<body[^>]*>(.*?)</body>', self.simulation_code, re.DOTALL | re.IGNORECASE)
        if body_match:
            return body_match.group(1).strip()
        
        # If can't isolate, return the whole thing
        return self.simulation_code
    
    def enhance_prompt(self):
        """Enhance the prompt template with specific content for website generation."""
        # Get content either from provided content or files
        aim = self.aim_content if self.aim_content is not None else self._read_content_file(self.aim_path)
        theory = self.theory_content if self.theory_content is not None else self._read_content_file(self.theory_path)
        objective = self.objective_content if self.objective_content is not None else self._read_content_file(self.objective_path)
        
        # Only read procedure if we're not generating it dynamically
        procedure_content = "" if self.generate_procedure else self._read_content_file(self.procedure_path)
        
        # Get pretest questions from direct content or file
        pretest_questions = self.pretest_content if self.pretest_content is not None else self._read_pretest_json(self.pretest_path)
        
        # Convert to string if needed
        if isinstance(aim, (dict, list)):
            aim = json.dumps(aim, indent=2)
        if isinstance(theory, (dict, list)):
            theory = json.dumps(theory, indent=2)
        if isinstance(objective, (dict, list)):
            objective = json.dumps(objective, indent=2)
        
        # Build the prompt template
        template = f"""
        You are a professional web developer tasked with creating a complete virtual lab website.
        
        # SIMULATION CODE
        {self.simulation_code}
        """
        
        # Build different template depending on whether we're generating content or using provided content
        if self.generate_content:
            template += """
            # CONTENT GENERATION REQUIREMENTS
            Based on the simulation code above, please generate the following content sections:
            1. AIM - Create a clear, concise statement of the purpose of this simulation
            2. THEORY - Provide comprehensive theoretical background relevant to this simulation
            3. OBJECTIVE - List specific learning objectives that this simulation addresses
            4. PRETEST QUESTIONS - Create 3-5 multiple-choice questions that test understanding of concepts
            
            # IMPORTANT SIMULATION PLACEMENT
            The simulation code provided should ONLY be placed in the SIMULATION tab of the website.
            DO NOT place the simulation code in any other tabs.
            Each tab should have its own distinct content without duplicating the simulation.
            """
        else:
            # Only include sections with content
            if aim:
                template += f"""
                # AIM
                {aim}
                """
            
            if theory:
                template += f"""
                # THEORY
                {theory}
                """
            
            if objective:
                template += f"""
                # OBJECTIVE
                {objective}
                """
                
            if pretest_questions:
                template += f"""
                # PRETEST QUESTIONS
                {json.dumps(pretest_questions, indent=2)}
                """
        
        if not self.generate_procedure and procedure_content:
            template += f"""
            # PROCEDURE
            {procedure_content}
            """
        
        if self.generate_procedure:
            template += """
            # PROCEDURE
            Please generate a detailed step-by-step procedure based on the simulation code.
            The procedure should clearly explain how to operate the simulation and understand the results.
            Include specific instructions for:
            1. Initial setup and configuration
            2. How to interact with each control element
            3. How to interpret the simulation results
            4. Common troubleshooting tips
            """
        
        # Add styling requirements
        if self.enhanced_css:
            template += """
            # CSS REQUIREMENTS
            Create an aesthetically pleasing, professional design with modern CSS.
            Use a clean, readable font and a color scheme appropriate for educational contexts.
            Ensure the UI is intuitive and responsive, working well on both desktop and mobile devices.
            Specific styling requirements:
            - Use a consistent color palette with complementary colors
            - Implement proper spacing and padding for all elements
            - Use subtle animations for tab transitions and interactive elements
            - Style buttons, inputs, and interactive elements with clear hover/active states
            - Use proper typography with readable fonts (at least 16px for body text)
            - Implement accessible contrast ratios for all text
            - Add subtle box shadows to create depth and visual hierarchy
            """
        
        # Add tab layout preference if specified
        if self.left_tabs:
            template += """
            # TAB LAYOUT
            Place the navigation tabs VERTICALLY ON THE LEFT SIDE of the page.
            The tabs should be styled as a sidebar with distinct active state.
            The main content area should take up at least 75% of the page width.
            """
        
        # Add custom enhancement text if provided
        if self.custom_enhancement:
            template += f"""
            # ADDITIONAL REQUIREMENTS
            {self.custom_enhancement}
            """
            
        # Add user feedback if provided
        if self.feedback and self.feedback.strip():
            template += f"""
            # USER FEEDBACK
            Please incorporate the following user feedback into the website design:
            {self.feedback}
            """
        
        # If we have previous website code, include it in the prompt
        if self.previous_website_code:
            template += f"""
            
            # PREVIOUS WEBSITE CODE
            Below is the previously generated website code. When implementing the feedback,
            modify this existing code rather than creating a completely new website.
            Focus on making targeted changes based on the feedback while preserving the
            overall structure and content.
            
            ```html
            {self.previous_website_code}
            ```
            
            IMPORTANT: Your task is to MODIFY the PREVIOUS WEBSITE CODE above according to the USER FEEDBACK.
            Do NOT generate a completely new website. Instead, identify the specific changes needed
            based on the feedback and update the existing code accordingly.
            """
            
        template += """
        # OUTPUT FORMAT
        Generate a complete HTML file with embedded CSS and JavaScript.
        The HTML should include all necessary components to run the virtual lab with the simulation.
        Organize the website with the following sections:
        1. Introduction/Home
        2. Aim
        3. Theory
        4. Objective
        5. Procedure
        6. Simulation (ONLY place the simulation code in this tab)
        7. Pretest Questions (implement as an interactive quiz)
        
        IMPORTANT: The simulation code should ONLY appear in the Simulation tab. Do not duplicate it across tabs.
        
        Make sure the website is fully functional, visually appealing, and educational.
        """
        
        self.prompt_template = template
        
        # Optional: let the LLM enhance the prompt further if needed
        if self.prompt_enhancer_llm:
            enhancer_prompt = f"Please enhance the following prompt template for a virtual lab website design task. Focus on clarity and detail:\n\n{template}"
            enhanced_prompt = self.prompt_enhancer_llm.invoke(enhancer_prompt).content
            self.prompt_template = enhanced_prompt

    def generate_website(self):
        """Generate the complete website HTML"""
        if not self.llm:
            return "Error: Language model not set. Please call set_llm() first."
            
        try:
            response = self.llm.invoke(self.prompt_template)
            # Extract HTML content from the response
            html_content = response.content
            
            # Try to extract just the HTML if it's wrapped in other text
            html_pattern = re.compile(r'```(?:html)?(.*?)```', re.DOTALL)
            match = html_pattern.search(html_content)
            if match:
                html_content = match.group(1).strip()
                
            # Ensure that the simulation is only in the Simulation tab by checking for duplicates
            # This is a fallback in case the model doesn't follow instructions
            if "<html" in html_content:
                # Simple heuristic: If we see the same simulation code snippet in multiple places,
                # we'll clean it up by keeping only the one in the simulation tab
                simulation_snippet = self._extract_simulation_content()
                
                # If we can identify key parts of the simulation
                if simulation_snippet and len(simulation_snippet) > 100:
                    # Try to identify a unique signature from the simulation (first 100 chars)
                    sim_signature = simulation_snippet[:100]
                    
                    # Count occurrences
                    occurrences = html_content.count(sim_signature)
                    
                    # If the simulation appears to be duplicated
                    if occurrences > 1:
                        # Try to identify the simulation tab content
                        sim_tab_pattern = re.compile(r'(id=["\']\w*simulation\w*["\'](.*?)' + re.escape(sim_signature), re.DOTALL | re.IGNORECASE)
                        sim_tab_match = sim_tab_pattern.search(html_content)
                        
                        if sim_tab_match:
                            # Keep the simulation in the simulation tab and remove others
                            for _ in range(occurrences - 1):
                                # Find the last occurrence and remove it if it's not in the simulation tab
                                last_index = html_content.rfind(sim_signature)
                                if last_index > 0 and sim_tab_match.start() != last_index:
                                    html_content = html_content[:last_index] + "<!-- Simulation code moved to Simulation tab -->" + html_content[last_index + len(simulation_snippet):]
                    
            return html_content
        except Exception as e:
            return f"<html><body><h1>Error generating website</h1><p>{str(e)}</p></body></html>"
    
    def get_output(self):
        """Generate and return the website HTML"""
        if not self.prompt_template:
            self.enhance_prompt()
        return self.generate_website()
