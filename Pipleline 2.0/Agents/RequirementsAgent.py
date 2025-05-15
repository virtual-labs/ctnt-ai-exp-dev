import PyPDF2
from langchain_google_genai import ChatGoogleGenerativeAI

from BaseAgent import BaseAgent

class RequirementsAgent(BaseAgent):
    """This agent is responsible for gathering requirements from the user.
    It will ask the user a series of questions to gather all the necessary information
    to create a prompt that meets the user's needs.
    """

    role = "Requirements Agent"
    basic_prompt = """
    You are an expert Requirements Agent. You have been provided with a detailed requirements document "
        "(extracted from a PDF) that describes a complex simulation system. Analyze the document and extract "
        "a clear, concise, and prioritized list of actionable requirements in bullet points. Format the output as plain text "
        "that can be embedded into a README file.
    """

    file_path = None

    def __init__(self, file_path):
        super(RequirementsAgent, self).__init__(self.role, self.basic_prompt,context=None,)
        self.file_path = file_path
        self.read_requirements()

    def read_requirements(self):
        with open(self.file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        self.context = text.strip()
        return  self.context

if __name__ == "__main__":
    req = RequirementsAgent("../1.pdf")
    llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro-exp-03-25",
            temperature=0.1,
            max_tokens=100000
        )
    req.set_llm(llm)
    req.set_prompt_enhancer_llm(llm)
    print("_"*50)
    print(req.enhance_prompt())
    print("_"*50)
    print(req.get_output())
    print("_"*50)
