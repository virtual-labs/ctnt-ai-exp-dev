from langchain_google_genai import ChatGoogleGenerativeAI

from BaseAgent import BaseAgent


class DocumentationAgent(BaseAgent):

    role = "Documentation Agent"
    basic_prompt_template = """
    You are a thorough DocumentationAgent.
    Generate comprehensive documentation for both end-users and developers in Markdown format.
    Your documentation should include:
      - An overview of the system architecture
      - Usage guidelines
      - Troubleshooting tips
      - Maintenance instructions
    """

    def __init__(self,integrated_system=None):
        super(DocumentationAgent, self).__init__(self.role, basic_prompt=self.basic_prompt_template, context=None)
        self.context = integrated_system


if __name__ == "__main__":
    agent = DocumentationAgent("<html>Hello</html>")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro-exp-03-25",
        temperature=0.1,
        max_tokens=100000
    )
    agent.set_llm(llm)
    agent.set_prompt_enhancer_llm(llm)
    print(agent.enhance_prompt())
    print("_" * 50)
    print(agent.get_output())