import sys

import PyPDF2
from langchain import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from BaseAgent import BaseAgent

class TestingAgent(BaseAgent):
    role = "Testing Agent"
    basic_prompt_template = """
    You are a meticulous TestingAgent.
    Test the following code module (designed to be embedded in an HTML file) by simulating its execution.
    Return a detailed report indicating "passed" if all tests succeed, or describe any failures clearly.

    Code Module:
    {code_module}
    """

    context= None

    def __init__(self, code_module):

        self.context = code_module.strip() if code_module else ""
        prompt = self.basic_prompt_template.format(code_module=self.code_module)
        super(TestingAgent, self).__init__(self.role, prompt, context=None)

if __name__ == "__main__":
    agent = TestingAgent(code_module=sys.argv[1])
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
    print("_" * 50)
    print(agent.get_output())