from langchain import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from typing_extensions import override

from BaseAgent import BaseAgent


class IntegrationAgent(BaseAgent):
    role = "Integration Agent"
    basic_prompt_template = """
    You are an experienced IntegrationAgent.
    Integrate the following code module into a final product that must be delivered as a single, self-contained HTML file.
    The HTML file should include inline CSS and JavaScript libraries.
   Merge the previous and the current modules to get the final system
    """

    previous_code_module = None
    current_code_module = None

    def __init__(self,previous_code_module=None, current_code_module=None):
        super(IntegrationAgent, self).__init__(self.role, basic_prompt=self.basic_prompt_template, context=None)
        self.previous_code_module = previous_code_module.strip() if previous_code_module else ""

        self.current_code_module = current_code_module.strip() if current_code_module else ""


    @override
    def get_output(self):
        if not self.llm:
            raise ValueError("LLM is not set.")

        # Use the enhanced prompt if available, else the basic one
        base_prompt = self.enhanced_prompt if self.enhanced_prompt else self.basic_prompt

        final_prompt_template = (
            "You are an expert in {role}.\n\n"
            "{context}\n\n"
            "Here is the task given to you: \n"
            "{base_prompt}\n"
            "Previous Code Module:\n"
            "{previous_code_module}\n"
            "Current Code Module:\n"
            "{current_code_module}\n"
        )

        prompt = PromptTemplate(
            input_variables=["role", "context", "base_prompt", "previous_code_module", "current_code_module"],
            template=final_prompt_template
        )

        chain = LLMChain(llm=self.llm, prompt=prompt)
        return chain.invoke({
            "role": self.role,
            "context": self.context,
            "base_prompt": base_prompt,
            "previous_code_module": self.previous_code_module,
            "current_code_module": self.current_code_module
        })['text']

if __name__ == "__main__":
    code_module = """
    def add(a, b):
        return a + b
    """
    previous_code_module = """
    def subtract(a, b):
        return a - b
    """

    integration_agent = IntegrationAgent(previous_code_module=previous_code_module, current_code_module=code_module)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro-exp-03-25",
        temperature=0.1,
        max_tokens=10000
    )

    integration_agent.set_llm(llm)
    integration_agent.set_prompt_enhancer_llm(llm)
    print(integration_agent.enhance_prompt())
    print("" + "-" * 50)
    print(integration_agent.get_output())