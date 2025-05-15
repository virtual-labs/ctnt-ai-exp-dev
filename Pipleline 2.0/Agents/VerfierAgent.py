from langchain_google_genai import ChatGoogleGenerativeAI
from typing_extensions import override

from langchain import PromptTemplate
from langchain.chains.llm import LLMChain
from BaseAgent import BaseAgent

class VerifierAgent(BaseAgent):


    role = "Verifier Agent"
    basic_prompt_template = """
    You are a rigorous VerifierAgent.
    Validate the integrated system below (the final HTML product) by checking functionality, design, and adherence to requirements.
    If the system meets all criteria, simply respond with "passed". Otherwise, provide detailed feedback on any issues.
    """

    integrated_system = None
    req_doc = None
    def __init__(self):
        super(VerifierAgent, self).__init__(self.role, basic_prompt=self.basic_prompt_template, context=None)
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
            "System:\n"
            "{integrated_system}\n"
            "Requirements Document:\n"
            "{req_doc}\n"
        )

        prompt = PromptTemplate(
            input_variables=["role", "context", "base_prompt", "integrated_system", "req_doc"],
            template=final_prompt_template
        )

        chain = LLMChain(llm=self.llm, prompt=prompt)
        return chain.invoke({
            "role": self.role,
            "context": self.context,
            "base_prompt": base_prompt,
            "integrated_system": self.integrated_system,
            "req_doc": self.req_doc
        })['text']
if __name__ == "__main__":
    agent = VerifierAgent()
    agent.integrated_system = "<html>Hello</html>"
    agent.req_doc = "print Hello"
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