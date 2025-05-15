import dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI

dotenv.load_dotenv()


class BaseAgent:
    llm = None
    prompt_enhancer_llm = None
    enhanced_prompt = None

    def __init__(self, role: str, basic_prompt: str, context: str = ""):
        self.role = role
        self.basic_prompt = basic_prompt
        self.context = context

    def set_llm(self, llm):
        self.llm = llm

    def set_prompt_enhancer_llm(self, llm):
        self.prompt_enhancer_llm = llm
    def enhance_prompt(self):
        if  self.prompt_enhancer_llm is None:
            raise ValueError("Prompt enhancer LLM is not set.")

        enhanced_prompt_template = (
            "You are an expert prompt engineer for the role of '{role}'.\n\n"
            "The basic prompt that needs enhancing is:\n"
            "{basic_prompt}\n\n"
            "Using the above details, refine and improve the basic prompt by providing clear instructions, suggestions, and guidelines "
            "on how to approach the task effectively. Ensure the enhanced prompt is actionable and provides hints on what the agent should expect, "
            "but do not include any content that is part of the context since i will send it again\n\n"
        )

        prompt = PromptTemplate(
            input_variables=["role", "basic_prompt", "context"],
            template=enhanced_prompt_template
        )

        chain = LLMChain(llm=self.prompt_enhancer_llm, prompt=prompt)
        enhanced_prompt = chain.invoke({
            "role": self.role,
            "basic_prompt": self.basic_prompt,
            "context": self.context
        })
        self.enhanced_prompt = enhanced_prompt['text']
        return self.enhanced_prompt

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
        )

        prompt = PromptTemplate(
            input_variables=["role", "context", "base_prompt"],
            template=final_prompt_template
        )

        chain = LLMChain(llm=self.llm, prompt=prompt)
        return chain.invoke({
            "role": self.role,
            "context": self.context,
            "base_prompt": base_prompt
        })['text']


if __name__ == "__main__":
    role = "Requirements Agent"
    context = (
        "Requirements Document:\n"
        "Give me a 3D game for the following requirements:\n"
        "The game should be a 3D simulation of a car racing game.\n"
        "The game should have realistic physics and graphics.\n"
        "The game should support multiplayer mode.\n"
        "The game should have a leaderboard system.\n"
        "The game should have a tutorial mode for new players."
    )
    basic_prompt = (
        "You are an expert Requirements Agent. You have been provided with a detailed requirements document "
        "(extracted from a PDF) that describes a complex simulation system. Analyze the document and extract "
        "a clear, concise, and prioritized list of actionable requirements in bullet points. Format the output as plain text "
        "that can be embedded into a README file."
    )

    agent = BaseAgent(role, basic_prompt, context)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro-exp-03-25",
        temperature=0.1,
        max_tokens=100000
    )

    agent.set_llm(llm)
    agent.set_prompt_enhancer_llm(llm)

    enhanced_prompt = agent.enhance_prompt()

    print("\033[93m" + enhanced_prompt + "\033[0m")

    output = agent.get_output()
    print("Final Output:\n", output)
