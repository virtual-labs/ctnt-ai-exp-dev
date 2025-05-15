from sys import implementation

from Agents.CodingAgent import CodingAgent
from Agents.HumanReviewAgentForRequirement import HumanReviewAgentForRequirement
from Agents.ImplementationAgent import ImplementationAgent
from Agents.RequirementsAgent import RequirementsAgent
from Agents.VerfierAgent import VerifierAgent
from Agents.DocumentationAgent import DocumentationAgent
from BaseAgent import BaseAgent
from langchain_google_genai import ChatGoogleGenerativeAI

class Pipeline:
    llm = None
    max_loop = 3

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro-exp-03-25",
            temperature=0.1,
            max_tokens=100000
        )

    def run(self):
        reqAgent = RequirementsAgent("1.pdf")
        reqAgent.set_llm(self.llm)
        reqAgent.set_prompt_enhancer_llm(self.llm)
        reqAgent.enhance_prompt()
        req_Agent_output = reqAgent.get_output()
        print("[\033[91mRequirements OUTPUT\033[0m")
        print(req_Agent_output)
        human_review_output = ""
        while True:
            review_1 = input(">>> Enter your review for the requirements: Press Enter to skip: ")
            if review_1 == "":
                if human_review_output == "":
                    human_review_output = req_Agent_output
                break


            human_review = HumanReviewAgentForRequirement(req_Agent_output, review_1)
            human_review.set_llm(self.llm)
            human_review.set_prompt_enhancer_llm(self.llm)
            human_review.enhance_prompt()
            human_review_output = human_review.get_output()
            print(human_review_output)
        print("\033[91mHuman Review Output\033[0m")
        print(human_review_output)
        implementation_agent = ImplementationAgent(human_review_output)
        implementation_agent.set_llm(self.llm)
        implementation_agent.set_prompt_enhancer_llm(self.llm)

        impl_agent_output = implementation_agent.get_output()
        print("\033[91mImplementation OUTPUT\033[0m")
        print(impl_agent_output)

        loop = 0
        code_review = ""
        coding_agent_output = ""
        while loop < self.max_loop:
            coding_agent = CodingAgent(impl_agent_output, code_review)
            coding_agent.set_llm(self.llm)
            coding_agent.set_prompt_enhancer_llm(self.llm)
            coding_agent.enhance_prompt()
            coding_agent_output = coding_agent.get_output()
            with open("code.html", "w") as f:
                f.write(coding_agent_output)

                print()
                print("-"*100)
                print(coding_agent_output)
            code_review = input(">>> Enter your review for the code: ")

        documentation_agent = DocumentationAgent(coding_agent_output)
        documentation_agent.set_llm(self.llm)
        documentation_agent.set_prompt_enhancer_llm(self.llm)
        documentation_agent.enhance_prompt()
        documentation_agent_output = documentation_agent.get_output()
        with open("documentation.md", "w") as f:
            f.write(documentation_agent_output)


if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.run()
