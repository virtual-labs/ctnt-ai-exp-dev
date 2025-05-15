from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from typing_extensions import override

from BaseAgent import BaseAgent


class HumanReviewAgentForRequirement(BaseAgent):
    role = "Human Requirements Agent"
    basic_prompt_template = """
    Your will get a requirements document and a review of the requirements. Now, modify the requirements document based on the review.
    """

    current_requirements = None
    review = None

    def __init__(self, requirements_review,current_requirements):
        super(HumanReviewAgentForRequirement, self).__init__(self.role, basic_prompt=self.basic_prompt_template, context=None)
        self.current_requirements = current_requirements
        self.review = requirements_review


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
            "Current Requirements:\n"
            "{current_requirements}\n"
            "Review:\n"
            "{review}\n"
        )

        prompt = PromptTemplate(
            input_variables=["role", "context", "base_prompt", "current_requirements", "review"],
            template=final_prompt_template
        )

        chain = LLMChain(llm=self.llm, prompt=prompt)
        return chain.invoke({
            "role": self.role,
            "context": self.context,
            "base_prompt": base_prompt,
            "current_requirements": self.current_requirements,
            "review": self.review
        })['text']
