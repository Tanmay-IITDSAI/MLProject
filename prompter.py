class Prompter:
    def __init__(self, template_name: str = "alpaca"):
        # Define different prompt templates if needed
        self.template_name = template_name

    def generate_prompt(self, instruction: str, input_text: str = "", output_text: str = "") -> str:
        if self.template_name == "alpaca":
            if input_text:
                return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output_text}"
            else:
                return f"### Instruction:\n{instruction}\n\n### Response:\n{output_text}"
        else:
            # Add other templates as needed
            return f"{instruction}\n{input_text}\n{output_text}"

    def get_response(self, output: str) -> str:
        # Extract the response part from the generated output
        # This can be customized based on the prompt template
        return output.split("### Response:")[-1].strip()