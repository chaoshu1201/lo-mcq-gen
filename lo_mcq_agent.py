import os
import json
import argparse
import time
from dotenv import load_dotenv
from utils import count_los, plot_lo_statistics, json_to_excel, json_to_markdown, json_to_gift, setup_logger, execute_python_code
from model_loader import GeminiModel, QwenVLModel


# Load environment variables from .env file
load_dotenv()

# Set up logging for lo_mcq_agent
logger = setup_logger('lo_mcq_agent', 'lo_mcq_agent.log')

class LoMcqGenAgent:
    def __init__(self, args):
        self.input_dir = args.input_dir
        self.output_dir = args.output_dir
        self.prompt_template_dir = args.prompt_template_dir
        # Convert the string "None" to actual None
        self.system_prompt_file = None if args.system_prompt_file == "None" else args.system_prompt_file
        self.query_prompt_file = args.query_prompt_file
        self.module_name = args.module_name
        self.model_name = args.model_name
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.assistant_model_name = args.assistant_model_name
        self.assistant_query_prompt_file = args.assistant_query_prompt_file
        self.code_interpreter = args.code_interpreter

        if self.model_name in ["gemini-1.5-flash-002", "gemini-1.5-pro-002", "gemini-1.5-pro", "gemini-2.0-flash", "gemini-2.0-pro-exp-02-05"]:
            self.model = GeminiModel(
                self.prompt_template_dir,
                self.system_prompt_file,
                self.model_name,
                self.temperature,
                self.top_p,
                self.code_interpreter  # Pass the code_interpreter to the model
            )
        elif self.model_name in ["Qwen/Qwen2-VL-7B-Instruct", "Qwen/Qwen2-VL-72B-Instruct"]:
            self.model = QwenVLModel(
                self.prompt_template_dir,
                self.system_prompt_file,
                self.model_name,
                self.temperature,
                self.top_p,
                self.code_interpreter  # Pass the code_interpreter to the model
            )
        else:
            raise ValueError(f"Model {self.model_name} not supported")

        os.makedirs(self.output_dir, exist_ok=True)

    def post_process_response(self, response_data, filename):
        # Save the response to a JSON file
        json_output_file = os.path.join(self.output_dir, f"output_{filename.replace('.pdf', '.json')}")
        with open(json_output_file, 'w') as f:
            json.dump(response_data, f, indent=2)
        logger.info(f"Response saved to {json_output_file}")

        # Convert JSON to Excel
        excel_output_file = os.path.join(self.output_dir, f"output_{filename.replace('.pdf', '.xlsx')}")
        json_to_excel(response_data, excel_output_file)
        logger.info(f"Excel file saved to {excel_output_file}")

        # Convert JSON to Markdown
        markdown_output_file = os.path.join(self.output_dir, f"output_{filename.replace('.pdf', '.md')}")
        json_to_markdown(response_data, markdown_output_file)
        logger.info(f"Markdown file saved to {markdown_output_file}")

        # Convert JSON to GIFT format
        gift_output_file = os.path.join(self.output_dir, f"output_{filename.replace('.pdf', '.gift')}")
        json_to_gift(response_data, gift_output_file)
        logger.info(f"GIFT file saved to {gift_output_file}")

        # Count and plot LO statistics
        lo_counts = count_los(response_data)
        plot_output_file = os.path.join(self.output_dir, f"lo_stats_{filename.replace('.pdf', '.png')}")
        plot_lo_statistics(lo_counts, plot_output_file)
        logger.info("LO statistics plot saved in the output directory")

    def update_numerical_mcq(self, response_data):
        """Inserts numerical answers into the response data."""
        if 'mcqs' in response_data:
            for i, mcq in enumerate(response_data['mcqs'], start=1):
                python_code = mcq.get('python_code')
                if python_code:  # Check if python_code exists and is valid
                    logger.debug(f"Generated Numerical Question {i}: {json.dumps(mcq)}")
                    logger.info("Local Python code executing...")
                    result = execute_python_code(python_code)
                    logger.info(f"Local Python code execution output for Question {i}: {result}, model code execution output: {mcq.get('python_output')}")

                    if (result is not None) and (mcq.get('python_output') != result):
                        # Update the python_output field
                        mcq["python_output"] = result

                        # Read prompt from file
                        with open(os.path.join(self.prompt_template_dir, self.assistant_query_prompt_file), 'r') as f:
                            assistant_prompt_template = f.read()
                        assistant_query_prompt = assistant_prompt_template.format(mcq_blk=json.dumps(mcq))

                        # Revise the MCQ
                        logger.info("Waiting for 30 seconds before revising MCQ...")
                        time.sleep(30)
                        assistant_response = self.model.generate_json_text(assistant_query_prompt)
                        mcq_upd = assistant_response
                        response_data['mcqs'][i - 1] = mcq_upd
                        logger.info(f"Numerical Question {i} Updated")
                        logger.debug(f"Updated Numerical Question {i}: {json.dumps(mcq_upd)}")
                        # Clean the result by removing prefixes like 0x, 0b, or 0o
                        # cleaned_result = clean_code_output(result)

                        # Insert the cleaned result into the options
                        # if 'options' in mcq and isinstance(mcq['options'], list) and len(mcq['options']) > 0:
                            # update_correct_answer(i, mcq, cleaned_result)  # Call the new function
                        
        return response_data

    def process_pdf(self, filename):
        pdf_path = os.path.join(self.input_dir, filename)

        # Read prompt from file
        with open(os.path.join(self.prompt_template_dir, self.query_prompt_file), 'r') as f:
            prompt_template = f.read()
        # Replace module name in prompt
        query_prompt = prompt_template.format(module_name=self.module_name)

        logger.info(f"Querying {filename}...")
        response_data = self.model.query_pdf(pdf_path, query_prompt)

        if self.code_interpreter == "local":
            response_data = self.update_numerical_mcq(response_data)

        # Call the new post-process method
        self.post_process_response(response_data, filename)

    def run(self):
        for filename in os.listdir(self.input_dir):
            if filename.endswith('.pdf'):
                self.process_pdf(filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PDF files using a Gemini agent")
    parser.add_argument("--input_dir", default="./input", help="Directory containing input PDF files (default: ./input)")
    parser.add_argument("--output_dir", default="./output", help="Directory to save output files (default: ./output)")
    parser.add_argument("--prompt_template_dir", default="./prompt_templates", help="Directory containing prompt template files (default: ./prompt_templates)")
    parser.add_argument("--system_prompt_file", default="system.txt", help="System prompt file name (default: system.txt)")
    parser.add_argument("--query_prompt_file", default="query_hard.txt", help="Query prompt file name (default: query.txt)")
    parser.add_argument("--module_name", default="Data Design", help="Name of the module (default: General Module)")
    parser.add_argument("--model_name", default="gemini-2.0-flash", help="LLM to use (default: gemini-1.5-flash-002)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for the LLM (default: 1.0)")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p value for the LLM (default: 0.95)")
    parser.add_argument("--code_interpreter", default="inbuilt", choices=["inbuilt", "local", "cloud"], help="Code interpreter to use (default: inbuilt)")
    parser.add_argument("--assistant_model_name", default="gemini-1.5-flash-002", help="assistant model to use (default: gemini-1.5-flash-002)")
    parser.add_argument("--assistant_query_prompt_file", default="query_mcq_rev.txt", help="Assistant query prompt file name (default: query_assistant.txt)")

    args = parser.parse_args()
    
    agent = LoMcqGenAgent(args)
    agent.run()
