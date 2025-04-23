import os
import re
import json
from pdf2image import convert_from_path
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import logging
import subprocess

def setup_logger(name, log_file, level=logging.INFO):
    """Function to setup as many loggers as you want"""
    
    # Create log directory if it doesn't exist
    log_dir = os.path.join('.', 'log')
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels

    # Create a file handler
    file_handler = logging.FileHandler(os.path.join(log_dir, log_file))
    file_handler.setLevel(logging.DEBUG)  # Set file handler to DEBUG level

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Set console handler to INFO level

    # Create formatters and add them to the handlers
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

logger = setup_logger('utils', 'utils.log')  # Initialize logger

def count_los(data):
    lo_counts = Counter()
    for lo in data['learning_outcomes']:
        lo_counts[lo['level']] += 1
    return lo_counts

def plot_lo_statistics(lo_counts, plot_path):
    levels = list(lo_counts.keys())
    counts = list(lo_counts.values())

    plt.figure(figsize=(10, 6))
    plt.bar(levels, counts)
    plt.title('Learning Outcomes by Bloom\'s Taxonomy Level')
    plt.xlabel('Bloom\'s Taxonomy Level')
    plt.ylabel('Number of Learning Outcomes')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

def json_to_excel(json_data, excel_path):
    rows = []
    for i, (lo, mcq) in enumerate(zip(json_data['learning_outcomes'], json_data['mcqs']), start=1):
        row = {
            'ID': i,
            'LO Level': lo['level'],
            'LO Description': lo['description'],
            'MCQ Question': mcq['question'],
            'MCQ Options': '\n'.join(mcq['options']),
            'Correct Answer': mcq['correct_answer'],
            'Explanation': mcq['explanation'],
            'Related Pages': mcq.get('related_pages', '')
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_excel(excel_path, index=False)

def json_to_markdown(json_data, markdown_path):
    # Open file with UTF-8 encoding and newline handling
    with open(markdown_path, 'w', encoding='utf-8', newline='\n') as f:
        # Use a default module name if not present in json_data
        module_name = json_data.get('module_name', 'Unnamed Module')
        lesson_name = json_data.get('lesson_name', 'Unnamed Lesson')
        f.write(f"# {module_name} - {lesson_name}\n\n")
        
        if 'summary' in json_data:
            f.write("## Summary\n\n")
            f.write(f"{json_data['summary']}\n\n")
        
        if 'learning_outcomes' in json_data:
            f.write("## Learning Outcomes\n\n")
            for i, lo in enumerate(json_data['learning_outcomes'], 1):
                f.write(f"{i}. **{lo['level']}**: {lo['description']}\n\n")
        
        if 'mcqs' in json_data:
            f.write("## Multiple Choice Questions\n\n")
            for i, mcq in enumerate(json_data['mcqs'], 1):
                f.write(f"### Question {i}\n\n")
                f.write(f"{mcq['question']}\n\n")
                for option in mcq['options']:
                    f.write(f"- {option}\n")
                f.write(f"\n**Correct Answer:** {mcq['correct_answer']}\n\n")
                f.write(f"**Explanation:** {mcq['explanation']}\n\n")
                if 'related_pages' in mcq:
                    f.write(f"**Related Pages:** {mcq['related_pages']}\n\n")
                if 'python_code' in mcq and mcq['python_code']:
                    f.write("**Python Code:**\n")
                    f.write(f"```python\n{mcq['python_code']}\n```\n\n")
                if 'python_output' in mcq and mcq['python_output']:
                    f.write("**Python Output:**\n")
                    f.write(f"```python\n{mcq['python_output']}\n```\n\n")

def json_to_gift(json_data, gift_path):
    """Convert MCQs from JSON to GIFT format for Moodle import."""
    with open(gift_path, 'w', encoding='utf-8') as f:
        for i, mcq in enumerate(json_data['mcqs'], start=1):
            # Format question title and text
            f.write(f"::{i}::{mcq['question']}{{\n")
            
            # Write answer choices, stripping option letters
            for option in mcq['options']:
                # Strip first 3 chars ("A. ", "B. ", etc)
                option_text = option[3:]
                # Check if original option started with correct answer letter
                prefix = "=" if option.startswith(mcq['correct_answer']) else "~"
                f.write(f"{prefix}{option_text}\n")
            
            # Write feedback
            f.write(f"####${mcq['explanation']}\n")
            f.write("}\n\n")

def convert_pdf_to_images(pdf_path, output_dir, dpi=50, quality=50):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert PDF to images
    images = convert_from_path(pdf_path, dpi=dpi)
    
    # Save each image
    for i, image in enumerate(images):
        image_path = os.path.join(output_dir, f"page_{i + 1}.jpg")
        image.save(image_path, "JPEG", quality=quality, optimize=True)
    
    return len(images)

def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

def clean_json_string(s):
    # # Remove any text before the first '{'
    # s = re.sub(r'^[^{]*', '', s)
    # # Remove any text after the last '}'
    # s = re.sub(r'}[^}]*$', '}', s)

    # Remove ```json ... ``` wrapper
    if s.startswith('```json'):
        s = s[len('```json'):]
        s = s.lstrip()  # Remove leading whitespace/newlines
    if s.endswith('```'):
        s = s.rstrip()  # Remove trailing whitespace/newlines
        s = s[:-3]

    # Replaces the triple-quoted content with this JSON-encoded version. 
    # Sometimes the generated text will use triple-quote for the "python_code" field, 
    # which will lead to JSON decoding error when use json.loads() to parse the text (string). 
    # cleaned_json_string = re.sub(r'"""(.*?)"""', lambda m: json.dumps(m.group(1)), s, flags=re.DOTALL)
    cleaned_json_string = s

    return cleaned_json_string

def execute_python_code(python_code: str) -> str:
    """
    Executes the provided Python code and saves it with a name based on the question number.

    This function takes in a string of Python code and an integer representing a question number. It saves the Python code to a temporary file with a name in the format 'qX.py', where X is the question number. The function then executes the Python script and captures the output. If the execution is successful, the function returns the output of the script. If an error occurs during execution, the function logs the error and returns None.

    Parameters:
    - python_code (str): The Python code to be executed.
    - question_number (int): The question number used to name the temporary file.

    Returns:
    - str: The output of the executed Python script if successful, otherwise None.
    """
    try:
        # Create a filename based on the question number
        # filename = f'q{question_number}.py'
        filename = 'temp_script.py'
        # Save the code to a temporary file
        with open(filename, 'w') as f:
            f.write(python_code)

        # Run the Python script and capture the output
        result = subprocess.run(['python', filename], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()  # Return the output of the script
        else:
            logger.error(f"Error executing Python code: {result.stderr}")
            return None
    except Exception as e:
        logger.error(f"Exception occurred while executing Python code: {e}")
        return None
    finally:
        # Clean up the temporary file
        if os.path.exists(filename):
            os.remove(filename)

def clean_code_output(result):
    """Cleans the result by removing prefixes like 0x, 0b, or 0o."""
    if isinstance(result, str):
        if result.startswith('0x'):
            return result[2:]  # Remove '0x' prefix
        elif result.startswith('0b'):
            return result[2:]  # Remove '0b' prefix
        elif result.startswith('0o'):
            return result[2:]  # Remove '0o' prefix
    return result  # Return the result as is if no prefix is found

def update_correct_answer(q_no, mcq, cleaned_result):
    """
    Updates the correct answer in the mcq if necessary based on the cleaned result.

    Parameters:
    - mcq (dict): The multiple-choice question dictionary.
    - cleaned_result (str): The cleaned numerical result from the execution.
    """
    existing_option_letter = None

    # Check if the cleaned result already exists in the options
    for option in mcq['options']:
        # Extract the numerical part from the option
        option_letter = option.split('. ')[0]  # Get the option letter
        option_answer = option.split('. ')[1]  # Get the answer part after the option letter and dot
        if cleaned_result == option_answer:
            existing_option_letter = option_letter
            break

    if existing_option_letter:
        if existing_option_letter == mcq['correct_answer']:
            # Do nothing if the existing option is the correct answer
            logger.info(f"Calculated result {cleaned_result} for Question {q_no} already exists as the correct answer option {existing_option_letter}. No changes made.")
        else:
            # Update the correct answer to the existing option
            logger.info(f"Updating Question {q_no} correct answer option letter from {mcq['correct_answer']} to {existing_option_letter}.")
            mcq['correct_answer'] = existing_option_letter
    else:
        # Replace the answer in the mcq["correct_answer"] option
        calculated_answer = f"{mcq['correct_answer']}. {cleaned_result}"  # Format the calculated result
        option_pos = ord(mcq['correct_answer']) - 65  # Get the position of the correct answer
        logger.info(f"Replacing Question {q_no} original correct option {mcq['options'][option_pos]} with calculated answer {cleaned_result}.")
        mcq['options'][option_pos] = calculated_answer  # Replace the option
