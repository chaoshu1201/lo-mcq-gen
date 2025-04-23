# Learning Outcomes and MCQ Generator

This tool automatically extracts Learning Outcomes (LOs) from PDF lecture notes and generates associated Multiple Choice Questions (MCQs) based on the content using AI models.

## Features

- Extracts learning outcomes from lecture notes using Bloom's Taxonomy
- Generates multiple-choice questions with solutions
- Supports code execution for computational questions
- Outputs results in multiple formats (JSON, Excel, Markdown, GIFT)
- Visualizes learning outcome statistics

## Requirements

- Python 3.12
- Conda
- PDF lecture notes

## Installation

### Step 1: Clone the repository (if applicable)

```bash
git clone <repository-url>
cd lo-mcq-gen
```

### Step 2: Create and activate a conda environment

```bash
conda create -n lo-mcq-gen python=3.12
conda activate lo-mcq-gen
```

### Step 3: Install dependencies from requirements.txt

```bash
pip install -r requirement.txt
```

### Step 4: Setup API Keys

Rename the provided `.env_example` file to `.env`:

```bash
cp .env_example .env
```

Then open the `.env` file and replace the placeholder with your own Gemini API key:

