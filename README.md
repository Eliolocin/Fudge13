# LLM-as-Judge Translation Interface

A simple web interface for judging English-to-Filipino translations using LLM models (Gemini & GPT).

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Environment Setup

### 1. Create a Python Virtual Environment

Create and activate a virtual environment to isolate project dependencies:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Configuration

Create `.env` file with your API keys:
```bash
cp .env.example .env
```

Add your API keys to `.env`:
```
GOOGLE_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

**Note:** API keys are optional - the application will run in demo mode without them.

### 4. Create Required Directories

```bash
# Create prompts directory for system prompts
mkdir prompts

# Create results directory for saving judgments (optional - auto-created)
mkdir results
```

### 5. Run the Application

```bash
python app.py
```

The application will be available at:
- http://127.0.0.1:5000 (localhost)
- http://192.168.1.7:5000 (network interface)

## Usage

1. Select or create a system prompt with tokens
2. Input source text and Filipino translation
3. Choose your LLM judge
4. Click "Judge" to get structured evaluation results

## Tokens Available

- `{{source_text}}` - The English source text
- `{{fil_translation}}` - The Filipino translation to be judged
- `{{ref_translation}}` - Optional reference translation
Filipino Judge-as-LLM (Fudge) Test Application
