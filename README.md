# LLM-as-Judge Translation Interface

A simple web interface for judging English-to-Filipino translations using LLM models (Gemini & GPT).

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create `.env` file with your API keys:
```bash
cp .env.example .env
```

3. Add your API keys to `.env`:
```
GOOGLE_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

4. Create prompts folder and add your prompt files:
```bash
mkdir prompts
```

5. Run the application:
```bash
python app.py
```

6. Open either web link:
```bash
 * Running on http://127.0.0.1:5000
 * Running on http://192.168.1.7:5000
```

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
