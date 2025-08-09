# Web Application Architecture Flow

```mermaid
flowchart TD
    A[User Input\nSource Text + Translation\nSystem Prompt + Settings] --> B[POST /api/judge\nFlask Endpoint]
    
    B --> C{Input Validation\nRequired Fields Check}
    C -->|Missing Fields| C1[Return 400 Error\nMissing required fields]
    C -->|Valid| D[Token Validation\nvalidate tokens]
    
    D --> E{System Prompt\nToken Check}
    E -->|Missing Tokens| E1[Return 400 Error\nMissing required tokens]
    E -->|Valid| F[Token Replacement\nreplace tokens]
    
    F --> G[Token Processing\nsource_text to source_text\nfil_translation to fil_translation\nref_translation to ref_translation]
    
    G --> H[Position Randomization\nrandomize positions\nShuffle pos tags to reduce bias]
    
    H --> I{LLM Mode Selection}
    I -->|Standard| J[Create Standard LLM Client\ncreate LLM client]
    I -->|Agentic| K[Create Agentic LLM Client\ncreate agentic LLM client]
    
    J --> L[Standard LLM Processing\nGeminiClient or OpenAIClient]
    K --> M[Agentic LLM Processing\nAgenticGeminiClient]
    
    L --> N[Direct API Call\nStructured JSON Output\n6 Boolean Criteria + Explanations]
    M --> O[Two-Step Agentic Process\n1. Unstructured Generation + Tools\n2. Structure Conversion]
    
    N --> P[Response Validation\nPydantic Schema Check]
    O --> P
    
    P --> Q{Validation Success?}
    Q -->|Failed| Q1[Return 500 Error\nJSON Parse/Validation Error]
    Q -->|Success| R[Final Score Calculation\ncalculate final score]
    
    R --> S[Score Logic\nCount True Criteria\n5-6 = Excellent 5\n3-4 = Good 3-4\n0-2 = Poor 1-2]
    
    S --> T[Save Results\nsave judgment results\nTimestamped JSON file]
    
    T --> U[Response Assembly\njudgment + final score + metadata]
    
    U --> V[JSON Response to Frontend\nSuccess + Structured Data]
    V --> W[Frontend Display\nResults Rendering + UI Update]
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style F fill:#fff3e0
    style H fill:#fff3e0
    style I fill:#e8f5e8
    style L fill:#fce4ec
    style M fill:#fce4ec
    style R fill:#fff3e0
    style V fill:#e1f5fe
    style W fill:#e1f5fe

```

## Key Components

### Input Processing Flow
1. **User Input** - Source text, translation, system prompt, and LLM settings
2. **Validation** - Required field checks and token validation
3. **Token Replacement** - Replace `{{source_text}}`, `{{fil_translation}}`, `{{ref_translation}}` placeholders
4. **Position Randomization** - Shuffle `<pos>` tagged content to reduce position bias

### LLM Processing Paths

#### Standard Path (Prompt-Engineered)
- Direct API call to Gemini or OpenAI
- Structured JSON output with 6 boolean criteria
- Single-step processing

#### Agentic Path (Enhanced Reasoning)
- Two-step process: Unstructured generation → Structure conversion
- Tool usage: Custom functions + Google Search sub-agents
- Thought summary capture for transparency
- Function call logging

### Output Processing
1. **Response Validation** - Pydantic schema validation
2. **Score Calculation** - Boolean criteria → numerical score (1-5)
3. **Results Storage** - Timestamped JSON files in `/results/`
4. **Frontend Response** - Complete judgment data with metadata

## File References
- **app.py:64** - Main `/api/judge` endpoint
- **prompt_utils.py:92-106** - Token replacement logic  
- **prompt_utils.py:108-109** - Position randomization
- **eval_utils.py:6** - Final score calculation
- **llms/prompt_engineered_judge_main.py** - Standard LLM clients
- **llms/agentic_judge_main.py** - Agentic LLM processing