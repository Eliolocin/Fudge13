from flask import Flask, render_template, request, jsonify
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from llms.prompt_engineered_judge_main import create_llm_client
from llms.agentic_judge_main import create_agentic_llm_client
from utils.prompt_utils import replace_tokens, validate_tokens
from utils.eval_utils import calculate_final_score

# Load environment variables
load_dotenv()

app = Flask(__name__)

@app.route('/')
def index():
    """Main page with the LLM-as-Judge interface."""
    return render_template('index.html')

@app.route('/api/prompts')
def get_prompts():
    """Get list of available prompt files."""
    prompts_dir = 'prompts'
    prompts = []
    
    if os.path.exists(prompts_dir):
        for filename in os.listdir(prompts_dir):
            if filename.endswith('.txt'):
                prompts.append({
                    'name': filename[:-4],  # Remove .txt extension
                    'filename': filename
                })
    
    return jsonify(prompts)

@app.route('/api/prompt/<filename>')
def get_prompt_content(filename):
    """Get content of a specific prompt file."""
    filepath = os.path.join('prompts', filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'Prompt file not found'}), 404
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return jsonify({'content': content})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/judge', methods=['POST'])
def judge_translation():
    """Process translation judgment request."""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['system_prompt', 'source_text', 'fil_translation', 'llm_model', 'llm_provider']
        missing_fields = [field for field in required_fields if not data.get(field)]
        
        if missing_fields:
            return jsonify({
                'success': False,
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        # Extract data
        system_prompt = data['system_prompt']
        source_text = data['source_text']
        fil_translation = data['fil_translation']
        ref_translation = data.get('ref_translation', '')
        llm_model = data['llm_model']
        llm_provider = data['llm_provider']
        agentic_mode = data.get('agentic_mode', False)
        
        print(f"[DEBUG] Judge request received: provider={llm_provider}, model={llm_model}, agentic_mode={agentic_mode}")
        print(f"[DEBUG] Text lengths: source={len(source_text)}, translation={len(fil_translation)}")
        
        # Validate tokens in system prompt
        token_validation = validate_tokens(system_prompt)
        if not token_validation['valid']:
            return jsonify({
                'success': False,
                'error': f'System prompt missing required tokens: {", ".join(token_validation["missing_tokens"])}'
            }), 400
        
        # Replace tokens in system prompt
        final_prompt = replace_tokens(
            system_prompt,
            source_text=source_text,
            fil_translation=fil_translation,
            ref_translation=ref_translation
        )
        
        print(f"[DEBUG] Final prompt length: {len(final_prompt)}")
        
        # Create LLM client - use agentic client if agentic mode is enabled
        try:
            if agentic_mode:
                print(f"[DEBUG] Creating Agentic LLM client for provider: {llm_provider}")
                llm_client = create_agentic_llm_client(llm_provider, llm_model)
                print(f"[DEBUG] Created agentic client type: {type(llm_client).__name__}")
            else:
                print(f"[DEBUG] Creating standard LLM client for provider: {llm_provider}")
                llm_client = create_llm_client(llm_provider, llm_model)
                print(f"[DEBUG] Created standard client type: {type(llm_client).__name__}")
        except ValueError as e:
            print(f"[DEBUG] Error creating LLM client: {str(e)}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 400
        
        # Generate judgment
        print("[DEBUG] Calling llm_client.generate_judgment...")
        judgment_result = llm_client.generate_judgment(final_prompt)
        print(f"[DEBUG] Judgment result success: {judgment_result.get('success', False)}")
        
        if not judgment_result['success']:
            return jsonify({
                'success': False,
                'error': judgment_result['error'],
                'raw_response': judgment_result.get('raw_response')
            }), 500
        
        # Calculate final score
        judgment_data = judgment_result['data']
        final_score = calculate_final_score(judgment_data)
          # Save judgment results
        save_path = save_judgment_results(data, judgment_result, final_score, final_prompt)
        if not save_path:
            return jsonify({
                'success': False,
                'error': 'Failed to save judgment results'
            }), 500
          # Return complete result
        response_data = {
            'success': True,
            'judgment': judgment_data,
            'final_score': final_score,
            'raw_response': judgment_result.get('raw_response'),  # Include raw response for frontend logging
            'metadata': {
                'model': llm_model,
                'provider': llm_provider,
                'prompt_length': len(final_prompt),
                'saved_file': os.path.basename(save_path) if save_path else None,
                'agentic_mode': agentic_mode
            }
        }
        
        # Add agentic-specific data if available
        if agentic_mode and 'thought_summary' in judgment_result:
            response_data['thought_summary'] = judgment_result['thought_summary']
        
        if agentic_mode and 'agentic_features' in judgment_result:
            response_data['metadata']['agentic_features'] = judgment_result['agentic_features']
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

def save_judgment_results(request_data, llm_response, final_score_data, raw_prompt):
    """Save complete judgment results to a timestamped JSON file in ./results folder"""
    try:
        # Ensure results directory exists
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Create timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        filename = f"judgment_{timestamp}.json"
        filepath = os.path.join(results_dir, filename)
        
        # Prepare complete results data
        results_data = {
            "timestamp": datetime.now().isoformat(),            "request_data": {
                "source_text": request_data.get("source_text"),
                "fil_translation": request_data.get("fil_translation"),
                "ref_translation": request_data.get("ref_translation"),
                "system_prompt": request_data.get("system_prompt"),
                "raw_prompt": raw_prompt,  # The final processed prompt sent to LLM
                "llm_provider": request_data.get("llm_provider"),
                "llm_model": request_data.get("llm_model")
            },
            "llm_response": {
                "success": llm_response.get("success"),
                "data": llm_response.get("data"),
                "raw_response": llm_response.get("raw_response"),
                "error": llm_response.get("error"),
                "thought_summary": llm_response.get("thought_summary")
            },
            "final_score": final_score_data,
            "metadata": {
                "app_version": "1.0",
                "filename": filename,
                "saved_at": datetime.now().isoformat()
            }
        }
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"[DEBUG] Results saved to: {filepath}")
        return filepath
        
    except Exception as e:
        print(f"[ERROR] Failed to save results: {str(e)}")
        return None

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
