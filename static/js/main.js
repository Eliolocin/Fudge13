// Main JavaScript for LLM-as-Judge Translation Interface

class LLMJudgeApp {    constructor() {
        this.initializeElements();
        this.bindEvents();
        this.loadPrompts();
        this.updateLLMInfo();
        this.updateTokenStatus(); // Initial token status check
        this.updateValidationSummary(); // Initial validation summary
    }initializeElements() {
        // Prompt elements
        this.promptButtons = document.getElementById('prompt-buttons');
        this.systemPrompt = document.getElementById('system-prompt');
        this.llmPicker = document.getElementById('llm-picker');
        this.llmInfo = document.getElementById('llm-info');
        
        // Agentic mode elements
        this.agenticModeContainer = document.getElementById('agentic-mode-container');
        this.agenticModeToggle = document.getElementById('agentic-mode-toggle');

        // Token legend elements
        this.tokenItems = document.querySelectorAll('.token-item');

        // Text input elements
        this.sourceText = document.getElementById('source-text');
        this.filTranslation = document.getElementById('fil-translation');
        this.refTranslation = document.getElementById('ref-translation');
        this.judgeBtn = document.getElementById('judge-btn');

        // Results elements
        this.resultsContainer = document.getElementById('results-container');

        // Modal elements
        this.modal = document.getElementById('explanation-modal');
        this.modalTitle = document.getElementById('modal-title');
        this.modalExplanation = document.getElementById('modal-explanation');
        this.closeModal = document.querySelector('.close');        // Current state
        this.selectedPrompt = null;
        this.currentTokens = {};
        this.currentResults = null;
    }bindEvents() {
        // Judge button click
        this.judgeBtn.addEventListener('click', () => this.handleJudge());

        // Modal close events
        this.closeModal.addEventListener('click', () => this.hideModal());
        this.modal.addEventListener('click', (e) => {
            if (e.target === this.modal) {
                this.hideModal();
            }
        });

        // Escape key to close modal
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.modal.style.display === 'block') {
                this.hideModal();
            }
        });        // Real-time token detection (no highlighting overlay)
        this.systemPrompt.addEventListener('input', () => this.updateTokenStatus());

        // LLM picker change
        this.llmPicker.addEventListener('change', () => this.updateLLMInfo());
        
        // Agentic mode toggle change
        this.agenticModeToggle.addEventListener('change', () => this.updateLLMInfo());
        
        // Real-time validation for text inputs (simplified - no visual effects)
        this.sourceText.addEventListener('input', () => this.updateValidationSummary());
        this.filTranslation.addEventListener('input', () => this.updateValidationSummary());
        this.refTranslation.addEventListener('input', () => this.updateValidationSummary());
        this.systemPrompt.addEventListener('input', () => {
            this.updateTokenStatus();
            this.updateValidationSummary();
        });// Token copy button clicks
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('copy-btn')) {
                // Get the token text from the data attribute, but decode HTML entities
                const tokenText = e.target.dataset.token;
                const decodedToken = tokenText.replace(/&#123;/g, '{').replace(/&#125;/g, '}');
                this.copyTokenToClipboard(decodedToken, e.target);
            }
        });
    }    async loadPrompts() {
        try {
            const response = await fetch('/api/prompts');
            const prompts = await response.json();

            if (prompts.length === 0) {
                this.promptButtons.innerHTML = `
                    <div class="no-prompts">
                        📝 No prompts found in ./prompts folder<br>
                        <small>Add .txt files to the prompts directory</small>
                    </div>
                `;
            } else {
                this.promptButtons.innerHTML = prompts.map(prompt => `
                    <button class="prompt-btn" data-filename="${prompt.filename}">
                        <div class="prompt-btn-name">${prompt.name}</div>
                        <div class="prompt-btn-preview">Click to load prompt...</div>
                    </button>
                `).join('');

                // Bind prompt button events
                this.promptButtons.querySelectorAll('.prompt-btn').forEach(btn => {
                    btn.addEventListener('click', () => this.loadPromptContent(btn.dataset.filename, btn));
                });

                // Load preview for each prompt
                this.loadPromptPreviews(prompts);
            }
        } catch (error) {
            console.error('Error loading prompts:', error);
            this.promptButtons.innerHTML = `
                <div class="no-prompts">
                    ❌ Error loading prompts<br>
                    <small>Check console for details</small>
                </div>
            `;
        }
    }

    async loadPromptPreviews(prompts) {
        for (const prompt of prompts) {
            try {
                const response = await fetch(`/api/prompt/${prompt.filename}`);
                const data = await response.json();
                
                if (data.content) {
                    const preview = data.content.substring(0, 80) + (data.content.length > 80 ? '...' : '');
                    const btn = this.promptButtons.querySelector(`[data-filename="${prompt.filename}"]`);
                    if (btn) {
                        const previewEl = btn.querySelector('.prompt-btn-preview');
                        previewEl.textContent = preview;
                    }
                }
            } catch (error) {
                console.error(`Error loading preview for ${prompt.filename}:`, error);
            }
        }
    }    async loadPromptContent(filename, buttonElement = null) {
        try {
            // Update button states
            this.promptButtons.querySelectorAll('.prompt-btn').forEach(btn => {
                btn.classList.remove('selected');
            });
            
            if (buttonElement) {
                buttonElement.classList.add('selected');
                this.selectedPrompt = filename;
            }

            const response = await fetch(`/api/prompt/${filename}`);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();

            if (data.content) {
                this.systemPrompt.value = data.content;
                this.updateTokenStatus();
                this.updateValidationSummary();
            } else {
                this.showError('No content found in prompt file');
            }
        } catch (error) {
            console.error('Error loading prompt content:', error);
            this.showError(`Error loading prompt: ${error.message}`);
        }
    }updateTokenStatus() {
        const text = this.systemPrompt.value;
        const tokenPattern = /\{\{([^}]+)\}\}/g;
        
        // Find all tokens in the text
        const foundTokens = {};
        let match;
        while ((match = tokenPattern.exec(text)) !== null) {
            const tokenName = match[1];
            foundTokens[tokenName] = (foundTokens[tokenName] || 0) + 1;
        }

        // Update current tokens state
        this.currentTokens = foundTokens;

        // Update token legend
        this.updateTokenLegend(foundTokens);
          // Update validation summary only
        this.updateValidationSummary();
    }

    updateValidationSummary() {
        const validation = this.validateInputs();
        const summaryEl = document.querySelector('.validation-summary');
        const judgeBtn = document.getElementById('judge-btn');
        
        if (validation.valid) {
            summaryEl.innerHTML = '<i class="fas fa-check-circle"></i> All fields are valid. Ready to judge!';
            summaryEl.className = 'validation-summary valid';
            judgeBtn.disabled = false;
            judgeBtn.classList.remove('disabled');
        } else {
            summaryEl.innerHTML = `<i class="fas fa-exclamation-triangle"></i> ${validation.message}`;
            summaryEl.className = 'validation-summary error';
            judgeBtn.disabled = true;
            judgeBtn.classList.add('disabled');
        }    }

    updateTokenLegend(foundTokens) {
        this.tokenItems.forEach(item => {
            const tokenName = item.dataset.token;
            const statusEl = item.querySelector('.token-status');
            const usageInfoEl = item.querySelector('.token-usage-info');
            const countEl = item.querySelector('.token-usage-count');
            
            // Remove existing count
            if (countEl) {
                countEl.remove();
            }

            if (foundTokens[tokenName]) {
                statusEl.textContent = 'Found';
                statusEl.className = 'token-status found';
                
                // Update usage info text
                const usageText = usageInfoEl.querySelector('span:first-child');
                if (foundTokens[tokenName] === 1) {
                    usageText.textContent = 'Found once in prompt above';
                } else {
                    usageText.textContent = `Found ${foundTokens[tokenName]} times in prompt above`;
                }
                
                // Add usage count if more than 1
                if (foundTokens[tokenName] > 1) {
                    const newCountEl = document.createElement('span');
                    newCountEl.className = 'token-usage-count';
                    newCountEl.textContent = foundTokens[tokenName];
                    statusEl.appendChild(newCountEl);
                }
            } else {
                if (tokenName === 'ref_translation') {
                    statusEl.textContent = 'Optional';
                    statusEl.className = 'token-status missing';
                    const usageText = usageInfoEl.querySelector('span:first-child');
                    usageText.textContent = 'Copy this token into your prompt';
                } else {
                    statusEl.textContent = 'Required';
                    statusEl.className = 'token-status missing';
                    const usageText = usageInfoEl.querySelector('span:first-child');
                    usageText.textContent = 'Copy this token into your prompt';
                }
            }
        });
    }    updateLLMInfo() {
        const selectedOption = this.llmPicker.selectedOptions[0];
        const model = selectedOption.value;
        const provider = selectedOption.dataset.provider;
        
        // Check if current model supports agentic features (Gemini 2.5 series)
        const agenticSupportedModels = [
            'gemini-2.5-pro',
            'gemini-2.5-flash',
            'gemini-2.5-flash-preview-05-20'
        ];
        const isAgenticSupported = provider === 'google' && agenticSupportedModels.includes(model);
        
        // Show/hide agentic mode toggle based on model support
        if (isAgenticSupported) {
            this.agenticModeContainer.style.display = 'block';
        } else {
            this.agenticModeContainer.style.display = 'none';
            this.agenticModeToggle.checked = false; // Reset toggle when not supported
        }
        
        const providerInfo = {
            google: {
                name: 'Google',
                badge: 'provider-google',
                capabilities: 'Structured output, fast inference, multimodal'
            },
            openai: {
                name: 'OpenAI',
                badge: 'provider-openai', 
                capabilities: 'Function calling, structured output, reasoning'
            }
        };

        const info = providerInfo[provider];
        let capabilities = info.capabilities;
        
        // Add agentic capabilities if enabled
        if (isAgenticSupported && this.agenticModeToggle.checked) {
            capabilities += ', Google Search grounding, Thought summaries';
        }
        
        this.llmInfo.innerHTML = `
            <strong>Provider:</strong> <span class="provider-badge ${info.badge}">${info.name}</span><br>
            <strong>Model:</strong> ${model}<br>
            <strong>Capabilities:</strong> ${capabilities}
        `;
    }    copyTokenToClipboard(tokenText, buttonElement = null) {
        navigator.clipboard.writeText(tokenText).then(() => {
            this.showClipboardNotification(tokenText);
            
            // Show visual feedback on button
            if (buttonElement) {
                const originalText = buttonElement.textContent;
                buttonElement.textContent = 'Copied!';
                buttonElement.classList.add('copied');
                
                setTimeout(() => {
                    buttonElement.textContent = originalText;
                    buttonElement.classList.remove('copied');
                }, 1500);
            }
        }).catch(() => {
            this.showError('Failed to copy token to clipboard');
        });
    }

    showClipboardNotification(tokenText) {
        // Remove any existing notifications
        const existingNotifications = document.querySelectorAll('.clipboard-notification');
        existingNotifications.forEach(notification => {
            notification.remove();
        });

        // Create new notification
        const notification = document.createElement('div');
        notification.className = 'clipboard-notification';
        notification.innerHTML = `Copied <code>${tokenText}</code> to clipboard!`;
        
        document.body.appendChild(notification);
        
        // Auto-remove after 3 seconds
        setTimeout(() => {
            notification.classList.add('hiding');
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300); // Wait for animation to complete
        }, 3000);
    }

    showSuccessMessage(message) {
        // Create a temporary success notification
        const notification = document.createElement('div');
        notification.className = 'success-notification';
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: var(--success-color);
            color: white;
            padding: 12px 20px;
            border-radius: 8px;
            box-shadow: var(--shadow-lg);
            z-index: 1000;
            animation: slideInRight 0.3s ease-out;
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.style.animation = 'slideOutRight 0.3s ease-in';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 2000);
    }

    async handleJudge() {
        // Validate inputs
        const validation = this.validateInputs();
        if (!validation.valid) {
            this.showError(validation.message);
            return;
        }

        // Show loading state
        this.setLoadingState(true);        try {
            const requestData = {
                system_prompt: this.systemPrompt.value,
                source_text: this.sourceText.value,
                fil_translation: this.filTranslation.value,
                ref_translation: this.refTranslation.value,
                llm_model: this.llmPicker.value,
                llm_provider: this.llmPicker.selectedOptions[0].dataset.provider,
                agentic_mode: this.agenticModeToggle.checked || false
            };

            console.log('[DEBUG] Sending judge request:', requestData);
            console.log('[DEBUG] Provider:', requestData.llm_provider, 'Model:', requestData.llm_model);

            const response = await fetch('/api/judge', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            });            const result = await response.json();
            console.log('[DEBUG] Received response:', result);
            
            // Log the raw LLM response for debugging
            if (result.raw_response) {
                console.log('[DEBUG] Raw LLM Response (full):', result.raw_response);
            }

            if (response.ok) {
                this.displayResults(result);
            } else {
                this.showError(result.error || 'Failed to judge translation');
            }
        } catch (error) {
            console.error('[DEBUG] Error during judgment:', error);
            this.showError('Network error occurred');
        } finally {
            this.setLoadingState(false);
        }
    }

    validateInputs() {
        // Check required fields
        if (!this.sourceText.value.trim()) {
            return { valid: false, message: 'Source text is required' };
        }

        if (!this.filTranslation.value.trim()) {
            return { valid: false, message: 'Filipino translation is required' };
        }

        if (!this.systemPrompt.value.trim()) {
            return { valid: false, message: 'System prompt is required' };
        }

        // Check for required tokens
        const prompt = this.systemPrompt.value;
        if (!prompt.includes('{{source_text}}')) {
            return { valid: false, message: 'System prompt must contain {{source_text}} token' };
        }

        if (!prompt.includes('{{fil_translation}}')) {
            return { valid: false, message: 'System prompt must contain {{fil_translation}} token' };
        }

        return { valid: true };
    }

    setLoadingState(loading) {
        this.judgeBtn.disabled = loading;
        if (loading) {
            this.judgeBtn.classList.add('loading');
        } else {
            this.judgeBtn.classList.remove('loading');
        }
    }    displayResults(result) {
        if (!result.success) {
            this.showError(result.error || 'Failed to process judgment');
            return;
        }

        const { judgment, final_score, metadata } = result;
        
        // Create metrics data for easier handling
        const metrics = [
            {
                key: 'accuracy',
                name: 'Accuracy',
                description: 'Is the translation semantically accurate?',
                value: judgment.accuracy,
                explanation: judgment.accuracy_explanation
            },
            {
                key: 'fluency',
                name: 'Fluency',
                description: 'Is the translation fluent in Filipino?',
                value: judgment.fluency,
                explanation: judgment.fluency_explanation
            },
            {
                key: 'coherence',
                name: 'Coherence',
                description: 'Is the translation logically coherent?',
                value: judgment.coherence,
                explanation: judgment.coherence_explanation
            },
            {
                key: 'cultural_appropriateness',
                name: 'Cultural Appropriateness',
                description: 'Is the translation culturally appropriate?',
                value: judgment.cultural_appropriateness,
                explanation: judgment.cultural_appropriateness_explanation
            },
            {
                key: 'guideline_adherence',
                name: 'Guideline Adherence',
                description: 'Does the translation follow guidelines?',
                value: judgment.guideline_adherence,
                explanation: judgment.guideline_adherence_explanation
            },
            {
                key: 'completeness',
                name: 'Completeness',
                description: 'Is the translation complete?',
                value: judgment.completeness,
                explanation: judgment.completeness_explanation
            }
        ];

        this.resultsContainer.innerHTML = `
            <div class="results-header">
                <div class="final-score ${final_score.color}">
                    <div class="score-number">${final_score.score}</div>
                    <div class="score-label">${final_score.label}</div>
                    <div class="score-details">${final_score.true_count}/${final_score.total_criteria} criteria met</div>
                </div>
            </div>
              <div class="metrics-grid">
                ${metrics.map(metric => `                    <button class="metric-card ${metric.value ? 'positive' : 'negative'}" 
                            data-metric="${metric.key}"
                            data-explanation="${metric.explanation.replace(/"/g, '&quot;').replace(/'/g, '&#39;').replace(/\n/g, '\\n').replace(/\r/g, '\\r')}"
                            onclick="app.showMetricExplanation('${metric.name}', this.dataset.explanation)">
                        <div class="metric-icon">
                            <i class="fas fa-${metric.value ? 'check-circle' : 'times-circle'}"></i>
                        </div>
                        <div class="metric-content">
                            <div class="metric-name">${metric.name}</div>
                            <div class="metric-status">${metric.value ? 'Pass' : 'Fail'}</div>
                            <div class="read-more">READ MORE</div>
                        </div>
                    </button>
                `).join('')}
            </div>
            
            ${result.thought_summary ? `
                <div class="thought-summary-section">
                    <div class="thought-summary-header">
                        <h3>
                            <i class="fas fa-brain"></i> 
                            LLM Thought Process
                            <span class="agentic-badge">Agentic Mode</span>
                        </h3>
                        <button class="toggle-thought-btn" onclick="app.toggleThoughtSummary()">
                            <i class="fas fa-chevron-down"></i>
                            <span>Show Details</span>
                        </button>
                    </div>
                    <div class="thought-summary-content" style="display: none;">
                        <div class="thought-summary-text">
                            ${result.thought_summary.replace(/\n/g, '<br>')}
                        </div>
                    </div>
                </div>
            ` : ''}
            
            <div class="judgment-metadata">
                <div class="metadata-item">
                    <strong>Model:</strong> ${metadata.model} (${metadata.provider})
                    ${metadata.agentic_mode ? '<span class="agentic-indicator"><i class="fas fa-robot"></i> Agentic</span>' : ''}
                </div>
                <div class="metadata-item">
                    <strong>Prompt Length:</strong> ${metadata.prompt_length} characters
                </div>
                ${metadata.agentic_features ? `
                    <div class="metadata-item">
                        <strong>Agentic Features:</strong> 
                        ${metadata.agentic_features.google_search_enabled ? '<i class="fas fa-search"></i> Search' : ''}
                        ${metadata.agentic_features.thought_summary_captured ? '<i class="fas fa-brain"></i> Thoughts' : ''}
                    </div>
                ` : ''}
            </div>
        `;
        
        // Store current results for metric explanations
        this.currentResults = { judgment, final_score, metadata };
    }    showMetricExplanation(metricName, explanation) {
        // Decode HTML entities and line breaks
        const decodedExplanation = explanation
            .replace(/&quot;/g, '"')
            .replace(/&#39;/g, "'")
            .replace(/\\n/g, '\n')
            .replace(/\\r/g, '\r');
        this.showModal(metricName, decodedExplanation);
    }

    toggleThoughtSummary() {
        const content = document.querySelector('.thought-summary-content');
        const button = document.querySelector('.toggle-thought-btn');
        const icon = button.querySelector('i');
        const text = button.querySelector('span');
        
        if (content.style.display === 'none') {
            // Show the thought summary
            content.style.display = 'block';
            icon.className = 'fas fa-chevron-up';
            text.textContent = 'Hide Details';
        } else {
            // Hide the thought summary
            content.style.display = 'none';
            icon.className = 'fas fa-chevron-down';
            text.textContent = 'Show Details';
        }
    }

    showError(message) {
        // Remove any existing error notifications
        const existingErrors = document.querySelectorAll('.error-notification');
        existingErrors.forEach(error => error.remove());

        // Create new error notification
        const notification = document.createElement('div');
        notification.className = 'error-notification';
        notification.innerHTML = `
            <i class="fas fa-exclamation-circle"></i>
            <span>${message}</span>
            <button class="close-error" onclick="this.parentElement.remove()">
                <i class="fas fa-times"></i>
            </button>
        `;
        
        document.body.appendChild(notification);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.classList.add('hiding');
                setTimeout(() => {
                    if (notification.parentNode) {
                        notification.parentNode.removeChild(notification);
                    }
                }, 300);
            }
        }, 5000);
    }

    showModal(title, explanation) {
        this.modalTitle.textContent = title;
        this.modalExplanation.textContent = explanation;
        this.modal.style.display = 'block';
    }

    hideModal() {
        this.modal.style.display = 'none';
    }
}

// Initialize the app when DOM is loaded
let app; // Global app instance
document.addEventListener('DOMContentLoaded', () => {
    app = new LLMJudgeApp();
});
