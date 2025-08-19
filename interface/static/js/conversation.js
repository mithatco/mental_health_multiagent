// Global variables
let conversationId = null;
let conversationInterval = null;
let lastMessageCount = 0;

// Add mode toggle variables
let currentMode = 'single'; // 'single', 'oneshot', or 'batch'
let batchId = null;
let batchInterval = null;

// Document ready function
document.addEventListener('DOMContentLoaded', function() {
    // Initialize the form with available options
    initializeForm();
    
    // Add event listener for form submission
    document.getElementById('conversation-form').addEventListener('submit', function(e) {
        e.preventDefault();
        if (currentMode === 'single') {
            startConversation();
        } else if (currentMode === 'oneshot') {
            startOneshotConversation();
        } else {
            startBatchGeneration();
        }
    });
    
    // Add event listener for stop buttons
    document.getElementById('stop-btn').addEventListener('click', stopConversation);
    document.getElementById('stop-oneshot-btn').addEventListener('click', stopOneshotConversation);
    document.getElementById('stop-batch-btn').addEventListener('click', stopBatchGeneration);
    
    // Add event listener for view log buttons
    document.getElementById('view-log-btn').addEventListener('click', viewConversationLog);
    document.getElementById('view-oneshot-log-btn').addEventListener('click', viewOneshotLog);
    document.getElementById('view-batch-btn').addEventListener('click', viewBatchResults);
    
    // Add mode toggle handlers
    document.getElementById('single-mode-btn').addEventListener('click', () => switchMode('single'));
    document.getElementById('oneshot-mode-btn').addEventListener('click', () => switchMode('oneshot'));
    document.getElementById('batch-mode-btn').addEventListener('click', () => switchMode('batch'));
    
    // Store providers data from the server
    window.providersData = providersData || {};
    
    // Add event listeners for model selection changes
    document.getElementById('assistant-model-select').addEventListener('change', checkForGroqModels);
    document.getElementById('patient-model-select').addEventListener('change', checkForGroqModels);
    document.getElementById('agent-model-select').addEventListener('change', checkForGroqModels);
    
    // Check initially
    checkForGroqModels();
});

// Initialize form with available questionnaires, profiles and models
function initializeForm() {
    // Fetch available questionnaires
    fetch('/api/questionnaires')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            const select = document.getElementById('questionnaire-select');
            select.innerHTML = '';
            
            // Check for API errors
            if (data.error) {
                console.error('API Error:', data.error);
                const option = document.createElement('option');
                option.value = '';
                option.textContent = `Error: ${data.error}`;
                option.disabled = true;
                select.appendChild(option);
                
                // Also show the error in the conversation display for visibility
                document.getElementById('conversation-display').innerHTML = `
                    <div class="error-message">
                        <h3>Error Loading Questionnaires</h3>
                        <p>${data.error}</p>
                        <p>Please check that your questionnaire files are properly formatted PDF files in the documents/questionnaires directory.</p>
                        ${data.traceback ? `<details><summary>Technical Details</summary><pre>${data.traceback}</pre></details>` : ''}
                    </div>
                `;
                return;
            }
            
            if (data.questionnaires && data.questionnaires.length) {
                console.log('Questionnaires loaded:', data.questionnaires);
                data.questionnaires.forEach((questionnaire, index) => {
                    const option = document.createElement('option');
                    option.value = questionnaire.id;
                    option.textContent = `${questionnaire.name} (${questionnaire.question_count} questions)`;
                    // Select the first questionnaire by default
                    if (index === 0) {
                        option.selected = true;
                    }
                    select.appendChild(option);
                });
                
                // If there's a warning, show it but still allow selection
                if (data.warning) {
                    console.warn('API Warning:', data.warning);
                }
            } else {
                const option = document.createElement('option');
                option.value = '';
                option.textContent = 'No questionnaires available';
                option.disabled = true;
                select.appendChild(option);
                
                // Show help message in conversation display
                document.getElementById('conversation-display').innerHTML = `
                    <div class="notice-message">
                        <h3>No Questionnaires Found</h3>
                        <p>To get started:</p>
                        <ol>
                            <li>Create a <code>documents/questionnaires</code> directory in your project folder</li>
                            <li>Add PDF files containing questions to this directory</li>
                            <li>Questions should end with a question mark (?)</li>
                            <li>Refresh this page</li>
                        </ol>
                    </div>
                `;
            }
        })
        .catch(error => {
            console.error('Error fetching questionnaires:', error);
            const select = document.getElementById('questionnaire-select');
            select.innerHTML = '<option value="" disabled selected>Error loading questionnaires</option>';
            
            // Show error in conversation display
            document.getElementById('conversation-display').innerHTML = `
                <div class="error-message">
                    <h3>Error Loading Questionnaires</h3>
                    <p>${error.message || error}</p>
                    <p>Please check your network connection and ensure the server is running.</p>
                </div>
            `;
        });
    
    // Fetch available profiles
    fetch('/api/profiles')
        .then(response => response.json())
        .then(data => {
            const select = document.getElementById('profile-select');
            select.innerHTML = '';
            
            if (data.profiles && data.profiles.length) {
                data.profiles.forEach(profile => {
                    const option = document.createElement('option');
                    option.value = profile;
                    option.textContent = profile;
                    select.appendChild(option);
                });
            } else {
                const option = document.createElement('option');
                option.value = '';
                option.textContent = 'No profiles available';
                option.disabled = true;
                select.appendChild(option);
            }
        })
        .catch(error => {
            console.error('Error fetching profiles:', error);
            const select = document.getElementById('profile-select');
            select.innerHTML = '<option value="" disabled selected>Error loading profiles</option>';
        });
    
    // Fetch available models
    fetch('/api/models')
        .then(response => response.json())
        .then(data => {
            const assistantSelect = document.getElementById('assistant-model-select');
            const patientSelect = document.getElementById('patient-model-select');
            const agentSelect = document.getElementById('agent-model-select');
            
            assistantSelect.innerHTML = '';
            patientSelect.innerHTML = '';
            agentSelect.innerHTML = '';
            
            if (data.models && data.models.length) {
                // Create option groups for each provider
                const ollamaAssistantGroup = document.createElement('optgroup');
                ollamaAssistantGroup.label = 'Ollama (Local)';
                
                const ollamaPatientGroup = document.createElement('optgroup');
                ollamaPatientGroup.label = 'Ollama (Local)';
                
                const ollamaAgentGroup = document.createElement('optgroup');
                ollamaAgentGroup.label = 'Ollama (Local)';
                
                const groqAssistantGroup = document.createElement('optgroup');
                groqAssistantGroup.label = 'Groq (Cloud API)';
                
                const groqPatientGroup = document.createElement('optgroup');
                groqPatientGroup.label = 'Groq (Cloud API)';
                
                const groqAgentGroup = document.createElement('optgroup');
                groqAgentGroup.label = 'Groq (Cloud API)';
                
                // Add models to appropriate groups
                data.models.forEach(model => {
                    // Get the provider and model name
                    const provider = model.provider;
                    const name = model.name;
                    const displayName = model.display_name || name;
                    
                    // Create option elements for each select
                    if (provider === 'ollama') {
                        // Assistant select
                        const assistantOption = document.createElement('option');
                        assistantOption.value = name;
                        assistantOption.textContent = displayName;
                        assistantOption.dataset.provider = provider;
                                            // Default to qwen3:4b if available
                    if (name === 'qwen3:4b') {
                            assistantOption.selected = true;
                        }
                        ollamaAssistantGroup.appendChild(assistantOption);
                        
                        // Patient select
                        const patientOption = document.createElement('option');
                        patientOption.value = name;
                        patientOption.textContent = displayName;
                        patientOption.dataset.provider = provider;
                                            // Default to qwen3:4b if available
                    if (name === 'qwen3:4b') {
                            patientOption.selected = true;
                        }
                        ollamaPatientGroup.appendChild(patientOption);
                        
                        // Agent select for one-shot mode
                        const agentOption = document.createElement('option');
                        agentOption.value = name;
                        agentOption.textContent = displayName;
                        agentOption.dataset.provider = provider;
                                            // Default to qwen3:4b if available
                    if (name === 'qwen3:4b') {
                            agentOption.selected = true;
                        }
                        ollamaAgentGroup.appendChild(agentOption);
                    } else if (provider === 'groq') {
                        // Assistant select
                        const assistantOption = document.createElement('option');
                        assistantOption.value = name;
                        assistantOption.textContent = displayName;
                        assistantOption.dataset.provider = provider;
                        groqAssistantGroup.appendChild(assistantOption);
                        
                        // Patient select
                        const patientOption = document.createElement('option');
                        patientOption.value = name;
                        patientOption.textContent = displayName;
                        patientOption.dataset.provider = provider;
                        groqPatientGroup.appendChild(patientOption);
                        
                        // Agent select for one-shot mode
                        const agentOption = document.createElement('option');
                        agentOption.value = name;
                        agentOption.textContent = displayName;
                        agentOption.dataset.provider = provider;
                        groqAgentGroup.appendChild(agentOption);
                    }
                });
                
                // Add option groups to selects if they have children
                if (ollamaAssistantGroup.children.length > 0) {
                    assistantSelect.appendChild(ollamaAssistantGroup);
                }
                
                if (ollamaPatientGroup.children.length > 0) {
                    patientSelect.appendChild(ollamaPatientGroup);
                }
                
                if (ollamaAgentGroup.children.length > 0) {
                    agentSelect.appendChild(ollamaAgentGroup);
                }
                
                if (groqAssistantGroup.children.length > 0) {
                    assistantSelect.appendChild(groqAssistantGroup);
                }
                
                if (groqPatientGroup.children.length > 0) {
                    patientSelect.appendChild(groqPatientGroup);
                }
                
                if (groqAgentGroup.children.length > 0) {
                    agentSelect.appendChild(groqAgentGroup);
                }
                
                // Check if any models were added
                if (assistantSelect.children.length === 0) {
                    const option = document.createElement('option');
                    option.value = '';
                    option.textContent = 'No models available';
                    option.disabled = true;
                    assistantSelect.appendChild(option);
                    patientSelect.appendChild(option.cloneNode(true));
                    agentSelect.appendChild(option.cloneNode(true));
                }
                
                // Add change listeners to check for Groq models
                assistantSelect.addEventListener('change', checkForGroqModels);
                patientSelect.addEventListener('change', checkForGroqModels);
                agentSelect.addEventListener('change', checkForGroqModels);
                
                // Initial check for Groq models
                checkForGroqModels();
                
                // Add API key toggle button functionality
                document.getElementById('toggle-api-key-btn').addEventListener('click', function() {
                    const apiKeyField = document.getElementById('groq-api-key');
                    const btnText = this.textContent;
                    
                    if (apiKeyField.type === 'password') {
                        apiKeyField.type = 'text';
                        this.textContent = 'Hide';
                    } else {
                        apiKeyField.type = 'password';
                        this.textContent = 'Show';
                    }
                });
                
                // Add providers info
                if (data.providers) {
                    // Check Ollama availability
                    if (data.providers.ollama && !data.providers.ollama.available) {
                        document.getElementById('conversation-display').innerHTML += `
                            <div class="notice-message">
                                <h3>Ollama Not Available</h3>
                                <p>The Ollama server doesn't appear to be running.</p>
                                <p>Please make sure to start Ollama locally before using Ollama models.</p>
                            </div>
                        `;
                    }
                    
                    // Check Groq availability
                    if (data.providers.groq && !data.providers.groq.available) {
                        document.getElementById('groq-api-section').style.display = 'block';
                        document.getElementById('conversation-display').innerHTML += `
                            <div class="notice-message">
                                <h3>Groq API Key Required</h3>
                                <p>To use Groq models, you need to provide an API key.</p>
                                <p>You can enter it in the "Advanced Options" section.</p>
                            </div>
                        `;
                    }
                }
            } else {
                // Error options
                const option1 = document.createElement('option');
                option1.value = '';
                option1.textContent = 'No models available';
                option1.disabled = true;
                assistantSelect.appendChild(option1);
                
                const option2 = document.createElement('option');
                option2.value = '';
                option2.textContent = 'No models available';
                option2.disabled = true;
                patientSelect.appendChild(option2);
                
                const option3 = document.createElement('option');
                option3.value = '';
                option3.textContent = 'No models available';
                option3.disabled = true;
                agentSelect.appendChild(option3);
            }
        })
        .catch(error => {
            console.error('Error fetching models:', error);
            const assistantSelect = document.getElementById('assistant-model-select');
            const patientSelect = document.getElementById('patient-model-select');
            const agentSelect = document.getElementById('agent-model-select');
            
            assistantSelect.innerHTML = '<option value="" disabled selected>Error loading models</option>';
            patientSelect.innerHTML = '<option value="" disabled selected>Error loading models</option>';
            agentSelect.innerHTML = '<option value="" disabled selected>Error loading models</option>';
        });
}

// Start a new conversation
function startConversation() {
    // Get form data BEFORE disabling the form
    const form = document.getElementById('conversation-form');
    const formData = new FormData(form);
    
    // Now disable form and show loading state
    setFormEnabled(false);
    updateStatus('Starting conversation...', 'loading');
    
    // Clear previous conversation
    document.getElementById('conversation-display').innerHTML = '';
    document.getElementById('diagnosis-content').innerHTML = '<p>The diagnosis will appear after the conversation completes.</p>';
    
    // Get provider data from data attributes
    const assistantSelect = document.getElementById('assistant-model-select');
    const patientSelect = document.getElementById('patient-model-select');
    
    const assistantOption = assistantSelect.options[assistantSelect.selectedIndex];
    const patientOption = patientSelect.options[patientSelect.selectedIndex];
    
    const assistantProvider = assistantOption ? assistantOption.dataset.provider || 'ollama' : 'ollama';
    const patientProvider = patientOption ? patientOption.dataset.provider || 'ollama' : 'ollama'; 
    
    const groqApiKey = document.getElementById('groq-api-key').value;
    
    // Validate Groq API key if needed
    if ((assistantProvider === 'groq' || patientProvider === 'groq') && !groqApiKey) {
        updateStatus('Error: Groq API key is required for Groq models', 'error');
        setFormEnabled(true);
        
        // Show error message in conversation display
        document.getElementById('conversation-display').innerHTML = `
            <div class="error-message">
                <h3>Groq API Key Required</h3>
                <p>Please enter your Groq API key in the Advanced Options section.</p>
                <p>You can get a key from <a href="https://console.groq.com/" target="_blank">console.groq.com</a></p>
            </div>
        `;
        return;
    }
    
    // Debug log form entries
    console.log("Form entries:");
    for (const pair of formData.entries()) {
        console.log(pair[0] + ': ' + pair[1]);
    }
    
    // Debug questionnaire selection
    const questionnaireSelect = document.getElementById('questionnaire-select');
    console.log('Questionnaire select element:', questionnaireSelect);
    console.log('Selected index:', questionnaireSelect.selectedIndex);
    console.log('Selected value:', questionnaireSelect.value);
    console.log('All options:', Array.from(questionnaireSelect.options).map(opt => ({value: opt.value, text: opt.text, selected: opt.selected})));
    
    // Build JSON payload
    const payload = {
        questionnaire: formData.get('questionnaire'),
        profile: formData.get('profile'),
        assistant_model: formData.get('assistant_model'),
        patient_model: formData.get('patient_model'),
        save_logs: formData.get('save_logs') === 'on',
        refresh_cache: formData.get('refresh_cache') === 'on',
        disable_rag: formData.get('disable_rag') === 'on',
        disable_rag_evaluation: formData.get('disable_rag_evaluation') === 'on',
        // Add provider information
        assistant_provider: assistantProvider,
        patient_provider: patientProvider,
        groq_api_key: groqApiKey
    };
    
    console.log('Request payload:', payload);
    
    // Additional validation
    if (!payload.questionnaire) {
        console.error('No questionnaire in payload - questionnaire value is:', payload.questionnaire);
        console.error('FormData questionnaire value:', formData.get('questionnaire'));
    }
    
    // Send the request
    fetch('/api/conversations/start', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(data => {
                throw new Error(data.error || `HTTP error! Status: ${response.status}`);
            });
        }
        return response.json();
    })
    .then(data => {
        conversationId = data.conversation_id;
        console.log(`Conversation started with ID: ${conversationId}`);
        updateStatus('Conversation in progress...', 'loading');
        
        // Start polling for updates
        startPolling();
        
        // Enable stop button
        document.getElementById('stop-btn').disabled = false;
    })
    .catch(error => {
        console.error('Error starting conversation:', error);
        updateStatus(`Error: ${error.message}`, 'error');
        setFormEnabled(true);
        
        // Show error message in conversation display
        document.getElementById('conversation-display').innerHTML = `
            <div class="error-message">
                <h3>Error Starting Conversation</h3>
                <p>${error.message}</p>
                <p>Please check your settings and try again.</p>
            </div>
        `;
    });
}

// Stop the conversation
function stopConversation() {
    if (!conversationId) return;
    
    updateStatus('Stopping conversation...', 'loading');
    
    fetch(`/api/conversations/${conversationId}/stop`, {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            updateStatus('Conversation stopped', 'inactive');
            stopPolling();
            
            // Re-enable form
            setFormEnabled(true);
        } else {
            updateStatus(`Error: ${data.error || 'Failed to stop conversation'}`, 'error');
        }
    })
    .catch(error => {
        console.error('Error stopping conversation:', error);
        updateStatus('Error stopping conversation', 'error');
    });
}

// Start polling for conversation updates
function startPolling() {
    if (conversationInterval) {
        clearInterval(conversationInterval);
    }
    
    lastMessageCount = 0;
    
    conversationInterval = setInterval(() => {
        if (!conversationId) return;
        
        fetch(`/api/conversations/${conversationId}/status`)
            .then(response => response.json())
            .then(data => {
                console.log(`[POLLING] Status response for conversation ${conversationId}:`, data);
                console.log(`[POLLING] Conversation length: ${data.conversation ? data.conversation.length : 'N/A'}`);
                console.log(`[POLLING] Status: ${data.status}`);
                console.log(`[POLLING] Last message count: ${lastMessageCount}`);
                
                if (data.status === 'completed') {
                    console.log('[POLLING] Conversation completed, updating display');
                    // Conversation is complete
                    updateConversation(data.conversation);
                    updateDiagnosis(data.diagnosis);
                    updateStatus('Conversation completed', 'completed');
                    stopPolling();
                    
                    // Enable view log button if log is saved
                    if (data.log_saved) {
                        document.getElementById('view-log-btn').disabled = false;
                        document.getElementById('view-log-btn').dataset.logId = data.log_id;
                    }
                    
                    // Re-enable form
                    setFormEnabled(true);
                } else if (data.status === 'error') {
                    console.log('[POLLING] Conversation had error:', data.error);
                    // Conversation had an error
                    updateStatus(`Error: ${data.error || 'An error occurred'}`, 'error');
                    stopPolling();
                    setFormEnabled(true);
                } else if (data.status === 'in_progress') {
                    console.log('[POLLING] Conversation in progress, updating messages');
                    // Update conversation with new messages
                    updateConversation(data.conversation);
                    updateStatus('Conversation in progress...', 'active');
                } else {
                    console.log('[POLLING] Unknown status:', data.status);
                }
            })
            .catch(error => {
                console.error('Error polling conversation status:', error);
            });
    }, 1000);  // Poll every second
}

// Stop polling for updates
function stopPolling() {
    if (conversationInterval) {
        clearInterval(conversationInterval);
        conversationInterval = null;
    }
    
    // Disable stop button
    document.getElementById('stop-btn').disabled = true;
}

// Update the conversation display with messages
function updateConversation(conversation) {
    console.log('[UPDATE] updateConversation called with:', conversation);
    console.log('[UPDATE] Conversation length:', conversation ? conversation.length : 'N/A');
    console.log('[UPDATE] Last message count:', lastMessageCount);
    
    if (!conversation || !conversation.length) {
        console.log('[UPDATE] No conversation data, returning early');
        return;
    }
    
    const display = document.getElementById('conversation-display');
    
    // Only update if we have new messages
    if (conversation.length <= lastMessageCount) {
        console.log('[UPDATE] No new messages, returning early');
        return;
    }
    
    console.log('[UPDATE] Adding new messages from index', lastMessageCount, 'to', conversation.length - 1);
    
    // Clear placeholder if it exists
    const placeholder = display.querySelector('.conversation-placeholder');
    if (placeholder) {
        display.innerHTML = '';
    }
    
    // Add new messages
    for (let i = lastMessageCount; i < conversation.length; i++) {
        const message = conversation[i];
        console.log(`[UPDATE] Processing message ${i}:`, message);
        
        // Skip system messages
        if (message.role === 'system') {
            console.log(`[UPDATE] Skipping system message ${i}`);
            continue;
        }
        
        console.log(`[UPDATE] Adding message ${i} with role: ${message.role}`);
        
        // Create message element
        const messageDiv = document.createElement('div');
        messageDiv.className = `message message-${message.role} clearfix`;
        
        const roleLabel = document.createElement('div');
        roleLabel.className = 'role-label';
        roleLabel.textContent = message.role.charAt(0).toUpperCase() + message.role.slice(1);
        
        const contentDiv = document.createElement('div');
        contentDiv.className = `message-content ${message.role}-bubble`;
        contentDiv.innerHTML = formatText(message.content);
        
        messageDiv.appendChild(roleLabel);
        messageDiv.appendChild(contentDiv);
        display.appendChild(messageDiv);
        
        console.log(`[UPDATE] Successfully added message ${i} to display`);
    }
    
    // Update last message count
    lastMessageCount = conversation.length;
    console.log(`[UPDATE] Updated lastMessageCount to: ${lastMessageCount}`);
    
    // Scroll to bottom
    display.scrollTop = display.scrollHeight;
}

// Update diagnosis display
function updateDiagnosis(diagnosis) {
    if (!diagnosis) return;
    
    const diagnosisContent = document.getElementById('diagnosis-content');
    diagnosisContent.innerHTML = formatText(diagnosis);
}

// Format text with line breaks
function formatText(text) {
    if (!text) return '';
    return text.split('\n').join('<br>');
}

// Update status message
function updateStatus(message, state) {
    const status = document.getElementById('status');
    status.textContent = message;
    
    // Remove all state classes
    status.classList.remove('loading', 'active', 'completed', 'error', 'inactive');
    
    // Add appropriate state class
    if (state) {
        status.classList.add(state);
    }
}

// Enable or disable form elements
function setFormEnabled(enabled) {
    const form = document.getElementById('conversation-form');
    const elements = form.elements;
    
    for (let i = 0; i < elements.length; i++) {
        elements[i].disabled = !enabled;
    }
    
    // Update start button text
    const startBtn = document.getElementById('start-btn');
    startBtn.textContent = enabled ? 'Start Conversation' : 'Conversation in Progress...';
}

// View saved conversation log
function viewConversationLog() {
    const viewLogBtn = document.getElementById('view-log-btn');
    const logId = viewLogBtn.dataset.logId;
    
    if (logId) {
        window.location.href = `/?log=${logId}`;
    }
}

// Function to switch between modes
function switchMode(mode) {
    currentMode = mode;
    
    // Update button states
    document.getElementById('single-mode-btn').classList.toggle('active', mode === 'single');
    document.getElementById('oneshot-mode-btn').classList.toggle('active', mode === 'oneshot');
    document.getElementById('batch-mode-btn').classList.toggle('active', mode === 'batch');
    
    // Show/hide appropriate views
    document.getElementById('single-conversation-view').style.display = mode === 'single' ? 'block' : 'none';
    document.getElementById('oneshot-conversation-view').style.display = mode === 'oneshot' ? 'block' : 'none';
    document.getElementById('batch-conversation-view').style.display = mode === 'batch' ? 'block' : 'none';
    
    // Show/hide batch-only form elements
    const batchOnlyElements = document.querySelectorAll('.batch-only');
    batchOnlyElements.forEach(element => {
        element.style.display = (mode === 'batch') ? 'block' : 'none';
    });
    
    // Show/hide oneshot-only form elements
    const oneshotOnlyElements = document.querySelectorAll('.oneshot-only');
    oneshotOnlyElements.forEach(element => {
        element.style.display = (mode === 'oneshot') ? 'block' : 'none';
    });
    
    // Show/hide turn-based-only form elements
    const turnBasedOnlyElements = document.querySelectorAll('.turn-based-only');
    turnBasedOnlyElements.forEach(element => {
        element.style.display = (mode === 'single' || mode === 'batch') ? 'block' : 'none';
    });
    
    // Ensure form fields maintain their values when switching modes
    const questionnaireSelect = document.getElementById('questionnaire-select');
    if (questionnaireSelect.selectedIndex > 0) {
        console.log(`Current questionnaire selection: ${questionnaireSelect.value}`);
    }
    
    // Update form submit button text
    const startBtn = document.getElementById('start-btn');
    if (mode === 'single') {
        startBtn.textContent = 'Start Turn-based Conversation';
    } else if (mode === 'oneshot') {
        startBtn.textContent = 'Generate One-shot Conversation';
    } else {
        startBtn.textContent = 'Start Full Simulation';
    }
}

// Start batch generation
function startBatchGeneration() {
    // Get form data BEFORE disabling the form
    const form = document.getElementById('conversation-form');
    const formData = new FormData(form);
    
    // Now disable form and show loading state
    setFormEnabled(false);
    updateBatchStatus('Starting simulation...', 'loading');
    
    // Reset progress bar
    document.getElementById('batch-progress-bar').style.width = '0%';
    document.getElementById('batch-progress-text').textContent = '0 / 0 conversations completed';
    document.getElementById('batch-progress-details').textContent = 'Initializing simulation...';
    
    // Clear previous results
    document.getElementById('batch-results-container').innerHTML = '<p>Simulation results will appear here after completion.</p>';
    
    // Get provider data from data attributes
    const assistantSelect = document.getElementById('assistant-model-select');
    const patientSelect = document.getElementById('patient-model-select');
    
    const assistantOption = assistantSelect.options[assistantSelect.selectedIndex];
    const patientOption = patientSelect.options[patientSelect.selectedIndex];
    
    const assistantProvider = assistantOption ? assistantOption.dataset.provider || 'ollama' : 'ollama';
    const patientProvider = patientOption ? patientOption.dataset.provider || 'ollama' : 'ollama'; 
    
    const groqApiKey = document.getElementById('groq-api-key').value;
    
    // Validate Groq API key if needed
    if ((assistantProvider === 'groq' || patientProvider === 'groq') && !groqApiKey) {
        updateBatchStatus('Error: Groq API key is required for Groq models', 'error');
        setFormEnabled(true);
        
        // Show error message in batch results
        document.getElementById('batch-results-container').innerHTML = `
            <div class="error-message">
                <h3>Groq API Key Required</h3>
                <p>Please enter your Groq API key in the Advanced Options section.</p>
                <p>You can get a key from <a href="https://console.groq.com/" target="_blank">console.groq.com</a></p>
            </div>
        `;
        return;
    }
    
    // Build JSON payload
    const payload = {
        questionnaire: formData.get('questionnaire'),
        profile: formData.get('profile'),
        batch_count: parseInt(formData.get('batch_count'), 10) || 5,
        randomize_profiles: formData.get('randomize_profiles') === 'on',
        assistant_model: formData.get('assistant_model'),
        patient_model: formData.get('patient_model'),
        save_logs: formData.get('save_logs') === 'on',
        refresh_cache: formData.get('refresh_cache') === 'on',
        disable_rag: formData.get('disable_rag') === 'on',
        disable_rag_evaluation: formData.get('disable_rag_evaluation') === 'on',
        // Add provider information
        assistant_provider: assistantProvider,
        patient_provider: patientProvider,
        groq_api_key: groqApiKey
    };
    
    console.log('Batch generation payload:', payload);
    
    // Send the request
    fetch('/api/batches/start', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(data => {
                throw new Error(data.error || `HTTP error! Status: ${response.status}`);
            });
        }
        return response.json();
    })
    .then(data => {
        batchId = data.batch_id;
        const totalConversations = data.total_conversations || payload.batch_count;
        
        console.log(`Batch generation started with ID: ${batchId}, total: ${totalConversations} conversations`);
        updateBatchStatus('Simulation in progress...', 'loading');
        
        // Start polling for updates
        startBatchPolling(batchId, totalConversations);
        
        // Enable stop button
        document.getElementById('stop-batch-btn').disabled = false;
    })
    .catch(error => {
        console.error('Error starting batch generation:', error);
        updateBatchStatus(`Error: ${error.message}`, 'error');
        setFormEnabled(true);
        
        // Show error message in batch results
        document.getElementById('batch-results-container').innerHTML = `
            <div class="error-message">
                <h3>Error Starting Simulation</h3>
                <p>${error.message}</p>
                <p>Please check your settings and try again.</p>
            </div>
        `;
    });
}

// Start polling for batch generation updates
function startBatchPolling(batchId, totalConversations) {
    if (batchInterval) {
        clearInterval(batchInterval);
    }
    
    // Add console log when polling starts
    console.log(`Starting full simulation polling for batch ID: ${batchId}, total: ${totalConversations}`);
    
    let consecutiveErrors = 0;
    
    batchInterval = setInterval(() => {
        if (!batchId) return;
        
        console.log(`Polling batch status for ${batchId}...`);
        
        fetch(`/api/batches/${batchId}/status`, {
            cache: 'no-store', // Add cache control
            headers: {
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache'
            }
        })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                // Log the data received for debugging
                console.log(`Batch status data received:`, data);
                
                // Reset error counter on success
                consecutiveErrors = 0;
                
                // Update progress with enhanced information
                const completed = data.completed_conversations || 0;
                const total = totalConversations || data.total_conversations || 0;
                const inProgressIdx = data.in_progress_conversation;
                
                // Calculate percentages
                const completedPercent = total > 0 ? (completed / total) * 100 : 0;
                const inProgressPercent = (inProgressIdx !== null && inProgressIdx !== undefined && total > 0) ? (1 / total) * 100 : 0;
                
                console.log(`Progress: ${completed}/${total} completed (${completedPercent.toFixed(1)}%), in progress: ${inProgressIdx}`);
                
                // Update DOM elements for the progress bars
                const progressContainer = document.getElementById('batch-progress-bar');
                
                // Clear existing progress bars
                progressContainer.innerHTML = '';
                
                // Add completed progress bar
                if (completedPercent > 0) {
                    const completedBar = document.createElement('div');
                    completedBar.className = 'progress-bar-completed';
                    completedBar.style.width = `${completedPercent}%`;
                    progressContainer.appendChild(completedBar);
                }
                
                // Add in-progress bar if there's an active conversation
                if (inProgressIdx !== null && inProgressIdx !== undefined && inProgressPercent > 0) {
                    const inProgressBar = document.createElement('div');
                    inProgressBar.className = 'progress-bar-in-progress';
                    inProgressBar.style.width = `${inProgressPercent}%`;
                    inProgressBar.style.left = `${completedPercent}%`;
                    progressContainer.appendChild(inProgressBar);
                }
                
                // Update progress text
                document.getElementById('batch-progress-text').textContent = 
                    `${completed} / ${total} conversations completed`;
                
                // Add more detailed progress information
                let detailsText = 'Full simulation in progress...';
                if (inProgressIdx !== null && inProgressIdx !== undefined) {
                    detailsText = `Processing conversation ${inProgressIdx + 1} of ${total}`;
                } else if (completed === total) {
                    detailsText = 'All conversations completed';
                }
                
                // Try to update details element if it exists
                const progressDetails = document.getElementById('batch-progress-details');
                if (progressDetails) {
                    progressDetails.textContent = detailsText;
                }
                
                if (data.status === 'completed') {
                    // Batch is complete
                    updateBatchStatus('Full simulation completed', 'completed');
                    updateBatchResults(data.results);
                    stopBatchPolling();
                    
                    // Enable view batch button
                    document.getElementById('view-batch-btn').disabled = false;
                    document.getElementById('view-batch-btn').dataset.batchId = batchId;
                    
                    // Re-enable form
                    setFormEnabled(true);
                } else if (data.status === 'error') {
                    // Batch had an error
                    updateBatchStatus(`Error: ${data.error || 'An error occurred'}`, 'error');
                    stopBatchPolling();
                    setFormEnabled(true);
                } else if (data.status === 'in_progress') {
                    // Update batch status
                    updateBatchStatus('Full simulation in progress...', 'active');
                }
            })
            .catch(error => {
                consecutiveErrors++;
                console.error(`Error polling batch status (attempt ${consecutiveErrors}):`, error);
                
                // After 5 consecutive errors, stop polling
                if (consecutiveErrors >= 5) {
                    console.error('Too many consecutive errors, stopping batch polling');
                    updateBatchStatus('Error: Lost connection to server', 'error');
                    stopBatchPolling();
                    setFormEnabled(true);
                }
            });
    }, 2000);  // Poll every 2 seconds
}

// Stop polling for batch updates
function stopBatchPolling() {
    if (batchInterval) {
        clearInterval(batchInterval);
        batchInterval = null;
    }
    
    // Disable stop button
    document.getElementById('stop-batch-btn').disabled = true;
}

// Stop batch generation
function stopBatchGeneration() {
    if (!batchId) return;
    
    updateBatchStatus('Stopping full simulation...', 'loading');
    
    fetch(`/api/batches/${batchId}/stop`, {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            updateBatchStatus('Full simulation stopped', 'inactive');
            stopBatchPolling();
            
            // Re-enable form
            setFormEnabled(true);
        } else {
            updateBatchStatus(`Error: ${data.error || 'Failed to stop full simulation'}`, 'error');
        }
    })
    .catch(error => {
        console.error('Error stopping full simulation:', error);
        updateBatchStatus('Error stopping full simulation', 'error');
    });
}

// Update batch status display
function updateBatchStatus(message, state) {
    const status = document.getElementById('batch-status');
    status.textContent = message;
    
    // Remove all state classes
    status.classList.remove('loading', 'active', 'completed', 'error', 'inactive');
    
    // Add appropriate state class
    if (state) {
        status.classList.add(state);
    }
}

// Update batch results display
function updateBatchResults(results) {
    console.log('Updating full simulation results:', results);
    
    if (!results || !results.conversations) {
        document.getElementById('batch-results-container').innerHTML = `
            <div class="notice-message">
                <p>No results available for this full simulation.</p>
            </div>
        `;
        return;
    }
    
    // Sort conversations by timestamp (newest first)
    const conversations = [...results.conversations].sort((a, b) => {
        // Sort by timestamp if available
        const timeA = a.timestamp || '';
        const timeB = b.timestamp || '';
        if (timeA < timeB) return 1;
        if (timeA > timeB) return -1;
        return 0; // timestamps are equal
    });
    
    // Check if batch has any conversations
    if (conversations.length === 0) {
        document.getElementById('batch-results-container').innerHTML = `
            <div class="notice-message">
                <p>This full simulation did not generate any conversations.</p>
            </div>
        `;
        return;
    }
    
    let html = '<div class="batch-summary">';
    html += `<p><strong>Total Conversations:</strong> ${conversations.length}</p>`;
    
    // Calculate average duration
    const durations = conversations.filter(c => c.duration).map(c => c.duration);
    const avgDuration = durations.length > 0 
        ? (durations.reduce((a, b) => a + b, 0) / durations.length).toFixed(2) 
        : 'N/A';
    
    html += `<p><strong>Average Duration:</strong> ${avgDuration}s</p>`;
    html += '</div>';
    
    // Add table of results
    html += '<table class="results-table">';
    html += '<thead><tr><th>ID</th><th>Profile</th><th>Duration</th><th>Status</th></tr></thead>';
    html += '<tbody>';
    
    conversations.forEach(conversation => {
        html += `<tr>
            <td>${conversation.conversation_id || 'Unknown'}</td>
            <td>${conversation.profile || 'Default'}</td>
            <td>${conversation.duration ? conversation.duration.toFixed(2) + 's' : 'N/A'}</td>
            <td>${conversation.status || 'Unknown'}</td>
        </tr>`;
    });
    
    html += '</tbody></table>';
    
    document.getElementById('batch-results-container').innerHTML = html;
}

// View batch results
function viewBatchResults() {
    const batchBtn = document.getElementById('view-batch-btn');
    const batchId = batchBtn.dataset.batchId;
    
    if (batchId) {
        window.location.href = `/?batch=${batchId}`;
    }
}

// Start a one-shot conversation (generate entire conversation in one call)
function startOneshotConversation() {
    // Get form data BEFORE disabling the form
    const form = document.getElementById('conversation-form');
    const formData = new FormData(form);
    
    // Now disable form and show loading state
    setFormEnabled(false);
    updateOneshotStatus('Starting generation...', 'loading');
    
    // Clear previous conversation
    document.getElementById('oneshot-conversation-display').innerHTML = '';
    document.getElementById('oneshot-diagnosis-content').innerHTML = '<p>The diagnosis will appear after generation completes.</p>';
    
    // Get provider data from data attributes for agent model (one-shot mode only uses one model)
    const agentSelect = document.getElementById('agent-model-select');
    const agentOption = agentSelect.options[agentSelect.selectedIndex];
    const agentProvider = agentOption ? agentOption.dataset.provider || 'ollama' : 'ollama';
    
    const groqApiKey = document.getElementById('groq-api-key').value;
    
    // Validate Groq API key if needed
    if (agentProvider === 'groq' && !groqApiKey) {
        updateOneshotStatus('Error: Groq API key is required for Groq models', 'error');
        setFormEnabled(true);
        
        // Show error message in conversation display
        document.getElementById('oneshot-conversation-display').innerHTML = `
            <div class="error-message">
                <h3>Groq API Key Required</h3>
                <p>Please enter your Groq API key in the Advanced Options section.</p>
                <p>You can get a key from <a href="https://console.groq.com/" target="_blank">console.groq.com</a></p>
            </div>
        `;
        return;
    }
    
    // Debug log form entries
    console.log("Form entries for one-shot mode:");
    for (const pair of formData.entries()) {
        console.log(pair[0] + ': ' + pair[1]);
    }
    
    // Build JSON payload
    const payload = {
        questionnaire: formData.get('questionnaire'),
        profile: formData.get('profile'),
        agent_model: formData.get('agent_model'),  // For one-shot mode, we use the agent model
        save_logs: formData.get('save_logs') === 'on',
        refresh_cache: formData.get('refresh_cache') === 'on',
        full_conversation: true,  // This is what makes it one-shot
        disable_rag: formData.get('disable_rag') === 'on',
        disable_rag_evaluation: formData.get('disable_rag_evaluation') === 'on',
        // Add provider information
        assistant_provider: agentProvider,  // For one-shot, assistant_provider is used
        patient_provider: agentProvider,    // Use same provider for consistency
        groq_api_key: groqApiKey
    };
    
    console.log('One-shot payload:', payload);
    
    // Send the request
    fetch('/api/conversations/start', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(data => {
                throw new Error(data.error || `HTTP error! Status: ${response.status}`);
            });
        }
        return response.json();
    })
    .then(data => {
        conversationId = data.conversation_id;
        console.log(`One-shot conversation started with ID: ${conversationId}`);
        updateOneshotStatus('Generating full conversation...', 'loading');
        
        // Start polling for updates
        startOneshotPolling();
        
        // Enable stop button
        document.getElementById('stop-oneshot-btn').disabled = false;
    })
    .catch(error => {
        console.error('Error starting one-shot conversation:', error);
        updateOneshotStatus(`Error: ${error.message}`, 'error');
        setFormEnabled(true);
        
        // Show error message in conversation display
        document.getElementById('oneshot-conversation-display').innerHTML = `
            <div class="error-message">
                <h3>Error Starting Generation</h3>
                <p>${error.message}</p>
                <p>Please check your settings and try again.</p>
            </div>
        `;
    });
}

// Stop one-shot conversation
function stopOneshotConversation() {
    if (!conversationId) return;
    
    updateOneshotStatus('Stopping generation...', 'loading');
    
    fetch(`/api/conversations/${conversationId}/stop`, {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            updateOneshotStatus('Generation stopped', 'inactive');
            stopOneshotPolling();
            
            // Re-enable form
            setFormEnabled(true);
        } else {
            updateOneshotStatus(`Error: ${data.error || 'Failed to stop generation'}`, 'error');
        }
    })
    .catch(error => {
        console.error('Error stopping one-shot generation:', error);
        updateOneshotStatus('Error stopping generation', 'error');
    });
}

// Start polling for one-shot conversation updates
function startOneshotPolling() {
    if (conversationInterval) {
        clearInterval(conversationInterval);
    }
    
    lastMessageCount = 0;
    
    conversationInterval = setInterval(() => {
        if (!conversationId) return;
        
        fetch(`/api/conversations/${conversationId}/status`)
            .then(response => response.json())
            .then(data => {
                if (data.status === 'completed') {
                    // Conversation is complete
                    updateOneshotConversation(data.conversation);
                    updateOneshotDiagnosis(data.diagnosis);
                    updateOneshotStatus('Generation completed', 'completed');
                    stopOneshotPolling();
                    
                    // Enable view log button if log is saved
                    if (data.log_saved) {
                        document.getElementById('view-oneshot-log-btn').disabled = false;
                        document.getElementById('view-oneshot-log-btn').dataset.logId = data.log_id;
                    }
                    
                    // Re-enable form
                    setFormEnabled(true);
                } else if (data.status === 'error') {
                    // Conversation had an error
                    updateOneshotStatus(`Error: ${data.error || 'An error occurred'}`, 'error');
                    stopOneshotPolling();
                    setFormEnabled(true);
                } else if (data.status === 'in_progress') {
                    // Update status to show progress
                    updateOneshotStatus('Generation in progress...', 'active');
                }
            })
            .catch(error => {
                console.error('Error polling one-shot status:', error);
            });
    }, 1000);  // Poll every second
}

// Stop polling for one-shot updates
function stopOneshotPolling() {
    if (conversationInterval) {
        clearInterval(conversationInterval);
        conversationInterval = null;
    }
    
    // Disable stop button
    document.getElementById('stop-oneshot-btn').disabled = true;
}

// Update the one-shot conversation display with messages
function updateOneshotConversation(conversation) {
    if (!conversation || !conversation.length) return;
    
    const display = document.getElementById('oneshot-conversation-display');
    
    // Clear placeholder if it exists
    const placeholder = display.querySelector('.conversation-placeholder');
    if (placeholder) {
        display.innerHTML = '';
    } else if (display.querySelector('.loading-indicator')) {
        display.innerHTML = '';
    }
    
    // Add all messages at once
    conversation.forEach(message => {
        // Skip system messages
        if (message.role === 'system') return;
        
        // Create message element
        const messageDiv = document.createElement('div');
        messageDiv.className = `message message-${message.role} clearfix`;
        
        const roleLabel = document.createElement('div');
        roleLabel.className = 'role-label';
        roleLabel.textContent = message.role.charAt(0).toUpperCase() + message.role.slice(1);
        
        const contentDiv = document.createElement('div');
        contentDiv.className = `message-content ${message.role}-bubble`;
        contentDiv.innerHTML = formatText(message.content);
        
        messageDiv.appendChild(roleLabel);
        messageDiv.appendChild(contentDiv);
        display.appendChild(messageDiv);
    });
    
    // Scroll to bottom
    display.scrollTop = display.scrollHeight;
}

// Update one-shot diagnosis display
function updateOneshotDiagnosis(diagnosis) {
    if (!diagnosis) return;
    
    const diagnosisContent = document.getElementById('oneshot-diagnosis-content');
    diagnosisContent.innerHTML = formatText(diagnosis);
}

// Update one-shot status message
function updateOneshotStatus(message, state) {
    const status = document.getElementById('oneshot-status');
    status.textContent = message;
    
    // Remove all state classes
    status.classList.remove('loading', 'active', 'completed', 'error', 'inactive');
    
    // Add appropriate state class
    if (state) {
        status.classList.add(state);
    }
}

// View saved one-shot conversation log
function viewOneshotLog() {
    const viewLogBtn = document.getElementById('view-oneshot-log-btn');
    const logId = viewLogBtn.dataset.logId;
    
    if (logId) {
        window.location.href = `/?log=${logId}`;
    }
}

// Function to check if any of the selected models uses Groq provider
function checkForGroqModels() {
    const assistantModel = document.getElementById('assistant-model-select').value;
    const patientModel = document.getElementById('patient-model-select').value;
    const agentModel = document.getElementById('agent-model-select').value;
    
    const assistantModelData = models.find(model => model.name === assistantModel);
    const patientModelData = models.find(model => model.name === patientModel);
    const agentModelData = models.find(model => model.name === agentModel);
    
    const usesGroq = (assistantModelData && assistantModelData.provider === "groq") || 
                    (patientModelData && patientModelData.provider === "groq") ||
                    (agentModelData && agentModelData.provider === "groq");
    
    const groqKeySection = document.getElementById('groq-api-section');
    const groqNoteElement = document.getElementById('groq_api_key_env_note');
    
    if (usesGroq) {
        groqKeySection.style.display = 'block';
        
        // Check if the key is available from environment variable
        if (window.providersData && window.providersData.groq && window.providersData.groq.available) {
            document.getElementById('groq-api-key').disabled = true;
            document.getElementById('groq-api-key').placeholder = "API Key from environment variable";
            
            // Create or show the note element
            if (!groqNoteElement) {
                const noteElement = document.createElement('div');
                noteElement.id = 'groq_api_key_env_note';
                noteElement.className = 'text-info small mt-1';
                noteElement.textContent = "Using Groq API key from environment variable";
                groqKeySection.appendChild(noteElement);
            } else {
                groqNoteElement.style.display = 'block';
            }
        } else {
            document.getElementById('groq-api-key').disabled = false;
            document.getElementById('groq-api-key').placeholder = "Enter your Groq API key";
            
            // Hide the note if it exists
            if (groqNoteElement) {
                groqNoteElement.style.display = 'none';
            }
        }
    } else {
        groqKeySection.style.display = 'none';
        
        // Hide the note if it exists
        if (groqNoteElement) {
            groqNoteElement.style.display = 'none';
        }
    }
}
