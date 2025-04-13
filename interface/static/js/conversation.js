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
                data.questionnaires.forEach(questionnaire => {
                    const option = document.createElement('option');
                    option.value = questionnaire.id;
                    option.textContent = `${questionnaire.name} (${questionnaire.question_count} questions)`;
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
                // Add models to selects
                data.models.forEach(model => {
                    // Assistant select
                    const assistantOption = document.createElement('option');
                    assistantOption.value = model;
                    assistantOption.textContent = model;
                    // Default to qwen2.5:3b if available
                    if (model === 'qwen2.5:3b') {
                        assistantOption.selected = true;
                    }
                    assistantSelect.appendChild(assistantOption);
                    
                    // Patient select
                    const patientOption = document.createElement('option');
                    patientOption.value = model;
                    patientOption.textContent = model;
                    // Default to qwen2.5:3b if available
                    if (model === 'qwen2.5:3b') {
                        patientOption.selected = true;
                    }
                    patientSelect.appendChild(patientOption);
                    
                    // Agent select for one-shot mode
                    const agentOption = document.createElement('option');
                    agentOption.value = model;
                    agentOption.textContent = model;
                    // Default to qwen2.5:3b if available
                    if (model === 'qwen2.5:3b') {
                        agentOption.selected = true;
                    }
                    agentSelect.appendChild(agentOption);
                });
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
    // Disable form and show loading state
    setFormEnabled(false);
    updateStatus('Starting conversation...', 'loading');
    
    // Clear previous conversation
    document.getElementById('conversation-display').innerHTML = '';
    document.getElementById('diagnosis-content').innerHTML = '<p>The diagnosis will appear after the conversation completes.</p>';
    
    // Get form data
    const form = document.getElementById('conversation-form');
    const formData = new FormData(form);
    
    // Debug log form entries
    console.log("Form entries:");
    for (let entry of formData.entries()) {
        console.log(entry[0] + ": " + entry[1]);
    }
    
    // Convert FormData to JSON object - improved implementation
    const data = {};
    for (let [key, value] of formData.entries()) {
        console.log(`Processing form field: ${key} = ${value}`);
        
        if (key === 'save_logs' || key === 'refresh_cache') {
            data[key] = value === 'on';  // Convert checkbox to boolean
        } else {
            data[key] = value;
        }
    }
    
    // Debug log the final data object
    console.log("Data object to be sent:", data);
    const jsonData = JSON.stringify(data);
    console.log("JSON string to be sent:", jsonData);
    
    // Add fallback for empty data (this shouldn't normally happen)
    if (Object.keys(data).length === 0) {
        console.error("Form data is empty! Using fallback values.");
        
        // Get values directly from form elements
        const questionnaire = document.getElementById('questionnaire-select').value;
        const profile = document.getElementById('profile-select').value;
        const assistantModel = document.getElementById('assistant-model-select').value;
        const patientModel = document.getElementById('patient-model-select').value;
        const saveLogs = document.getElementById('save-logs').checked;
        const refreshCache = document.getElementById('refresh-cache').checked;
        
        // Create data object manually
        data.questionnaire = questionnaire;
        data.profile = profile;
        data.assistant_model = assistantModel;
        data.patient_model = patientModel;
        data.save_logs = saveLogs;
        data.refresh_cache = refreshCache;
        
        console.log("Created fallback data:", data);
    }
    
    // Verify we have a questionnaire selected
    if (!data.questionnaire) {
        updateStatus('Error: No questionnaire selected', 'error');
        setFormEnabled(true);
        return;
    }
    
    // Send request to start conversation
    fetch('/api/conversations/start', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        },
        credentials: 'same-origin',
        body: JSON.stringify(data),
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.conversation_id) {
            conversationId = data.conversation_id;
            updateStatus('Conversation started', 'active');
            
            // Enable stop button
            document.getElementById('stop-btn').disabled = false;
            
            // Start polling for updates
            startPolling();
        } else {
            updateStatus(`Error: ${data.error || 'Failed to start conversation'}`, 'error');
            setFormEnabled(true);
        }
    })
    .catch(error => {
        console.error('Error starting conversation:', error);
        updateStatus('Error starting conversation: ' + error.message, 'error');
        setFormEnabled(true);
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
                if (data.status === 'completed') {
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
                    // Conversation had an error
                    updateStatus(`Error: ${data.error || 'An error occurred'}`, 'error');
                    stopPolling();
                    setFormEnabled(true);
                } else if (data.status === 'in_progress') {
                    // Update conversation with new messages
                    updateConversation(data.conversation);
                    updateStatus('Conversation in progress...', 'active');
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
    if (!conversation || !conversation.length) return;
    
    const display = document.getElementById('conversation-display');
    
    // Only update if we have new messages
    if (conversation.length <= lastMessageCount) return;
    
    // Clear placeholder if it exists
    const placeholder = display.querySelector('.conversation-placeholder');
    if (placeholder) {
        display.innerHTML = '';
    }
    
    // Add new messages
    for (let i = lastMessageCount; i < conversation.length; i++) {
        const message = conversation[i];
        
        // Skip system messages
        if (message.role === 'system') continue;
        
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
    }
    
    // Update last message count
    lastMessageCount = conversation.length;
    
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
    // First check if a questionnaire is selected - do this explicitly before anything else
    const questionnaire = document.getElementById('questionnaire-select').value;
    if (!questionnaire) {
        updateBatchStatus('Error: No questionnaire selected', 'error');
        return;
    }
    
    // Check batch count before proceeding
    const batchCountInput = document.getElementById('batch-count');
    const batchCount = parseInt(batchCountInput.value, 10);
    if (isNaN(batchCount) || batchCount < 1) {
        updateBatchStatus('Error: Invalid batch count', 'error');
        return;
    }
    
    console.log(`Starting full simulation with questionnaire: ${questionnaire} and count: ${batchCount}`);
    
    // Disable form and show loading state
    setFormEnabled(false);
    updateBatchStatus('Starting full simulation...', 'loading');
    
    // Clear previous batch results
    document.getElementById('batch-results-container').innerHTML = 
        '<p>Processing full simulation...</p>';
    document.getElementById('batch-progress-bar').style.width = '0%';
    document.getElementById('batch-progress-text').textContent = '0 / 0 conversations completed';
    
    // Manually construct the data payload instead of using FormData
    const data = {
        questionnaire: questionnaire,
        profile: document.getElementById('profile-select').value,
        assistant_model: document.getElementById('assistant-model-select').value,
        patient_model: document.getElementById('patient-model-select').value,
        save_logs: document.getElementById('save-logs').checked,
        refresh_cache: document.getElementById('refresh-cache').checked,
        randomize_profiles: document.getElementById('randomize-profiles')?.checked || false,
        batch_count: batchCount // Explicitly use the parsed integer
    };
    
    // Log the data we're about to send
    console.log("Batch data to be sent:", data);
    console.log("JSON string to be sent:", JSON.stringify(data));
    
    // Send request to start batch generation
    fetch('/api/batches/start', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        },
        credentials: 'same-origin',
        body: JSON.stringify(data),
    })
    .then(response => {
        if (!response.ok) {
            // For better debugging, try to get the error message from the response
            return response.text().then(text => {
                console.error(`Server returned ${response.status}: ${text}`);
                throw new Error(`HTTP error! Status: ${response.status}. Details: ${text}`);
            });
        }
        return response.json();
    })
    .then(data => {
        if (data.batch_id) {
            batchId = data.batch_id;
            updateBatchStatus(`Full simulation started (${batchCount} conversations)`, 'active');
            
            // Enable stop button
            document.getElementById('stop-batch-btn').disabled = false;
            
            // Start polling for updates
            startBatchPolling(data.batch_id, data.total_conversations);
        } else {
            updateBatchStatus(`Error: ${data.error || 'Failed to start full simulation'}`, 'error');
            setFormEnabled(true);
        }
    })
    .catch(error => {
        console.error('Error starting full simulation:', error);
        updateBatchStatus('Error starting full simulation: ' + error.message, 'error');
        setFormEnabled(true);
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

// Start a one-shot conversation
function startOneshotConversation() {
    // First check if a questionnaire is selected
    const questionnaire = document.getElementById('questionnaire-select').value;
    if (!questionnaire) {
        updateOneshotStatus('Error: No questionnaire selected', 'error');
        return;
    }
    
    // Check if agent model is selected
    const agentModel = document.getElementById('agent-model-select').value;
    if (!agentModel) {
        updateOneshotStatus('Error: No agent model selected', 'error');
        return;
    }
    
    // Disable form and show loading state
    setFormEnabled(false);
    updateOneshotStatus('Generating one-shot conversation...', 'loading');
    
    // Clear previous conversation
    document.getElementById('oneshot-conversation-display').innerHTML = 
        '<div class="loading-indicator"><div class="spinner"></div>Generating full conversation...</div>';
    document.getElementById('oneshot-diagnosis-content').innerHTML = 
        '<p>The diagnosis will appear here after generation completes.</p>';
    
    // Create data object with only the necessary parameters
    const data = {
        questionnaire: questionnaire,
        profile: document.getElementById('profile-select').value,
        agent_model: agentModel,
        save_logs: document.getElementById('save-logs').checked,
        refresh_cache: document.getElementById('refresh-cache').checked,
        full_conversation: true // Signal that this is a full conversation
    };
    
    console.log("One-shot data to be sent:", data);
    
    // Send request to start conversation
    fetch('/api/conversations/start', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        },
        credentials: 'same-origin',
        body: JSON.stringify(data)
    })
    .then(response => {
        if (!response.ok) {
            return response.text().then(text => {
                console.error(`Server returned ${response.status}: ${text}`);
                throw new Error(`HTTP error! Status: ${response.status}. Details: ${text}`);
            });
        }
        return response.json();
    })
    .then(data => {
        if (data.conversation_id) {
            conversationId = data.conversation_id;
            updateOneshotStatus('Generating conversation...', 'active');
            
            // Enable stop button
            document.getElementById('stop-oneshot-btn').disabled = false;
            
            // Start polling for conversation updates
            startOneshotPolling();
        } else {
            updateOneshotStatus(`Error: ${data.error || 'Failed to start generation'}`, 'error');
            setFormEnabled(true);
        }
    })
    .catch(error => {
        console.error('Error starting one-shot generation:', error);
        updateOneshotStatus('Error starting generation: ' + error.message, 'error');
        setFormEnabled(true);
        
        // Display error in conversation panel
        document.getElementById('oneshot-conversation-display').innerHTML = `
            <div class="error-message">
                <h3>Error Starting Generation</h3>
                <p>${error.message}</p>
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
