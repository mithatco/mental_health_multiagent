// Global variables
let chatId = null;
let isConversationActive = false;
let initialMessagePolled = false;

// Document ready function
document.addEventListener('DOMContentLoaded', function() {
    // Initialize the form with available options
    initializeForm();
    
    // Add event listener for form submission
    document.getElementById('chat-form').addEventListener('submit', function(e) {
        e.preventDefault();
        startChat();
    });
    
    // Add event listener for send button
    document.getElementById('send-btn').addEventListener('click', sendUserMessage);
    
    // Add event listener for user input - allow Enter to send
    document.getElementById('user-input').addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendUserMessage();
        }
    });
    
    // Add event listener for stop button
    document.getElementById('stop-btn').addEventListener('click', endChat);
    
    // Add event listener for view log button
    document.getElementById('view-log-btn').addEventListener('click', viewChatLog);
});

// Initialize form with available questionnaires and models
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
            
            if (data.questionnaires && data.questionnaires.length) {
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
                        <p>Please add questionnaire PDFs to the documents/questionnaires directory.</p>
                    </div>
                `;
            }
        })
        .catch(error => {
            console.error('Error fetching questionnaires:', error);
            const select = document.getElementById('questionnaire-select');
            select.innerHTML = '<option value="" disabled selected>Error loading questionnaires</option>';
        });
    
    // Fetch available models
    fetch('/api/models')
        .then(response => response.json())
        .then(data => {
            const assistantSelect = document.getElementById('assistant-model-select');
            assistantSelect.innerHTML = '';
            
            if (data.models && data.models.length) {
                // Create option groups for each provider
                const ollamaGroup = document.createElement('optgroup');
                ollamaGroup.label = 'Ollama (Local)';
                
                const groqGroup = document.createElement('optgroup');
                groqGroup.label = 'Groq (Cloud API)';
                
                // Add models to appropriate groups
                data.models.forEach(model => {
                    const provider = model.provider;
                    const name = model.name;
                    const displayName = model.display_name || name;
                    
                    const assistantOption = document.createElement('option');
                    assistantOption.value = name;
                    assistantOption.textContent = displayName;
                    assistantOption.dataset.provider = provider;
                    
                    // Default to qwen3:4b if available
                    if (name === 'qwen3:4b') {
                        assistantOption.selected = true;
                    }
                    
                    // Add to appropriate group
                    if (provider === 'ollama') {
                        ollamaGroup.appendChild(assistantOption);
                    } else if (provider === 'groq') {
                        groqGroup.appendChild(assistantOption);
                    }
                });
                
                // Add option groups to select if they have children
                if (ollamaGroup.children.length > 0) {
                    assistantSelect.appendChild(ollamaGroup);
                }
                
                if (groqGroup.children.length > 0) {
                    assistantSelect.appendChild(groqGroup);
                }
            } else {
                const option = document.createElement('option');
                option.value = '';
                option.textContent = 'No models available';
                option.disabled = true;
                assistantSelect.appendChild(option);
            }
        })
        .catch(error => {
            console.error('Error fetching models:', error);
            const assistantSelect = document.getElementById('assistant-model-select');
            assistantSelect.innerHTML = '<option value="" disabled selected>Error loading models</option>';
        });
}

// Start a new chat
function startChat() {
    if (isConversationActive) return;
    
    // Get form data BEFORE disabling the form
    const form = document.getElementById('chat-form');
    const questionnaireSelect = document.getElementById('questionnaire-select');
    const data = {
        questionnaire: questionnaireSelect.value,
        assistant_model: document.getElementById('assistant-model-select').value,
        save_logs: document.getElementById('save-logs').checked,
        refresh_cache: document.getElementById('refresh-cache').checked,
        use_rag: document.getElementById('use-rag').checked,
        chat_mode: true  // Signal that this is a chat where user is the patient
    };
    
    // Now disable form and show loading state
    setFormEnabled(false);
    updateStatus('Starting chat...', 'loading');
    
    // Clear previous conversation
    document.getElementById('conversation-display').innerHTML = '';
    document.getElementById('diagnosis-panel').style.display = 'none';
    
    // Reset the initialMessagePolled flag
    initialMessagePolled = false;
    
    // Debug logging
    console.log('Questionnaire select element:', questionnaireSelect);
    console.log('Selected index:', questionnaireSelect.selectedIndex);
    console.log('Selected value:', questionnaireSelect.value);
    console.log('All options:', Array.from(questionnaireSelect.options).map(opt => ({value: opt.value, text: opt.text, selected: opt.selected})));
    console.log('Form data being sent:', data);
    
    // Verify we have a questionnaire selected
    if (!data.questionnaire) {
        console.error('No questionnaire selected - questionnaire value is:', data.questionnaire);
        updateStatus('Error: No questionnaire selected', 'error');
        setFormEnabled(true);
        return;
    }
    
    // Send request to start chat
    fetch('/api/chat/start', {
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
        if (data.chat_id) {
            chatId = data.chat_id;
            isConversationActive = true;
            updateStatus('Chat started', 'active');
            
            // Enable chat input
            document.getElementById('chat-input-container').style.display = 'flex';
            
            // Enable stop button
            document.getElementById('stop-btn').disabled = false;
            
            // If there's an initial message, display it
            if (data.initial_message) {
                addMessage('assistant', data.initial_message);
                // Immediately enable input for user response
                document.getElementById('user-input').disabled = false;
                document.getElementById('send-btn').disabled = false;
                document.getElementById('user-input').focus();
            } else {
                // Poll for the first message
                console.log("No initial message found, polling for first message...");
                // Disable input until we get the first message
                document.getElementById('user-input').disabled = true;
                document.getElementById('send-btn').disabled = true;
                getNextAssistantMessage();
            }
        } else {
            updateStatus(`Error: ${data.error || 'Failed to start chat'}`, 'error');
            setFormEnabled(true);
        }
    })
    .catch(error => {
        console.error('Error starting chat:', error);
        updateStatus('Error starting chat: ' + error.message, 'error');
        setFormEnabled(true);
    });
}

// Send user message to the assistant
function sendUserMessage() {
    if (!isConversationActive || !chatId) return;
    
    const userInput = document.getElementById('user-input');
    const userMessage = userInput.value.trim();
    
    if (!userMessage) return;
    
    // Add user message to the conversation
    addMessage('patient', userMessage);
    
    // Clear input field
    userInput.value = '';
    
    // Disable input while waiting for response
    userInput.disabled = true;
    document.getElementById('send-btn').disabled = true;
    
    // Log the message being sent
    console.log(`Sending message to chat ${chatId}: "${userMessage}"`);
    
    // Create message data as an object
    const messageData = {
        message: userMessage
    };
    
    // Send message to the server
    fetch(`/api/chat/${chatId}/message`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        },
        body: JSON.stringify(messageData)
    })
    .then(response => {
        // Check for HTTP errors
        if (!response.ok) {
            return response.text().then(text => {
                // Try to parse as JSON if possible
                try {
                    const errorData = JSON.parse(text);
                    throw new Error(`Server error (${response.status}): ${errorData.error || 'Unknown error'}`);
                } catch (parseError) {
                    // If not JSON, use text directly
                    throw new Error(`Server error (${response.status}): ${text || 'Unknown error'}`);
                }
            });
        }
        return response.json();
    })
    .then(data => {
        console.log("Message sent successfully:", data);
        
        if (data.error) {
            updateStatus(`Error: ${data.error}`, 'error');
            // Re-enable input
            userInput.disabled = false;
            document.getElementById('send-btn').disabled = false;
            return;
        }
        
        // Get the assistant's response
        getNextAssistantMessage();
    })
    .catch(error => {
        console.error('Error sending message:', error);
        updateStatus('Error sending message: ' + error.message, 'error');
        
        // Re-enable input
        userInput.disabled = false;
        document.getElementById('send-btn').disabled = false;
    });
}

// Get the next message from the assistant
function getNextAssistantMessage() {
    if (!isConversationActive || !chatId) return;
    
    console.log("Polling for assistant message...");
    
    fetch(`/api/chat/${chatId}/response`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log("Response data:", data);
            
            if (data.status === 'error') {
                // Show error message in conversation display
                const errorDiv = document.createElement('div');
                errorDiv.className = 'error-message';
                errorDiv.innerHTML = `<h3>Error</h3><p>${data.error}</p>`;
                
                if (data.output) {
                    // Add a collapsible section for the technical details
                    const detailsButton = document.createElement('button');
                    detailsButton.innerText = 'Show technical details';
                    detailsButton.className = 'details-button';
                    detailsButton.onclick = function() {
                        const details = document.getElementById('error-details');
                        if (details.style.display === 'none') {
                            details.style.display = 'block';
                            this.innerText = 'Hide technical details';
                        } else {
                            details.style.display = 'none';
                            this.innerText = 'Show technical details';
                        }
                    };
                    
                    const detailsContent = document.createElement('pre');
                    detailsContent.id = 'error-details';
                    detailsContent.style.display = 'none';
                    detailsContent.className = 'error-details';
                    detailsContent.innerText = data.output;
                    
                    errorDiv.appendChild(detailsButton);
                    errorDiv.appendChild(detailsContent);
                }
                
                document.getElementById('conversation-display').appendChild(errorDiv);
                
                // Update status
                updateStatus(`Error: ${data.error.split(':')[0]}`, 'error');
                
                // End conversation
                isConversationActive = false;
                
                // Re-enable form
                setFormEnabled(true);
                
                return;
            }
            
            if (data.status === 'completed') {
                // Conversation is complete - show diagnosis
                if (data.message) {
                    addMessage('assistant', data.message);
                }
                
                if (data.diagnosis) {
                    showDiagnosis(data.diagnosis);
                }
                
                // End the chat
                isConversationActive = false;
                updateStatus('Chat completed', 'completed');
                
                // Disable input
                document.getElementById('user-input').disabled = true;
                document.getElementById('send-btn').disabled = true;
                
                // Enable view log button if log is saved
                if (data.log_id) {
                    document.getElementById('view-log-btn').disabled = false;
                    document.getElementById('view-log-btn').dataset.logId = data.log_id;
                }
                
                // Re-enable form
                setFormEnabled(true);
                
                return;
            } else if (data.status === 'waiting_for_user') {
                // Show assistant message if available
                if (data.message) {
                    addMessage('assistant', data.message);
                    initialMessagePolled = true;
                }
                
                // Re-enable input for user response
                document.getElementById('user-input').disabled = false;
                document.getElementById('send-btn').disabled = false;
                document.getElementById('user-input').focus();
            } else if (data.status === 'thinking' || data.status === 'in_progress' || data.status === 'starting') {
                // Assistant is still thinking or initializing, poll again after a delay
                console.log("Assistant is still thinking, polling again...");
                setTimeout(getNextAssistantMessage, 1000);
            } else if (data.status === 'error') {
                updateStatus(`Error: ${data.error || 'An error occurred'}`, 'error');
                
                // Re-enable input
                document.getElementById('user-input').disabled = false;
                document.getElementById('send-btn').disabled = false;
            }
        })
        .catch(error => {
            console.error('Error getting assistant response:', error);
            updateStatus('Error getting response', 'error');
            
            // Display error in conversation
            const errorDiv = document.createElement('div');
            errorDiv.className = 'error-message';
            errorDiv.innerHTML = `<h3>Connection Error</h3><p>${error.message}</p>`;
            document.getElementById('conversation-display').appendChild(errorDiv);
            
            // End conversation
            isConversationActive = false;
            
            // Re-enable form
            setFormEnabled(true);
        });
}

// Add a message to the conversation display
function addMessage(role, content) {
    const display = document.getElementById('conversation-display');
    
    // Clear placeholder if it exists
    const placeholder = display.querySelector('.conversation-placeholder');
    if (placeholder) {
        display.innerHTML = '';
    }
    
    // Create message element
    const messageDiv = document.createElement('div');
    messageDiv.className = `message message-${role} clearfix`;
    
    const roleLabel = document.createElement('div');
    roleLabel.className = 'role-label';
    roleLabel.textContent = role.charAt(0).toUpperCase() + role.slice(1);
    
    const contentDiv = document.createElement('div');
    contentDiv.className = `message-content ${role}-bubble`;
    contentDiv.innerHTML = formatText(content);
    
    messageDiv.appendChild(roleLabel);
    messageDiv.appendChild(contentDiv);
    display.appendChild(messageDiv);
    
    // Scroll to bottom
    display.scrollTop = display.scrollHeight;
}

// Format text with line breaks and markdown-like formatting
function formatText(text) {
    if (!text) return '';
    
    // Replace line breaks with <br>
    text = text.replace(/\n/g, '<br>');
    
    // Format bold text
    text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Format medical terms, symptoms, and quotes
    text = text.replace(/<med>(.*?)<\/med>/g, '<span class="medical-term">$1</span>');
    text = text.replace(/<sym>(.*?)<\/sym>/g, '<span class="symptom">$1</span>');
    text = text.replace(/<quote>(.*?)<\/quote>/g, '<span class="patient-quote">$1</span>');
    
    return text;
}

// Show the diagnosis panel
function showDiagnosis(diagnosis) {
    const diagnosisPanel = document.getElementById('diagnosis-panel');
    const diagnosisContent = document.getElementById('diagnosis-content');
    
    diagnosisContent.innerHTML = formatText(diagnosis);
    diagnosisPanel.style.display = 'block';
}

// End the chat
function endChat() {
    if (!isConversationActive || !chatId) return;
    
    updateStatus('Ending chat...', 'loading');
    
    fetch(`/api/chat/${chatId}/end`, {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        isConversationActive = false;
        
        if (data.success) {
            updateStatus('Chat ended', 'inactive');
            
            // Disable input
            document.getElementById('user-input').disabled = true;
            document.getElementById('send-btn').disabled = true;
            
            // Show diagnosis if available
            if (data.diagnosis) {
                showDiagnosis(data.diagnosis);
            }
            
            // Enable view log button if log is saved
            if (data.log_id) {
                document.getElementById('view-log-btn').disabled = false;
                document.getElementById('view-log-btn').dataset.logId = data.log_id;
            }
            
            // Re-enable form
            setFormEnabled(true);
        } else {
            updateStatus(`Error: ${data.error || 'Failed to end chat'}`, 'error');
        }
    })
    .catch(error => {
        console.error('Error ending chat:', error);
        updateStatus('Error ending chat', 'error');
    });
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
    const form = document.getElementById('chat-form');
    const elements = form.elements;
    
    for (let i = 0; i < elements.length; i++) {
        elements[i].disabled = !enabled;
    }
    
    // Update start button text
    const startBtn = document.getElementById('start-btn');
    startBtn.textContent = enabled ? 'Start Chat' : 'Chat in Progress...';
}

// View saved chat log
function viewChatLog() {
    const viewLogBtn = document.getElementById('view-log-btn');
    const logId = viewLogBtn.dataset.logId;
    
    if (logId) {
        window.location.href = `/?log=${logId}`;
    }
} 