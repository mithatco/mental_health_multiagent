// Global variables
let currentChatId = null;
let logData = [];

// Event listeners
document.addEventListener('DOMContentLoaded', function() {
    // Load logs when page loads
    refreshLogs();
    
    // Add event listeners for filters
    document.getElementById('profile-filter').addEventListener('change', applyFilters);
    document.getElementById('date-filter').addEventListener('change', applyFilters);
    document.getElementById('search-filter').addEventListener('input', applyFilters);
    
    // Add event listener for refresh button
    document.getElementById('refresh-btn').addEventListener('click', refreshLogs);
    
    // Add event listener for export button
    document.getElementById('export-btn').addEventListener('click', exportChat);
});

// Refresh logs
function refreshLogs() {
    // Show loading state
    const listElement = document.getElementById('log-list');
    listElement.innerHTML = '<div class="loading"><div class="spinner"></div><p>Loading logs...</p></div>';
    
    // Fetch logs from server
    fetch('/api/logs?refresh=true')
        .then(response => response.json())
        .then(data => {
            // Store log data globally
            logData = data.logs;
            
            // Update profile filter options
            updateProfileOptions(data.profiles);
            
            // Apply any active filters
            applyFilters();
        })
        .catch(error => {
            console.error('Error loading logs:', error);
            listElement.innerHTML = '<div class="no-logs">Failed to load logs</div>';
        });
}

// Update profile filter options
function updateProfileOptions(profiles) {
    const profileSelect = document.getElementById('profile-filter');
    const currentValue = profileSelect.value;
    
    // Clear all options except "All"
    while (profileSelect.options.length > 1) {
        profileSelect.remove(1);
    }
    
    // Add profile options
    profiles.forEach(profile => {
        const option = document.createElement('option');
        option.value = profile;
        option.textContent = profile;
        profileSelect.appendChild(option);
    });
    
    // Restore previous selection if possible
    if (profiles.includes(currentValue)) {
        profileSelect.value = currentValue;
    }
}

// Apply filters to log list
function applyFilters() {
    if (!logData || !logData.length) return;
    
    const listElement = document.getElementById('log-list');
    const profileFilter = document.getElementById('profile-filter').value;
    const dateFilter = document.getElementById('date-filter').value;
    const searchFilter = document.getElementById('search-filter').value.toLowerCase();
    
    // Clear existing items
    listElement.innerHTML = '';
    
    // Prepare filter date objects
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    
    const sevenDaysAgo = new Date(today);
    sevenDaysAgo.setDate(today.getDate() - 7);
    
    const thirtyDaysAgo = new Date(today);
    thirtyDaysAgo.setDate(today.getDate() - 30);
    
    // Filter logs
    const filteredLogs = logData.filter(log => {
        // Profile filter
        if (profileFilter !== 'All' && log.profile !== profileFilter) {
            return false;
        }
        
        // Date filter
        if (dateFilter !== 'All time') {
            const logDate = new Date(log.timestamp);
            
            if (dateFilter === 'Today') {
                const logDay = new Date(logDate);
                logDay.setHours(0, 0, 0, 0);
                if (logDay.getTime() !== today.getTime()) {
                    return false;
                }
            } else if (dateFilter === 'Last 7 days' && logDate < sevenDaysAgo) {
                return false;
            } else if (dateFilter === 'Last 30 days' && logDate < thirtyDaysAgo) {
                return false;
            }
        }
        
        // Search filter
        if (searchFilter) {
            const searchableText = `${log.filename} ${log.profile} ${log.questionnaire}`.toLowerCase();
            if (!searchableText.includes(searchFilter)) {
                return false;
            }
        }
        
        return true;
    });
    
    // Add filtered logs to list
    if (filteredLogs.length === 0) {
        listElement.innerHTML = '<div class="no-logs">No logs match the filters</div>';
    } else {
        filteredLogs.forEach(log => {
            const logItem = document.createElement('div');
            logItem.className = 'log-item';
            logItem.dataset.id = log.id;
            logItem.textContent = `${log.formatted_date} - ${log.profile} - ${log.questionnaire}`;
            logItem.addEventListener('click', () => loadChat(log.id));
            listElement.appendChild(logItem);
        });
    }
}

// Load chat details
function loadChat(chatId) {
    currentChatId = chatId;
    
    // Update selected item
    document.querySelectorAll('.log-item').forEach(item => {
        item.classList.remove('selected');
        if (item.dataset.id === chatId) {
            item.classList.add('selected');
        }
    });
    
    // Show loading state
    document.getElementById('chat-container').innerHTML = '<div class="loading"><div class="spinner"></div><p>Loading conversation...</p></div>';
    document.getElementById('metadata-display').innerHTML = '';
    document.getElementById('diagnosis-content').innerHTML = 'Loading...';
    
    // Fetch chat data
    fetch(`/api/logs/${chatId}`)
        .then(response => response.json())
        .then(data => {
            // Update header
            document.querySelector('.metadata .header').textContent = `Conversation: ${data.questionnaire}`;
            
            // Update metadata
            updateMetadata(data);
            
            // Update chat messages
            displayChat(data.conversation);
            
            // Update diagnosis
            document.getElementById('diagnosis-content').innerHTML = formatText(data.diagnosis || 'No diagnosis available');
            
            // Update RAG summary
            displayRagSummary(data.rag_summary);
        })
        .catch(error => {
            console.error('Error loading chat:', error);
            document.getElementById('chat-container').innerHTML = '<div class="no-logs">Failed to load conversation</div>';
        });
}

// Update metadata display
function updateMetadata(data) {
    const metadataDisplay = document.getElementById('metadata-display');
    metadataDisplay.innerHTML = '';
    
    // Format date
    let dateStr = 'Unknown';
    try {
        dateStr = new Date(data.timestamp).toLocaleString();
    } catch (e) {}
    
    // Create metadata fields
    const fields = [
        { label: 'Date', value: dateStr },
        { label: 'Questionnaire', value: data.questionnaire || 'Unknown' },
        { label: 'Patient Profile', value: data.metadata?.patient_profile || 'Unknown' },
        { label: 'Assistant Model', value: data.metadata?.assistant_model || 'Unknown' },
        { label: 'Patient Model', value: data.metadata?.patient_model || 'Unknown' },
        { label: 'Question Count', value: data.metadata?.question_count || 'Unknown' }
    ];
    
    fields.forEach(field => {
        const item = document.createElement('div');
        item.className = 'metadata-item';
        item.innerHTML = `<strong>${field.label}:</strong> ${field.value}`;
        metadataDisplay.appendChild(item);
    });
}

// Display chat messages
function displayChat(conversation) {
    const chatContainer = document.getElementById('chat-container');
    chatContainer.innerHTML = '';
    
    if (!conversation || conversation.length === 0) {
        chatContainer.innerHTML = '<div class="no-logs">No messages in this conversation</div>';
        return;
    }
    
    conversation.forEach(message => {
        const role = message.role;
        const content = message.content;
        
        // Skip system messages
        if (role === 'system' || !content.trim()) {
            return;
        }
        
        // Create message elements
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
        chatContainer.appendChild(messageDiv);
    });
    
    // Scroll to top
    chatContainer.scrollTop = 0;
}

// Format text with line breaks
function formatText(text) {
    if (!text) return '';
    
    // Replace newlines with <br> tags
    return text.split('\n').join('<br>');
}

// Export current chat as text
function exportChat() {
    if (!currentChatId) {
        alert('Please select a chat to export first');
        return;
    }
    
    window.open(`/api/logs/${currentChatId}/export`, '_blank');
}

// Add this function to your existing JavaScript file
function displayRagSummary(ragSummary) {
    const ragSummaryContent = document.getElementById('rag-summary-content');
    
    if (!ragSummary || Object.keys(ragSummary).length === 0) {
        ragSummaryContent.innerHTML = '<p>No RAG summary available</p>';
        return;
    }
    
    let html = `
        <div class="rag-stats">
            <p><strong>Total RAG Queries:</strong> ${ragSummary.total_rag_queries}</p>
            <p><strong>Total Documents Accessed:</strong> ${ragSummary.total_documents_accessed}</p>
        </div>
        <h4>Documents Accessed:</h4>
    `;
    
    if (ragSummary.documents_accessed && Object.keys(ragSummary.documents_accessed).length > 0) {
        html += '<div class="documents-list">';
        
        for (const [path, info] of Object.entries(ragSummary.documents_accessed)) {
            html += `
                <div class="document-item">
                    <div class="document-path">${path}</div>
                    <div class="access-count">Accessed ${info.access_count} time${info.access_count !== 1 ? 's' : ''}</div>
                    ${info.example_excerpt ? `
                        <div class="document-excerpt">
                            <strong>Example excerpt:</strong> "${info.example_excerpt}"
                        </div>
                    ` : ''}
                </div>
            `;
        }
        
        html += '</div>';
    } else {
        html += '<p>No documents were accessed</p>';
    }
    
    ragSummaryContent.innerHTML = html;
}
