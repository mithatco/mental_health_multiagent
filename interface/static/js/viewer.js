// Global variables
let currentChatId = null;
let currentBatchId = null;
let logData = [];
let batchData = [];
let currentView = 'individual'; // 'individual' or 'batch'

// Event listeners
document.addEventListener('DOMContentLoaded', function() {
    // Load logs when page loads
    refreshLogs();
    
    // Add event listeners for view toggle buttons
    document.getElementById('individual-logs-btn').addEventListener('click', () => switchView('individual'));
    document.getElementById('batch-logs-btn').addEventListener('click', () => switchView('batch'));
    
    // Add event listeners for individual filters
    document.getElementById('profile-filter').addEventListener('change', applyFilters);
    document.getElementById('date-filter').addEventListener('change', applyFilters);
    document.getElementById('search-filter').addEventListener('input', applyFilters);
    
    // Add event listeners for batch filters
    document.getElementById('batch-date-filter').addEventListener('change', applyBatchFilters);
    document.getElementById('batch-search-filter').addEventListener('input', applyBatchFilters);
    
    // Add event listener for refresh button
    document.getElementById('refresh-btn').addEventListener('click', refreshData);
    
    // Add event listener for export button
    document.getElementById('export-btn').addEventListener('click', exportData);
});

// Switch between individual and batch views
function switchView(view) {
    currentView = view;
    
    // Update button states
    document.getElementById('individual-logs-btn').classList.toggle('active', view === 'individual');
    document.getElementById('batch-logs-btn').classList.toggle('active', view === 'batch');
    
    // Show/hide appropriate views
    document.getElementById('individual-logs-view').classList.toggle('hidden', view !== 'individual');
    document.getElementById('batch-logs-view').classList.toggle('hidden', view !== 'batch');
    document.getElementById('chat-view').classList.toggle('hidden', view !== 'individual');
    document.getElementById('batch-view').classList.toggle('hidden', view !== 'batch');
    
    // Load data if needed
    if (view === 'individual' && logData.length === 0) {
        refreshLogs();
    } else if (view === 'batch' && batchData.length === 0) {
        refreshBatches();
    }
    
    // Update button text
    document.getElementById('refresh-btn').textContent = view === 'individual' ? 'Refresh Logs' : 'Refresh Batches';
    document.getElementById('export-btn').textContent = view === 'individual' ? 'Export as Text' : 'Export Batch Summary';
}

// Refresh data based on current view
function refreshData() {
    if (currentView === 'individual') {
        refreshLogs();
    } else {
        refreshBatches();
    }
}

// Export data based on current view
function exportData() {
    if (currentView === 'individual') {
        exportChat();
    } else {
        exportBatchSummary();
    }
}

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

// Refresh batches
function refreshBatches() {
    // Show loading state
    const listElement = document.getElementById('batch-list');
    listElement.innerHTML = '<div class="loading"><div class="spinner"></div><p>Loading batches...</p></div>';
    
    // Fetch batches from server
    fetch('/api/batches?refresh=true')
        .then(response => response.json())
        .then(data => {
            // Store batch data globally
            batchData = data.batches;
            
            // Apply any active filters
            applyBatchFilters();
        })
        .catch(error => {
            console.error('Error loading batches:', error);
            listElement.innerHTML = '<div class="no-logs">Failed to load batches</div>';
        });
}

// Apply filters to batch list
function applyBatchFilters() {
    if (!batchData || !batchData.length) return;
    
    const listElement = document.getElementById('batch-list');
    const dateFilter = document.getElementById('batch-date-filter').value;
    const searchFilter = document.getElementById('batch-search-filter').value.toLowerCase();
    
    // Clear existing items
    listElement.innerHTML = '';
    
    // Prepare filter date objects
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    
    const sevenDaysAgo = new Date(today);
    sevenDaysAgo.setDate(today.getDate() - 7);
    
    const thirtyDaysAgo = new Date(today);
    thirtyDaysAgo.setDate(today.getDate() - 30);
    
    // Filter batches
    const filteredBatches = batchData.filter(batch => {
        // Date filter
        if (dateFilter !== 'All time') {
            const batchDate = new Date(batch.timestamp);
            
            if (dateFilter === 'Today') {
                const batchDay = new Date(batchDate);
                batchDay.setHours(0, 0, 0, 0);
                if (batchDay.getTime() !== today.getTime()) {
                    return false;
                }
            } else if (dateFilter === 'Last 7 days' && batchDate < sevenDaysAgo) {
                return false;
            } else if (dateFilter === 'Last 30 days' && batchDate < thirtyDaysAgo) {
                return false;
            }
        }
        
        // Search filter
        if (searchFilter) {
            const searchableText = `${batch.id} ${batch.description || ''}`.toLowerCase();
            if (!searchableText.includes(searchFilter)) {
                return false;
            }
        }
        
        return true;
    });
    
    // Add filtered batches to list
    if (filteredBatches.length === 0) {
        listElement.innerHTML = '<div class="no-logs">No batches match the filters</div>';
    } else {
        filteredBatches.forEach(batch => {
            const batchItem = document.createElement('div');
            batchItem.className = 'batch-item';
            batchItem.dataset.id = batch.id;
            
            // Format date for display
            const batchDate = new Date(batch.timestamp);
            const formattedDate = batchDate.toLocaleDateString() + ' ' + batchDate.toLocaleTimeString();
            
            batchItem.textContent = `${formattedDate} - ${batch.conversation_count} conversations`;
            batchItem.addEventListener('click', () => loadBatch(batch.id));
            listElement.appendChild(batchItem);
        });
    }
}

// Load batch details
function loadBatch(batchId) {
    currentBatchId = batchId;
    
    // Update selected item
    document.querySelectorAll('.batch-item').forEach(item => {
        item.classList.remove('selected');
        if (item.dataset.id === batchId) {
            item.classList.add('selected');
        }
    });
    
    // Show loading state
    document.getElementById('batch-summary-content').innerHTML = '<div class="loading"><div class="spinner"></div><p>Loading batch summary...</p></div>';
    document.getElementById('batch-results-table').querySelector('tbody').innerHTML = '<tr><td colspan="5" class="loading-cell"><div class="spinner"></div><p>Loading batch results...</p></td></tr>';
    document.getElementById('batch-metadata-display').innerHTML = '';
    
    // Fetch batch data
    fetch(`/api/batches/${batchId}`)
        .then(response => response.json())
        .then(data => {
            // Update header
            document.querySelector('#batch-view .metadata .header').textContent = `Batch: ${data.id}`;
            
            // Update metadata
            updateBatchMetadata(data);
            
            // Update batch summary
            displayBatchSummary(data);
            
            // Update batch results
            displayBatchResults(data);
        })
        .catch(error => {
            console.error('Error loading batch:', error);
            document.getElementById('batch-summary-content').innerHTML = '<div class="error">Failed to load batch summary</div>';
            document.getElementById('batch-results-table').querySelector('tbody').innerHTML = '<tr><td colspan="5" class="error">Failed to load batch results</td></tr>';
        });
}

// Update batch metadata display
function updateBatchMetadata(data) {
    const metadataDisplay = document.getElementById('batch-metadata-display');
    metadataDisplay.innerHTML = '';
    
    // Format date
    let dateStr = 'Unknown';
    try {
        dateStr = new Date(data.timestamp).toLocaleString();
    } catch (e) {}
    
    // Create metadata fields
    const fields = [
        { label: 'Date', value: dateStr },
        { label: 'Conversation Count', value: data.conversation_count || 'Unknown' },
        { label: 'Profile', value: data.profile || 'Mixed' },
        { label: 'Average Duration', value: data.avg_duration ? `${data.avg_duration.toFixed(2)}s` : 'Unknown' }
    ];
    
    fields.forEach(field => {
        const item = document.createElement('div');
        item.className = 'metadata-item';
        item.innerHTML = `<strong>${field.label}:</strong> ${field.value}`;
        metadataDisplay.appendChild(item);
    });
}

// Display batch summary
function displayBatchSummary(data) {
    const summaryContainer = document.getElementById('batch-summary-content');
    
    if (!data.summary || Object.keys(data.summary).length === 0) {
        summaryContainer.innerHTML = '<p>No batch summary available</p>';
        return;
    }
    
    // If summary is CSV format, display as table
    let html = '<table class="batch-summary-table">';
    
    // Add headers
    html += '<thead><tr>';
    for (const header of data.summary.headers) {
        html += `<th>${header}</th>`;
    }
    html += '</tr></thead>';
    
    // Add rows
    html += '<tbody>';
    for (const row of data.summary.rows) {
        html += '<tr>';
        for (const cell of row) {
            html += `<td>${cell}</td>`;
        }
        html += '</tr>';
    }
    html += '</tbody></table>';
    
    summaryContainer.innerHTML = html;
}

// Display batch results
function displayBatchResults(data) {
    const resultsTable = document.getElementById('batch-results-table').querySelector('tbody');
    
    if (!data.results || data.results.length === 0) {
        resultsTable.innerHTML = '<tr><td colspan="5">No batch results available</td></tr>';
        return;
    }
    
    let html = '';
    data.results.forEach(result => {
        html += `<tr>
            <td>${result.conversation_id}</td>
            <td>${result.profile}</td>
            <td>${result.question_count}</td>
            <td>${result.duration.toFixed(2)}s</td>
            <td>
                <button class="view-log-btn" data-path="${result.log_path}">View Log</button>
            </td>
        </tr>`;
    });
    
    resultsTable.innerHTML = html;
    
    // Add event listeners to view log buttons
    document.querySelectorAll('.view-log-btn').forEach(button => {
        button.addEventListener('click', () => {
            // Extract filename from path
            const path = button.dataset.path;
            const filename = path.split('/').pop();
            
            // Find log ID from filename
            const logId = filename.replace(/\.json$/, '');
            
            // Switch to individual view and load the chat
            switchView('individual');
            loadChat(logId);
        });
    });
}

// Export batch summary
function exportBatchSummary() {
    if (!currentBatchId) {
        alert('Please select a batch to export first');
        return;
    }
    
    window.open(`/api/batches/${currentBatchId}/export`, '_blank');
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
