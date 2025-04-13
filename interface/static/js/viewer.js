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

    // Add section toggle functionality
    setupSectionToggles();
    
    // Fetch available models for evaluation
    fetchAvailableModels();
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

// Fix the loadChat function to properly check for the element before accessing it
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
    
    // Clear any existing evaluation content
    document.getElementById('evaluation-content').innerHTML = '<p>No evaluation results available</p>';
    
    // Reset evaluate button to default state when loading a chat
    const evaluateBtn = document.getElementById('evaluate-btn');
    evaluateBtn.disabled = false;
    evaluateBtn.textContent = "Evaluate with Ollama";
    
    // Remove any existing event listeners by cloning and replacing the button
    const newEvaluateBtn = evaluateBtn.cloneNode(true);
    evaluateBtn.parentNode.replaceChild(newEvaluateBtn, evaluateBtn);
    
    // Refresh available models
    fetchAvailableModels();
    
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
            
            // Update diagnosis with new function
            updateDiagnosis(data.diagnosis || 'No diagnosis available');
            
            // Update RAG summary
            displayRagSummary(data.rag_summary);
            
            // Check if evaluation exists and show it
            if (data.evaluation) {
                displayEvaluationResults(data.evaluation);
            } else {
                // Explicitly clear evaluation content if no evaluation exists
                document.getElementById('evaluation-content').innerHTML = '<p>No evaluation results available</p>';
            }
            
            // Add event listener for evaluate button (to the new button instance)
            document.getElementById('evaluate-btn').addEventListener('click', () => {
                startEvaluation(chatId);
            });
            
            // Check if an evaluation is already in progress
            // But only if the evaluate button is visible (not in a minimized section)
            if (!document.getElementById('evaluation-container').classList.contains('section-minimized')) {
                checkEvaluationStatus(chatId);
            }
        })
        .catch(error => {
            console.error('Error loading chat:', error);
            document.getElementById('chat-container').innerHTML = '<div class="no-logs">Failed to load conversation</div>';
        });
}

// Add a function to check if evaluation is already in progress
function checkEvaluationStatus(chatId) {
    fetch(`/api/logs/${chatId}/evaluation?nocache=${Date.now()}`)
    .then(response => response.json())
    .then(data => {
        const evaluateBtn = document.getElementById('evaluate-btn');
        
        if (data.status === 'in_progress') {
            // If an evaluation is already running, update button
            evaluateBtn.disabled = true;
            evaluateBtn.textContent = "Evaluating...";
            
            // Start polling
            pollEvaluationStatus(chatId);
        } else if (data.status === 'completed' && data.results) {
            // If evaluation is already complete, show results
            displayEvaluationResults(data.results);
        }
        // For any other status, leave the button as-is
    })
    .catch(error => {
        console.error('Error checking evaluation status:', error);
    });
}

// Fix the evaluation polling to handle timeouts and errors better
function pollEvaluationStatus(chatId) {
    const evaluateBtn = document.getElementById('evaluate-btn');
    let dots = 0;
    
    // Update the dots animation on the button
    const updateDots = setInterval(() => {
        dots = (dots + 1) % 4;
        evaluateBtn.textContent = `Evaluating${'.'.repeat(dots)}`;
    }, 500);
    
    // Set a timeout counter to prevent infinite polling
    let pollCount = 0;
    const maxPolls = 60; // Max 2 minutes (at 2 second intervals)
    
    // Check evaluation status every 2 seconds
    const statusCheck = setInterval(() => {
        pollCount++;
        
        // Stop polling after maxPolls to prevent infinite polling
        if (pollCount >= maxPolls) {
            clearInterval(statusCheck);
            clearInterval(updateDots);
            showEvaluationError('Evaluation timed out. Please try again later.');
            // Reset button
            evaluateBtn.disabled = false;
            evaluateBtn.textContent = "Evaluate with Ollama";
            return;
        }
        
        // Add cache-busting parameter to avoid cached responses
        fetch(`/api/logs/${chatId}/evaluation?nocache=${Date.now()}`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            // Add debugging to see exactly what's coming back
            console.log("Evaluation status response:", data);
            
            if (data.status === 'completed') {
                // Evaluation completed
                clearInterval(statusCheck);
                clearInterval(updateDots);
                // Reset button
                evaluateBtn.disabled = false;
                evaluateBtn.textContent = "Evaluate with Ollama";
                
                // FIX: Use loadChat to reload the chat with fresh evaluation data
                // This ensures we get the full data structure
                loadChat(chatId);
            } else if (data.status === 'error') {
                // Evaluation failed
                clearInterval(statusCheck);
                clearInterval(updateDots);
                // Reset button
                evaluateBtn.disabled = false;
                evaluateBtn.textContent = "Evaluate with Ollama";
                showEvaluationError(data.message || 'Evaluation failed');
            } else if (data.status !== 'in_progress') {
                // Any other status means it's not running
                clearInterval(statusCheck);
                clearInterval(updateDots);
                // Reset button
                evaluateBtn.disabled = false;
                evaluateBtn.textContent = "Evaluate with Ollama";
            }
            // If status is 'in_progress', continue polling
        })
        .catch(error => {
            console.error('Error checking evaluation status:', error);
            pollCount += 5; // Increment poll count more on error to timeout sooner
        });
    }, 2000);
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
    
    // Check if this is a full conversation (single agent) or a multi-agent conversation
    const isFullConversation = data.metadata?.hasOwnProperty('full_conversation') ? 
                              data.metadata.full_conversation : false;
    
    // Create metadata fields
    const fields = [
        { label: 'Date', value: dateStr },
        { label: 'Questionnaire', value: data.questionnaire || 'Unknown' },
        { label: 'Patient Profile', value: data.metadata?.patient_profile || 'Unknown' }
    ];
    
    // Add appropriate model information based on conversation type
    if (isFullConversation) {
        // For full conversations, show just the agent model
        fields.push({ label: 'Agent Model', value: data.metadata?.assistant_model || 'Unknown' });
        fields.push({ label: 'Conversation Type', value: 'Full Conversation (Single Agent)' });
    } else {
        // For multi-agent conversations, show both models
        fields.push({ label: 'Assistant Model', value: data.metadata?.assistant_model || 'Unknown' });
        fields.push({ label: 'Patient Model', value: data.metadata?.patient_model || 'Unknown' });
    }
    
    // Add timing information if available
    if (data.timing_metrics && data.timing_metrics.total_time) {
        const totalTime = parseFloat(data.timing_metrics.total_time).toFixed(2);
        
        // Create tooltip content with timing breakdown
        let tooltipContent = `
            <div class="timing-tooltip">
                <p>Conversation: ${parseFloat(data.timing_metrics.conversation_generation_time || 0).toFixed(2)}s</p>
                <p>Diagnosis: ${parseFloat(data.timing_metrics.diagnosis_generation_time || 0).toFixed(2)}s</p>
                ${data.timing_metrics.rag_evaluation_time ? 
                  `<p>RAG Evaluation: ${parseFloat(data.timing_metrics.rag_evaluation_time).toFixed(2)}s</p>` : ''}
            </div>
        `;
        
        // Add timing field with info icon
        fields.push({ 
            label: 'Total Time', 
            value: `${totalTime}s <span class="info-icon" data-tooltip="${encodeURIComponent(tooltipContent)}">ⓘ</span>` 
        });
    }
    
    // Add question count if available
    fields.push({ label: 'Question Count', value: data.metadata?.question_count || 'Unknown' });
    
    // Create and append each metadata item
    fields.forEach(field => {
        const item = document.createElement('div');
        item.className = 'metadata-item';
        item.innerHTML = `<strong>${field.label}:</strong> ${field.value}`;
        metadataDisplay.appendChild(item);
    });
    
    // Initialize tooltips for any info icons
    initializeTooltips();
}

// Add tooltip initialization function
function initializeTooltips() {
    document.querySelectorAll('.info-icon').forEach(icon => {
        icon.addEventListener('mouseenter', function(e) {
            // Create tooltip element if it doesn't exist
            let tooltip = document.getElementById('tooltip');
            if (!tooltip) {
                tooltip = document.createElement('div');
                tooltip.id = 'tooltip';
                tooltip.className = 'tooltip';
                document.body.appendChild(tooltip);
            }
            
            // Set tooltip content and position
            tooltip.innerHTML = decodeURIComponent(this.dataset.tooltip);
            
            // Position tooltip near the icon
            const rect = this.getBoundingClientRect();
            tooltip.style.top = `${rect.bottom + 5}px`;
            tooltip.style.left = `${rect.left}px`;
            
            // Show tooltip
            tooltip.classList.add('visible');
        });
        
        icon.addEventListener('mouseleave', function() {
            // Hide tooltip
            const tooltip = document.getElementById('tooltip');
            if (tooltip) {
                tooltip.classList.remove('visible');
            }
        });
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
    
    // Create a copy of the conversation to avoid modifying the original
    const conversationCopy = [...conversation];
    
    // Find the last diagnosis message from assistant (typically the last assistant message)
    let lastAssistantIndex = -1;
    for (let i = conversationCopy.length - 1; i >= 0; i--) {
        if (conversationCopy[i].role === 'assistant') {
            lastAssistantIndex = i;
            break;
        }
    }
    
    // Remove the last assistant message (diagnosis) if found
    if (lastAssistantIndex >= 0) {
        conversationCopy.splice(lastAssistantIndex, 1);
    }
    
    // Display the filtered conversation
    conversationCopy.forEach(message => {
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

// Format text with Markdown support
function formatText(text) {
    if (!text) return '';
    
    // Check if marked.js is loaded, if not use basic formatting
    if (typeof marked === 'undefined') {
        // Basic formatting - just replace newlines with <br> tags
        return text.split('\n').join('<br>');
    }
    
    // First, process our custom tags and replace them with HTML
    // This must be done before Markdown processing to prevent tag interference
    text = processCustomTags(text);
    
    // Configure marked options for security and GitHub flavor
    marked.setOptions({
        breaks: true,           // Add <br> on single line breaks
        gfm: true,              // Use GitHub Flavored Markdown
        headerIds: false,       // Don't add IDs to headers (for security)
        mangle: false,          // Don't mangle email links
        sanitize: false,        // Let DOMPurify handle sanitization
        smartLists: true,       // Use smarter list behavior
        smartypants: true,      // Use smart typography like quotes and dashes
    });
    
    // Parse markdown
    let html = marked.parse(text);
    
    // Return parsed HTML with special handling for diagnosis content
    return html;
}

// Process custom tags for medical terms, symptoms, and quotes
function processCustomTags(text) {
    // Replace medical term tags
    text = text.replace(/<med>(.*?)<\/med>/g, '<span class="medical-term">$1</span>');
    
    // Replace symptom tags
    text = text.replace(/<sym>(.*?)<\/sym>/g, '<span class="symptom">$1</span>');
    
    // Replace patient quote tags
    text = text.replace(/<quote>(.*?)<\/quote>/g, '<span class="patient-quote">$1</span>');
    
    return text;
}

// Update diagnosis display with better markdown handling
function updateDiagnosis(diagnosisText) {
    const diagnosisContent = document.getElementById('diagnosis-content');
    
    if (!diagnosisText || diagnosisText === 'No diagnosis available') {
        diagnosisContent.innerHTML = '<p>No diagnosis available</p>';
        return;
    }
    
    // Add diagnosis content with markdown support
    diagnosisContent.innerHTML = formatText(diagnosisText);
    
    // Add special class to diagnosis content for styling
    diagnosisContent.classList.add('markdown-content');
    
    // Add color coding to diagnosis sections
    colorCodeDiagnosisSections(diagnosisContent);
    
    // Add collapsible behavior to sections
    setupCollapsibleSections(diagnosisContent);
}

// Color code diagnosis sections by content
function colorCodeDiagnosisSections(container) {
    // Apply color coding to Clinical Impression section
    applyColorToElements(container, ['h2', 'h3', 'strong'], 
                        ['Clinical Impression', 'clinical impression'], 
                        'clinical-impression-section');
    
    // Apply color coding to Reasoning section
    applyColorToElements(container, ['h2', 'h3', 'strong'], 
                        ['Reasoning', 'reasoning'], 
                        'reasoning-section');
    
    // Apply color coding to Recommendations section
    applyColorToElements(container, ['h2', 'h3', 'strong'], 
                        ['Recommend', 'recommend', 'Treatment', 'treatment', 'Next Steps'], 
                        'recommendations-section');
}

// Apply color coding class to elements containing specific text
function applyColorToElements(container, selectors, textPhrases, className) {
    // Query for all specified elements
    selectors.forEach(selector => {
        const elements = container.querySelectorAll(selector);
        
        elements.forEach(element => {
            const content = element.textContent.toLowerCase();
            
            // Check if element text contains any of the specified phrases
            if (textPhrases.some(phrase => content.includes(phrase.toLowerCase()))) {
                element.classList.add(className);
            }
        });
    });
}

// Setup collapsible sections in diagnosis content
function setupCollapsibleSections(container) {
    // Find all headings that might represent sections
    const headings = container.querySelectorAll('h1, h2, h3, h4');
    
    headings.forEach(heading => {
        // Only process headings that have content following them
        let sectionContent = [];
        let nextEl = heading.nextElementSibling;
        
        // Collect elements until we hit another heading or run out of siblings
        while (nextEl && !nextEl.matches('h1, h2, h3, h4')) {
            sectionContent.push(nextEl);
            nextEl = nextEl.nextElementSibling;
        }
        
        if (sectionContent.length > 0) {
            // Make the heading clickable
            heading.classList.add('collapsible-heading');
            
            // Add expand/collapse icon
            const icon = document.createElement('span');
            icon.className = 'collapse-icon';
            icon.innerHTML = '▼';
            heading.appendChild(icon);
            
            // Create a container for the section content
            const sectionDiv = document.createElement('div');
            sectionDiv.className = 'collapsible-section';
            
            // Move content into the container
            sectionContent.forEach(el => {
                sectionDiv.appendChild(el);
            });
            
            // Insert the container after the heading
            heading.after(sectionDiv);
            
            // Add click event to toggle section visibility
            heading.addEventListener('click', () => {
                sectionDiv.classList.toggle('collapsed');
                icon.innerHTML = sectionDiv.classList.contains('collapsed') ? '►' : '▼';
            });
        }
    });
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
    
    // let html = `
    //     <div class="rag-stats">
    //         <p><strong>Total RAG Queries:</strong> ${ragSummary.total_rag_queries}</p>
    //         <p><strong>Total Documents Accessed:</strong> ${ragSummary.total_documents_accessed}</p>
    //     </div>
    // `;

    let html = ``;
    
    // Add evaluation metrics section if available
    if (ragSummary.evaluation_metrics && Object.keys(ragSummary.evaluation_metrics).length > 0) {
        html += `<h4>RAG Evaluation Metrics</h4>
        <div class="evaluation-metrics-summary">`;
        
        for (const [metric, data] of Object.entries(ragSummary.evaluation_metrics)) {
            if (data.average_score !== null) {
                html += `
                <div class="metric-item">
                    <div class="metric-name">${formatMetricName(metric)}</div>
                    <div class="metric-score">Avg. Score: ${data.average_score.toFixed(4)}</div>
                    <div class="metric-count">(${data.count || 0} evaluations)</div>
                    ${metric !== "overall" ? `<button class="show-reasons-btn" data-metric="${metric}">Show Explanations</button>` : ''}
                </div>`;
            }
        }
        
        html += `</div>`;
        
        // Add section for metric reasons that will be shown/hidden on demand
        html += `<div id="metric-reasons-container" class="metric-reasons-container">`;
        for (const [metric, data] of Object.entries(ragSummary.evaluation_metrics)) {
            if (data.reasons && data.reasons.length > 0) {
                html += `
                <div id="reasons-${metric}" class="metric-reasons hidden">
                    <h5>${formatMetricName(metric)} Explanations</h5>
                    <div class="reasons-list">`;
                
                // Add each reason with its score
                data.reasons.forEach((item, index) => {
                    const scoreClass = item.passed ? 'score-passed' : 'score-failed';
                    html += `
                    <div class="reason-item">
                        <div class="reason-header">
                            <span class="reason-score ${scoreClass}">${item.score.toFixed(2)}</span>
                            <span class="reason-index">Evaluation #${index + 1}</span>
                        </div>
                        <div class="reason-content">${item.reason}</div>
                    </div>`;
                });
                
                html += `</div></div>`;
            }
        }
        html += `</div>`;
    }
    
    html += `<h4>Documents Accessed:</h4>`;
    
    if (ragSummary.documents_accessed && Object.keys(ragSummary.documents_accessed).length > 0) {
        html += '<div class="documents-list">';
        
        // Convert object to array and sort by average_score (highest first)
        const sortedDocuments = Object.entries(ragSummary.documents_accessed)
            .map(([path, info]) => ({ path, ...info }))
            .sort((a, b) => {
                // Handle missing scores (null, undefined) by placing them at the end
                if (!a.average_score) return 1;
                if (!b.average_score) return -1;
                // Sort in descending order (highest score first)
                return b.average_score - a.average_score;
            });
        
        // Generate HTML for sorted documents
        for (const doc of sortedDocuments) {
            html += `
                <div class="document-item">
                    <div class="document-header">
                        <div class="document-path">${doc.path}</div>
                        <div class="access-count">Accessed ${doc.access_count} time${doc.access_count !== 1 ? 's' : ''}</div>
                    </div>
                    <div class="document-scores">
                        ${doc.average_score ? `<div class="score"><strong>Average Score:</strong> ${doc.average_score.toFixed(4)}</div>` : ''}
                    </div>
                    ${doc.example_excerpt ? `
                        <!-- <div class="document-excerpt">
                            <strong>Example excerpt:</strong> "${doc.example_excerpt}"
                        </div> -->
                    ` : ''}
                    ${doc.relevance_explanation ? `
                        <div class="relevance-explanation">
                            <strong>Why it's relevant:</strong> ${doc.relevance_explanation}
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
    
    // Add event listeners for show reasons buttons
    document.querySelectorAll('.show-reasons-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const metric = this.dataset.metric;
            const reasonsDiv = document.getElementById(`reasons-${metric}`);
            
            // Hide all other reason divs
            document.querySelectorAll('.metric-reasons').forEach(div => {
                div.classList.add('hidden');
            });
            
            // Show this one
            reasonsDiv.classList.remove('hidden');
            
            // Update button text
            document.querySelectorAll('.show-reasons-btn').forEach(b => {
                b.textContent = "Show Explanations";
            });
            this.textContent = "Hide Explanations";
        });
    });
}

// Add these functions to your existing JavaScript file
function startEvaluation(chatId) {
    // Disable evaluate button and update text to show progress
    const evaluateBtn = document.getElementById('evaluate-btn');
    evaluateBtn.disabled = true;
    evaluateBtn.textContent = "Evaluating...";
    
    // Get selected model
    const modelSelect = document.getElementById('model-select');
    const selectedModel = modelSelect.value;
    
    // Call API to start evaluation with the selected model
    fetch(`/api/logs/${chatId}/evaluate`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ model: selectedModel })
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'started' || data.status === 'in_progress') {
            // Poll for evaluation status
            pollEvaluationStatus(chatId);
        } else {
            // Show error
            showEvaluationError(data.message || 'Failed to start evaluation');
            // Reset button
            evaluateBtn.disabled = false;
            evaluateBtn.textContent = "Evaluate with Ollama";
        }
    })
    .catch(error => {
        console.error('Error starting evaluation:', error);
        showEvaluationError('Failed to start evaluation');
        // Reset button
        evaluateBtn.disabled = false;
        evaluateBtn.textContent = "Evaluate with Ollama";
    });
}

// Fix the evaluation polling
function pollEvaluationStatus(chatId) {
    const evaluateBtn = document.getElementById('evaluate-btn');
    let dots = 0;
    
    // Update the dots animation on the button
    const updateDots = setInterval(() => {
        dots = (dots + 1) % 4;
        evaluateBtn.textContent = `Evaluating${'.'.repeat(dots)}`;
    }, 500);
    
    // Check evaluation status every 2 seconds
    const statusCheck = setInterval(() => {
        fetch(`/api/logs/${chatId}/evaluation`)
        .then(response => response.json())
        .then(data => {
            if (data.status === 'completed') {
                // Evaluation completed
                clearInterval(statusCheck);
                clearInterval(updateDots);
                // Reset button
                evaluateBtn.disabled = false;
                evaluateBtn.textContent = "Evaluate with Ollama";
                displayEvaluationResults(data.results);
            } else if (data.status === 'error') {
                // Evaluation failed
                clearInterval(statusCheck);
                clearInterval(updateDots);
                // Reset button
                evaluateBtn.disabled = false;
                evaluateBtn.textContent = "Evaluate with Ollama";
                showEvaluationError(data.message || 'Evaluation failed');
            }
            // If status is 'in_progress', continue polling
        })
        .catch(error => {
            console.error('Error checking evaluation status:', error);
            clearInterval(statusCheck);
            clearInterval(updateDots);
            showEvaluationError('Failed to check evaluation status');
        });
    }, 2000);
}

function showEvaluationError(message) {
    // Enable evaluate button and reset text
    const evaluateBtn = document.getElementById('evaluate-btn');
    evaluateBtn.disabled = false;
    evaluateBtn.textContent = "Evaluate with Ollama";
    
    // Show error message
    document.getElementById('evaluation-content').innerHTML = `
        <div class="error-message">
            <p>Error: ${message}</p>
            <p>Please try again later.</p>
        </div>
    `;
}

function displayEvaluationResults(results) {
    // Enable evaluate button and reset text
    const evaluateBtn = document.getElementById('evaluate-btn');
    evaluateBtn.disabled = false;
    evaluateBtn.textContent = "Evaluate with Ollama";
    
    // Check if we have valid results
    if (!results || results.error) {
        showEvaluationError(results.error || 'Invalid evaluation results');
        return;
    }
    
    // Debug log the complete results structure first
    console.log("Full evaluation results:", JSON.stringify(results, null, 2));
    
    // Extract evaluation data from the proper location in the results structure
    let evaluationData = null;
    
    if (results.evaluation) {
        // If the results contain an 'evaluation' field (from the API)
        console.log("Using results.evaluation");
        evaluationData = results.evaluation;
    } else if (results.results && results.results.evaluation) {
        // If the results are nested as results.results.evaluation (from polling)
        console.log("Using results.results.evaluation");
        evaluationData = results.results.evaluation;
    } else {
        // Otherwise, use the results object directly
        console.log("Using results directly");
        evaluationData = results;
    }
    
    // Log the extracted evaluation data
    console.log("Extracted evaluation data:", evaluationData);
    
    // Check if we have rubric scores to display
    if (!evaluationData) {
        showEvaluationError('No evaluation data found in results');
        return;
    }
    
    // Handle different field names/locations for scores
    const rubricScores = evaluationData.rubric_scores || 
                        (evaluationData.evaluation && evaluationData.evaluation.rubric_scores) ||
                        {};
    
    const averageScore = evaluationData.average_score || 
                        (evaluationData.evaluation && evaluationData.evaluation.average_score) ||
                        "N/A";
                        
    const explanations = evaluationData.explanations || 
                        (evaluationData.evaluation && evaluationData.evaluation.explanations) ||
                        {};
                        
    const overallComments = evaluationData.overall_comments || 
                        (evaluationData.evaluation && evaluationData.evaluation.overall_comments) ||
                        "No overall comments available";
                        
    const diagnosisAccuracy = evaluationData.diagnosis_accuracy || 
                            (evaluationData.evaluation && evaluationData.evaluation.diagnosis_accuracy);
    
    // Get question evaluation data if available
    const questionEvaluation = evaluationData.question_evaluation || 
                             (evaluationData.evaluation && evaluationData.evaluation.question_evaluation);
    
    // Log what we found
    console.log("Extracted components:", {
        rubricScores,
        averageScore,
        explanations,
        overallComments,
        diagnosisAccuracy,
        questionEvaluation
    });
    
    // Format the evaluation time
    const evalTime = evaluationData.evaluation_time ? 
        `${Math.round(evaluationData.evaluation_time)} seconds` : 
        'Unknown';
    
    // Generate HTML for evaluation results
    let html = `
        <div class="evaluation-summary">
            <p><strong>Evaluation performed:</strong> ${evaluationData.timestamp || 'Unknown'}</p>
            <p><strong>Model used:</strong> ${evaluationData.model || 'Unknown'}</p>
            <p><strong>Evaluation time:</strong> ${evalTime}</p>
            <p><strong>Average score:</strong> ${averageScore}</p>
        </div>
    `;

    // Add question evaluation section if available
    if (questionEvaluation) {
        const percentageAsked = parseFloat(questionEvaluation.percentage_asked).toFixed(1);
        const qualityScore = parseInt(questionEvaluation.quality_score);
        const totalQuestions = questionEvaluation.total_questions || 0;
        const questionsAsked = questionEvaluation.questions_asked || 0;
        
        html += `
            <h4>Question Coverage Analysis</h4>
            <div class="question-evaluation">
                <div class="question-metrics">
                    <div class="question-metric">
                        <div class="metric-header">Questions Asked</div>
                        <div class="percentage-circle">
                            <svg viewBox="0 0 36 36">
                                <path class="circle-bg"
                                    d="M18 2.0845
                                    a 15.9155 15.9155 0 0 1 0 31.831
                                    a 15.9155 15.9155 0 0 1 0 -31.831"
                                />
                                <path class="circle"
                                    stroke-dasharray="${percentageAsked}, 100"
                                    d="M18 2.0845
                                    a 15.9155 15.9155 0 0 1 0 31.831
                                    a 15.9155 15.9155 0 0 1 0 -31.831"
                                />
                                <text x="18" y="20.35" class="percentage">${percentageAsked}%</text>
                            </svg>
                        </div>
                        <div class="question-count">${questionsAsked} of ${totalQuestions}</div>
                    </div>
                    <div class="question-metric">
                        <div class="metric-header">Question Quality</div>
                        <div class="quality-score">
                            <div class="quality-stars">
                                ${generateStars(qualityScore, 5)}
                            </div>
                            <div class="quality-value">${qualityScore}/5</div>
                        </div>
                    </div>
                </div>
                <div class="question-explanation">
                    <p>${questionEvaluation.explanation}</p>
                </div>
                <div class="question-details-toggle">
                    <button class="show-questions-btn">Show Question Details</button>
                </div>
                <div class="question-details hidden">
                    <table class="question-table">
                        <thead>
                            <tr>
                                <th>Original Question</th>
                                <th>Asked</th>
                                <th>Therapist Version</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${questionEvaluation.question_analysis ? questionEvaluation.question_analysis.map(q => `
                                <tr class="${q.asked ? 'question-asked' : 'question-not-asked'}">
                                    <td>${q.original}</td>
                                    <td>${q.asked ? '✓' : '✗'}</td>
                                    <td>${q.asked ? q.therapist_version || '-' : '-'}</td>
                                </tr>
                            `).join('') : ''}
                        </tbody>
                    </table>
                </div>
            </div>
        `;
        
        // Add event listener for the show details button after HTML is added to DOM
        setTimeout(() => {
            const showQuestionsBtn = document.querySelector('.show-questions-btn');
            if (showQuestionsBtn) {
                showQuestionsBtn.addEventListener('click', function() {
                    const questionDetails = document.querySelector('.question-details');
                    if (questionDetails) {
                        questionDetails.classList.toggle('hidden');
                        this.textContent = questionDetails.classList.contains('hidden') ? 
                            'Show Question Details' : 'Hide Question Details';
                    }
                });
            }
        }, 100);
    }

    // Check for rubric scores
    if (rubricScores && Object.keys(rubricScores).length > 0) {
        html += `<h4>Mental Health Evaluation Scores</h4>
        <div class="evaluation-metrics">`;
        
        // Display each rubric score
        for (const key in rubricScores) {
            const score = rubricScores[key];
            const explanation = explanations && explanations[key] 
                ? explanations[key] 
                : 'No explanation provided';
            
            html += `
                <div class="metric-card">
                    <div class="metric-header">${formatMetricName(key)}</div>
                    <div class="metric-value">${score}</div>
                    <div class="metric-description">${explanation}</div>
                </div>
            `;
        }
        html += `</div>`;
    }
    
    // Add overall comments if available
    if (overallComments) {
        html += `
            <h4>Overall Assessment</h4>
            <div class="overall-comments">
                ${overallComments}
            </div>
        `;
    }
    
    // Add diagnosis accuracy if available - improved visual display
    if (diagnosisAccuracy) {
        const accuracy = diagnosisAccuracy;
        const matchStatus = accuracy.matches_profile ? "matches" : "does not match";
        const matchClass = accuracy.matches_profile ? "match-success" : "match-failure";
        const confidenceScore = parseInt(accuracy.confidence) || 0;
        
        html += `
            <h4>Diagnosis Accuracy</h4>
            <div class="diagnosis-accuracy ${matchClass}">
                <div class="match-indicator">
                    <div class="match-status">
                        <span class="match-icon">${accuracy.matches_profile ? '✓' : '✗'}</span>
                        <span class="match-text">Diagnosis ${matchStatus} expected profile</span>
                    </div>
                    <div class="confidence-meter">
                        <div class="confidence-label">Confidence:</div>
                        <div class="confidence-stars">
                            ${generateStars(confidenceScore, 5)}
                        </div>
                        <div class="confidence-value">${confidenceScore}/5</div>
                    </div>
                </div>
                <div class="match-explanation">
                    <strong>Explanation:</strong> ${accuracy.explanation || 'No explanation provided'}
                </div>
            </div>
        `;
    }
    
    // Update the content
    document.getElementById('evaluation-content').innerHTML = html;
}

// Helper function to generate star ratings for confidence
function generateStars(count, total) {
    let stars = '';
    
    // Add filled stars
    for (let i = 0; i < count; i++) {
        stars += '<span class="star filled">★</span>';
    }
    
    // Add empty stars
    for (let i = count; i < total; i++) {
        stars += '<span class="star empty">☆</span>';
    }
    
    return stars;
}

function formatMetricName(name) {
    // Convert snake_case to Title Case
    return name.split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}

function getMetricDescription(metricName) {
    // Return description for common metrics
    const descriptions = {
        'answer_relevancy': 'How relevant the response is to the question',
        'faithfulness': 'Whether the response contains information supported by the context',
        'context_precision': 'How relevant the context is to the question',
        'context_recall': 'How much relevant information from the context is used in the response'
    };
    
    return descriptions[metricName] || 'No description available';
}

function setupSectionToggles() {
    // Add toggle elements to each section header
    const sections = [
        { 
            id: 'chat-container-section', 
            title: 'Conversation', 
            container: 'chat-container' 
        },
        { 
            id: 'diagnosis-section', 
            title: 'Diagnosis', 
            container: 'diagnosis-content' 
        },
        { 
            id: 'rag-summary-section', 
            title: 'RAG Summary', 
            container: 'rag-summary-content' 
        },
        { 
            id: 'evaluation-section', 
            title: 'Conversation Evaluation', 
            container: 'evaluation-container' 
        }
    ];
    
    sections.forEach(section => {
        const container = document.getElementById(section.container);
        if (!container) return; // Skip if container not found
        
        const header = document.createElement('div');
        header.className = 'section-header';
        
        const title = document.createElement('h3');
        title.textContent = section.title;
        
        const toggle = document.createElement('button');
        toggle.className = 'section-toggle';
        toggle.innerHTML = '−'; // Default to minus sign (expanded)
        toggle.setAttribute('aria-label', 'Toggle section visibility');
        toggle.setAttribute('data-section-id', section.id);
        
        header.appendChild(title);
        header.appendChild(toggle);
        
        // Create a section wrapper if it doesn't exist
        let wrapper = document.getElementById(section.id);
        if (!wrapper) {
            wrapper = document.createElement('div');
            wrapper.id = section.id;
            wrapper.className = 'section';
            container.parentNode.insertBefore(wrapper, container);
            wrapper.appendChild(container);
        }
        
        // Insert header before the container within the wrapper
        wrapper.insertBefore(header, wrapper.firstChild);
        
        // Add click event to toggle
        toggle.addEventListener('click', () => {
            wrapper.classList.toggle('section-minimized');
            toggle.innerHTML = wrapper.classList.contains('section-minimized') ? '+' : '−';
            
            // Store section state in localStorage
            localStorage.setItem(`section-${section.id}-minimized`, 
                wrapper.classList.contains('section-minimized') ? 'true' : 'false');
        });
        
        // Restore previously saved state
        const isMinimized = localStorage.getItem(`section-${section.id}-minimized`) === 'true';
        if (isMinimized) {
            wrapper.classList.add('section-minimized');
            toggle.innerHTML = '+';
        }
    });
}

// Function to fetch available models from Ollama
function fetchAvailableModels() {
    fetch('/api/models')
        .then(response => response.json())
        .then(data => {
            const modelSelect = document.getElementById('model-select');
            modelSelect.innerHTML = ''; // Clear existing options
            
            if (data.error) {
                console.error('Error fetching models:', data.error);
                addDefaultModelOption(modelSelect);
                return;
            }
            
            // Add models to dropdown
            if (data.models && data.models.length > 0) {
                data.models.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model;
                    option.textContent = model;
                    modelSelect.appendChild(option);
                });
                
                // Set default selection (prefer smaller models for evaluation)
                const preferredModels = [
                    'qwen2.5:3b', 'gemma:2b', 'llama2:7b', 'mistral:7b', 'phi3:3b'
                ];
                
                for (const modelName of preferredModels) {
                    if (data.models.includes(modelName)) {
                        modelSelect.value = modelName;
                        break;
                    }
                }
            } else {
                addDefaultModelOption(modelSelect);
            }
        })
        .catch(error => {
            console.error('Error fetching models:', error);
            const modelSelect = document.getElementById('model-select');
            addDefaultModelOption(modelSelect);
        });
}

// Helper function to add default model option
function addDefaultModelOption(selectElement) {
    selectElement.innerHTML = ''; // Clear existing options
    const option = document.createElement('option');
    option.value = 'qwen2.5:3b';
    option.textContent = 'qwen2.5:3b (default)';
    selectElement.appendChild(option);
}
