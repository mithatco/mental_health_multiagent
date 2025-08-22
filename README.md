# Mental Health Multi-Agent System

A research framework accompanying the CIKM 2025 paper:

**Trustworthy AI Psychotherapy: Multi-Agent LLM Workflow for Counseling and Explainable Mental Disorder Diagnosis**

This repository provides the source code, datasets, and evaluation resources to reproduce the results from the paper.

## Datasets

The `datasets/` directory contains the evaluation dataset used in the CIKM 2025 paper:

- **`datasets/chat_logs/`**: Contains conversation logs from the evaluation experiments, organized by different LLM providers and models
- **`datasets/profiles/`**: Client profiles used to simulate various psychiatric conditions during evaluation
- **`datasets/documents/`**: Reference documents and questionnaires used in the evaluation process

This dataset enables reproduction of the paper's results and provides a foundation for further research in AI-assisted mental health assessment.

---

## Table of Contents

- [System Overview](#system-overview)
- [Architecture Diagram](#architecture-diagram)
- [Key Components](#key-components)
- [Installation](#installation)
- [Usage](#usage)
  - [Basic Usage](#basic-usage)
  - [Command-Line Options](#command-line-options)
  - [Chat Log Viewer](#chat-log-viewer)
  - [Batch Processing](#batch-processing)
  - [Viewing Saved Conversations](#viewing-saved-conversations)
  - [Analyzing Batch Results](#analyzing-batch-results)
- [Directory Structure](#directory-structure)
- [How It Works](#how-it-works)
- [Customization Options](#customization-options)
- [Evaluation System](#evaluation-system)
  - [Standard Metrics](#standard-metrics)
  - [Rubric-Based Evaluation](#rubric-based-evaluation)
  - [Running Evaluations](#running-evaluations)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)

## System Overview

The Mental Health Multi-Agent System is designed to simulate psychiatric assessments by creating a conversation between two specialized AI agents:

1. **Mental Health Assistant**: Uses clinical knowledge to ask questions, interpret responses, and provide a diagnosis based on a questionnaire
2. **Client**: Responds to questions based on a simulated psychiatric condition profile

The system can leverage large language models (LLMs) from multiple providers:
- **Ollama**: Run models locally through Ollama
- **Groq**: Access models through the cloud-based Groq API for faster inference

The system employs Retrieval-Augmented Generation (RAG) to enhance responses with domain knowledge and stores conversations for further analysis.

## Architecture Diagram

```mermaid
graph TD
    subgraph "User Inputs"
        A[Questionnaire PDF] --> D
        B[Client Profile] --> E
    end

    subgraph "Document Processing"
        D[Document Processor] --> D1[Extract Questions]
        D[Document Processor] --> D2[Vector Embedding]
        D2 --> V[Vector Store]
    end

    subgraph "Agent System"
        E[Client Agent] <--> F[Conversation Handler]
        G[Mental Health Assistant Agent] <--> F
        G <--> R[RAG Engine]
        R <--> V
    end

    subgraph "LLM Providers"
        F <--> O[Ollama - Local Models]
        F <--> GR[Groq - Cloud API]
        R <--> O
        R <--> GR
    end

    subgraph "Outputs"
        F --> L[Chat Logger]
        F --> I[Diagnosis]
        L --> S[Saved Conversations]
        S --> EV[Conversation Evaluator]
    end
```

## Key Components

### 1. Document Processing & RAG

- **PDF Processor**: Extracts questions from questionnaire PDFs
- **Document Processor**: Processes various document types (PDF, TXT, DOCX, JSON)
- **RAG Engine**: Provides relevant context from medical literature
- **Vector Store**: Stores document embeddings for semantic search

### 2. Agent System

- **Mental Health Assistant**: Conducts assessments using predefined questionnaires
- **Client**: Simulates responses based on selected psychiatric condition profiles
- **Conversation Handler**: Manages the interaction between agents

### 3. Infrastructure

- **LLM Client System**: Provides unified interface to multiple LLM providers
  - **Ollama Client**: Interfaces with locally deployed LLMs
  - **Groq Client**: Interfaces with Groq API for cloud-based inference
- **Chat Logger**: Records conversations and diagnoses
- **Profile System**: Manages different client profiles

### 4. Analysis Tools

- **Batch Processor**: Generates multiple conversations for research and analysis
- **Batch Analyzer**: Analyzes patterns and statistics across multiple conversations
- **Conversation Evaluator**: Assesses quality and clinical validity of conversations
- **PDF Debugger**: Helps diagnose issues with questionnaire extraction

## Installation

### Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.ai/download) installed and running
- Required Python packages (see requirements.txt)

### Setup Steps

1. Clone this repository or download the source code
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

3. Download required models with Ollama:

```bash
# For embeddings (required for RAG):
ollama pull nomic-embed-text

# For the agent models (choose based on your hardware):
ollama pull qwen3:4b
# or a larger model with better performance if your environment supports it
```

4. Create the necessary directories:

```bash
mkdir -p documents chat_logs cache
```

5. Add questionnaire PDFs to the `documents` directory

## Usage

This section covers everything you need to know to use the Mental Health Multi-Agent System effectively. The typical workflow is:

1. **Run conversations** using the basic usage commands
2. **View and analyze results** using the chat log viewer
3. **Generate multiple conversations** using batch processing for research or testing

### Basic Usage

Run the application with default settings (using Ollama):

```bash
python main.py
```

Using Groq instead of Ollama:

```bash
# Set Groq API key as environment variable
export GROQ_API_KEY=your_groq_api_key

# Run with Groq for both assistant and client
python main.py --assistant_provider groq --patient_provider groq --assistant_model llama3-70b-8192 --patient_model llama3-70b-8192
```

Or pass the API key directly:

```bash
python main.py --assistant_provider groq --patient_provider groq --groq_api_key your_groq_api_key --assistant_model llama3-70b-8192 --patient_model llama3-70b-8192
```

### Command-Line Options

#### Basic Options
```bash
# Use a specific questionnaire PDF
python main.py --pdf_path path/to/questionnaire.pdf

# Specify document directories
python main.py --docs_dir path/to/docs_folder
python main.py --questionnaires_dir path/to/questionnaires_folder

# Choose a specific client profile
python main.py --patient_profile depression

# Don't save the conversation log
python main.py --no-save

# Specify a different logs directory
python main.py --logs-dir path/to/logs_folder

# Refresh the document cache
python main.py --refresh_cache
```

#### LLM Provider Options
```bash
# LLM Provider selection
python main.py --assistant_provider groq --patient_provider ollama

# Use different models for the agents
python main.py --assistant_model qwen3:4b --patient_model qwen3:4b

# Using Groq models
python main.py --assistant_provider groq --assistant_model llama3-70b-8192

# Custom Ollama URL
python main.py --ollama_url http://localhost:11434

# API keys for cloud providers
python main.py --groq_api_key your_groq_api_key
python main.py --openai_api_key your_openai_api_key
```

#### Advanced Options
```bash
# Generate full conversation in a single LLM call
python main.py --full_conversation

# Interactive mode with real user
python main.py --interactive true

# Disable RAG completely
python main.py --disable-rag

# Disable only RAG evaluation (keep document retrieval)
python main.py --disable-rag-evaluation

# Skip loading SentenceTransformer models for faster startup
python main.py --skip-transformers

# API mode with state file
python main.py --state-file path/to/state.json
```

#### Batch Processing Options
```bash
# Generate multiple conversations (using --batch or -n)
python main.py --batch 5 --patient_profile depression
python main.py -n 10 --randomize-profiles

# Save batch to specific directory
python main.py --batch 3 --logs-dir ./my_batch_results
```

### Supported LLM Providers

The system currently supports three LLM providers:

1. **Ollama** (default)
   - Runs models locally on your machine
   - No API key required
   - Configure with `--ollama_url` (default: http://localhost:11434)

2. **Groq**
   - Cloud-based API with extremely fast inference
   - Requires API key (get one at https://console.groq.com/)
   - Set with `--groq_api_key` or environment variable `GROQ_API_KEY`

3. **OpenAI**
   - Access to GPT models through OpenAI API
   - Requires API key (get one at https://platform.openai.com/)
   - Set with `--openai_api_key` or environment variable `OPENAI_API_KEY`

You can mix providers, using different providers for each agent:

```bash
# Use Groq for assistant (higher quality) and Ollama for client (cost savings)
python main.py --assistant_provider groq --patient_provider ollama --assistant_model llama3-70b-8192 --patient_model qwen3:4b --groq_api_key your_groq_api_key

# Use OpenAI for assistant and Groq for patient
python main.py --assistant_provider openai --patient_provider groq --assistant_model gpt-4 --patient_model llama3-70b-8192 --openai_api_key your_openai_key --groq_api_key your_groq_key
```

### Batch Processing

The batch processing feature allows you to generate multiple conversations automatically, which is useful for research, testing, or creating datasets. See the "Batch Processing Options" section above for command examples.

### Chat Log Viewer

The project includes a web-based chat log viewer to easily browse, filter, and analyze your conversations. This is the primary way to view and analyze your saved conversations.

#### Running the Chat Viewer

To launch the chat viewer:

```bash
python chat_viewer.py
```

Additional options:
- `--port`: Specify the port to run the server on (default: 5000)
- `--logs-dir`: Directory containing chat logs (default: ./chat_logs)

#### Using the Chat Viewer

1. Open your browser and navigate to http://127.0.0.1:5000
2. You'll see the chat logs listed in the sidebar
3. Use the filters to narrow down the log list:
   - Select a client profile from the dropdown
   - Choose a date range
   - Enter search terms
4. Click on any log to view the full conversation
5. The diagnosis is shown in a separate panel at the bottom
6. Use the "Export as Text" button to download a text version of the conversation
7. Click "Evaluate with Ollama" to run an automated evaluation of the conversation

### Viewing Saved Conversations

For quick command-line viewing, you can also use:

```bash
python view_chats.py --list
```

### Analyzing Batch Results

After generating a batch of conversations, you can analyze the results to find patterns:

```bash
python analyze_batch.py chat_logs/batch_20231215_120000
```

This will provide:
- Profile distribution statistics
- Analysis of diagnoses across conversations 
- Correlation between client profiles and diagnoses
- Conversation statistics (length, duration, etc.)

## Directory Structure

```
mental_health_multiagent/
├── main.py                       # Main application entry point
├── agents/                       # Agent implementations
│   ├── mental_health_assistant.py
│   └── patient.py
├── utils/                        # Utility modules
│   ├── conversation_handler.py   # Handles agent interactions
│   ├── chat_logger.py            # Conversation logging
│   ├── document_processor.py     # Document handling
│   ├── llm_client_base.py        # Base LLM client interface
│   ├── ollama_client.py          # Ollama LLM interface
│   ├── groq_client.py            # Groq API interface
│   ├── pdf_processor.py          # PDF extraction
│   ├── rag_engine.py             # RAG functionality
│   ├── vector_store.py           # Vector database
│   ├── batch_processor.py        # Batch conversation processing
│   ├── chat_evaluator.py         # Conversation evaluation
│   ├── ragas_evaluation.py       # Ragas-based evaluation
│   ├── ollama_evaluation.py      # Fallback evaluator
├── prompts/                      # System prompts
│   ├── mental_health_assistant_prompt.txt
│   └── patient_prompt.txt
├── profiles/                     # Client profiles
│   ├── anxiety.txt
│   ├── bipolar.txt
│   ├── depression.txt
│   ├── ptsd.txt
│   └── schizophrenia.txt
├── rubrics/                      # Evaluation rubrics
│   ├── rubrics.py
├── documents/                    # Reference documents for knowledge retrieval
│   └── questionnaires/           # Questionnaires used for assessment
├── interface/                    # Components for the chat log viewer
├── chat_logs/                    # Saved conversations
├── cache/                        # Embedding cache
├── analyze_batch.py              # Batch analysis tool
├── debug_pdf.py                  # PDF debugging tool
├── create_profile.py             # Client profile creation tool
└── chat_viewer.py                # Chat log viewer entry point
```

## How It Works

The system functions through the following workflow:

1. **Questionnaire Loading**
   - The system loads one or more questionnaires from PDF files
   - Questions are extracted using text analysis
   - Questions become the structure for the conversation

2. **Client Profile Selection**
   - A profile is selected that defines the client's psychiatric condition
   - The profile contains symptoms, history, and response patterns
   - This determines how the client agent will respond to questions

3. **Document Processing**
   - All documents are processed and split into chunks
   - Embeddings are created for each chunk
   - A vector store allows semantic search for relevant information

4. **Conversation Cycle**
   - The assistant asks questions from the questionnaire
   - The client responds based on their profile
   - The conversation handler manages this exchange

5. **RAG Enhancement**
   - The RAG engine retrieves relevant information from documents:
     1. Client responses are summarized into clinical observations using AI
     2. These observations form the basis for a focused RAG query
     3. Documents are retrieved based on semantic similarity to this query
     4. Retrieved documents' content is provided to the assistant for diagnosis generation
   - This improves the quality and relevance of the diagnosis

6. **Diagnosis Generation**
   - After all questions are asked, the assistant generates a diagnosis
   - The diagnosis considers all client responses
   - RAG provides additional clinical context

7. **Conversation Logging**
   - The entire conversation is saved as JSON and plain text
   - Includes metadata about models and profiles used
   - **Enhanced RAG Metrics**:
     - Document access counts and relevance scores
     - Per-document highest and average relevance scores 
     - Example excerpts showing why documents were retrieved
   - Enables review and analysis of past sessions

8. **Conversation Evaluation**
   - The system can evaluate the quality of conversations
   - Provides metrics on clinical accuracy, empathy, and therapeutic approach
   - Offers quantitative insights into conversation effectiveness

## Customization Options

### Adding New Client Profiles

Create a new profile file in the `profiles` directory following this format:

```
You are roleplaying as a client with [DISORDER] seeking mental health assessment. You have the following characteristics:

- [SYMPTOM 1]
- [SYMPTOM 2]
- ...

When answering questions:
- [BEHAVIOR 1]
- [BEHAVIOR 2]
- ...
```

Alternatively, use the provided script:

```bash
python create_profile.py --interactive
```

### Adding Reference Documents and Questionnaires

The system now clearly separates two types of documents:

1. **Questionnaires**: Place assessment questionnaires in the `documents/questionnaires/` directory. These are used to structure the conversation.

2. **Reference Materials**: Place clinical reference materials in the main `documents/` directory. These are used by the RAG system to enhance the assistant's knowledge and improve diagnostic accuracy. Good examples include:
   - DSM-5 excerpts
   - Clinical guidelines
   - Research papers
   - Treatment protocols

This separation makes it easier to manage your documents and ensures that only questionnaires appear in the selection menu when starting an assessment.

### Modifying System Prompts

Edit the files in the `prompts` directory to change the behavior of the agents:
- `mental_health_assistant_prompt.txt`: Controls the assistant's approach
- `patient_prompt.txt`: Default client behavior (if no profile is selected)

### Customizing Evaluation Rubrics

The system uses customizable rubrics for evaluating conversation quality, stored in `utils/rubrics.py`. You can modify existing rubrics or add new ones to match your specific evaluation needs.

## Evaluation System

The Mental Health Multi-Agent System includes a sophisticated conversation evaluation capability that analyzes the quality, clinical accuracy, and therapeutic approach of the mental health assistant's responses.

### Standard Metrics

The evaluation system can provide several standard metrics when context information is available:

1. **Answer/Response Relevancy** - Measures how directly the assistant's responses address the client's questions or statements
2. **Faithfulness** - Assesses whether the assistant's responses contain information that is supported by the available context
3. **Context Precision** - Evaluates how relevant the context information is to the client's questions
4. **Context Recall** - Measures how effectively the assistant incorporates relevant context information in their responses

### Rubric-Based Evaluation

The system also performs specialized mental health evaluation using clinical rubrics:

1. **Empathy & Rapport** - Evaluates how well the responses demonstrate empathy and build rapport with the client
2. **Clinical Accuracy** - Assesses the clinical accuracy and appropriateness of responses
3. **Therapeutic Approach** - Evaluates the appropriate use of therapeutic techniques 
4. **Safety & Risk Assessment** - Measures how well responses address safety concerns or risk factors
5. **Communication Clarity** - Evaluates the clarity and accessibility of language used

Each rubric uses a 5-point scale with detailed criteria for each level, allowing for nuanced assessment of conversation quality.

### Running Evaluations

You can evaluate conversations in several ways:

1. Launch the chat viewer: `python chat_viewer.py`
2. Open a conversation from the sidebar
3. Select an evaluation model from the dropdown menu
   - The system will automatically detect all available Ollama models on your system
   - Smaller models (like Qwen 3B or Gemma 2B) are preferred for faster evaluation
   - Larger models may provide more detailed analysis but take longer to process
4. Click "Evaluate with Ollama" to begin the evaluation
5. Results will appear in the evaluation panel showing:
   - Overall scores across different evaluation dimensions
   - Detailed explanations for each score
   - Diagnosis accuracy assessment (when applicable)
   - Overall assessment of the conversation quality

### Customizing the Evaluation Model

The evaluation system allows you to choose which locally installed Ollama model to use:

1. Install models via Ollama CLI: `ollama pull model_name`
2. They will automatically appear in the model selection dropdown

If no specific model is selected, the system will default to using qwen2.5:3b for evaluation.

## Advanced Features

### Batch Generation and Analysis

The batch processing system allows you to:

1. Generate multiple conversations automatically
2. Use the same client profile for all conversations or randomize profiles
3. Save all conversations with metadata for analysis
4. Generate statistical analyses across conversations
5. Compare diagnoses between different client profiles
6. Identify patterns in the AI's diagnostic approach

This is especially useful for:
- Evaluating the consistency of the mental health assistant
- Creating datasets for research or training
- Testing system performance with different profiles
- Analyzing how different symptoms lead to different diagnoses

### Embedding Model Selection

The system uses `nomic-embed-text` by default for embeddings, but you can modify the `vector_store.py` file to use a different model.

### Custom Model Integration

The system's modular LLM client architecture allows you to easily add support for new language model providers. To integrate a new provider:

1. Create a new client class inheriting from `LLMClientBase`
2. Implement the required methods (`generate_response`, `get_available_models`, etc.)
3. Register the client in the LLM evaluator factory
4. Add command-line arguments for your new provider

### RAG System Customization

Advanced users can customize the RAG (Retrieval-Augmented Generation) system:

- **Custom Embedding Models**: Modify `vector_store.py` to use different embedding models
- **Chunk Size Optimization**: Adjust document chunking parameters for your specific documents
- **Retrieval Strategies**: Customize how relevant documents are selected and ranked
- **Context Integration**: Modify how retrieved context is integrated into agent responses

## Troubleshooting

### No Questions Extracted from PDF

- Ensure the PDF contains searchable text (not scanned images)
- Check that questions end with question marks
- Try using the `debug_pdf.py` tool to examine the PDF
- Consider creating a text file with questions instead

#### Debugging PDF Extraction

If you're having trouble with PDF question extraction, use the debugging tool:

```bash
python debug_pdf.py path/to/your/questionnaire.pdf
```

This will analyze the PDF and show what text and questions are being extracted.

#### Debugging Question Extraction

If you're having issues with questionnaire parsing or want to verify how questions are being extracted from your documents, use the `debug_question_extraction.py` utility:

```bash
python -m utils.debug_question_extraction path/to/your/questionnaire.pdf
```

This will display:
1. Basic document information
2. The first 200 characters of content
3. A list of all extracted questions
4. If no questions were found, the full document content for manual inspection

Example output:

```
Document: documents/questionnaires/dsm5.pdf
Content length: 9843 characters
First 200 characters of content:
DSM-5 Self-Rated Level 1 Cross-Cutting Symptom Measure—Adult

Instructions

For each question below, circle the number that best describes how much (or how often) you have been bothered by each proble

Extracted 23 questions:
1. Little interest or pleasure in doing things?
2. Feeling down, depressed, or hopeless?
...
23. Using any of the following medicines ON YOUR OWN (without a doctor's prescription) in greater amounts or longer than prescribed (e.g., painkillers like Vicodin, stimulants like Ritalin or Adderall, sedatives like sleeping pills or Valium, or drugs such as marijuana, cocaine, crack, ecstasy, hallucinogens, heroin, inhalants, or methamphetamine)?
```

#### Troubleshooting Question Extraction

If questions are not being extracted properly:

1. Check if the document is properly formatted with clear question marks
2. Try preprocessing the document (e.g., convert PDF to text)
3. For multi-line questions, ensure they follow standard formatting
4. Run the test suite to verify the extraction logic: `python -m tests.test_question_extraction`

### Embedding Errors

- Make sure Ollama is running (`ollama serve`)
- Check that the embedding model is installed (`ollama pull nomic-embed-text`)
- Try using a different embedding model by modifying `vector_store.py`

### Poor Quality Responses

- Try a larger LLM model
- Add more detailed client profiles
- Provide more comprehensive reference materials in the documents directory
- Adjust the system prompts to be more specific

### Out of Memory Errors

- Use smaller models
- Reduce chunk size in `vector_store.py`
- Limit the number of documents processed

## Understanding RAG Logs

The system now produces comprehensive RAG metrics in conversation logs:

```json
"rag_summary": {
  "total_rag_queries": 5,
  "total_documents_accessed": 12,
  "documents_accessed": {
    "APA_DSM-5-Schizophrenia.pdf": {
      "access_count": 4,
      "highest_score": 0.7449,
      "average_score": 0.6823,
      "example_excerpt": "Schizophrenia The upcoming fifth edition of the Diagnostic and Statistical..."
    },
    "APA_DSM-5-Substance-Use-Disorder.pdf": {
      "access_count": 3,
      "highest_score": 0.7272,
      "average_score": 0.6853,
      "example_excerpt": "Substance-Related and Addictive Disorders In the fifth edition of..."
    }
  }
}
```

This format provides:
- Count of queries where RAG was used
- Total document access instances
- Per-document statistics:
  - Access frequency
  - Highest relevance score achieved
  - Average relevance score
  - Example text showing why the document was deemed relevant

During execution, the system will also display RAG information:

## License

This project is provided as open source software for research and educational purposes.