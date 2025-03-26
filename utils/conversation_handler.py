import json
import time
from typing import List, Dict, Any
from collections import defaultdict

from .rag_evaluator import RAGEvaluator

class ConversationHandler:
    def __init__(self, assistant, patient, state_file=None):
        """
        Initialize the conversation handler.
        
        Args:
            assistant (MentalHealthAssistant): The mental health assistant agent
            patient (Patient): The patient agent
            state_file: Optional path to a file to write conversation state to (for API mode)
        """
        self.assistant = assistant
        self.patient = patient
        self.conversation_log = []
        self.state_file = state_file
        self.rag_metrics = {
            "total_queries": 0,
            "avg_impact_score": 0.0,
            "document_usage": {}
        }
        
        # Initialize RAG evaluator if deepeval is available
        try:
            from .rag_evaluator import RAGEvaluator
            self.rag_evaluator = RAGEvaluator()
            self.has_evaluator = self.rag_evaluator.is_available
            
            if not self.has_evaluator:
                print("RAG evaluation disabled: deepeval library is not available in this environment")
        except ImportError:
            print("RAG evaluation disabled: couldn't import RAGEvaluator")
            self.has_evaluator = False
            self.rag_evaluator = None
        
        # Add rag_evaluation_results to track metrics
        self.rag_evaluation_results = []
    
    def run(self, disable_output: bool = False) -> str:
        """
        Run the conversation between the mental health assistant and patient.
        
        Args:
            disable_output: Whether to disable printing to console
            
        Returns:
            str: Final diagnosis from the mental health assistant
        """
        if not disable_output:
            print("=== Starting Mental Health Assessment ===\n")
        
        # Start with the assistant's first question
        if not disable_output:
            print("[DEBUG] Getting first message from assistant")
        question = self.assistant.get_next_message()
        
        # Check if the response is a dictionary with RAG usage info
        if isinstance(question, dict) and "content" in question:
            question_content = question["content"]
            rag_usage = question.get("rag_usage", {})
            
            # Enhanced RAG metrics handling
            if rag_usage:
                self._update_rag_metrics(question_content, rag_usage)
            
            # Add to conversation log with RAG info
            self.conversation_log.append({
                "role": "assistant",
                "content": question_content,
                "rag_usage": rag_usage
            })
            
            if not disable_output:
                print(f"Assistant: {question_content}")
            
            # Display enhanced RAG usage info if available
            if rag_usage and rag_usage.get("documents", []):
                documents = rag_usage.get("documents", [])
                stats = rag_usage.get("stats", {})
                if not disable_output and documents:
                    print(f"\n[RAG Engine accessed {len(documents)} documents, avg relevance: {stats.get('avg_score', 0):.2f}]")
                    for i, doc in enumerate(documents[:3]):  # Limit display to top 3
                        print(f"  - Doc {i+1}: {doc.get('title', 'Unknown')} (score: {doc.get('score', 0):.2f})")
                        if 'highlight' in doc:
                            print(f"     Highlight: \"{doc['highlight']}\"")
                
            question = question_content  # Use just the content for patient interaction
        else:
            # Legacy format support
            self.conversation_log.append({
                "role": "assistant",
                "content": question
            })
            if not disable_output:
                print(f"Assistant: {question}")
        
        # Track if we've gone through all questions
        all_questions_asked = False
        
        # Check if the question is incomplete (missing a question mark or only fragments)
        if not self._is_complete_question(question):
            if not disable_output:
                print("[DEBUG] Detected incomplete question. Fixing...")
            question = self._fix_incomplete_question(question, self.assistant.current_question_idx - 1, disable_output)
        
        while True:
            # If we've already gone through all questions and received the patient's last response,
            # it's time to generate the diagnosis
            if all_questions_asked:
                if not disable_output:
                    print("[DEBUG] All questions have been asked. Requesting final diagnosis...")
                # Force the assistant to generate a diagnosis
                diagnosis = self.assistant.generate_diagnosis()
                
                # Check if the diagnosis is a dictionary with RAG usage info
                if isinstance(diagnosis, dict) and "content" in diagnosis:
                    diagnosis_content = diagnosis["content"]
                    rag_usage = diagnosis.get("rag_usage", {})
                    
                    # Enhanced RAG metrics handling for diagnosis
                    if rag_usage:
                        self._update_rag_metrics(diagnosis_content, rag_usage)
                    
                    # Add to conversation log with RAG info
                    self.conversation_log.append({
                        "role": "assistant",
                        "content": diagnosis_content,
                        "rag_usage": rag_usage
                    })
                    
                    if not disable_output:
                        print(f"\n=== Final Diagnosis ===\n")
                        print(f"Assistant: {diagnosis_content}")
                        
                        # Display enhanced RAG usage for diagnosis
                        if rag_usage and rag_usage.get("documents", []):
                            documents = rag_usage.get("documents", [])
                            stats = rag_usage.get("stats", {})
                            impact = rag_usage.get("impact", {})
                            
                            print(f"\n[Diagnosis used {len(documents)} reference documents]")
                            print(f"[RAG impact score: {impact.get('impact_score', 0):.2f}]")
                            for i, doc in enumerate(documents[:3]):  # Limit display to top 3
                                print(f"  - Doc {i+1}: {doc.get('title', 'Unknown')} (score: {doc.get('score', 0):.2f})")
                    
                    # Add RAG summary to the state file
                    self._add_rag_summary_to_state()
                    
                    return diagnosis_content
                else:
                    # Legacy format support
                    self.conversation_log.append({
                        "role": "assistant", 
                        "content": diagnosis
                    })
                    if not disable_output:
                        print(f"\n=== Final Diagnosis ===\n")
                        print(f"Assistant: {diagnosis}")
                    return diagnosis
            
            # Get patient's response to the current question
            patient_response = self.patient.respond_to_question(question)
            self.conversation_log.append({
                "role": "patient",
                "content": patient_response
            })
            
            if not disable_output:
                print(f"Patient: {patient_response}")
            
            # Small delay for more natural conversation flow
            time.sleep(1)
            
            # Check if we've asked the last question
            if self.assistant.current_question_idx >= len(self.assistant.questions):
                if not disable_output:
                    print("[DEBUG] Just asked the final question. Next step will be diagnosis.")
                all_questions_asked = True
            
            if not disable_output:
                print("[DEBUG] Getting next message from assistant")
            # Get next question from assistant
            question = self.assistant.get_next_message(patient_response)
            
            # Check if the question is incomplete
            if not all_questions_asked and not self._is_complete_question(question):
                if not disable_output:
                    print("[DEBUG] Detected incomplete question. Fixing...")
                question = self._fix_incomplete_question(question, self.assistant.current_question_idx - 1, disable_output)
            
            # Check if the response is a dictionary with RAG usage info
            if isinstance(question, dict) and "content" in question:
                question_content = question["content"]
                rag_usage = question.get("rag_usage", {})
                
                # Enhanced RAG metrics handling
                if rag_usage:
                    self._update_rag_metrics(question_content, rag_usage)
                
                if not disable_output:
                    print("[DEBUG] Assistant response includes RAG info")
                
                # Add to conversation log with RAG info
                self.conversation_log.append({
                    "role": "assistant",
                    "content": question_content,
                    "rag_usage": rag_usage
                })
                
                # Display response to user
                if not disable_output:
                    print(f"Assistant: {question_content}")
                
                # Display enhanced RAG usage info if available
                if rag_usage and rag_usage.get("documents", []):
                    documents = rag_usage.get("documents", [])
                    stats = rag_usage.get("stats", {})
                    impact = rag_usage.get("impact", {})
                    
                    if not disable_output and documents:
                        print(f"\n[RAG Engine accessed {len(documents)} documents, avg relevance: {stats.get('avg_score', 0):.2f}]")
                        if impact:
                            print(f"[RAG impact score: {impact.get('impact_score', 0):.2f}]")
                        for i, doc in enumerate(documents[:3]):  # Limit display to top 3
                            print(f"  - Doc {i+1}: {doc.get('title', 'Unknown')} (score: {doc.get('score', 0):.2f})")
                            if 'highlight' in doc:
                                print(f"     Highlight: \"{doc['highlight']}\"")
                
                # If this is the diagnosis, return it
                if all_questions_asked:
                    # Add RAG summary to the state file
                    self._add_rag_summary_to_state()
                    return question_content
                
                question = question_content  # Use just the content for next iteration
            else:
                # Legacy format support
                if not disable_output:
                    print("[DEBUG] Assistant response does NOT include RAG info")
                self.conversation_log.append({
                    "role": "assistant",
                    "content": question
                })
                if not disable_output:
                    print(f"Assistant: {question}")
                
                # If this is the diagnosis, return it
                if all_questions_asked:
                    return question

            # Update state file if provided
            self._update_state_file()
    
    def _update_rag_metrics(self, response: str, rag_usage: Dict[str, Any]) -> None:
        """Update RAG metrics based on usage data."""
        documents = rag_usage.get("documents", [])
        impact = rag_usage.get("impact", {})
        
        if documents:
            # Update total queries
            self.rag_metrics["total_queries"] += 1
            
            # Update impact score
            impact_score = impact.get("impact_score", 0)
            current_avg = self.rag_metrics["avg_impact_score"]
            total_queries = self.rag_metrics["total_queries"]
            
            # Calculate running average
            self.rag_metrics["avg_impact_score"] = ((current_avg * (total_queries - 1)) + impact_score) / total_queries
            
            # Update document usage stats
            for doc in documents:
                doc_title = doc.get("title", "Unknown")
                if doc_title in self.rag_metrics["document_usage"]:
                    self.rag_metrics["document_usage"][doc_title] += 1
                else:
                    self.rag_metrics["document_usage"][doc_title] = 1
        
        # Add RAG evaluation if deepeval is available and we have enough context
        if self.has_evaluator and len(self.conversation_log) >= 2:
            try:
                documents = rag_usage.get("documents", [])
                
                if documents:
                    # Get the last user query (input to RAG)
                    last_user_message = None
                    for msg in reversed(self.conversation_log):
                        if msg.get("role") == "patient":  # Note: changed from "user" to "patient"
                            last_user_message = msg.get("content")
                            break
                    
                    if last_user_message:
                        # Extract context content from documents
                        context = []
                        for doc in documents:
                            if "content" in doc:
                                context.append(doc["content"])
                            elif "highlight" in doc:
                                context.append(doc["highlight"])
                        
                        # Skip evaluation if we don't have enough context
                        if not context:
                            return
                        
                        # Evaluate RAG performance
                        eval_results = self.rag_evaluator.evaluate_rag(
                            query=last_user_message,
                            response=response,
                            context=context
                        )
                        
                        # Handle case where evaluation returns an error
                        if "error" in eval_results:
                            print(f"RAG Evaluation error: {eval_results['error']}")
                            # Don't stop the conversation, just log the error and continue
                            
                            # If we still have metrics despite the error, we'll use those
                            if "metrics" not in eval_results or not eval_results["metrics"]:
                                return
                        
                        # Store evaluation results
                        self.rag_evaluation_results.append(eval_results)
                        
                        # Add evaluation results to the rag_usage to be included in logs
                        rag_usage["evaluation"] = {
                            "contextual_relevancy": eval_results.get("metrics", {}).get("contextual_relevancy", {}),
                            "faithfulness": eval_results.get("metrics", {}).get("faithfulness", {}),
                            "answer_relevancy": eval_results.get("metrics", {}).get("answer_relevancy", {}),
                            "average_score": eval_results.get("average_score", 0)
                        }
                        
                        # Print summary of evaluation results
                        print(f"[RAG Evaluation] Average Score: {eval_results.get('average_score', 'N/A')}")
                        for metric_name, metric_results in eval_results.get("metrics", {}).items():
                            print(f"[RAG Evaluation] {metric_name}: {metric_results.get('score', 'N/A')}")
            except Exception as e:
                print(f"Error during RAG evaluation: {str(e)}")
                # Continue with the conversation even if evaluation fails
    
    def _add_rag_summary_to_state(self) -> None:
        """Add RAG usage summary to state file if it exists."""
        if not self.state_file:
            return
            
        try:
            # Load existing state
            with open(self.state_file, 'r') as f:
                state = json.load(f)
                
            # Add RAG metrics
            state["rag_metrics"] = {
                "total_queries": self.rag_metrics["total_queries"],
                "avg_impact_score": round(self.rag_metrics["avg_impact_score"], 4),
                "top_documents": dict(sorted(
                    self.rag_metrics["document_usage"].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5])  # Top 5 documents
            }
            
            # Write updated state
            with open(self.state_file, 'w') as f:
                json.dump(state, f)
        except Exception as e:
            print(f"Error updating RAG metrics in state file: {str(e)}")
    
    def _is_complete_question(self, text):
        """Check if a question appears complete."""
        if not text:
            return False
            
        # Questions with complete sentences typically end with these punctuation marks
        if text.strip().endswith('?'):
            return True
            
        # Check if it's at least a complete sentence
        if text.strip().endswith('.') or text.strip().endswith('!'):
            # Make sure it has a reasonable length
            words = text.split()
            return len(words) >= 3
            
        # Check for fragments that are clearly incomplete
        if text.strip().startswith(', ') or text.strip().startswith('or '):
            return False
            
        # If it has reasonable length and structure, consider it complete
        words = text.split()
        return len(words) >= 5
    
    def _fix_incomplete_question(self, text, question_idx, disable_output=False):
        """Fix an incomplete question by referencing the original questionnaire."""
        # If question index is valid, get the original question
        if 0 <= question_idx < len(self.assistant.questions):
            original = self.assistant.questions[question_idx]
            if not disable_output:
                print(f"[DEBUG] Using original question from questionnaire: {original}")
            return original
            
        # If we can't get the original, try to make the fragment more question-like
        if text.strip().startswith(', ') or text.strip().startswith('or '):
            fixed = f"Do you experience {text.strip()}?"
            if not disable_output:
                print(f"[DEBUG] Fixed fragment to: {fixed}")
            return fixed
            
        # Add a question mark if it's missing
        if not text.strip().endswith('?'):
            fixed = f"{text.strip()}?"
            if not disable_output:
                print(f"[DEBUG] Added question mark: {fixed}")
            return fixed
            
        return text
    
    def get_conversation_log(self) -> List[Dict[str, str]]:
        """
        Get the conversation log.
        
        Returns:
            List of conversation messages
        """
        return self.conversation_log

    def get_rag_metrics(self) -> Dict[str, Any]:
        """
        Get RAG usage metrics for the conversation.
        
        Returns:
            Dictionary of RAG metrics
        """
        metrics = {
            "total_queries": self.rag_metrics["total_queries"],
            "avg_impact_score": round(self.rag_metrics["avg_impact_score"], 4),
            "top_documents": dict(sorted(
                self.rag_metrics["document_usage"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5])  # Top 5 documents
        }
        
        # Add deepeval metrics if available
        if self.rag_evaluation_results:
            # Calculate average scores across all evaluations
            eval_totals = defaultdict(list)
            for result in self.rag_evaluation_results:
                for metric_name, metric_data in result.get("metrics", {}).items():
                    if "score" in metric_data:
                        eval_totals[metric_name].append(metric_data["score"])
            
            # Add average scores to metrics
            metrics["deepeval_metrics"] = {
                metric: {
                    "avg_score": round(sum(scores) / len(scores), 4),
                    "evaluations": len(scores)
                }
                for metric, scores in eval_totals.items()
            }
            
        return metrics

    def _update_state_file(self):
        """Update the state file with the current conversation if provided."""
        if not self.state_file:
            return
        
        try:
            # Prepare state data
            state_data = {
                "conversation": self.conversation_log,
                "status": "in_progress",
                "timestamp": time.time()
            }
            
            # Write to file
            with open(self.state_file, 'w') as f:
                json.dump(state_data, f)
        except Exception as e:
            print(f"Error updating state file: {str(e)}")
