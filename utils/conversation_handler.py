import time

class ConversationHandler:
    def __init__(self, assistant, patient):
        """
        Initialize the conversation handler.
        
        Args:
            assistant (MentalHealthAssistant): The mental health assistant agent
            patient (Patient): The patient agent
        """
        self.assistant = assistant
        self.patient = patient
    
    def run(self):
        """
        Run the conversation between the mental health assistant and patient.
        
        Returns:
            str: Final diagnosis from the mental health assistant
        """
        print("=== Starting Mental Health Assessment ===\n")
        
        # Start with the assistant's first question
        question = self.assistant.get_next_message()
        
        while True:
            print(f"Assistant: {question}")
            
            # Check if this is the diagnosis (end of conversation)
            if "diagnosis" in question.lower() or len(self.assistant.questions) < self.assistant.current_question_idx:
                return question
            
            # Get patient's response
            patient_response = self.patient.respond_to_question(question)
            print(f"Patient: {patient_response}")
            
            # Small delay for more natural conversation flow
            time.sleep(1)
            
            # Get next question from assistant
            question = self.assistant.get_next_message(patient_response)
