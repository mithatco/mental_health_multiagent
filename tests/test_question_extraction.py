import sys
import os
from pathlib import Path

# Add the parent directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.document_processor import extract_questions_from_text

def test_simple_questions():
    text = """
    1. How have you been feeling lately?
    2. Do you have trouble sleeping at night?
    3. Have you experienced changes in your appetite?
    """
    questions = extract_questions_from_text(text)
    assert len(questions) == 3
    assert all('?' in q for q in questions)
    assert "How have you been feeling lately?" in questions
    
def test_multiline_questions():
    text = """
    1. Do you experience pain in various parts of your body
    (head, back, joints, abdomen, legs)?
    
    2. Do you have trouble with memory, such as remembering
    appointments or finding your way home?
    
    3. Have you used any substances like alcohol, tobacco, prescription medications 
    (painkillers like Vicodin, stimulants like Ritalin or Adderall, sedatives like sleeping pills or Valium), 
    or drugs such as marijuana, cocaine, crack, ecstasy, hallucinogens, heroin, inhalants, or methamphetamine?
    """
    questions = extract_questions_from_text(text)
    
    assert len(questions) == 3
    assert "Do you experience pain in various parts of your body (head, back, joints, abdomen, legs)?" in questions
    assert "Do you have trouble with memory, such as remembering appointments or finding your way home?" in questions
    assert any("Have you used any substances" in q and "marijuana" in q for q in questions)

def test_question_fragments():
    text = """
    Do you have difficulty:
    - Concentrating on things?
    - Making decisions?
    - Remembering information?
    """
    questions = extract_questions_from_text(text)
    # Should ideally combine these into proper questions
    print(questions)  # For debugging
    assert len(questions) > 0
    
if __name__ == "__main__":
    test_simple_questions()
    test_multiline_questions()
    test_question_fragments()
    print("All tests passed!")
