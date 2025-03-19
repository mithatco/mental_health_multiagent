"""
Utility to check method parameters for compatibility issues.
"""
import inspect
import sys
import os
import logging
from typing import Callable, Set, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Add the parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_method_parameters(method: Callable, expected_params: Set[str]) -> Dict[str, Any]:
    """
    Check if a method accepts the expected parameters.
    
    Args:
        method: The method to check
        expected_params: Set of parameter names to check
        
    Returns:
        Dictionary with results
    """
    signature = inspect.signature(method)
    method_params = set(signature.parameters.keys())
    
    # Check for missing parameters
    missing_params = expected_params - method_params
    
    # Check for parameters with default values
    default_params = {
        name for name, param in signature.parameters.items()
        if param.default != inspect.Parameter.empty
    }
    
    return {
        "method_name": method.__qualname__,
        "accepts_all_params": len(missing_params) == 0,
        "missing_params": missing_params,
        "all_params": method_params,
        "has_var_kwargs": any(param.kind == inspect.Parameter.VAR_KEYWORD 
                             for param in signature.parameters.values()),
        "default_params": default_params
    }

def check_critical_methods():
    """Check parameters for critical methods in our application."""
    # Import the necessary classes
    from utils.conversation_handler import ConversationHandler
    from utils.batch_runner import BatchRunner
    from main import run_conversation
    
    # Check ConversationHandler.run
    conversation_handler_run = ConversationHandler.run
    conversation_run_check = check_method_parameters(
        conversation_handler_run, 
        {"disable_output"}
    )
    
    logger.info(f"ConversationHandler.run check: {conversation_run_check}")
    if not conversation_run_check["accepts_all_params"] and not conversation_run_check["has_var_kwargs"]:
        logger.warning("⚠️ ConversationHandler.run does not accept 'disable_output' parameter!")
        logger.warning("Fix: Update ConversationHandler.run to include this parameter.")
    else:
        logger.info("✅ ConversationHandler.run properly accepts 'disable_output' parameter")
    
    # Check run_conversation
    run_conv_check = check_method_parameters(
        run_conversation,
        {"pdf_path", "patient_profile", "assistant_model", "patient_model", 
         "disable_output", "logs_dir", "log_filename", "refresh_cache"}
    )
    
    logger.info(f"run_conversation check: {run_conv_check}")
    if not run_conv_check["accepts_all_params"] and not run_conv_check["has_var_kwargs"]:
        missing = run_conv_check["missing_params"]
        logger.warning(f"⚠️ run_conversation does not accept parameters: {missing}")
    else:
        logger.info("✅ run_conversation accepts all expected parameters")
    
    # Check BatchRunner.run_batch
    batch_runner_run = BatchRunner.run_batch
    batch_run_check = check_method_parameters(
        batch_runner_run,
        {"batch_size", "pdf_path", "patient_profile", "randomize_profiles"}
    )
    
    logger.info(f"BatchRunner.run_batch check: {batch_run_check}")
    if not batch_run_check["accepts_all_params"] and not batch_run_check["has_var_kwargs"]:
        missing = batch_run_check["missing_params"]
        logger.warning(f"⚠️ BatchRunner.run_batch does not accept parameters: {missing}")
    else:
        logger.info("✅ BatchRunner.run_batch accepts all expected parameters")
        
    return (conversation_run_check, run_conv_check, batch_run_check)

if __name__ == "__main__":
    logger.info("Checking method parameters for compatibility issues...")
    check_critical_methods()
