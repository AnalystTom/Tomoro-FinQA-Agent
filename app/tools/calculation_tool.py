# app/tools/calculation_tool.py
import logging
from asteval import Interpreter
import numbers # To check for numeric types

# Configure logging
logger = logging.getLogger(__name__)

CALCULATOR_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluates a mathematical expression using a safe Python interpreter (asteval). "
                       "Supports basic arithmetic (+, -, *, /), powers (**), parentheses, and common math "
                       "functions (e.g., sin, cos, sqrt, log, abs, max, min). "
                       "Numerical results are rounded to one decimal place. "
                       "Input should be a single string representing the mathematical expression.",
        "parameters": {
            "type": "object",
            "properties": {
                "math_expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate. For example, '2 + 2', 'sqrt(16) * (min(5, 3) + 1)'."
                }
            },
            "required": ["math_expression"]
        }
    }
}

def safe_calculate(math_expression: str) -> str:
    """
    Safely evaluates a mathematical expression string using asteval.
    Numerical results are rounded to one decimal place.

    Args:
        math_expression: The mathematical expression string to evaluate.

    Returns:
        A string representing the result of the calculation (rounded if numeric) or an error message.
    """
    if not isinstance(math_expression, str):
        return "Error: Input expression must be a string."
    
    if not math_expression.strip():
        return f"Error: invalid syntax in expression '{math_expression}'. Details: Expression is empty."

    aeval = Interpreter(err_writer=None, out_writer=None) 
    
    result_val: Any = None # To store the raw result from asteval
    try:
        result_val = aeval.eval(math_expression)
    except SyntaxError as se:
        logger.error(f"SyntaxError during asteval.eval for '{math_expression}': {se}", exc_info=True)
        if aeval.error: aeval.error = []
        return f"Error: invalid syntax in expression '{math_expression}'. Details: {str(se)}"
    except ZeroDivisionError:
        logger.error(f"ZeroDivisionError during asteval.eval for '{math_expression}'", exc_info=True)
        if aeval.error: aeval.error = []
        return "Error: division by zero"
    except Exception as e:
        logger.error(f"Exception during asteval.eval for '{math_expression}': {e}", exc_info=True)
        if aeval.error:
            secondary_errors = [str(err_report.get_error()[1]) for err_report in aeval.error]
            logger.warning(f"Additional asteval.error details for '{math_expression}' after exception: {secondary_errors}")
            aeval.error = []
        return f"Error: {str(e)}"

    if aeval.error:
        error_messages = []
        has_syntax_error_type = False
        
        for err_report in aeval.error:
            error_details = err_report.get_error() 
            etype = error_details[0]
            emsg = str(error_details[1]) 

            if etype == ZeroDivisionError:
                logger.warning(f"ZeroDivisionError reported via aeval.error for '{math_expression}'")
                aeval.error = [] 
                return "Error: division by zero" 
            
            error_messages.append(emsg)
            if etype == SyntaxError:
                has_syntax_error_type = True
        
        aeval.error = [] 

        if has_syntax_error_type:
            return f"Error: invalid syntax in expression '{math_expression}'. Details: {'; '.join(error_messages)}"
        elif error_messages: 
            return f"Error evaluating expression '{math_expression}': {'; '.join(error_messages)}"

    # If no exceptions were raised and no errors were reported in aeval.error,
    # process the result for rounding.
    if result_val is not None:
        if isinstance(result_val, numbers.Number) and not isinstance(result_val, bool): # Check if it's a number (and not a boolean)
            try:
                rounded_result = round(float(result_val), 1)
                # Ensure .0 is kept for whole numbers after rounding, e.g., 4.0 not 4
                # However, round(x, 1) for an int like 4 gives 4.0.
                # If result_val was int, round(result_val, 1) is float.
                # If result_val was float, round(result_val, 1) is float.
                # The goal is one decimal place.
                # If rounded_result is an integer (e.g. 4.0 becomes 4), format it to have .0
                if rounded_result == int(rounded_result):
                    return f"{int(rounded_result)}.0" 
                return str(rounded_result)
            except Exception as e:
                logger.error(f"Error rounding numeric result '{result_val}' for expression '{math_expression}': {e}", exc_info=True)
                return str(result_val) # Fallback to string of original if rounding fails
        else:
            # Not a number, or it's a boolean, return as string
            return str(result_val)
    else:
        # Should ideally not happen if no errors/exceptions, but as a safeguard
        logger.warning(f"Asteval returned None for expression '{math_expression}' without explicit errors.")
        return "Error: Calculation resulted in an undefined value."


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger.info("Testing safe_calculate with rounding:")
    
    expressions_to_test = [
        "1 + 1",                            # Expected: 2.0
        "10 / 4",                           # Expected: 2.5
        "10 / 3",                           # Expected: 3.3
        "0.1234 * 100",                     # Expected: 12.3
        "0.99 * 100",                       # Expected: 99.0
        "sqrt(2)",                          # Expected: 1.4 (approx)
        "max(1.12, 1.18, 1.15)",            # Expected: 1.2
        "100 * (0.14136385986817752)",      # Expected: 14.1
        "((206588 - 181001) / 181001) * 100", # Expected: 14.1
        "'hello'",                          # Expected: 'hello' (not a number)
        "10 / 0",                           # Expected: Error: division by zero
        "invalid_syntax)",                  # Expected: Error: invalid syntax...
        "True"                              # Expected: True (boolean, not rounded)
    ]
    for expr in expressions_to_test:
        output = safe_calculate(expr)
        print(f"Expression: '{expr}' -> Result: '{output}'")

    print("\nSchema (description updated for rounding):")
    import json
    print(json.dumps(CALCULATOR_TOOL_SCHEMA, indent=2))
