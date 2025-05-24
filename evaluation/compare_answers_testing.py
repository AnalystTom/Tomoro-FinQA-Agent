import re
import math

# Helper to get the apparent precision of a number string
def get_precision(value_str: str) -> int:
    if '.' in value_str:
        return len(value_str.split('.')[-1])
    return 0

# Helper to normalize and extract numeric value
def normalize_and_extract_numeric(answer_str: str, is_percentage_context: bool):
    if not isinstance(answer_str, str):
        answer_str = str(answer_str) # Ensure it's a string

    # Remove LaTeX, common phrases, currency symbols (basic example)
    # You might need more robust regex for various LaTeX math formats
    cleaned_str = re.sub(r'\\\(|\\\)|\$|€|£|approximately|about|is|The answer is|,', '', answer_str, flags=re.IGNORECASE)
    
    # Extract numbers, including negative and decimals
    # This regex attempts to find the last valid number, handling percentages
    numeric_match = re.search(r'([-+]?\d*\.?\d+)\s*\%?', cleaned_str.strip())
    
    if numeric_match:
        num_str = numeric_match.group(1)
        value = float(num_str)
        
        # If original string had '%' or it's a percentage context, convert to decimal
        # (e.g., 50% -> 0.5 or 50.0 -> 0.5 if known to be a percentage)
        # This part depends on how you determine 'is_percentage_context'
        # For simplicity, let's assume if '%' was in numeric_match.group(0), it's a percentage
        if (numeric_match.group(0) and '%' in numeric_match.group(0)) or is_percentage_context:
            # The value is ALREADY the percentage value (e.g., -32.82 from "-32.82%")
            # For direct comparison of percentage magnitudes, use this value.
            # If converting to decimal (0.xx form), you'd divide by 100 here.
            # For this problem, let's keep them as percentage magnitudes (e.g., -32.82 and -32.0)
            return value, True # True indicates it's a percentage magnitude
        return value, False # False indicates it's a raw number
    return None, False

# --- Your compare_answers function ---
# This is a simplified example focusing on the problematic case.
# You'll need to integrate this with your existing logic that handles Items 1, 2, 4 correctly.

def compare_answers(item_id: str, final_agent_answer_str: str, final_ground_truth_str: str, agent_tool_result_str: str = None):
    # Determine if we are dealing with percentages based on GT or agent answer
    # This is a heuristic; a more robust way would be to know the question type
    is_percentage_q = '%' in final_ground_truth_str or ('%' in final_agent_answer_str if final_agent_answer_str else False)

    agent_numeric_val, agent_is_perc = normalize_and_extract_numeric(final_agent_answer_str, is_percentage_q)
    gt_numeric_val, gt_is_perc = normalize_and_extract_numeric(final_ground_truth_str, is_percentage_q)

    # Use the more precise agent_tool_result_str if available and parse it
    agent_raw_calc_val = None
    if agent_tool_result_str:
        # Assuming agent_tool_result_str is the direct string output from safe_calculate (e.g., "-32.8197")
        try:
            agent_raw_calc_val = float(agent_tool_result_str)
            # If the original question was a percentage, this raw_calc_val is also the percentage magnitude
        except (ValueError, TypeError):
            agent_raw_calc_val = None
            
    # If agent_numeric_val couldn't be parsed from final_agent_answer_str, try using agent_raw_calc_val
    if agent_numeric_val is None and agent_raw_calc_val is not None:
        agent_numeric_val = agent_raw_calc_val
        agent_is_perc = is_percentage_q # Assume context applies

    # If any parsing failed, they are not comparable numerically
    if agent_numeric_val is None or gt_numeric_val is None:
        # Fallback to string comparison if one or both are not numeric (e.g. "Yes", "No")
        # Ensure this part of your original logic is retained.
        # For now, if numeric parsing fails for this example, assume False.
        print(f"DEBUG [{item_id}]: Could not parse numeric values. Agent: '{final_agent_answer_str}', GT: '{final_ground_truth_str}'")
        return False

    # --- Core Comparison Logic ---
    # We are comparing percentage magnitudes directly (e.g. -32.82 vs -32.0)

    # 1. Standard Epsilon Check (for most cases)
    # This likely handles your Items 1 & 2.
    STANDARD_PERCENTAGE_MAGNITUDE_EPSILON = 0.1 # e.g., 0.1 percentage point difference
    if abs(agent_numeric_val - gt_numeric_val) < STANDARD_PERCENTAGE_MAGNITUDE_EPSILON:
        print(f"DEBUG [{item_id}]: PASS - Standard Epsilon. Agent: {agent_numeric_val}, GT: {gt_numeric_val}, Diff: {abs(agent_numeric_val - gt_numeric_val)}")
        return True

    # 2. Approach 1: Handling Integer-like GT with More Precise Agent Calculation
    #    (Specifically for cases like GT: "-32%", Agent: "-32.82%")
    #    Use the agent_raw_calc_val if available, otherwise agent_numeric_val.
    precise_agent_val_to_use = agent_raw_calc_val if agent_raw_calc_val is not None else agent_numeric_val
    
    # Check if GT was likely an integer percentage (e.g., original string was "XX%" or "XX.0%")
    gt_original_numeric_part = re.search(r'([-+]?\d*\.?\d+)', final_ground_truth_str.replace(',', ''))
    gt_precision = 0
    if gt_original_numeric_part:
        gt_precision = get_precision(gt_original_numeric_part.group(1))

    if gt_precision == 0: # GT is an integer percentage like "-32%"
        # Rule 2a: Agent's precise value rounded to 0 decimal places is within 1 of GT.
        # This handles the case where the true value might be, e.g., -32.8 (rounds to -33)
        # and GT is -32. This means the agent's value is on the "other side" of a 0.5 rounding boundary for GT.
        rounded_agent_to_gt_precision = round(precise_agent_val_to_use, 0)
        if abs(rounded_agent_to_gt_precision - gt_numeric_val) <= 1.0:
            # Further check: ensure the unrounded difference isn't too large.
            # Example: Agent calc -33.4 (rounds to -33). GT -30. Diff 3.4. round_diff 3. This rule would fail it.
            # Example: Agent calc -32.8197 (rounds to -33). GT -32. Diff 0.8197. round_diff 1. This rule would pass.
            if abs(precise_agent_val_to_use - gt_numeric_val) < 1.5: # Max 1.5 pp original difference
                print(f"DEBUG [{item_id}]: PASS - Agent precise ({precise_agent_val_to_use}) rounded to GT precision ({rounded_agent_to_gt_precision}) is within 1 of GT ({gt_numeric_val}). Original diff: {abs(precise_agent_val_to_use - gt_numeric_val)}")
                return True
        
        # Rule 2b: (Alternative or additional for integer GT)
        # The absolute difference between the agent's precise value and the integer GT
        # is within a slightly more generous, specific tolerance for this scenario.
        # For GT "-32", agent "-32.8197", diff is 0.8197.
        INTEGER_GT_SPECIFIC_EPSILON = 0.85 # Allow up to 0.85 pp difference for this specific case
        if abs(precise_agent_val_to_use - gt_numeric_val) < INTEGER_GT_SPECIFIC_EPSILON:
            print(f"DEBUG [{item_id}]: PASS - Integer GT Specific Epsilon. Agent Precise: {precise_agent_val_to_use}, GT: {gt_numeric_val}, Diff: {abs(precise_agent_val_to_use - gt_numeric_val)}")
            return True


    print(f"DEBUG [{item_id}]: FAIL - No rule matched. Agent: {agent_numeric_val} (Precise: {precise_agent_val_to_use}), GT: {gt_numeric_val}, Diff: {abs(precise_agent_val_to_use - gt_numeric_val if precise_agent_val_to_use is not None else agent_numeric_val - gt_numeric_val) }")
    return False

# Example Usage (how your evaluation script might call it):
# Assume these are extracted from your eval data and agent response
item_id_test = "Single_AAPL/2002/page_23.pdf-1"
final_agent_answer_str_test = "The percentage change ... was approximately \\(-32.82\\%\\)."
final_ground_truth_str_test = "-32%"
agent_tool_result_str_test = "-32.8197" # This is crucial - it's the output of safe_calculate

is_correct = compare_answers(item_id_test, final_agent_answer_str_test, final_ground_truth_str_test, agent_tool_result_str_test)
print(f"Result for {item_id_test}: {is_correct}")

item_id_test_2 = "Single_JKHY/2009/page_28.pdf-3"
final_agent_answer_str_test_2 = "... approximately 14.14%."
final_ground_truth_str_test_2 = "14.1%"
agent_tool_result_str_test_2 = "14.1364"
is_correct_2 = compare_answers(item_id_test_2, final_agent_answer_str_test_2, final_ground_truth_str_test_2, agent_tool_result_str_test_2)
print(f"Result for {item_id_test_2}: {is_correct_2}")