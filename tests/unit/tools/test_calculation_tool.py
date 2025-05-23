"""
Tests for the calculator tool.
"""
import pytest
from app.tools.calculation_tool import safe_calculate, CALCULATOR_TOOL_SCHEMA


class TestSafeCalculate:
    """Test cases for the safe_calculate function."""
    
    def test_basic_arithmetic(self):
        """Test basic arithmetic operations."""
        assert safe_calculate("2 + 2") == "4"
        assert safe_calculate("10 - 3") == "7"
        assert safe_calculate("5 * 6") == "30"
        assert safe_calculate("20 / 4") == "5.0"
        
    def test_complex_expressions(self):
        """Test more complex mathematical expressions."""
        assert safe_calculate("(5 + 3) * 2") == "16"
        assert safe_calculate("2 ** 3") == "8"
        assert safe_calculate("10 % 3") == "1"
        
    def test_mathematical_functions(self):
        """Test common mathematical functions."""
        assert safe_calculate("abs(-5)") == "5"
        assert safe_calculate("max(1, 5, 3)") == "5"
        assert safe_calculate("min(1, 5, 3)") == "1"
        assert safe_calculate("round(3.14159, 2)") == "3.14"
        
    def test_division_by_zero(self):
        """Test division by zero error handling."""
        result = safe_calculate("5 / 0")
        assert "Error" in result
        
    def test_syntax_errors(self):
        """Test handling of syntax errors."""
        result = safe_calculate("2 +")
        assert "Error" in result
        
        result = safe_calculate("2 * * 3")
        assert "Error" in result
        
    def test_invalid_expressions(self):
        """Test handling of invalid expressions."""
        result = safe_calculate("undefined_function()")
        assert "Error" in result
        
    def test_safe_evaluation(self):
        """Test that potentially unsafe operations are blocked."""
        # Test that eval-unsafe operations don't work
        result = safe_calculate("__import__('os').system('ls')")
        assert "Error" in result
        
        result = safe_calculate("exec('print(hello)')")
        assert "Error" in result
        
    def test_decimal_operations(self):
        """Test operations with decimal numbers."""
        assert safe_calculate("3.14 * 2") == "6.28"
        assert safe_calculate("1.5 + 2.5") == "4.0"
        
    def test_negative_numbers(self):
        """Test operations with negative numbers."""
        assert safe_calculate("-5 + 3") == "-2"
        assert safe_calculate("5 * -2") == "-10"


class TestCalculatorToolSchema:
    """Test cases for the calculator tool schema."""
    
    def test_schema_structure(self):
        """Test that the schema has the correct structure."""
        assert "type" in CALCULATOR_TOOL_SCHEMA
        assert CALCULATOR_TOOL_SCHEMA["type"] == "function"
        
        assert "function" in CALCULATOR_TOOL_SCHEMA
        function_def = CALCULATOR_TOOL_SCHEMA["function"]
        
        assert "name" in function_def
        assert function_def["name"] == "calculator"
        
        assert "description" in function_def
        assert "parameters" in function_def
        
    def test_parameters_schema(self):
        """Test the parameters schema structure."""
        params = CALCULATOR_TOOL_SCHEMA["function"]["parameters"]
        
        assert params["type"] == "object"
        assert "properties" in params
        assert "required" in params
        
        properties = params["properties"]
        assert "math_expression" in properties
        assert properties["math_expression"]["type"] == "string"
        
        assert params["required"] == ["math_expression"]