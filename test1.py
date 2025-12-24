import re
import sys
from pathlib import Path
from typing import Optional, Tuple, List

# Increase recursion limit for deeply nested structures
sys.setrecursionlimit(10000)

# ================================
# Token-based Parser for SAS IF-ELSE
# ================================

class SASParser:
    """
    Recursive descent parser for SAS IF-ELSE scoring logic
    Handles multiple top-level IF-ELSE trees
    """
    
    def __init__(self, sas_code: str, verbose: bool = False):
        self.original = sas_code
        self.verbose = verbose
        self.tokens = self._tokenize(sas_code)
        self.pos = 0
        self.tree_count = 0
    
    def _tokenize(self, code: str) -> list:
        """
        Tokenize SAS code into meaningful tokens
        """
        # Normalize whitespace first
        code = re.sub(r'\s+', ' ', code.upper().strip())
        
        tokens = []
        i = 0
        total_len = len(code)
        last_progress = 0
        
        while i < len(code):
            # Progress indicator for large files
            if self.verbose:
                progress = int((i / total_len) * 100)
                if progress >= last_progress + 10:
                    print(f"   Tokenizing: {progress}%")
                    last_progress = progress
            
            # Skip whitespace
            if code[i].isspace():
                i += 1
                continue
            
            remaining = code[i:]
            
            # Multi-character keywords/operators
            if remaining.startswith('THEN'):
                tokens.append(('THEN', 'THEN'))
                i += 4
            elif remaining.startswith('ELSE'):
                tokens.append(('ELSE', 'ELSE'))
                i += 4
            elif remaining.startswith('END;'):
                tokens.append(('END', 'END'))
                i += 4
            elif remaining.startswith('END'):
                tokens.append(('END', 'END'))
                i += 3
            elif remaining.startswith('DO;'):
                tokens.append(('DO', 'DO'))
                i += 3
            elif remaining.startswith('DO'):
                tokens.append(('DO', 'DO'))
                i += 2
            elif remaining.startswith('IF'):
                tokens.append(('IF', 'IF'))
                i += 2
            elif remaining.startswith('OR'):
                tokens.append(('OR', 'OR'))
                i += 2
            elif remaining.startswith('AND'):
                tokens.append(('AND', 'AND'))
                i += 3
            elif remaining.startswith('NOT'):
                tokens.append(('NOT', 'NOT'))
                i += 3
            elif remaining.startswith('IN'):
                tokens.append(('IN', 'IN'))
                i += 2
            elif remaining.startswith('SUM_SCORE'):
                tokens.append(('SUM_SCORE', 'SUM_SCORE'))
                i += 9
            elif remaining.startswith('<='):
                tokens.append(('LE', '<='))
                i += 2
            elif remaining.startswith('>='):
                tokens.append(('GE', '>='))
                i += 2
            elif remaining.startswith('<>'):
                tokens.append(('NE', '<>'))
                i += 2
            elif code[i] == '(':
                tokens.append(('LPAREN', '('))
                i += 1
            elif code[i] == ')':
                tokens.append(('RPAREN', ')'))
                i += 1
            elif code[i] == '=':
                tokens.append(('EQ', '='))
                i += 1
            elif code[i] == '<':
                tokens.append(('LT', '<'))
                i += 1
            elif code[i] == '>':
                tokens.append(('GT', '>'))
                i += 1
            elif code[i] == '+':
                tokens.append(('PLUS', '+'))
                i += 1
            elif code[i] == '-':
                tokens.append(('MINUS', '-'))
                i += 1
            elif code[i] == '*':
                tokens.append(('STAR', '*'))
                i += 1
            elif code[i] == '/':
                tokens.append(('SLASH', '/'))
                i += 1
            elif code[i] == ',':
                tokens.append(('COMMA', ','))
                i += 1
            elif code[i] == '.':
                tokens.append(('DOT', '.'))
                i += 1
            elif code[i] == ';':
                tokens.append(('SEMICOLON', ';'))
                i += 1
            elif code[i] == "'":
                # String literal
                j = i + 1
                while j < len(code) and code[j] != "'":
                    j += 1
                value = code[i:j+1]
                tokens.append(('STRING', value))
                i = j + 1
            elif code[i] == '"':
                # String literal (double quotes)
                j = i + 1
                while j < len(code) and code[j] != '"':
                    j += 1
                value = code[i:j+1]
                tokens.append(('STRING', value))
                i = j + 1
            elif code[i].isalnum() or code[i] == '_':
                # Identifier or number
                j = i
                while j < len(code) and (code[j].isalnum() or code[j] == '_' or code[j] == '.'):
                    j += 1
                value = code[i:j]
                
                # Check if it's a number
                try:
                    float(value)
                    tokens.append(('NUMBER', value))
                except ValueError:
                    tokens.append(('IDENT', value))
                i = j
            else:
                i += 1
        
        if self.verbose:
            print(f"   Tokenizing: 100% - {len(tokens)} tokens")
        
        return tokens
    
    def peek(self, offset: int = 0) -> Optional[Tuple[str, str]]:
        """Look at token without consuming"""
        if self.pos + offset < len(self.tokens):
            return self.tokens[self.pos + offset]
        return None
    
    def consume(self) -> Optional[Tuple[str, str]]:
        """Consume and return current token"""
        if self.pos < len(self.tokens):
            token = self.tokens[self.pos]
            self.pos += 1
            return token
        return None
    
    def expect(self, token_type: str) -> str:
        """Consume token of expected type"""
        token = self.consume()
        if token is None or token[0] != token_type:
            raise SyntaxError(f"Expected {token_type}, got {token} at position {self.pos}")
        return token[1]
    
    def skip_optional(self, *token_types):
        """Skip tokens if they match any of the given types"""
        while self.peek() and self.peek()[0] in token_types:
            self.consume()
    
    def parse(self) -> List[str]:
        """
        Parse ALL top-level IF-ELSE blocks in the SAS code
        Returns a list of CASE expressions
        """
        case_expressions = []
        
        while self.pos < len(self.tokens):
            # Skip until we find an IF
            while self.peek() and self.peek()[0] != 'IF':
                self.consume()
            
            if not self.peek():
                break
            
            # Parse this IF-ELSE tree
            try:
                case_expr = self.parse_if_else()
                case_expressions.append(case_expr)
                self.tree_count += 1
                
                if self.verbose:
                    print(f"   Parsed tree #{self.tree_count}")
                    
            except Exception as e:
                # Try to skip to next top-level IF
                if self.verbose:
                    print(f"   Warning: Error parsing tree, skipping: {str(e)[:50]}")
                while self.peek() and self.peek()[0] != 'IF':
                    self.consume()
                if self.peek():
                    self.consume()
        
        return case_expressions
    
    def parse_if_else(self) -> str:
        """
        Parse IF-ELSE block and convert to CASE
        """
        # Expect IF
        self.expect('IF')
        
        # Parse condition
        condition = self.parse_condition()
        
        # Expect THEN DO;
        self.expect('THEN')
        self.skip_optional('DO', 'SEMICOLON')
        
        # Parse THEN block
        then_result = self.parse_block()
        
        # Expect END; ELSE DO;
        self.skip_optional('END', 'SEMICOLON')
        self.expect('ELSE')
        self.skip_optional('DO', 'SEMICOLON')
        
        # Parse ELSE block
        else_result = self.parse_block()
        
        # Expect END;
        self.skip_optional('END', 'SEMICOLON')
        
        # Build CASE expression
        return f"(CASE WHEN {condition} THEN {then_result} ELSE {else_result} END)"
    
    def parse_condition(self) -> str:
        """
        Parse condition until THEN keyword
        Convert SAS syntax to SQL syntax
        """
        parts = []
        paren_depth = 0
        
        while self.peek():
            token = self.peek()
            
            if token[0] == 'THEN' and paren_depth == 0:
                break
            
            if token[0] == 'LPAREN':
                paren_depth += 1
                self.consume()
                if paren_depth > 1:
                    parts.append('(')
            elif token[0] == 'RPAREN':
                paren_depth -= 1
                self.consume()
                if paren_depth >= 1:
                    parts.append(')')
            elif token[0] == 'OR':
                parts.append(' OR ')
                self.consume()
            elif token[0] == 'AND':
                parts.append(' AND ')
                self.consume()
            elif token[0] == 'NOT':
                parts.append(' NOT ')
                self.consume()
            elif token[0] == 'IN':
                parts.append(' IN ')
                self.consume()
            elif token[0] == 'EQ':
                self.consume()
                # Check if next token is DOT (missing value check)
                if self.peek() and self.peek()[0] == 'DOT':
                    self.consume()
                    # Convert "field = ." to "field IS NULL"
                    if parts:
                        field = parts.pop().strip()
                        parts.append(f"{field} IS NULL")
                else:
                    parts.append(' = ')
            elif token[0] == 'NE':
                self.consume()
                # Check for <> .
                if self.peek() and self.peek()[0] == 'DOT':
                    self.consume()
                    if parts:
                        field = parts.pop().strip()
                        parts.append(f"{field} IS NOT NULL")
                else:
                    parts.append(' <> ')
            elif token[0] == 'LT':
                parts.append(' < ')
                self.consume()
            elif token[0] == 'GT':
                parts.append(' > ')
                self.consume()
            elif token[0] == 'LE':
                parts.append(' <= ')
                self.consume()
            elif token[0] == 'GE':
                parts.append(' >= ')
                self.consume()
            elif token[0] == 'IDENT':
                parts.append(token[1])
                self.consume()
            elif token[0] == 'NUMBER':
                parts.append(token[1])
                self.consume()
            elif token[0] == 'STRING':
                parts.append(token[1])
                self.consume()
            elif token[0] == 'MINUS':
                parts.append('-')
                self.consume()
            elif token[0] == 'DOT':
                parts.append('.')
                self.consume()
            elif token[0] == 'COMMA':
                parts.append(', ')
                self.consume()
            else:
                self.consume()
        
        condition = ''.join(parts).strip()
        condition = re.sub(r'\s+', ' ', condition)
        
        return condition
    
    def parse_block(self) -> str:
        """
        Parse a THEN or ELSE block
        Could be nested IF-ELSE or score assignment
        """
        if self.peek() and self.peek()[0] == 'IF':
            return self.parse_if_else()
        else:
            return self.parse_score_assignment()
    
    def parse_score_assignment(self) -> str:
        """
        Parse: SUM_SCORE = SUM_SCORE + (value);
        Return just the value
        """
        value_parts = []
        in_value = False
        paren_depth = 0
        
        while self.peek():
            token = self.peek()
            
            if token[0] in ('END', 'ELSE') and paren_depth == 0:
                break
            
            if token[0] == 'PLUS':
                self.consume()
                in_value = True
                continue
            
            if in_value:
                if token[0] == 'LPAREN':
                    paren_depth += 1
                    if paren_depth == 1:
                        self.consume()
                        continue
                    value_parts.append('(')
                elif token[0] == 'RPAREN':
                    paren_depth -= 1
                    if paren_depth == 0:
                        self.consume()
                        continue
                    value_parts.append(')')
                elif token[0] == 'NUMBER':
                    value_parts.append(token[1])
                elif token[0] == 'MINUS':
                    value_parts.append('-')
                elif token[0] == 'DOT':
                    value_parts.append('.')
            
            self.consume()
        
        value = ''.join(value_parts).strip()
        
        if not value:
            return '0'
        
        return value


# ================================
# Main Conversion Functions
# ================================

def sas_to_sql_expressions(sas_code: str, verbose: bool = False) -> List[str]:
    """
    Convert SAS IF-ELSE scoring logic to list of SQL CASE expressions
    """
    parser = SASParser(sas_code, verbose=verbose)
    return parser.parse()


def format_case_expression(case_expr: str, base_indent: int = 4) -> str:
    """
    Format CASE expression with proper indentation for readability
    """
    result = []
    indent_level = base_indent
    indent_step = 4
    
    expr = case_expr
    expr = re.sub(r'\(CASE', ' (CASE', expr)
    expr = re.sub(r'WHEN', ' WHEN ', expr)
    expr = re.sub(r'THEN', ' THEN ', expr)
    expr = re.sub(r'ELSE', ' ELSE ', expr)
    expr = re.sub(r'END\)', ' END) ', expr)
    
    expr = re.sub(r'\s+', ' ', expr).strip()
    
    tokens = expr.split()
    i = 0
    current_line = []
    
    while i < len(tokens):
        token = tokens[i]
        
        if token == '(CASE':
            if current_line:
                result.append(' ' * indent_level + ' '.join(current_line))
                current_line = []
            result.append(' ' * indent_level + '(CASE')
            indent_level += indent_step
        elif token == 'WHEN':
            if current_line:
                result.append(' ' * indent_level + ' '.join(current_line))
                current_line = []
            current_line = ['WHEN']
        elif token == 'THEN':
            current_line.append('THEN')
        elif token == 'ELSE':
            if current_line:
                result.append(' ' * indent_level + ' '.join(current_line))
                current_line = []
            current_line = ['ELSE']
        elif token == 'END)':
            if current_line:
                result.append(' ' * indent_level + ' '.join(current_line))
                current_line = []
            indent_level -= indent_step
            result.append(' ' * indent_level + 'END)')
        else:
            current_line.append(token)
        
        i += 1
    
    if current_line:
        result.append(' ' * indent_level + ' '.join(current_line))
    
    return '\n'.join(result)


def build_sql(sas_code: str, table_name: str, output_column: str = "SUM_SCORE", verbose: bool = False) -> str:
    """
    Generate complete SQL query from SAS scoring logic
    Handles multiple IF-ELSE trees by summing them
    """
    if verbose:
        print("🔄 Parsing SAS code...")
    
    case_expressions = sas_to_sql_expressions(sas_code, verbose=verbose)
    
    if not case_expressions:
        raise ValueError("No IF-ELSE blocks found in SAS code")
    
    if verbose:
        print(f"✓ Found {len(case_expressions)} scoring tree(s)")
        print("📝 Formatting SQL...")
    
    # Format each expression
    formatted_exprs = []
    for i, expr in enumerate(case_expressions):
        if verbose and (i + 1) % 100 == 0:
            print(f"   Formatting tree {i+1}/{len(case_expressions)}")
        formatted_exprs.append(format_case_expression(expr))
    
    # Combine all expressions with +
    if len(formatted_exprs) == 1:
        combined_expr = formatted_exprs[0]
    else:
        # Join multiple trees with + and newlines
        combined_expr = "\n    +\n".join(formatted_exprs)
    
    return f"""-- ================================================
-- AUTO-GENERATED FROM SAS SCORECARD LOGIC
-- Generated by: SAS to Snowflake SQL Converter
-- Number of scoring trees: {len(case_expressions)}
-- ================================================

SELECT
    *,
{combined_expr} AS {output_column}
FROM {table_name};"""


# ================================
# Validation
# ================================

def validate_sql(sql_code: str) -> tuple[bool, list[str]]:
    """
    Validate generated SQL and return issues
    """
    issues = []
    
    open_count = sql_code.count('(')
    close_count = sql_code.count(')')
    if open_count != close_count:
        issues.append(f"Unbalanced parentheses: {open_count} open, {close_count} close")
    
    case_count = len(re.findall(r'\bCASE\b', sql_code, re.IGNORECASE))
    end_count = len(re.findall(r'\bEND\b', sql_code, re.IGNORECASE))
    
    if case_count != end_count:
        issues.append(f"Unbalanced CASE/END: {case_count} CASE, {end_count} END")
    
    if re.search(r'SUM_SCORE\s*=\s*SUM_SCORE', sql_code, re.IGNORECASE):
        issues.append("Found SAS assignment syntax in SQL output")
    
    if re.search(r'\bDO\b', sql_code, re.IGNORECASE):
        issues.append("Found SAS 'DO' keyword in SQL output")
    
    return len(issues) == 0, issues


# ================================
# File-based runner
# ================================

def convert_file(
    input_path: str, 
    output_path: str, 
    table_name: str,
    validate: bool = True,
    verbose: bool = True
):
    """
    Convert SAS file to SQL file
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if verbose:
        print(f"📖 Reading SAS code from: {input_path}")
    
    sas_code = input_path.read_text(encoding="utf-8")
    
    if verbose:
        lines = len(sas_code.splitlines())
        print(f"   {lines:,} lines, {len(sas_code):,} characters")
    
    try:
        sql_code = build_sql(sas_code, table_name, verbose=verbose)
    except Exception as e:
        print(f"❌ Conversion failed: {str(e)}")
        raise

    if validate:
        if verbose:
            print(f"✓ Validating SQL...")
        
        is_valid, issues = validate_sql(sql_code)
        
        if not is_valid:
            print("⚠️  Validation warnings:")
            for issue in issues:
                print(f"   - {issue}")
        else:
            if verbose:
                print("   Validation PASSED")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(sql_code, encoding="utf-8")

    if verbose:
        case_count = sql_code.upper().count('CASE')
        print(f"\n✅ Conversion complete!")
        print(f"📄 Input : {input_path}")
        print(f"📄 Output: {output_path}")
        print(f"📊 Stats:")
        print(f"   - Lines in SAS: {len(sas_code.splitlines()):,}")
        print(f"   - Lines in SQL: {len(sql_code.splitlines()):,}")
        print(f"   - CASE statements: {case_count:,}")


# ================================
# Test with sample data
# ================================

def test_converter():
    """
    Test the converter with sample SAS code (multiple trees)
    """
    # Sample with 2 separate IF-ELSE trees (simulating multiple scorecard components)
    sample_sas = """
    IF (f26<1.5 OR f26 = .) THEN DO;
        IF (f8<0.5 OR f8 = .) THEN DO;
            sum_score = sum_score + (-0.141490221);
        END;
        ELSE DO;
            sum_score = sum_score + (-0.1768962);
        END;
    END;
    ELSE DO;
        IF (f88<3.5 OR f88 = .) THEN DO;
            sum_score = sum_score + (0.0328746177);
        END;
        ELSE DO;
            sum_score = sum_score + (-0.0433662049);
        END;
    END;
    
    IF (f100<5 OR f100 = .) THEN DO;
        IF (f200<10 OR f200 = .) THEN DO;
            sum_score = sum_score + (0.123);
        END;
        ELSE DO;
            sum_score = sum_score + (-0.456);
        END;
    END;
    ELSE DO;
        IF (f300<15.5 OR f300 = .) THEN DO;
            sum_score = sum_score + (0.789);
        END;
        ELSE DO;
            sum_score = sum_score + (-0.012);
        END;
    END;
    """
    
    print("🧪 Testing converter with multiple trees...\n")
    try:
        sql = build_sql(sample_sas, "test_table", verbose=True)
        print("\n" + "="*60)
        print(sql)
        print("="*60)
        
        is_valid, issues = validate_sql(sql)
        if is_valid:
            print("\n✅ Validation: PASSED")
        else:
            print("\n⚠️  Validation issues:")
            for issue in issues:
                print(f"   - {issue}")
                
    except Exception as e:
        import traceback
        print(f"❌ Test failed: {str(e)}")
        traceback.print_exc()


# ================================
# CLI entry
# ================================

if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--test":
        test_converter()
        sys.exit(0)

    if len(sys.argv) not in [4, 5]:
        print("="*60)
        print("SAS to Snowflake SQL Converter (Large File Edition)")
        print("="*60)
        print("\nUsage:")
        print("  python sas_to_sql_large.py <input.sas> <output.sql> <table_name> [--no-validate]")
        print("  python sas_to_sql_large.py --test")
        print("\nExample:")
        print("  python sas_to_sql_large.py scorecard.sas scorecard.sql MY_SCHEMA.MY_TABLE")
        print("\nFeatures:")
        print("  ✓ Handles 50K+ lines of SAS code")
        print("  ✓ Multiple IF-ELSE trees (combined with +)")
        print("  ✓ Deep nesting support")
        print("  ✓ Progress indicators")
        print("  ✓ Validation")
        sys.exit(1)

    validate = "--no-validate" not in sys.argv
    
    if validate:
        _, input_sas, output_sql, table = sys.argv
    else:
        args = [arg for arg in sys.argv if arg != "--no-validate"]
        _, input_sas, output_sql, table = args

    try:
        convert_file(input_sas, output_sql, table, validate=validate)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)
