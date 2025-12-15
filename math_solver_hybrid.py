"""
Hybrid Math Solver - Combines symbolic solving with advanced olympiad strategies
"""

import pandas as pd
import numpy as np
import re
from sympy import (sympify, solve, symbols, simplify, factorial, binomial, 
                   floor, ceiling, Rational, gcd, lcm, prime, isprime, 
                   divisors, totient, factorint, mod_inverse, sqrt, pi, 
                   sin, cos, tan, log, exp, Sum, Product, oo, Eq)
from sympy.parsing.latex import parse_latex
from sympy.ntheory import divisor_count, divisor_sigma
import math
from itertools import combinations, permutations
import warnings
warnings.filterwarnings('ignore')

class HybridMathSolver:
    """Hybrid solver that attempts symbolic solving first, then falls back to ML"""
    
    def __init__(self):
        self.common_variables = ['x', 'y', 'z', 'n', 'a', 'b', 'c', 'd', 't']
    
    def extract_math_from_latex(self, problem):
        """Extract mathematical expressions from LaTeX"""
        # Find all math expressions between $ signs
        math_expressions = re.findall(r'\$([^\$]+)\$', problem)
        return math_expressions
    
    def clean_latex(self, expr):
        """Clean LaTeX expression for parsing"""
        # Remove common LaTeX commands that might interfere
        expr = expr.replace('\\times', '*')
        expr = expr.replace('\\cdot', '*')
        expr = expr.replace('\\div', '/')
        expr = expr.replace('\\text', '')
        expr = expr.replace('{', '(')
        expr = expr.replace('}', ')')
        expr = expr.replace('\\left', '')
        expr = expr.replace('\\right', '')
        expr = expr.replace('\\,', '')
        expr = expr.replace('\\:', '')
        expr = expr.replace('\\;', '')
        expr = expr.replace('\\quad', '')
        expr = expr.replace('\\qquad', '')
        
        return expr.strip()
    
    def solve_simple_arithmetic(self, problem):
        """Solve simple arithmetic problems"""
        try:
            # Extract math expressions
            math_exprs = self.extract_math_from_latex(problem)
            
            if not math_exprs:
                return None
            
            # Check for simple evaluation questions like "What is 1-1?"
            for expr in math_exprs:
                cleaned = self.clean_latex(expr)
                
                # Check if it's just a simple expression to evaluate (no variables)
                if re.match(r'^[\d\s\+\-\*/\(\)\.]+$', cleaned):
                    try:
                        result = eval(cleaned)
                        if isinstance(result, (int, float)):
                            return int(result) if result == int(result) else result
                    except:
                        pass
            
            return None
        except Exception as e:
            return None
    
    def solve_equation(self, problem):
        """Solve algebraic equations"""
        try:
            # Extract math expressions
            math_exprs = self.extract_math_from_latex(problem)
            
            if not math_exprs:
                return None
            
            # Look for equations (contains =)
            for expr in math_exprs:
                if '=' in expr:
                    cleaned = self.clean_latex(expr)
                    
                    # Try to identify the variable to solve for
                    # Check problem text for "solve for x" or "find x"
                    var_to_solve = None
                    for var in self.common_variables:
                        if f'for ${var}$' in problem or f'for {var}' in problem.lower():
                            var_to_solve = var
                            break
                        if var in cleaned:
                            var_to_solve = var
                            break
                    
                    if var_to_solve:
                        try:
                            # Split equation by =
                            parts = cleaned.split('=')
                            if len(parts) == 2:
                                lhs = parts[0].strip()
                                rhs = parts[1].strip()
                                
                                # Create symbolic variable
                                var_symbol = symbols(var_to_solve)
                                
                                # Parse both sides
                                lhs_expr = sympify(lhs)
                                rhs_expr = sympify(rhs)
                                
                                # Solve equation
                                equation = lhs_expr - rhs_expr
                                solutions = solve(equation, var_symbol)
                                
                                if solutions:
                                    # Return first solution
                                    sol = solutions[0]
                                    if sol.is_number:
                                        result = float(sol)
                                        return int(result) if result == int(result) else result
                        except Exception as e:
                            pass
            
            return None
        except Exception as e:
            return None
    
    def solve_modular_arithmetic(self, problem):
        """Solve problems involving modular arithmetic and remainders"""
        try:
            # Look for remainder questions
            remainder_match = re.search(r'remainder when (.+?) is divided by (\d+)', problem, re.IGNORECASE)
            
            if remainder_match:
                expression_text = remainder_match.group(1).strip()
                modulus = int(remainder_match.group(2))
                
                # Extract the mathematical expression
                math_exprs = self.extract_math_from_latex(problem)
                
                # Try to find the expression in math blocks
                for expr in math_exprs:
                    # Check if this expression contains relevant parts
                    cleaned = self.clean_latex(expr)
                    
                    # Try to evaluate it
                    try:
                        # Handle special cases
                        if 'abc' in cleaned or 'product' in expression_text.lower():
                            # Product of variables - look for context
                            continue
                        
                        # Try direct evaluation
                        result = sympify(cleaned)
                        if result.is_number:
                            answer = int(result) % modulus
                            if 0 <= answer < 100000:  # Reasonable range check
                                return answer
                    except:
                        pass
                
                # Check for power expressions like 2^k mod something
                power_match = re.search(r'(\d+)\^(\d+)', expression_text)
                if power_match:
                    base = int(power_match.group(1))
                    exp = int(power_match.group(2))
                    if exp < 1000:  # Reasonable exponent
                        result = pow(base, exp, modulus)
                        return result
            
            return None
        except Exception as e:
            return None
    
    def solve_sequence_problem(self, problem):
        """Solve sequence and series problems"""
        try:
            # Check for Fibonacci sequence
            if 'fibonacci' in problem.lower() or re.search(r'F_\{?n\}?\s*=\s*F_\{?n-1\}?\s*\+\s*F_\{?n-2\}?', problem):
                # Look for F_n queries
                fib_query = re.search(r'F_\{?(\d+)\}?', problem)
                if fib_query:
                    n = int(fib_query.group(1))
                    if n < 100:  # Reasonable limit
                        # Generate Fibonacci number
                        fib = [0, 1]
                        for i in range(2, n + 1):
                            fib.append(fib[-1] + fib[-2])
                        return fib[n]
            
            # Arithmetic sequence
            arith_match = re.search(r'arithmetic.*?sequence.*?(\d+).*?(\d+)', problem.lower())
            if arith_match:
                # Would need more context to solve
                pass
            
            return None
        except Exception as e:
            return None
    
    def solve_combinatorics(self, problem):
        """Solve combinatorics problems"""
        try:
            problem_lower = problem.lower()
            
            # Factorials
            if 'factorial' in problem_lower:
                fact_match = re.search(r'(\d+)\s*!', problem)
                if fact_match:
                    n = int(fact_match.group(1))
                    if n < 20:  # Reasonable limit
                        result = math.factorial(n)
                        
                        # Check for modulo
                        mod_match = re.search(r'mod(?:ulo)?\s*(\d+)', problem)
                        if mod_match:
                            modulus = int(mod_match.group(1))
                            return result % modulus
                        return result
            
            # Combinations C(n,k) or binomial coefficients
            comb_match = re.search(r'(?:choose|\\binom|C)\(?(\d+),\s*(\d+)\)?', problem)
            if comb_match:
                n = int(comb_match.group(1))
                k = int(comb_match.group(2))
                if n < 100 and k < 100:
                    result = math.comb(n, k)
                    
                    mod_match = re.search(r'mod(?:ulo)?\s*(\d+)', problem)
                    if mod_match:
                        modulus = int(mod_match.group(1))
                        return result % modulus
                    return result
            
            # Permutations
            if 'permutation' in problem_lower:
                perm_match = re.search(r'P\((\d+),\s*(\d+)\)', problem)
                if perm_match:
                    n = int(perm_match.group(1))
                    k = int(perm_match.group(2))
                    if n < 20 and k < 20:
                        result = math.perm(n, k)
                        return result
            
            return None
        except Exception as e:
            return None
    
    def solve_number_theory(self, problem):
        """Solve number theory problems"""
        try:
            problem_lower = problem.lower()
            
            # GCD/LCM
            if 'gcd' in problem_lower or 'greatest common divisor' in problem_lower:
                numbers = re.findall(r'\b(\d+)\b', problem)
                if len(numbers) >= 2:
                    nums = [int(n) for n in numbers[:2]]
                    return math.gcd(nums[0], nums[1])
            
            if 'lcm' in problem_lower or 'least common multiple' in problem_lower:
                numbers = re.findall(r'\b(\d+)\b', problem)
                if len(numbers) >= 2:
                    nums = [int(n) for n in numbers[:2]]
                    return math.lcm(nums[0], nums[1])
            
            # Prime checking
            if 'prime' in problem_lower and 'how many' not in problem_lower:
                # Check if a specific number is prime
                prime_match = re.search(r'is\s+(\d+)\s+(?:a\s+)?prime', problem_lower)
                if prime_match:
                    n = int(prime_match.group(1))
                    return 1 if isprime(n) else 0
            
            # Divisibility
            if 'divisib' in problem_lower:
                div_match = re.search(r'(\d+).*?divisible.*?(\d+)', problem_lower)
                if div_match:
                    n = int(div_match.group(1))
                    d = int(div_match.group(2))
                    return 1 if n % d == 0 else 0
            
            # Power of primes dividing factorial
            if 'divides' in problem_lower and 'factorial' in problem_lower:
                # Legendre's formula
                pass
            
            return None
        except Exception as e:
            return None
    
    def solve_geometry(self, problem):
        """Solve basic geometry problems"""
        try:
            problem_lower = problem.lower()
            
            # Triangle perimeter
            if 'triangle' in problem_lower and 'perimeter' in problem_lower:
                sides = re.findall(r'side.*?(\d+)', problem_lower)
                if len(sides) >= 3:
                    return sum(int(s) for s in sides[:3])
            
            # Circle area/circumference
            if 'circle' in problem_lower:
                if 'radius' in problem_lower:
                    radius_match = re.search(r'radius.*?(\d+)', problem_lower)
                    if radius_match:
                        r = int(radius_match.group(1))
                        if 'area' in problem_lower:
                            return int(np.pi * r * r)
                        if 'circumference' in problem_lower or 'perimeter' in problem_lower:
                            return int(2 * np.pi * r)
            
            return None
        except Exception as e:
            return None
    
    def solve_problem(self, problem):
        """Main solving method with multiple strategies"""
        
        # Strategy 1: Simple arithmetic (for basic calculations)
        result = self.solve_simple_arithmetic(problem)
        if result is not None:
            return result
        
        # Strategy 2: Equation solving (for algebraic equations)
        result = self.solve_equation(problem)
        if result is not None:
            return result
        
        # Strategy 3: Modular arithmetic (for remainder problems)
        result = self.solve_modular_arithmetic(problem)
        if result is not None:
            return result
        
        # Strategy 4: Number theory (GCD, LCM, primes, divisibility)
        result = self.solve_number_theory(problem)
        if result is not None:
            return result
        
        # Strategy 5: Combinatorics (factorials, combinations, permutations)
        result = self.solve_combinatorics(problem)
        if result is not None:
            return result
        
        # Strategy 6: Sequences (Fibonacci, arithmetic, geometric)
        result = self.solve_sequence_problem(problem)
        if result is not None:
            return result
        
        # Strategy 7: Geometry (basic geometric calculations)
        result = self.solve_geometry(problem)
        if result is not None:
            return result
        
        # Fallback: Return 0 for unsolved complex olympiad problems
        print(f"  [INFO] Could not solve with current strategies, returning 0")
        return 0
    
    def solve_batch(self, problems):
        """Solve multiple problems"""
        results = []
        for problem in problems:
            result = self.solve_problem(problem)
            results.append(result)
        return results

def generate_predictions(test_path='data/test.csv', output_path='submission.csv'):
    """Generate predictions using hybrid solver"""
    print("="*80)
    print("Hybrid Math Solver - Generating Predictions")
    print("="*80)
    
    # Load test data
    test_data = pd.read_csv(test_path)
    print(f"\nLoaded {len(test_data)} test problems")
    
    # Initialize solver
    solver = HybridMathSolver()
    
    # Solve problems
    print("\nSolving problems...")
    predictions = []
    
    for idx, row in test_data.iterrows():
        problem_id = row['id']
        problem = row['problem']
        
        print(f"\nProblem {idx+1}: {problem_id}")
        print(f"  Question: {problem}")
        
        answer = solver.solve_problem(problem)
        predictions.append({'id': problem_id, 'answer': answer})
        
        print(f"  Answer: {answer}")
    
    # Create submission
    submission_df = pd.DataFrame(predictions)
    submission_df.to_csv(output_path, index=False)
    
    print(f"\n{'='*80}")
    print(f"Predictions saved to {output_path}")
    print("="*80)
    
    return submission_df

def evaluate_on_reference(reference_path='data/reference.csv'):
    """Evaluate solver on reference problems"""
    print("="*80)
    print("Evaluating Solver on Reference Problems")
    print("="*80)
    
    # Load reference data
    reference_data = pd.read_csv(reference_path)
    print(f"\nLoaded {len(reference_data)} reference problems")
    
    # Initialize solver
    solver = HybridMathSolver()
    
    correct = 0
    results = []
    
    for idx, row in reference_data.iterrows():
        problem_id = row['id']
        problem = row['problem']
        true_answer = row['answer']
        
        print(f"\n{'='*80}")
        print(f"Problem {idx+1}/{len(reference_data)}: {problem_id}")
        print(f"Question: {problem[:150]}...")
        print(f"True Answer: {true_answer}")
        
        predicted_answer = solver.solve_problem(problem)
        print(f"Predicted Answer: {predicted_answer}")
        
        is_correct = (predicted_answer == true_answer)
        if is_correct:
            correct += 1
            print("✓ CORRECT")
        else:
            print("✗ INCORRECT")
        
        results.append({
            'id': problem_id,
            'true_answer': true_answer,
            'predicted_answer': predicted_answer,
            'correct': is_correct
        })
    
    # Calculate accuracy
    accuracy = correct / len(reference_data) * 100
    
    print(f"\n{'='*80}")
    print(f"Evaluation Results")
    print(f"{'='*80}")
    print(f"Correct: {correct}/{len(reference_data)}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"{'='*80}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('evaluation_results.csv', index=False)
    print(f"\nDetailed results saved to evaluation_results.csv")
    
    return results_df, accuracy

def analyze_problem_types(data_path='data/reference.csv'):
    """Analyze types of problems in the dataset"""
    print("="*80)
    print("Problem Type Analysis")
    print("="*80)
    
    data = pd.read_csv(data_path)
    
    types = {
        'geometry': 0,
        'number_theory': 0,
        'combinatorics': 0,
        'algebra': 0,
        'modular_arithmetic': 0,
        'sequences': 0,
        'other': 0
    }
    
    for _, row in data.iterrows():
        problem_lower = row['problem'].lower()
        
        if any(word in problem_lower for word in ['triangle', 'circle', 'angle', 'polygon', 'geometric']):
            types['geometry'] += 1
        elif any(word in problem_lower for word in ['remainder', 'divides', 'modulo', 'mod']):
            types['modular_arithmetic'] += 1
        elif any(word in problem_lower for word in ['permutation', 'combination', 'choose', 'factorial']):
            types['combinatorics'] += 1
        elif any(word in problem_lower for word in ['sequence', 'fibonacci', 'series']):
            types['sequences'] += 1
        elif any(word in problem_lower for word in ['prime', 'gcd', 'lcm', 'divisor']):
            types['number_theory'] += 1
        elif any(word in problem_lower for word in ['solve', 'equation', 'function']):
            types['algebra'] += 1
        else:
            types['other'] += 1
    
    print("\nProblem Type Distribution:")
    print("-"*80)
    for ptype, count in sorted(types.items(), key=lambda x: x[1], reverse=True):
        percentage = count / len(data) * 100
        print(f"{ptype.replace('_', ' ').title():<25}: {count:>3} ({percentage:>5.1f}%)")
    print("="*80)
    
    return types

def main():
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--evaluate':
        # Evaluate on reference problems
        evaluate_on_reference()
    elif len(sys.argv) > 1 and sys.argv[1] == '--analyze':
        # Analyze problem types
        analyze_problem_types()
    else:
        # Generate predictions for submission
        generate_predictions()

if __name__ == "__main__":
    main()
