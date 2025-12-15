"""
LaTeX Problem Viewer for AI Mathematical Olympiad
This script provides utilities to view and process LaTeX-formatted math problems
"""

import pandas as pd
from IPython.display import display, Markdown, Latex
import re

def clean_latex(text):
    """Clean and format LaTeX for better visibility"""
    # Ensure proper spacing around equations
    text = re.sub(r'\\begin{equation\*?}', r'\n$$', text)
    text = re.sub(r'\\end{equation\*?}', r'$$\n', text)
    
    # Convert inline math to display properly
    text = re.sub(r'(?<!\$)\$(?!\$)([^\$]+)(?<!\$)\$(?!\$)', r'`$\1$`', text)
    
    # Add line breaks for better readability
    text = text.replace('. ', '.\n\n')
    
    return text

def display_problem(problem_id, df):
    """Display a single problem with enhanced LaTeX rendering"""
    row = df[df['id'] == problem_id].iloc[0]
    
    print("="*80)
    print(f"Problem ID: {problem_id}")
    print("="*80)
    print()
    
    # Format the problem text
    problem_text = clean_latex(row['problem'])
    
    # Display with markdown rendering
    display(Markdown(problem_text))
    
    if 'answer' in df.columns:
        print()
        print(f"**Answer:** {row['answer']}")
    print()
    print("="*80)

def display_all_problems(df, limit=None):
    """Display multiple problems with LaTeX rendering"""
    problems = df.head(limit) if limit else df
    
    for idx, row in problems.iterrows():
        display_problem(row['id'], df)
        print("\n")

def export_to_markdown(df, output_file):
    """Export problems to a markdown file for better viewing"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# AI Mathematical Olympiad Problems\n\n")
        
        for idx, row in df.iterrows():
            f.write(f"## Problem {idx + 1}: {row['id']}\n\n")
            f.write(f"{row['problem']}\n\n")
            
            if 'answer' in df.columns:
                f.write(f"**Answer:** {row['answer']}\n\n")
            
            f.write("---\n\n")
    
    print(f"Exported {len(df)} problems to {output_file}")

def create_latex_preview(df, output_file):
    """Create an HTML file with proper LaTeX rendering"""
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Mathematical Olympiad Problems</title>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .problem {
            background-color: white;
            padding: 30px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .problem-header {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        .problem-text {
            line-height: 1.8;
            font-size: 16px;
            color: #333;
        }
        .answer {
            margin-top: 20px;
            padding: 15px;
            background-color: #e8f4f8;
            border-left: 4px solid #3498db;
            border-radius: 4px;
        }
        h1 {
            text-align: center;
            color: #2c3e50;
        }
    </style>
</head>
<body>
    <h1>AI Mathematical Olympiad Problems</h1>
"""
    
    for idx, row in df.iterrows():
        problem_text = row['problem'].replace('$', '\\$')
        
        html_content += f"""
    <div class="problem">
        <div class="problem-header">
            <h2>Problem {idx + 1}</h2>
            <p><strong>ID:</strong> {row['id']}</p>
        </div>
        <div class="problem-text">
            {problem_text}
        </div>
"""
        
        if 'answer' in df.columns:
            html_content += f"""
        <div class="answer">
            <strong>Answer:</strong> {row['answer']}
        </div>
"""
        
        html_content += "    </div>\n"
    
    html_content += """
</body>
</html>
"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Created HTML preview with {len(df)} problems: {output_file}")

if __name__ == "__main__":
    # Load the data
    print("Loading data...")
    reference_df = pd.read_csv('data/reference.csv')
    test_df = pd.read_csv('data/test.csv')
    
    print(f"\nLoaded {len(reference_df)} reference problems")
    print(f"Loaded {len(test_df)} test problems")
    
    # Create HTML preview for reference problems
    create_latex_preview(reference_df, 'data/reference_problems.html')
    
    # Create HTML preview for test problems
    create_latex_preview(test_df, 'data/test_problems.html')
    
    print("\nYou can now open the HTML files in your browser for better LaTeX visualization!")
