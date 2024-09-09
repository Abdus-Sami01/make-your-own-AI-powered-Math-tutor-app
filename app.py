from flask import Flask, render_template, request
import sympy as sp
import numpy as np
import io
import base64
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
from scipy import stats
from transformers import pipeline

app = Flask(__name__)

# Load the transformers model for math word problem parsing
nlp_model = pipeline('text2text-generation', model='google/flan-t5-base')

def solve_polynomial(problem):
    try:
        expr = sp.sympify(problem)
        solutions = sp.solve(expr)
        return f"Solutions: {solutions}"
    except Exception as e:
        return f"Error: Could not solve the polynomial equation. {str(e)}"

def explain_solution(problem):
    try:
        expr = sp.sympify(problem)
        solution_steps = sp.solve(expr, show_steps=True)
        explanation = "\n".join(str(step) for step in solution_steps)
        return f"Explanation:\n{explanation}"
    except Exception as e:
        return f"Error: Could not generate explanation. {str(e)}"

def generate_explanation_with_nlp(problem):
    try:
        explanation = nlp_model(problem)[0]['generated_text']
        return f"Explanation (NLP model):\n{explanation}"
    except Exception as e:
        return f"Error: Could not generate explanation using NLP model. {str(e)}"

def plot_graph(expression):
    try:
        x = sp.Symbol('x')
        expr = sp.sympify(expression)
        f = sp.lambdify(x, expr, modules='numpy')

        x_vals = np.linspace(-10, 10, 400)
        y_vals = f(x_vals)

        fig, ax = plt.subplots()
        ax.plot(x_vals, y_vals, label=str(expr))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Graph of the function')
        ax.legend()
        ax.grid(True)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # Save the plot to a BytesIO object and encode as base64
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        plt.close(fig)

        return plot_url
    except Exception as e:
        return f"Error: Could not plot the graph. {str(e)}"

def differentiate(problem):
    try:
        expr = sp.sympify(problem)
        derivative = sp.diff(expr)
        return f"Derivative: {derivative}"
    except Exception as e:
        return f"Error: Could not compute the derivative. {str(e)}"

def integrate(problem):
    try:
        expr = sp.sympify(problem)
        integral = sp.integrate(expr)
        return f"Integral: {integral}"
    except Exception as e:
        return f"Error: Could not compute the integral. {str(e)}"

def simplify_expression(problem):
    try:
        expr = sp.sympify(problem)
        simplified_expr = sp.simplify(expr)
        return f"Simplified Expression: {simplified_expr}"
    except Exception as e:
        return f"Error: Could not simplify the expression. {str(e)}"

def expand_expression(problem):
    try:
        expr = sp.sympify(problem)
        expanded_expr = sp.expand(expr)
        return f"Expanded Expression: {expanded_expr}"
    except Exception as e:
        return f"Error: Could not expand the expression. {str(e)}"

def solve_equation(problem):
    try:
        expr = sp.sympify(problem)
        solutions = sp.solve(expr)
        return f"Solutions: {solutions}"
    except Exception as e:
        return f"Error: Could not solve the equation. {str(e)}"

def calculate_limit(problem):
    try:
        expr = sp.sympify(problem)
        limit = sp.limit(expr, sp.Symbol('x'), 0)  # Default to limit as x approaches 0
        return f"Limit: {limit}"
    except Exception as e:
        return f"Error: Could not compute the limit. {str(e)}"

def compute_series(problem):
    try:
        expr = sp.sympify(problem)
        series = sp.series(expr, sp.Symbol('x'), n=5)  # Compute the first 5 terms of the series
        return f"Series Expansion: {series}"
    except Exception as e:
        return f"Error: Could not compute the series expansion. {str(e)}"

def trigonometric_functions(problem):
    try:
        expr = sp.sympify(problem)
        sine = sp.sin(expr)
        cosine = sp.cos(expr)
        tangent = sp.tan(expr)
        
        sine_num = sp.N(sine)
        cosine_num = sp.N(cosine)
        tangent_num = sp.N(tangent)
        
        return (f"Sine: {sine} ≈ {sine_num}\n"
                f"Cosine: {cosine} ≈ {cosine_num}\n"
                f"Tangent: {tangent} ≈ {tangent_num}")
    except Exception as e:
        return f"Error: Could not compute trigonometric functions. {str(e)}"

def trigonometric_identities(problem):
    try:
        expr = sp.sympify(problem)
        # Example: verify if an identity holds
        identity_check = sp.simplify(expr)
        return f"Identity Verification: {identity_check}"
    except Exception as e:
        return f"Error: Could not verify trigonometric identities. {str(e)}"

def convert_angle(angle, unit_from, unit_to):
    try:
        if unit_from == 'degrees' and unit_to == 'radians':
            return np.deg2rad(angle)
        elif unit_from == 'radians' and unit_to == 'degrees':
            return np.rad2deg(angle)
        else:
            return angle
    except Exception as e:
        return f"Error: Could not convert angles. {str(e)}"

def statistical_summary(data):
    try:
        data = np.array([float(x) for x in data.split(',')])
        mean = np.mean(data)
        median = np.median(data)
        std_dev = np.std(data)
        variance = np.var(data)
        return (f"Mean: {mean}\nMedian: {median}\nStandard Deviation: {std_dev}\nVariance: {variance}")
    except Exception as e:
        return f"Error: Could not compute statistical summary. {str(e)}"

def hypothesis_testing(data, test_type):
    try:
        data = np.array([float(x) for x in data.split(',')])
        if test_type == 't_test':
            t_stat, p_value = stats.ttest_1samp(data, 0)  # Test against the mean of 0
            return f"T-Test Statistics: {t_stat}\nP-Value: {p_value}"
        elif test_type == 'chi_square':
            chi2_stat, p_value = stats.chisquare(data)  # Chi-Square Test for goodness of fit
            return f"Chi-Square Statistics: {chi2_stat}\nP-Value: {p_value}"
        else:
            return "Error: Unknown hypothesis test type."
    except Exception as e:
        return f"Error: Could not perform hypothesis testing. {str(e)}"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/solve', methods=['POST'])
def solve():
    problem = request.form['problem']
    operation = request.form.get('operation')
    data = request.form.get('data', '')
    hypothesis_test = request.form.get('hypothesis_test', '')

    plot_url= None

    try:
        if operation == 'differentiate':
            solution = differentiate(problem)
        elif operation == 'integrate':
            solution = integrate(problem)
        elif operation == 'simplify':
            solution = simplify_expression(problem)
        elif operation == 'expand':
            solution = expand_expression(problem)
        elif operation == 'solve':
            solution = solve_equation(problem)
        elif operation == 'limit':
            solution = calculate_limit(problem)
        elif operation == 'series':
            solution = compute_series(problem)
        elif operation == 'trigonometric':
            solution = trigonometric_functions(problem)
        elif operation == 'identity':
            solution = trigonometric_identities(problem)
        elif operation == 'convert_angle':
            angle = float(request.form['angle'])
            unit_from = request.form['unit_from']
            unit_to = request.form['unit_to']
            solution = f"Converted Angle: {convert_angle(angle, unit_from, unit_to)}"
        elif operation == 'statistical_summary':
            solution = statistical_summary(data)
        elif operation == 'hypothesis_testing':
            solution = hypothesis_testing(data, hypothesis_test)
        elif operation == 'plot':
            plot_url = plot_graph(problem)
            return render_template('index.html', plot_url=plot_url)
        elif "polynomial" in problem.lower():
            solution = solve_polynomial(problem)
        else:
            expr = sp.sympify(problem)
            solution = sp.simplify(expr)
            solution = f"Solution: {solution}"
    except Exception as e:
        try:
            parsed_problem = nlp_model(problem)[0]['generated_text']
            expr = sp.sympify(parsed_problem)
            solution = sp.simplify(expr)
            solution = f"Solution (word problem parsed): {solution}"
        except Exception as e:
            solution = f"Error: Could not parse the problem. {str(e)}"

    return render_template('index.html', solution=solution, plot_url=plot_url)


@app.route('/feedback', methods=['POST'])
def feedback():
    rating = request.form['rating']
    comments = request.form['comments']
    # Save feedback to database or file
    # For now, just print the feedback to the console
    print(f"Rating: {rating}")
    print(f"Comments: {comments}")
    return render_template('index.html', feedback_message="Thank you for your feedback!")

if __name__ == '__main__':
    app.run(debug=True)
