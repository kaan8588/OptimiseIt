import io
import base64
import warnings
import numpy as np
from scipy import stats
from flask import Flask, render_template, request, jsonify
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')

app = Flask(__name__)

def parse_array(data_str):
    return np.array([float(val.strip()) for val in data_str.split(',') if val.strip()])

def fig_to_base64(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', facecolor='white', bbox_inches='tight')
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/regression', methods=['POST'])
def api_regression():
    try:
        data = request.json
        x = parse_array(data['x']).reshape(-1, 1)
        y = parse_array(data['y'])
        x_name = data.get('x_name', 'X-Axis')
        y_name = data.get('y_name', 'Y-Axis')

        if len(x) != len(y) or len(x) < 3:
            return jsonify({"error": "X and Y arrays must have the same length (minimum 3)."}), 400

        x_smooth = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
        models = {}

        # 1. Simple Linear Regression (y = mx + b)
        lin_model = LinearRegression().fit(x, y)
        m = lin_model.coef_[0]
        b = lin_model.intercept_
        lin_eq = f"y = {m:.4f}x {'+' if b >= 0 else '-'} {abs(b):.4f}"
        
        models['Simple Linear Regression'] = {
            'r2': r2_score(y, lin_model.predict(x)), 
            'plot': lin_model.predict(x_smooth), 
            'color': '#ef4444',
            'equation': lin_eq
        }

        # 2. Polynomial Regression (y = ax^2 + bx + c)
        poly = PolynomialFeatures(degree=2)
        poly_model = LinearRegression().fit(poly.fit_transform(x), y)
        c_poly = poly_model.intercept_
        b_poly = poly_model.coef_[1]
        a_poly = poly_model.coef_[2]
        poly_eq = f"y = {a_poly:.4f}x² {'+' if b_poly >= 0 else '-'} {abs(b_poly):.4f}x {'+' if c_poly >= 0 else '-'} {abs(c_poly):.4f}"
        
        models['Polynomial Regression (2nd Degree)'] = {
            'r2': r2_score(y, poly_model.predict(poly.transform(x))), 
            'plot': poly_model.predict(poly.transform(x_smooth)), 
            'color': '#f59e0b',
            'equation': poly_eq
        }

        # Üstel ve Kuvvet modelleri için pozitiflik kontrolü
        if np.all(x > 0) and np.all(y > 0):
            # 3. Exponential Growth (y = a * e^(bx))
            p_exp = np.polyfit(x.flatten(), np.log(y), 1)
            a_exp = np.exp(p_exp[1])
            b_exp = p_exp[0]
            exp_eq = f"y = {a_exp:.4f} * e^({b_exp:.4f}x)"
            
            models['Exponential Growth'] = {
                'r2': r2_score(y, a_exp * np.exp(b_exp * x.flatten())), 
                'plot': a_exp * np.exp(b_exp * x_smooth.flatten()), 
                'color': '#10b981',
                'equation': exp_eq
            }
            
            # 4. Power Equation (y = a * x^b)
            p_pow = np.polyfit(np.log10(x.flatten()), np.log10(y), 1)
            a_pow = 10**p_pow[1]
            b_pow = p_pow[0]
            pow_eq = f"y = {a_pow:.4f} * x^({b_pow:.4f})"
            
            models['Power Equation'] = {
                'r2': r2_score(y, (a_pow) * (x.flatten() ** b_pow)), 
                'plot': (a_pow) * (x_smooth.flatten() ** b_pow), 
                'color': '#8b5cf6',
                'equation': pow_eq
            }

        # En iyi modeli seçme algoritması
        best_name = max(models, key=lambda k: models[k]['r2'])
        if best_name != 'Simple Linear Regression' and (models[best_name]['r2'] - models['Simple Linear Regression']['r2'] < 0.02):
            best_name = 'Simple Linear Regression'
            
        best_model = models[best_name]

        # Grafik Çizimi
        fig = Figure(figsize=(8, 4.5), dpi=100)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.scatter(x, y, color='#2563eb', label='Actual Data', zorder=5)
        ax.plot(x_smooth, best_model['plot'], color=best_model['color'], linewidth=3, label=best_name)
        ax.set_xlabel(x_name, fontweight='bold')
        ax.set_ylabel(y_name, fontweight='bold')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

        return jsonify({
            "model_name": best_name,
            "r2_score": round(best_model['r2'] * 100, 2),
            "equation": best_model['equation'],
            "image": fig_to_base64(fig)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/descriptive', methods=['POST'])
def api_descriptive():
    try:
        data = parse_array(request.json['data'])
        
        fig = Figure(figsize=(8, 4), dpi=100)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.boxplot(data, vert=False, patch_artist=True, boxprops=dict(facecolor='#bfdbfe', color='#1e3a8a'))
        ax.set_title("Data Distribution (Boxplot)", fontweight='bold')
        ax.set_yticks([]) 
        ax.grid(True, axis='x', linestyle='--', alpha=0.7)

        return jsonify({
            "Mean": round(np.mean(data), 3),
            "Median": round(np.median(data), 3),
            "Standard Deviation": round(np.std(data, ddof=1), 3),
            "Minimum": round(np.min(data), 3),
            "Maximum": round(np.max(data), 3),
            "Sample Size (N)": len(data),
            "image": fig_to_base64(fig)
        })
    except Exception as e:
        return jsonify({"error": "Invalid data format."}), 400

@app.route('/api/ttest', methods=['POST'])
def api_ttest():
    try:
        group_a = parse_array(request.json['grup_a'])
        group_b = parse_array(request.json['grup_b'])
        t_stat, p_val = stats.ttest_ind(group_a, group_b)
        
        result_text = "SIGNIFICANT difference exists." if p_val < 0.05 else "NO significant difference."
        
        fig = Figure(figsize=(8, 4.5), dpi=100)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        
        means = [np.mean(group_a), np.mean(group_b)]
        stds = [np.std(group_a, ddof=1), np.std(group_b, ddof=1)]
        
        ax.bar(['Group A', 'Group B'], means, yerr=stds, capsize=10, color=['#3b82f6', '#10b981'], alpha=0.8, edgecolor='black')
        ax.set_title(f"Group Means Comparison (p-value={p_val:.3f})", fontweight='bold')
        ax.set_ylabel("Mean Value")
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)

        return jsonify({
            "T-Statistic": round(t_stat, 4),
            "P-Value": round(p_val, 4),
            "Conclusion": result_text,
            "image": fig_to_base64(fig)
        })
    except Exception as e:
        return jsonify({"error": "Invalid data format."}), 400

@app.route('/api/correlation', methods=['POST'])
def api_correlation():
    try:
        x = parse_array(request.json['x'])
        y = parse_array(request.json['y'])
        corr, p_val = stats.pearsonr(x, y)
        
        strength = "Strong" if abs(corr) > 0.7 else "Moderate" if abs(corr) > 0.3 else "Weak"
        direction = "Positive" if corr > 0 else "Negative"
        
        fig = Figure(figsize=(8, 4.5), dpi=100)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        
        ax.scatter(x, y, color="#000000", label='Data Points', zorder=5)
        m, b = np.polyfit(x, y, 1)
        ax.plot(x, m*np.array(x) + b, color='#ef4444', linewidth=2, label='Trend Line')
        
        ax.set_title(f"Scatter Plot (r={corr:.3f})", fontweight='bold')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

        return jsonify({
            "Pearson Coefficient (r)": round(corr, 4),
            "Relationship": f"{strength} {direction}",
            "P-Value": round(p_val, 4),
            "image": fig_to_base64(fig)
        })
    except Exception as e:
        return jsonify({"error": "Arrays must be of the same length."}), 400

if __name__ == '__main__':
    app.run(debug=True)