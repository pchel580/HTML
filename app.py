from flask import Flask, render_template, request, jsonify
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import re
import math
from typing import Tuple, Optional, Dict
import time
import logging
import os

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

MAX_POINTS = 10_000_000
MAX_TIME_SECONDS = 3
DEFAULT_VARIABLES = {'x'}


class FunctionEvaluator:
    """Безопасный вычислитель математических функций с поддержкой произвольных переменных"""

    ALLOWED_NAMES = {
        **{k: v for k, v in math.__dict__.items() if not k.startswith('_')},
        'abs': abs,
        'min': min,
        'max': max,
        'pow': pow,
        'round': round,
        'log': math.log,
        'log10': math.log10,
        'sqrt': math.sqrt,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'exp': math.exp,
        'pi': math.pi,
        'e': math.e,
    }

    def __init__(self):
        self._last_error = None

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def validate(self, expr: str) -> Tuple[bool, set]:
        """Проверяет валидность выражения и возвращает набор переменных"""
        try:
            expr = self._replace_power_operator(expr)
            code = self._compile(expr)
            variables = self._extract_variables(code)
            return True, variables
        except (SyntaxError, ValueError, TypeError) as e:
            self._last_error = str(e)
            return False, set()

    def evaluate(self, expr: str, var_values: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """Вычисляет значения функции для массива значений переменных"""

        try:
            expr = self._replace_power_operator(expr)
            compiled = self._compile(expr)
            start_time = time.time()

            lengths = {len(v) for v in var_values.values()}
            if len(lengths) != 1:
                raise ValueError("Все переменные должны иметь одинаковое количество значений")

            result = np.zeros_like(next(iter(var_values.values())), dtype=float)
            for i in range(len(result)):
                try:
                    context = {k: v[i] for k, v in var_values.items()}
                    context.update({k: v for k, v in self.ALLOWED_NAMES.items()})

                    result[i] = float(eval(compiled, {'__builtins__': None}, context))
                except (ValueError, ZeroDivisionError):
                    result[i] = np.nan
                except Exception as e:
                    self._last_error = f"Ошибка в точке {i}: {str(e)}"
                    return None

            if time.time() - start_time > MAX_TIME_SECONDS:
                raise TimeoutError("Вычисление заняло слишком много времени")

            return result
        except Exception as e:
            self._last_error = str(e)
            return None

    def _replace_power_operator(self, expr: str) -> str:
        return re.sub(r'(?<=\w|\\)\^(?=\w|\(|\d)', '**', expr)

    def _compile(self, expr: str):
        expr = self._replace_power_operator(expr)
        if not re.fullmatch(r'^[\w\s\.\+\-\*/\(\)\^\,\=\>\<\!&|\~\:\%]+$', expr):
            raise ValueError("Выражение содержит недопустимые символы")

        code = compile(expr, '<string>', 'eval')
        # for name in code.co_names:
        #     if name not in self.ALLOWED_NAMES:
        #         raise ValueError(f"Использование '{name}' не разрешено")
        return code

    def _extract_variables(self, code) -> set:
        """Извлекает имена переменных из скомпилированного кода"""
        variables = set()
        for name in code.co_names:
            if name not in self.ALLOWED_NAMES and name.isidentifier():
                variables.add(name)
        return variables


def generate_plot_image(func: str, var_ranges: Dict[str, Tuple[float, float]]) -> Tuple[Optional[str], Optional[str]]:
    """Генерирует изображение графика для функции с несколькими переменными"""
    try:
        if not var_ranges:
            raise ValueError("Не указаны диапазоны для переменных")

        evaluator = FunctionEvaluator()
        is_valid, variables = evaluator.validate(func)
        if not is_valid:
            raise ValueError(evaluator.last_error)

        missing_vars = variables - set(var_ranges.keys())
        if missing_vars:
            raise ValueError(f"Не указаны диапазоны для переменных: {', '.join(missing_vars)}")

        main_var = next((v for v in DEFAULT_VARIABLES if v in variables), next(iter(variables), None))
        if not main_var:
            raise ValueError("Не удалось определить основную переменную для построения графика")

        x_min, x_max = var_ranges[main_var]
        range_width = x_max - x_min
        num_points = min(MAX_POINTS, max(500, int(range_width * 50)))

        var_values = {}
        for var, (v_min, v_max) in var_ranges.items():
            if var == main_var:
                var_values[var] = np.linspace(v_min, v_max, num_points)
            else:

                var_values[var] = np.full(num_points, (v_min + v_max) / 2)

        y = evaluator.evaluate(func, var_values)
        if y is None:
            raise ValueError(evaluator.last_error)

        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

        mask = np.isfinite(y)
        if np.any(~mask):
            discontinuities = np.where(~mask)[0]
            segments = np.split(np.arange(len(var_values[main_var])), discontinuities)

            for seg in segments:
                if len(seg) > 1:
                    ax.plot(var_values[main_var][seg], y[seg], 'b-', linewidth=2)
        else:
            ax.plot(var_values[main_var], y, 'b-', linewidth=2)

        title_vars = ', '.join(f"{var}∈[{vmin}, {vmax}]" for var, (vmin, vmax) in var_ranges.items())
        ax.set_title(f'График функции: {func}\n({title_vars})', pad=20)
        ax.set_xlabel(main_var, labelpad=10)
        ax.set_ylabel('f(' + ', '.join(var_ranges.keys()) + ')', labelpad=10)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)

        x_min, x_max = var_values[main_var].min(), var_values[main_var].max()
        x_margin = max((x_max - x_min) * 0.1, 0.1)  # 10% от диапазона или минимум 0.1
        ax.set_xlim(x_min - x_margin, x_max + x_margin)

        y_valid = y[np.isfinite(y)]
        if len(y_valid) > 0:
            y_min, y_max = np.nanmin(y_valid), np.nanmax(y_valid)
            if y_min > 0:
                y_min = -0.1 * y_max
            elif y_max < 0:
                y_max = -0.1 * y_min
            margin = max((y_max - y_min) * 0.1, 0.1)
            ax.set_ylim(y_min - margin, y_max + margin)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        plt.close(fig)
        buf.seek(0)

        return base64.b64encode(buf.read()).decode('utf-8'), None
    except Exception as e:
        return None, str(e)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/plot', methods=['POST'])
def plot_function():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Неверный формат данных'}), 400

    func = data.get('function', '').strip()
    var_ranges = data.get('var_ranges', {})

    if not func:
        return jsonify({'error': 'Функция не указана'}), 400

    try:

        processed_ranges = {}
        for var, (vmin, vmax) in var_ranges.items():
            processed_ranges[var] = (float(vmin), float(vmax))
    except (ValueError, TypeError) as e:
        return jsonify({'error': 'Неверный формат диапазонов переменных'}), 400

    image_data, error = generate_plot_image(func, processed_ranges)
    if error:
        return jsonify({'error': error}), 400

    return jsonify({'image': image_data})


if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    app.run(host='0.0.0.0', port=5000, threaded=True)
