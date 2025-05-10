from flask import Flask, render_template, request, jsonify
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import re
import math
from typing import Tuple, Optional
import time
import logging
import os

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# Конфигурация
MAX_POINTS = 10_000
MAX_TIME_SECONDS = 3


class FunctionEvaluator:
    """Безопасный вычислитель математических функций"""

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
        'x': None
    }

    def __init__(self):
        self._last_error = None

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def validate(self, expr: str) -> bool:
        """Проверяет валидность математического выражения"""
        try:
            expr = self._replace_power_operator(expr)
            self._compile(expr)
            return True
        except (SyntaxError, ValueError, TypeError) as e:
            self._last_error = str(e)
            return False

    def evaluate(self, expr: str, x_values: np.ndarray) -> Optional[np.ndarray]:
        """Вычисляет значения функции для массива x"""
        try:
            expr = self._replace_power_operator(expr)
            compiled = self._compile(expr)
            start_time = time.time()

            result = np.zeros_like(x_values, dtype=float)
            for i, x in enumerate(x_values):
                try:
                    result[i] = self._evaluate_single(compiled, x)
                except (ValueError, ZeroDivisionError):
                    result[i] = np.nan
                except Exception as e:
                    self._last_error = f"Ошибка при x={x}: {str(e)}"
                    return None

            if time.time() - start_time > MAX_TIME_SECONDS:
                raise TimeoutError("Вычисление заняло слишком много времени")

            return result
        except Exception as e:
            self._last_error = str(e)
            return None

    def _replace_power_operator(self, expr: str) -> str:
        """Заменяет ^ на ** для корректного вычисления"""
        # Заменяем только если ^ не в строковых литералах
        return re.sub(r'(?<!\w)\^(?!\w)', '**', expr)

    def _compile(self, expr: str):
        """Компилирует выражение с проверкой безопасности"""
        if not re.fullmatch(r'^[\w\s\.\+\-\*/\(\)\^\,\=\>\<\!&|\~\:\%]+$', expr):
            raise ValueError("Выражение содержит недопустимые символы")

        code = compile(expr, '<string>', 'eval')
        for name in code.co_names:
            if name not in self.ALLOWED_NAMES:
                raise ValueError(f"Использование '{name}' не разрешено")

        return code

    def _evaluate_single(self, compiled_code, x: float) -> float:
        """Вычисляет значение функции для одного x"""
        try:
            return float(eval(
                compiled_code,
                {'__builtins__': None},
                {'x': x, **{k: v for k, v in self.ALLOWED_NAMES.items() if v is not None}}
            ))
        except (ValueError, ZeroDivisionError) as e:
            raise ValueError(f"Функция не определена при x={x}")
        except Exception as e:
            raise ValueError(str(e))


def generate_plot_image(func: str, x_min: float, x_max: float) -> Tuple[Optional[str], Optional[str]]:
    """Генерирует изображение графика"""
    try:
        if x_min >= x_max:
            raise ValueError("Минимальное значение x должно быть меньше максимального")

        evaluator = FunctionEvaluator()
        if not evaluator.validate(func):
            raise ValueError(evaluator.last_error)

        # Автоматический подбор количества точек
        range_width = x_max - x_min
        num_points = min(MAX_POINTS, max(500, int(range_width * 50)))
        x = np.linspace(x_min, x_max, num_points)

        y = evaluator.evaluate(func, x)
        if y is None:
            raise ValueError(evaluator.last_error)

        # Создание графика
        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

        # Разделяем график на непрерывные участки
        mask = np.isfinite(y)
        if np.any(~mask):
            # Находим границы разрывов
            discontinuities = np.where(~mask)[0]
            segments = np.split(np.arange(len(x)), discontinuities)

            # Рисуем каждый непрерывный участок
            for seg in segments:
                if len(seg) > 1:  # Игнорируем одиночные точки
                    ax.plot(x[seg], y[seg], 'b-', linewidth=2)
        else:
            ax.plot(x, y, 'b-', linewidth=2)

        # Настройки графика
        ax.set_title(f'График функции: {func}', pad=20)
        ax.set_xlabel('x', labelpad=10)
        ax.set_ylabel('f(x)', labelpad=10)
        ax.grid(True, alpha=0.3)

        # Автоматическое масштабирование
        y_valid = y[np.isfinite(y)]
        if len(y_valid) > 0:
            y_min, y_max = np.nanpercentile(y_valid, [5, 95])
            margin = max((y_max - y_min) * 0.1, 0.1)
            ax.set_ylim(y_min - margin, y_max + margin)

        # Сохранение в буфер
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
    try:
        x_min = float(data.get('x_min', -10))
        x_max = float(data.get('x_max', 10))
    except ValueError:
        return jsonify({'error': 'Неверный диапазон значений x'}), 400

    if not func:
        return jsonify({'error': 'Функция не указана'}), 400

    image_data, error = generate_plot_image(func, x_min, x_max)
    if error:
        return jsonify({'error': error}), 400

    return jsonify({'image': image_data})


if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    app.run(host='0.0.0.0', port=5000, threaded=True)