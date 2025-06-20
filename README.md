# GIT_stencil-net

## Описание скриптов:

### 0. Общие файлы для нейросети:

- stencilnet/network.py - скрипт с описанием класса нейронной сети stencil-net
- stencilnet/timestepping.py - скрипт с функциями схем временных аппроксимаций
- my_funcs.py - отдельный py файл с функциями

### 1. Задача адвекции (/1_Advection/)

- 1_1_Advection_finite_diff_schemes.ipynb - решение с помощью конечно-разностных схем
- 1_2_Advection_stencil_net.ipynb - решение с помощью stencil-net
- 1_3_Advection_stencil_net_OptunaCycle.ipynb - решение с помощью stencil-net + optuna для подбора оптимальных гиперпараметров и оптимальной архитектуры нейросети
- 1_4_Advection_FINAL_RESULTS - отдельный скрипт для визуализации итогового решения
- init_params.xlsx - набор гепирпараметров и параметров для архитектуры нейронной сети
- my_advection_funcs.py - отдельный py файл с функциями для задачи адвекции

### 2. Задача теплопроводности (/2_Thermal_conductivity/)

- 2_1_Thermal_finite_diff_schemes.ipynb - решение с помощью конечно-разностных схем
- 2_2_Thermal_stencil_net.ipynb - решение с помощью stencil-net
- 2_3_Thermal_stencil_net_OptunaCycle.ipynb - решение с помощью stencil-net + optuna для подбора оптимальных гиперпараметров и оптимальной архитектуры нейросети
- 2_4_Thermal_FINAL_RESULTS - отдельный скрипт для визуализации итогового решения
- init_params.xlsx - набор гепирпараметров и параметров для архитектуры нейронной сети
- my_thermal_funcs.py - отдельный py файл с функциями для задачи теплопроводности

### 3. Уравнение Бюргерса (/3_Burgers/)

- models/ - предобученные модели из стать
- основные результаты в ветке 'experiments'

### 4. Двумерная задача адвекции (/4_2d_advection/)

- funcs_2d_advection.py - Функция для генерации данных
- 2d_generate_data.ipynb - Пример генерации данных
- 2d_linreg.ipynb - конечно-разностная схема (аналогия)

### 5. Двумерная задача теплопроводности (/5_2d_thermal_conductivity/)

- funcs_2d_thermal.py - Функция для генерации данных
- 2d_generate_data.ipynb - Пример генерации данных
- 2d_linreg.ipynb - конечно-разностная схема (аналогия)