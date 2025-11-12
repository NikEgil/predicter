import pandas as pd
import numpy as np


def compute_degree_days(df, T_min=5.0, T_max=100.0, countlim=60, timelim=120):
    """
    Вычисляет значение градусо-день для одного дня.

    Параметры:
        df (pd.DataFrame): Данные за один день с колонками 'datetime' и 'temp_air'.
        T_min (float): Пороговая температура (по умолчанию 5.0°C).
        T_max (float): Пороговая температура (по умолчанию 100.0°C), если градусо-день выше данного значения он равен ему.
        countlim (int): минимальное число измерений для рассчета, если данных меньше, то градусо-день равен 0.
        timelim (float): максимальное время между измерениями, если в данных время больше, то градусо-день равен 0.

    Возвращает:
        float: Значение градуса-дня.
    """
    if type(dd) != pd.DataFrame:
        df = pd.DataFrame(df, collumns=["datetime", "temp_air"])
    if df.empty:
        return 0.0
    if len(df) < countlim or df["datetime"].diff().max().total_seconds() < timelim * 60:
        return 0

    df = df.sort_values("datetime").copy()
    # Время в часах с начала дня
    start_date = df["datetime"].min().normalize()
    df["hour"] = (df["datetime"] - start_date).dt.total_seconds() / 3600.0

    df["excess"] = np.maximum(df["temp_air"] - T_min, 0)
    # Интеграл методом трапеций
    degree_hours = np.trapezoid(df["excess"], x=df["hour"])
    dd = degree_hours / 24.0
    if dd > T_max:
        dd = T_max
    return dd

def cumulative_sum(data):
    return np.cumsum(data)

def interpolate_zeros(arr):
    """
    Интерполирует нули в 1D-массиве линейно по соседним ненулевым значениям.
    Параметры:
        arr (array-like): входной массив чисел.
    Возвращает:
        np.ndarray: массив с интерполированными нулями.
    """
    s = pd.Series(arr, dtype=float)
    s.replace(0, np.nan, inplace=True)
    # 'linear', 'time', 'index', 'values', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'barycentric', 'krogh',
    #  'spline', 'polynomial', 'from_derivatives', 'piecewise_polynomial', 'pchip', 'akima', 'cubicspline'
    s.interpolate(method="linear", limit_direction="both", inplace=True)
    return s

def calc_dd_all_data(df, T_min=5, T_max=100, timelim=120,countlim=60):
    '''
    Рассчитывает degree-days по всем датчикам за каждый день.
    
    Parameters:
    df (pd.DataFrame): DataFrame с колонками 'datetime', 'sensor_id', 'temp_air'
    T_min (float): минимальная температура для расчёта degree-days
    T_max (float): максимальная температура (не используется в текущей реализации)
    time (int): временной интервал (не используется в текущей реализации)
    
    Returns:
    pd.DataFrame: DataFrame с колонками:
        'date' - день
        sensor id + 'dd' - вычесленный градусо-день
        sensor id + 'int' - интерполяция значений
        sensor id + 'cs' - сумма градусо-дней за период
    '''
    date_min = df["datetime"].min()
    date_max = df["datetime"].max()
    dd_df = pd.DataFrame()
    dd_df["date"] = pd.date_range(date_min.date(), date_max.date(), freq="D")
    for id in df["sensor_id"].unique():
        sensdata = df[df["sensor_id"] == id]
        dd = []
        for day in pd.date_range(date_min.date(), date_max.date(), freq="D"):
            mask = sensdata["datetime"].dt.normalize() == day
            day_data = sensdata[mask]
            dd.append(compute_degree_days(day_data, T_min, T_max,countlim=countlim,timelim=timelim))
        dd_df[id + "dd"] = dd
        dd_df[id + "int"] = interpolate_zeros(dd)
        dd_df[id + "cs"] = dd_df[id + "ddint"].cumsum()
    return dd_df
