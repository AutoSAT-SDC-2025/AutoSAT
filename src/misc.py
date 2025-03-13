import numpy
from src.car_variables import CarType

async def calc_axis_angle(x_axis: float, y_axis: float):
    if x_axis == 0.0:
        return 0.0
    else:
        return round(numpy.atan2(y_axis,x_axis),2)

def calculate_hunter_throttle(controller_axis_value: float, car_type: CarType) -> float | None:
    if car_type == CarType.kart:
        return controller_axis_value * 100
    elif car_type == CarType.hunter:
        return controller_axis_value * 1500

def calculate_hunter_steering(controller_axis_value: float, car_type: CarType) -> float | None:
    if car_type == CarType.kart:
        return controller_axis_value * 1.25
    elif car_type == CarType.hunter:
        return controller_axis_value * 576
