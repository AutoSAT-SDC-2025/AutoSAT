import cv2


class Renderer:
    def __init__(self):
        self.draw_ops = []

    def add_drawings(self, new_ops: list):
        self.draw_ops.extend(new_ops)

    def clear(self):
        self.draw_ops = []

    def render(self, frame):
        for op in self.draw_ops:
            try:
                if op["type"] == "line":
                    cv2.line(frame, op["start"], op["end"], op["color"], op["thickness"])
                elif op["type"] == "arrow":
                    cv2.arrowedLine(frame, op["start"], op["end"], op["color"], op["thickness"])
                elif op["type"] == "text":
                    cv2.putText(frame, op["text"], op["position"], cv2.FONT_HERSHEY_SIMPLEX, op.get("scale", 0.5), op["color"], op.get("thickness", 1))
                elif op["type"] == "rect":
                    cv2.rectangle(frame, op["top_left"], op["bottom_right"], op["color"], op["thickness"])
                elif op["type"] == "circle":
                    cv2.circle(frame, op["center"], op["radius"], op["color"], op["thickness"])
            except Exception as e:
                print(f"[Renderer] Failed to draw op {op}: {e}")
            cv2.imshow('Traffic Detection', frame)
