import cv2


class Renderer:
    def __init__(self):
        self.line_ops = []
        self.object_ops = []

    def add_drawings_linedetection(self, ops: list):
        """Add drawing operations related to line detection."""
        self.line_ops.extend(ops)

    def add_drawings_objectdetection(self, ops: list):
        """Add drawing operations related to object detection."""
        self.object_ops.extend(ops)

    def clear(self):
        """Clear all drawing operations."""
        self.line_ops = []
        self.object_ops = []

    def _apply_draw_ops(self, frame, ops):
        for op in ops:
            try:
                if op["type"] == "line":
                    cv2.line(frame, op["start"], op["end"], op["color"], op["thickness"])
                elif op["type"] == "arrow":
                    cv2.arrowedLine(frame, op["start"], op["end"], op["color"], op["thickness"])
                elif op["type"] == "text":
                    cv2.putText(frame, op["text"], op["position"], cv2.FONT_HERSHEY_SIMPLEX,
                                op.get("scale", 0.5), op["color"], op.get("thickness", 1))
                elif op["type"] == "rect":
                    cv2.rectangle(frame, op["top_left"], op["bottom_right"], op["color"], op["thickness"])
                elif op["type"] == "circle":
                    cv2.circle(frame, op["center"], op["radius"], op["color"], op["thickness"])
            except Exception as e:
                print(f"[Renderer] Failed to draw op {op}: {e}")
        return frame

    def render_lines(self, frame):
        """Render line detection drawings onto the frame."""
        return self._apply_draw_ops(frame, self.line_ops)

    def render_objects(self, frame):
        """Render object detection drawings onto the frame."""
        return self._apply_draw_ops(frame, self.object_ops)
