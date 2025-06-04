import cv2


class Renderer:
    def __init__(self):
        self.line_ops = []
        self.object_ops = []
        self.last_rendered_lines_image = None
        self.last_rendered_objects_image = None

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
        frame_copy = frame.copy()
        for op in ops:
            try:
                if op["type"] == "line":
                    cv2.line(frame_copy, op["start"], op["end"], op["color"], op["thickness"])
                elif op["type"] == "arrow":
                    cv2.arrowedLine(frame_copy, op["start"], op["end"], op["color"], op["thickness"])
                elif op["type"] == "text":
                    cv2.putText(frame_copy, op["text"], op["position"], cv2.FONT_HERSHEY_SIMPLEX,
                                op.get("scale", 0.5), op["color"], op.get("thickness", 1))
                elif op["type"] == "rect":
                    cv2.rectangle(frame_copy, op["top_left"], op["bottom_right"], op["color"], op["thickness"])
                elif op["type"] == "circle":
                    cv2.circle(frame_copy, op["center"], op["radius"], op["color"], op["thickness"])
            except Exception as e:
                print(f"[Renderer] Failed to draw op {op}: {e}")
        return frame_copy

    def render_lines(self, frame):
        """Render line detection drawings onto the frame."""
        self.last_rendered_lines_image = self._apply_draw_ops(frame, self.line_ops)

    def render_objects(self, frame):
        """Render object detection drawings onto the frame."""
        self.last_rendered_objects_image = self._apply_draw_ops(frame, self.object_ops)

    def get_last_linedetection_image(self):
        """Return last rendered image for line detection."""
        return self.last_rendered_lines_image

    def get_last_objectdetection_image(self):
        """Return last rendered image for object detection."""
        return self.last_rendered_objects_image
