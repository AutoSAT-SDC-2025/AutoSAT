import cv2

class Renderer:
    """
    Utility class for managing and applying drawing operations (lines, arrows, text, rectangles, circles)
    on images, typically for visualizing line and object detection results.
    Stores drawing operations in separate lists for line and object detection, and can render them onto frames.
    """

    def __init__(self):
        # Lists of drawing operations for line and object detection, respectively.
        self.line_ops = []
        self.object_ops = []
        # Stores the most recently rendered images for each type.
        self.last_rendered_lines_image = None
        self.last_rendered_objects_image = None

    def add_drawings_linedetection(self, ops: list):
        """
        Add drawing operations related to line detection.
        Args:
            ops (list): List of drawing operation dicts (see _apply_draw_ops for format).
        """
        self.line_ops.extend(ops)

    def add_drawings_objectdetection(self, ops: list):
        """
        Add drawing operations related to object detection.
        Args:
            ops (list): List of drawing operation dicts (see _apply_draw_ops for format).
        """
        self.object_ops.extend(ops)

    def clear(self):
        """
        Clear all stored drawing operations for both line and object detection.
        Call this before starting a new frame or detection cycle.
        """
        self.line_ops = []
        self.object_ops = []

    def _apply_draw_ops(self, frame, ops):
        """
        Internal helper to apply a list of drawing operations to a frame.
        Args:
            frame (np.ndarray): The image to draw on.
            ops (list): List of drawing operation dicts, each specifying type and parameters.
        Returns:
            np.ndarray: The frame with all drawing operations applied.
        """
        frame_copy = frame.copy()
        for op in ops:
            try:
                # Draw a line between two points.
                if op["type"] == "line":
                    cv2.line(frame_copy, op["start"], op["end"], op["color"], op["thickness"])
                # Draw an arrowed line.
                elif op["type"] == "arrow":
                    cv2.arrowedLine(frame_copy, op["start"], op["end"], op["color"], op["thickness"])
                # Draw text at a given position.
                elif op["type"] == "text":
                    cv2.putText(frame_copy, op["text"], op["position"], cv2.FONT_HERSHEY_SIMPLEX,
                                op.get("scale", 0.5), op["color"], op.get("thickness", 1))
                # Draw a rectangle.
                elif op["type"] == "rect":
                    cv2.rectangle(frame_copy, op["top_left"], op["bottom_right"], op["color"], op["thickness"])
                # Draw a circle.
                elif op["type"] == "circle":
                    cv2.circle(frame_copy, op["center"], op["radius"], op["color"], op["thickness"])
            except Exception as e:
                # Log and skip any drawing operation that fails.
                print(f"[Renderer] Failed to draw op {op}: {e}")
        return frame_copy

    def render_lines(self, frame):
        """
        Render all stored line detection drawing operations onto the given frame.
        The result is stored for later retrieval.
        Args:
            frame (np.ndarray): The image to draw on.
        """
        self.last_rendered_lines_image = self._apply_draw_ops(frame, self.line_ops)

    def render_objects(self, frame):
        """
        Render all stored object detection drawing operations onto the given frame.
        The result is stored for later retrieval.
        Args:
            frame (np.ndarray): The image to draw on.
        """
        self.last_rendered_objects_image = self._apply_draw_ops(frame, self.object_ops)

    def get_last_linedetection_image(self):
        """
        Get the most recently rendered image with line detection drawings.
        Returns:
            np.ndarray or None: The last rendered image, or None if not rendered yet.
        """
        return self.last_rendered_lines_image

    def get_last_objectdetection_image(self):
        """
        Get the most recently rendered image with object detection drawings.
        Returns:
            np.ndarray or None: The last rendered image, or None if not rendered yet.
        """
        return self.last_rendered_objects_image