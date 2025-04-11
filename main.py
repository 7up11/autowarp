from ultralytics import YOLO
from scipy.spatial.distance import euclidean
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2 as cv
import numpy as np
import tkinter as tk
import math


# Credit: ChatGPT
def perpendicular_distance(point, start, end):
    """Calculate the perpendicular distance from a point to a line segment."""
    if np.all(start == end):
        return euclidean(point, start)

    line_vec = end - start
    point_vec = point - start
    line_len = np.dot(line_vec, line_vec)
    t = max(0, min(1, np.dot(point_vec, line_vec) / line_len))
    projection = start + t * line_vec
    return euclidean(point, projection)


# Credit: ChatGPT
def rdp(contour, epsilon):
    """
    Applies the Ramer-Douglas-Peucker (RDP) algorithm to simplify a contour.

    Parameters:
        contour (np.ndarray): A Nx2 array of (x, y) coordinates.
        epsilon (float): Tolerance for point reduction (higher -> more aggressive).

    Returns:
        np.ndarray: Simplified contour as a Nx2 array.
    """
    if len(contour) < 3:
        return contour

    # Find the point with the maximum perpendicular distance
    start, end = contour[0], contour[-1]
    distances = np.array([perpendicular_distance(p, start, end) for p in contour[1:-1]])

    max_idx = np.argmax(distances)
    max_dist = distances[max_idx]

    if max_dist > epsilon:
        max_idx += 1  # Offset for skipping the first point
        # Recursive RDP on both segments
        left = rdp(contour[:max_idx + 1], epsilon)
        right = rdp(contour[max_idx:], epsilon)
        return np.vstack((left[:-1], right))
    else:
        return np.array([start, end])


def simplify(contour):
    points = contour.shape[0]
    if points == 4:
        return contour
    full_area = cv.contourArea(contour)
    d_min = math.inf
    point_min = None
    for point in range(points):
        area = cv.contourArea(np.delete(contour, point, axis=0))
        if (d := abs(full_area - area)) <= d_min:
            d_min = d
            point_min = point
    if point_min is None:
        raise ValueError("That's cool I guess.")
    return simplify(np.delete(contour, point_min, axis=0))


def warp(src, src_quad, dst_quad):
    h, _ = cv.findHomography(dst_quad, src_quad)
    _, _, w_dst, h_dst = cv.boundingRect(dst_quad)
    return cv.warpPerspective(src, h, (w_dst, h_dst), flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)


def normal_to_image(image_shape, contour):
    h, w = image_shape[:2]
    return (contour * (w, h)).astype(int)


class OutputWindow(tk.Toplevel):
    def __init__(self, parent, image):
        super().__init__(parent)
        self.title("Output")
        self.pil_image = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        self.tk_image = ImageTk.PhotoImage(self.pil_image)
        self.canvas = tk.Canvas(self)
        self.canvas.pack(fill=tk.BOTH, expand=tk.YES)
        self.canvas.bind("<Configure>", self.resize)
        self.canvas_image = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

    def resize(self, event):
        resized = self.pil_image.resize((event.width, event.height), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized)
        self.canvas.itemconfig(self.canvas_image, image=self.tk_image)
        self.canvas.coords(self.canvas_image, 0, 0)


class InputWindow(tk.Tk):
    def __init__(self, weights):
        super().__init__()
        self.yolo = YOLO(weights)
        self.quads = []

        self.title("Autowarp")
        self.geometry("1080x720")
        self.frame = ttk.Frame(self, padding=10)
        self.frame.grid(row=0, column=0, sticky=tk.NSEW)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.open_button = ttk.Button(self.frame, text="Open Image", command=self.open_image)
        self.open_button.grid(row=0, column=0)
        self.image = None
        self.tk_image = None
        self.label = tk.Label(self.frame)
        self.label.grid(row=1, column=0, sticky=tk.NSEW)
        self.label.bind("<Button-1>", self.click)
        self.frame.rowconfigure(1, weight=1)
        self.frame.columnconfigure(0, weight=1)

        self.output = None

    def inference(self, path):
        self.image = cv.imread(path)
        masks = self.yolo(self.image)[0].masks
        if not masks:
            self.quads = []
        else:
            self.quads = [simplify(rdp(mask, 0.01)) for mask in masks.xyn]

    def image_size(self):
        h_label = self.label.winfo_height()
        h_image, w_image = self.image.shape[:2]
        return int(w_image / h_image * h_label), h_label

    def open_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")]
        )
        if not path:
            return
        self.inference(path)
        quads = [normal_to_image(self.image.shape, quad) for quad in self.quads]
        image = self.image.copy()
        image = cv.polylines(image, quads, True, (0, 0, 255), thickness=3)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = Image.fromarray(image).resize(self.image_size(), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(image)
        self.label.config(image=self.tk_image)

    def click(self, event):
        if len(self.quads) == 0:
            return
        w, h = self.image_size()
        x = event.x - (self.label.winfo_width() - w) // 2
        y = event.y
        if x < 0 or y < 0 or x >= w or y >= h:
            return
        for quad in self.quads:
            quad_ui = normal_to_image(self.image_size()[::-1], quad)
            if cv.pointPolygonTest(quad_ui, (x, y), False) > -1:
                quad_image = normal_to_image(self.image.shape, quad)
                dst = np.array([
                    [0, 0], [0, 480], [640, 480], [640, 0]
                ])
                warped = warp(self.image, quad_image, dst)
                if self.output:
                    self.output.destroy()
                self.output = OutputWindow(self, warped)


if __name__ == "__main__":
    image = cv.imread("datasets/coco2017/val/images/000000195918.jpg")
    root = InputWindow("runs/segment/filtered-5e/weights/best.pt")
    root.mainloop()

