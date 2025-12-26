import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk


class TiffViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Python TIFF Viewer")
        self.root.geometry("800x600")

        self.image_path = None
        self.img_data = None
        self.current_frame = 0
        self.total_frames = 0

        self.create_widgets()

    def create_widgets(self):
        # Toolbar
        self.toolbar = tk.Frame(self.root, pady=5)
        self.toolbar.pack(side=tk.TOP, fill=tk.X)

        tk.Button(self.toolbar, text="Open TIFF", command=self.load_file).pack(side=tk.LEFT, padx=5)

        self.prev_btn = tk.Button(self.toolbar, text="◀ Prev", command=self.prev_page, state=tk.DISABLED)
        self.prev_btn.pack(side=tk.LEFT, padx=2)

        self.next_btn = tk.Button(self.toolbar, text="Next ▶", command=self.next_page, state=tk.DISABLED)
        self.next_btn.pack(side=tk.LEFT, padx=2)

        self.page_label = tk.Label(self.toolbar, text="Page: 0/0")
        self.page_label.pack(side=tk.LEFT, padx=10)

        # Image Display Area
        self.canvas = tk.Canvas(self.root, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def load_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("TIFF Files", "*.tif *.tiff"), ("All Files", "*.*")]
        )
        if file_path:
            try:
                self.image_path = file_path
                self.img_data = Image.open(self.image_path)
                self.total_frames = getattr(self.img_data, "n_frames", 1)
                self.current_frame = 0
                self.show_frame()

                if self.total_frames > 1:
                    self.next_btn.config(state=tk.NORMAL)
                    self.prev_btn.config(state=tk.NORMAL)
            except Exception as e:
                messagebox.showerror("Error", f"Could not open image: {e}")

    def show_frame(self):
        if self.img_data:
            self.img_data.seek(self.current_frame)

            # Resize image to fit canvas while maintaining aspect ratio
            display_img = self.img_data.copy()
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            # Initial load might have 1x1 canvas, use window size as fallback
            width = canvas_width if canvas_width > 1 else 800
            height = canvas_height if canvas_height > 1 else 600

            display_img.thumbnail((width, height), Image.Resampling.LANCZOS)

            self.photo = ImageTk.PhotoImage(display_img)
            self.canvas.delete("all")
            self.canvas.create_image(width // 2, height // 2, image=self.photo, anchor=tk.CENTER)

            self.page_label.config(text=f"Page: {self.current_frame + 1}/{self.total_frames}")

    def next_page(self):
        if self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self.show_frame()

    def prev_page(self):
        if self.current_frame > 0:
            self.current_frame -= 1
            self.show_frame()


if __name__ == "__main__":
    root = tk.Tk()
    app = TiffViewer(root)
    root.mainloop()