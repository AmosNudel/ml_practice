from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os

Tk().withdraw()  # Hide the root window

data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
csv_path = askopenfilename(
    title="Select CSV File",
    filetypes=[("CSV files", "*.csv")],
    initialdir=data_dir
)

if not csv_path:
    raise ValueError("No file selected!")
