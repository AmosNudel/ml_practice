import tkinter as tk
from tkinter import filedialog, messagebox
import os
import subprocess
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # Project root
PY_SCRIPTS_DIR = os.path.join(BASE_DIR, 'py_scripts')


def select_script():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    script_path = filedialog.askopenfilename(
        title='Select Python Script',
        initialdir=PY_SCRIPTS_DIR,
        filetypes=[('Python Files', '*.py')]
    )
    root.destroy()
    return script_path


def script_path_to_module(script_path):
    rel_path = os.path.relpath(script_path, PY_SCRIPTS_DIR)
    if rel_path.startswith('..'):
        return None
    module = rel_path[:-3].replace(os.sep, '.')
    return f'py_scripts.{module}'


def main():
    print(f'Current working directory: {os.getcwd()}')
    print(f'sys.path: {sys.path}')
    print(f'BASE_DIR: {BASE_DIR}')
    print(f'PY_SCRIPTS_DIR: {PY_SCRIPTS_DIR}')
    script_path = select_script()
    if not script_path:
        print('No script selected. Exiting.')
        sys.exit(0)
    module_name = script_path_to_module(script_path)
    if not module_name:
        print('Selected script is not inside py_scripts. Exiting.')
        sys.exit(1)
    print(f'Running module: {module_name}')
    env = os.environ.copy()
    env['PYTHONPATH'] = BASE_DIR
    print(f'PYTHONPATH for subprocess: {env["PYTHONPATH"]}')
    try:
        result = subprocess.run([
            sys.executable, '-m', module_name
        ], cwd=BASE_DIR, env=env)
        sys.exit(result.returncode)
    except Exception as e:
        print(f'Error running script: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()
