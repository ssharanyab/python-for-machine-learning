import os
import glob
import subprocess

def convert_py_to_ipynb(directory='.'):
    """
    Finds all .py files in the given directory and its subdirectories
    (excluding the script itself and venv if present) and converts them to .ipynb using jupytext.
    """
    print("Starting conversion from .py to .ipynb...")
    py_files = glob.glob(f'{directory}/**/*.py', recursive=True)
    
    for py_file in py_files:
        if 'convert_notebooks.py' in py_file or 'venv' in py_file:
            continue
            
        print(f"Converting {py_file}...")
        try:
            # Using jupytext to convert the file
            subprocess.run(['jupytext', '--to', 'notebook', py_file], check=True)
            print(f"Successfully converted {py_file}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to convert {py_file}: {e}")

if __name__ == '__main__':
    convert_notebooks_dir = os.path.dirname(os.path.abspath(__file__))
    convert_py_to_ipynb(convert_notebooks_dir)
    print("Conversion process completed.")
