import os
from datetime import datetime

# Add any directories or files you want to skip here
EXCLUDE = {'node_modules', '.git', '__pycache__', 'dist', 'build', '.venv', 'venv', 'structure'}

def build_tree(dir_path, prefix=""):
    lines = []
    contents = [c for c in os.listdir(dir_path) if c not in EXCLUDE]
    contents.sort()
    pointers = ['├── '] * (len(contents) - 1) + ['└── '] if contents else []
    
    for pointer, name in zip(pointers, contents):
        path = os.path.join(dir_path, name)
        lines.append(f"{prefix}{pointer}{name}")
        if os.path.isdir(path):
            extension = '│   ' if pointer == '├── ' else '    '
            lines.extend(build_tree(path, prefix + extension))
    return lines

def main():
    # Ensure 'structure' folder exists
    output_dir = "structure"
    os.makedirs(output_dir, exist_ok=True)

    # Create timestamped filename: structure_2026-07-29_11-11-05.txt
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filepath = os.path.join(output_dir, f"structure_{timestamp}.txt")

    # Generate tree lines
    tree_lines = [os.path.basename(os.getcwd()) + "/"] + build_tree(".")

    # Write to file
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(tree_lines))

    print(f"Directory structure saved to: {filepath}")

if __name__ == "__main__":
    main()