import os
import glob

def split_file(file_path, num_files):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Calculate number of lines per split
    total_lines = len(lines)
    lines_per_file = total_lines // num_files
    extra_lines = total_lines % num_files

    base_name, ext = os.path.splitext(file_path)

    start_index = 0
    for i in range(num_files):
        # Add one extra line to the first `extra_lines` files
        end_index = start_index + lines_per_file + (1 if i < extra_lines else 0)
        output_file_path = f"{base_name}{i + 1}{ext}"

        with open(output_file_path, 'w') as output_file:
            output_file.writelines(lines[start_index:end_index])

        print(f"Created file: {output_file_path} with lines {start_index} to {end_index - 1}")
        start_index = end_index


# Main logic to handle multiple files
def split_all_files_in_directory(directory, num_files, pattern="news*.txt"):
    # Find all files matching the pattern
    file_paths = glob.glob(os.path.join(directory, pattern))
    for file_path in file_paths:
        print(f"Splitting file: {file_path}")
        split_file(file_path, num_files)


def modify_text_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    total_lines = len(lines)

    if total_lines < 60000:
        with open(filename, 'a') as file:
            while total_lines < 60000:
                for line in lines:
                    file.write(line)
                    total_lines += 1
                    if total_lines >= 60000:
                        break
    elif total_lines > 60000:
        with open(filename, 'w') as file:
            file.writelines(lines[:60000])

def modify_all_files_in_directory(directory, pattern="news*.txt"):
    # Find all files matching the pattern
    file_paths = glob.glob(os.path.join(directory, pattern))
    for file_path in file_paths:
        print(f"modify file: {file_path}")
        modify_text_file(file_path)

# Usage
#split_all_files_in_directory(".", 2)  # "." for current directory, 10 for the number of parts
modify_all_files_in_directory(".","sport*.txt")