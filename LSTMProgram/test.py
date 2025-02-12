def trim_file(input_file, output_file, target_line):
    with open(input_file, 'r', encoding = 'utf-8') as f:
        lines = f.readlines()

    # Find the index of the target line
    start_index = next((i for i, line in enumerate(lines) if line.strip() == target_line.strip()),
                       None)

    if start_index is not None:
        # Write only the content from the target line onward
        with open(output_file, 'w', encoding = 'utf-8') as f:
            f.writelines(lines[start_index:])
        print(f"File trimmed successfully. Saved to {output_file}")
    else:
        print("Target line not found in the file.")


# Example usage
input_filename = "news4.txt"
output_filename = "news44.txt"
target_line = "Timestamp: 2/3/2025 10:53:12 PM, Millisecond: 420,current eyeGaze_left: Position (-0.1281794, 0.6299132, 0.1658499), Orientation(0.05937295, -0.02258949, -0.01263221, 0.9979004), Confidence: 0.9980469"

trim_file(input_filename, output_filename, target_line)
