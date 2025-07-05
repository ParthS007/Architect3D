input_file = "/work/courses/3dv/20/scannetpp_openmask3d-main/scannetpp/LABEL_FILES/architectural_instances.txt"          # Your source file
output_file = "/work/courses/3dv/20/scannetpp_openmask3d-main/scannetpp/LABEL_FILES/output.txt"        # File to save the LaTeX string

# Read and clean the input
with open(input_file, "r") as f:
    items = [line.strip() for line in f if line.strip()]

# Format into LaTeX-style italic string
latex_string = r"\\textit{" + ", ".join(items) + "}"

# Write to output file
with open(output_file, "w") as f:
    f.write(latex_string)

print(f"Formatted LaTeX string written to {output_file}")