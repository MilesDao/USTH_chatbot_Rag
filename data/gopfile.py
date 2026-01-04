from pathlib import Path

base_dir = Path("output")
input_dir = base_dir / "docai_text"
output_file = input_dir / "merged_ALLfile.txt"

all_files = sorted(input_dir.glob("*.ALL*.txt"))

with open(output_file, "w", encoding="utf-8") as outfile:
    for file in all_files:
        with open(file, "r", encoding="utf-8", errors="ignore") as infile:
            for line in infile:
                outfile.write(line)

print(f"Đã ghép {len(all_files)} file vào {output_file}")
