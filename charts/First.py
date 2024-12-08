import subprocess
import os

def parse_times(output):
    time_by_rows = time_by_columns = time_by_blocks = None
    for line in output.splitlines():
        if "Time for row-based partitioning" in line:
            time_by_rows = float(line.split(":")[1].split()[0])
        elif "Time for column-based partitioning" in line:
            time_by_columns = float(line.split(":")[1].split()[0])
        elif "Time for block-based partitioning" in line:
            time_by_blocks = float(line.split(":")[1].split()[0])
    return time_by_rows, time_by_columns, time_by_blocks

def run_and_collect_results(processes, rows, cols, executable, output_file):
    command = f"mpirun -np {processes} {executable} {rows} {cols}"
    try:
        result = subprocess.run(command, shell=True, text=True, capture_output=True, check=True)
        times = parse_times(result.stdout)
        if all(t is not None for t in times):
            with open(output_file, "a") as f:
                f.write(f"{processes},{rows},{cols},{times[0]},{times[1]},{times[2]}\n")
        else:
            print(f"Failed to parse times for command: {command}")
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}\n{e.stderr}")

def main():
    row_col_start = 1000
    row_col_end = 10000
    row_col_step = 1000
    num_processes = [1, 2, 4, 6, 8]
    executable = "./First.exe"
    output_dir = "charts/data_first"

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "results.csv")

    with open(output_file, "w") as f:
        # f.write("processes,rows,columns,tyme_by_rows,tyme_by_columns,tyme_by_blocks\n")
        pass

    for rc in range(row_col_start, row_col_end + 1, row_col_step):
        for np in num_processes:
            run_and_collect_results(np, rc, rc, executable, output_file)

    print(f"Results saved in '{output_file}'.")

if __name__ == "__main__":
    main()
