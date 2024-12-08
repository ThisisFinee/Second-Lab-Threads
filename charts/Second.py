import subprocess
import os
import math

def parse_execution_time(output):
    execution_time = None
    for line in output.splitlines():
        if "Время выполнения" in line:
            execution_time = float(line.split(":")[1].split()[0])
    return execution_time

def run_and_collect_results(processes, matrix_size, executable, output_file):
    command = f"mpirun -np {processes} {executable} {matrix_size}"
    try:
        result = subprocess.run(command, shell=True, text=True, capture_output=True, check=True)
        execution_time = parse_execution_time(result.stdout)
        if execution_time is not None:
            with open(output_file, "a") as f:
                f.write(f"{processes},{matrix_size},{execution_time}\n")
        else:
            print(f"Failed to parse execution time for command: {command}")
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}\n{e.stderr}")

def main():
    matrix_size_start = 512
    matrix_size_end = 2048
    matrix_size_step = 512
    max_processes = 15
    executable = "./Second.exe"
    output_dir = "charts/data_second"

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "results.csv")

    with open(output_file, "w") as f:
        # f.write("processes,matrix_size,execution_time\n")
        pass

    current_processes = 1
    while current_processes <= max_processes:
        root_processes = int(math.sqrt(current_processes))
        if root_processes ** 2 == current_processes:
            for matrix_size in range(matrix_size_start, matrix_size_end + 1, matrix_size_step):
                if matrix_size % root_processes == 0:
                    run_and_collect_results(current_processes, matrix_size, executable, output_file)
        current_processes += 1

    print(f"Results saved in '{output_file}'.")

if __name__ == "__main__":
    main()