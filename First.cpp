#include <mpi.h>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <chrono>

// Функция для замера времени выполнения
double measure_algorithm_time(const std::function<void()>& algorithm) {
    auto start_time = MPI_Wtime();
    algorithm();
    auto end_time = MPI_Wtime();
    return end_time - start_time;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3) {
        if (rank == 0)
            std::cerr << "Usage: " << argv[0] << " <rows> <cols>\n";
        MPI_Finalize();
        return 1;
    }

    int rows = std::atoi(argv[1]);
    int cols = std::atoi(argv[2]);

    // Инициализация матрицы и вектора
    std::vector<std::vector<double>> matrix(rows, std::vector<double>(cols, 1.0));
    std::vector<double> vector(cols, 1.0);
    std::vector<double> result(rows, 0.0);

    // Временные переменные для каждого подхода
    double time_by_rows = 0.0;
    double time_by_columns = 0.0;
    double time_by_blocks = 0.0;

    // 1. Разбиение по строкам
    time_by_rows = measure_algorithm_time([&]() {
        for (int i = rank; i < rows; i += size) {
            for (int j = 0; j < cols; ++j) {
                result[i] += matrix[i][j] * vector[j];
            }
        }
        MPI_Reduce(rank == 0 ? MPI_IN_PLACE : result.data(), result.data(), rows, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    });

    // 2. Разбиение по столбцам
    time_by_columns = measure_algorithm_time([&]() {
        std::vector<double> partial_result(rows, 0.0);
        for (int j = rank; j < cols; j += size) {
            for (int i = 0; i < rows; ++i) {
                partial_result[i] += matrix[i][j] * vector[j];
            }
        }
        MPI_Reduce(rank == 0 ? MPI_IN_PLACE : partial_result.data(), result.data(), rows, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    });

    // 3. Разбиение на блоки
    int block_size = rows / size;
    time_by_blocks = measure_algorithm_time([&]() {
        int start_row = rank * block_size;
        int end_row = (rank == size - 1) ? rows : start_row + block_size;

        for (int i = start_row; i < end_row; ++i) {
            for (int j = 0; j < cols; ++j) {
                result[i] += matrix[i][j] * vector[j];
            }
        }
        MPI_Reduce(rank == 0 ? MPI_IN_PLACE : result.data(), result.data(), rows, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    });

    // Вывод времени на главном процессе
    if (rank == 0) {
        std::cout << "Time for row-based partitioning: " << time_by_rows << " seconds\n";
        std::cout << "Time for column-based partitioning: " << time_by_columns << " seconds\n";
        std::cout << "Time for block-based partitioning: " << time_by_blocks << " seconds\n";
    }

    MPI_Finalize();
    return 0;
}
