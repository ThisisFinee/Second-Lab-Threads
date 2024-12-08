#include <mpi.h>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <functional> // Для std::function
#include <fstream>

// Функция для замера времени выполнения
double measure_algorithm_time(const std::function<void()>& algorithm) {
    double start_time = MPI_Wtime();
    algorithm();
    double end_time = MPI_Wtime();
    return end_time - start_time;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Установка диапазонов для перебора размеров матрицы
    int min_rows = 1000, max_rows = 10000, step_rows = 1000;
    int min_cols = 10000, max_cols = 10000, step_cols = 1;

    // Цикл для автоматического перебора значений rows и cols
    for (int rows = min_rows; rows <= max_rows; rows += step_rows) {
        for (int cols = min_cols; cols <= max_cols; cols += step_cols) {
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
                std::ofstream results_file("charts/data/results.csv", std::ios::app);
                results_file << size << "," << rows << "," << cols << "," << time_by_rows << "," << time_by_columns << "," << time_by_blocks << "\n";
                results_file.close();
            }
        }
    }

    MPI_Finalize();
    return 0;
}
