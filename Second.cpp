#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <chrono>

void initializeMatrix(std::vector<double>& matrix, int size, bool randomize = true) {
    if (randomize) {
        for (int i = 0; i < size * size; ++i) {
            matrix[i] = rand() % 100;
        }
    } else {
        std::fill(matrix.begin(), matrix.end(), 0.0);
    }
}

void multiplyBlock(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, int blockSize) {
    for (int i = 0; i < blockSize; ++i) {
        for (int j = 0; j < blockSize; ++j) {
            for (int k = 0; k < blockSize; ++k) {
                C[i * blockSize + j] += A[i * blockSize + k] * B[k * blockSize + j];
            }
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 512; // Размер матрицы (N x N)
    if (argc > 1) {
        N = atoi(argv[1]);
    }

    int sqrtSize = static_cast<int>(std::sqrt(size));
    if (sqrtSize * sqrtSize != size) {
        if (rank == 0) {
            std::cerr << "Количество процессов должно быть квадратом числа.\n";
        }
        MPI_Finalize();
        return -1;
    }

    int blockSize = N / sqrtSize;
    if (N % sqrtSize != 0) {
        if (rank == 0) {
            std::cerr << "Размер матрицы должен быть кратен корню из количества процессов.\n";
        }
        MPI_Finalize();
        return -1;
    }

    // Матрицы A, B и C для каждого процесса
    std::vector<double> A(blockSize * blockSize);
    std::vector<double> B(blockSize * blockSize);
    std::vector<double> C(blockSize * blockSize, 0.0);

    // Глобальные матрицы на корневом процессе
    std::vector<double> globalA, globalB, globalC;
    if (rank == 0) {
        globalA.resize(N * N);
        globalB.resize(N * N);
        globalC.resize(N * N, 0.0);
        initializeMatrix(globalA, N);
        initializeMatrix(globalB, N);
    }

    // Создание коммуникаторов 2D решетки
    MPI_Comm gridComm;
    int dims[2] = {sqrtSize, sqrtSize};
    int periods[2] = {1, 1}; // Циклические границы
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &gridComm);

    int coords[2];
    MPI_Cart_coords(gridComm, rank, 2, coords);

    int left, right, up, down;
    MPI_Cart_shift(gridComm, 1, -1, &right, &left);
    MPI_Cart_shift(gridComm, 0, -1, &down, &up);

    // Распределение блоков матриц A и B по процессам
    if (rank == 0) {
        for (int i = 0; i < sqrtSize; ++i) {
            for (int j = 0; j < sqrtSize; ++j) {
                int destRank;
                int destCoords[2] = {i, j};
                MPI_Cart_rank(gridComm, destCoords, &destRank);

                if (destRank == 0) {
                    for (int bi = 0; bi < blockSize; ++bi) {
                        for (int bj = 0; bj < blockSize; ++bj) {
                            A[bi * blockSize + bj] = globalA[(i * blockSize + bi) * N + (j * blockSize + bj)];
                            B[bi * blockSize + bj] = globalB[(i * blockSize + bi) * N + (j * blockSize + bj)];
                        }
                    }
                } else {
                    std::vector<double> tempA(blockSize * blockSize);
                    std::vector<double> tempB(blockSize * blockSize);
                    for (int bi = 0; bi < blockSize; ++bi) {
                        for (int bj = 0; bj < blockSize; ++bj) {
                            tempA[bi * blockSize + bj] = globalA[(i * blockSize + bi) * N + (j * blockSize + bj)];
                            tempB[bi * blockSize + bj] = globalB[(i * blockSize + bi) * N + (j * blockSize + bj)];
                        }
                    }
                    MPI_Send(tempA.data(), blockSize * blockSize, MPI_DOUBLE, destRank, 0, gridComm);
                    MPI_Send(tempB.data(), blockSize * blockSize, MPI_DOUBLE, destRank, 1, gridComm);
                }
            }
        }
    } else {
        MPI_Recv(A.data(), blockSize * blockSize, MPI_DOUBLE, 0, 0, gridComm, MPI_STATUS_IGNORE);
        MPI_Recv(B.data(), blockSize * blockSize, MPI_DOUBLE, 0, 1, gridComm, MPI_STATUS_IGNORE);
    }

    auto start = MPI_Wtime();

    // Инициализация сдвигов для алгоритма Кэннона
    for (int i = 0; i < coords[0]; ++i) {
        MPI_Sendrecv_replace(A.data(), blockSize * blockSize, MPI_DOUBLE, left, 0, right, 0, gridComm, MPI_STATUS_IGNORE);
    }
    for (int i = 0; i < coords[1]; ++i) {
        MPI_Sendrecv_replace(B.data(), blockSize * blockSize, MPI_DOUBLE, up, 0, down, 0, gridComm, MPI_STATUS_IGNORE);
    }

    // Основной цикл умножения с блоками
    for (int step = 0; step < sqrtSize; ++step) {
        multiplyBlock(A, B, C, blockSize);

        MPI_Sendrecv_replace(A.data(), blockSize * blockSize, MPI_DOUBLE, left, 0, right, 0, gridComm, MPI_STATUS_IGNORE);
        MPI_Sendrecv_replace(B.data(), blockSize * blockSize, MPI_DOUBLE, up, 0, down, 0, gridComm, MPI_STATUS_IGNORE);
    }

    auto end = MPI_Wtime();

    // Сбор результатов на корневом процессе
    if (rank == 0) {
        for (int i = 0; i < sqrtSize; ++i) {
            for (int j = 0; j < sqrtSize; ++j) {
                int sourceRank;
                int sourceCoords[2] = {i, j};
                MPI_Cart_rank(gridComm, sourceCoords, &sourceRank);

                if (sourceRank == 0) {
                    for (int bi = 0; bi < blockSize; ++bi) {
                        for (int bj = 0; bj < blockSize; ++bj) {
                            globalC[(i * blockSize + bi) * N + (j * blockSize + bj)] = C[bi * blockSize + bj];
                        }
                    }
                } else {
                    std::vector<double> tempC(blockSize * blockSize);
                    MPI_Recv(tempC.data(), blockSize * blockSize, MPI_DOUBLE, sourceRank, 2, gridComm, MPI_STATUS_IGNORE);
                    for (int bi = 0; bi < blockSize; ++bi) {
                        for (int bj = 0; bj < blockSize; ++bj) {
                            globalC[(i * blockSize + bi) * N + (j * blockSize + bj)] = tempC[bi * blockSize + bj];
                        }
                    }
                }
            }
        }

        std::cout << "Время выполнения: " << (end - start) << " секунд\n";
    } else {
        MPI_Send(C.data(), blockSize * blockSize, MPI_DOUBLE, 0, 2, gridComm);
    }

    MPI_Finalize();
    return 0;
}