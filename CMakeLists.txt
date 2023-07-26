cmake_minimum_required(VERSION 3.5)
project(document_scanner)
find_package(OpenCV REQUIRED)

add_executable(document_scanner ./src/scanner.h)
target_link_libraries(document_scanner ${OpenCV_LIBS})
