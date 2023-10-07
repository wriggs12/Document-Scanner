#include "../include/scanner.h"
#include <iostream>

int main(int argc, char** argv)
{
    std::string filename = argv[1];
    cv::VideoCapture cap;
    cap.open(filename);

    double fps = cap.get(cv::CAP_PROP_FPS);

    while (1)
    {
        cv::Mat frame;

        if (!cap.read(frame))
        {
            std::cout << "Error reading frame\n";
            break;
        }

        cv::imshow("Lane Detection", frame);

        if (cv::waitKey(30) == 27)
        {
            break;
        }
    }

    return 0;

    return 0;
}