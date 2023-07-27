#include "../include/scanner.h"
#include <iostream>

int main()
{
    cv::VideoCapture cap(0);

    if (!cap.isOpened())
    {
        std::cout << "Use a different camera port number!" << std::endl;
        return -1;
    }

    while (true)
    {
        cv::Mat frame;
        cap.read(frame);
        imshow("camera", frame);
        cv::waitKey(1);
    }

    return 0;
}