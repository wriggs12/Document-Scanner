#include "../include/scanner.h"
#include <iostream>

int main(int argc, char** argv)
{
    cv::namedWindow("Example 2-10", cv::WINDOW_AUTOSIZE);
    cv::VideoCapture cap;

    if (argc == 1)
    {
        cap.open(0);
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 320);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 260);
        cap.set(cv::CAP_PROP_FPS, 30);
        cap.set(cv::CAP_PROP_FOURCC,  cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    }
    else
    {
        std::cout << "HERE" << std::endl;
    }

    if (!cap.isOpened())
    {
        std::cerr << "Use a different camera port number!" << std::endl;
        return -1;
    }

    // while (true)
    // {
    //     cv::Mat frame;
    //     cap.read(frame);
    //     imshow("camera", frame);
    //     cv::waitKey(1);
    // }

    return 0;
}