#ifndef __APP__
#define __APP__

#include <Windows.h>
#include <Kinect.h>
#include <opencv2/opencv.hpp>

#include <vector>

#include <wrl/client.h>
using namespace Microsoft::WRL;

class Kinect
{
    public:

        Kinect();
        ~Kinect();

        void run();

    private:

        void initializeCapture();
        void initializeSensor();
        void initializeColor();
        void initializeDepth();
        void initializeVideoWriter();

        void finalize();

        bool readImages();
        bool readColor();
        bool readDepth();

        void accumulateBackground();
        void compositeScene();

        void render();

        static void fillDepthHoles(cv::Mat& im, UINT16 minReliableDistance, UINT16 maxReliableDistance);

    private:
        // Sensor
        ComPtr<IKinectSensor> kinect;

        // Coordinate Mapper
        ComPtr<ICoordinateMapper> coordinateMapper;

        // Reader
        ComPtr<IColorFrameReader> colorFrameReader;
        ComPtr<IDepthFrameReader> depthFrameReader;
        UINT16 minReliableDistance;
        UINT16 maxReliableDistance;

        // Video Writer
        cv::VideoWriter video_writer;

        // Color Buffer
        std::vector<BYTE> colorBuffer;
        int colorWidth;
        int colorHeight;
        unsigned int colorBytesPerPixel;
        cv::Mat colorMat;
        cv::Size colorMatSize;

        // Depth Buffer
        std::vector<UINT16> depthBuffer;
        int depthWidth;
        int depthHeight;
        unsigned int depthBytesPerPixel;
        cv::Mat depthMat;

        // Video storage
        static const size_t n_frames = 60;
        cv::Mat depth_frames[n_frames];
        cv::Mat color_frames[n_frames];
        int iFrame;

        cv::Mat depthMat0;

        const double scale = 0.6;
        const cv::Rect crop;

        std::string window_title;
};

#endif // __APP__
