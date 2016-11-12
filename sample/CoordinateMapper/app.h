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
        // Sensor
        ComPtr<IKinectSensor> kinect;

        // Coordinate Mapper
        ComPtr<ICoordinateMapper> coordinateMapper;

        // Reader
        ComPtr<IColorFrameReader> colorFrameReader;
        ComPtr<IDepthFrameReader> depthFrameReader;

        // Color Buffer
        std::vector<BYTE> colorBuffer;
        int colorWidth;
        int colorHeight;
        unsigned int colorBytesPerPixel;
        cv::Mat colorMat;

        // Depth Buffer
        std::vector<UINT16> depthBuffer;
        int depthWidth;
        int depthHeight;
        unsigned int depthBytesPerPixel;
        cv::Mat depthMat;

        cv::Mat depthMat0, colorMat0;
        bool got_background;
        int n_bg_frames_captured;
        const double scale = 0.4;

    private:

        void initialize();
        inline void initializeSensor();
        inline void initializeColor();
        inline void initializeDepth();

        void finalize();

        void update();
        inline void updateColor();
        inline void updateDepth();

        void draw();
        inline void drawColor();
        inline void drawDepth();

        void show();
        inline void showColor();
        inline void showDepth();
};

#endif // __APP__
