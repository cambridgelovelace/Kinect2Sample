#include "app.h"
#include "util.h"

#include <thread>
#include <chrono>

#include <omp.h>

Kinect::Kinect()
    : got_background(false)
    , n_bg_frames_captured(0)
{
    initialize();
}

Kinect::~Kinect()
{
    finalize();
}

void Kinect::run()
{
    while( true ){
        update();
        draw();
        show();
        const int key = cv::waitKey( 10 );
        if( key == VK_ESCAPE ){
            break;
        }
    }
}

void Kinect::initialize()
{
    cv::setUseOptimized( true );

    initializeSensor();
    initializeColor();
    initializeDepth();

    // Wait a Few Seconds until begins to Retrieve Data from Sensor ( about 2000-[ms] )
    std::this_thread::sleep_for( std::chrono::seconds( 2 ) );
}

inline void Kinect::initializeSensor()
{
    // Open Sensor
    ERROR_CHECK( GetDefaultKinectSensor( &kinect ) );
    ERROR_CHECK( kinect->Open() );

    // Check Open
    BOOLEAN isOpen = FALSE;
    ERROR_CHECK( kinect->get_IsOpen( &isOpen ) );
    if( !isOpen ){
        throw std::runtime_error( "failed IKinectSensor::get_IsOpen( &isOpen )" );
    }

    // Retrieve Coordinate Mapper
    ERROR_CHECK( kinect->get_CoordinateMapper( &coordinateMapper ) );
}

inline void Kinect::initializeColor()
{
    // Open Color Reader
    ComPtr<IColorFrameSource> colorFrameSource;
    ERROR_CHECK( kinect->get_ColorFrameSource( &colorFrameSource ) );
    ERROR_CHECK( colorFrameSource->OpenReader( &colorFrameReader ) );

    // Retrieve Color Description
    ComPtr<IFrameDescription> colorFrameDescription;
    ERROR_CHECK( colorFrameSource->CreateFrameDescription( ColorImageFormat::ColorImageFormat_Bgra, &colorFrameDescription ) );
    ERROR_CHECK( colorFrameDescription->get_Width( &colorWidth ) ); // 1920
    ERROR_CHECK( colorFrameDescription->get_Height( &colorHeight ) ); // 1080
    ERROR_CHECK( colorFrameDescription->get_BytesPerPixel( &colorBytesPerPixel ) ); // 4

    // Allocation Color Buffer
    colorBuffer.resize( colorWidth * colorHeight * colorBytesPerPixel );
}

inline void Kinect::initializeDepth()
{
    // Open Depth Reader
    ComPtr<IDepthFrameSource> depthFrameSource;
    ERROR_CHECK( kinect->get_DepthFrameSource( &depthFrameSource ) );
    ERROR_CHECK( depthFrameSource->OpenReader( &depthFrameReader ) );

    // Retrieve Depth Description
    ComPtr<IFrameDescription> depthFrameDescription;
    ERROR_CHECK( depthFrameSource->get_FrameDescription( &depthFrameDescription ) );
    ERROR_CHECK( depthFrameDescription->get_Width( &depthWidth ) ); // 512
    ERROR_CHECK( depthFrameDescription->get_Height( &depthHeight ) ); // 424
    ERROR_CHECK( depthFrameDescription->get_BytesPerPixel( &depthBytesPerPixel ) ); // 2

    // Allocation Depth Buffer
    depthBuffer.resize( depthWidth * depthHeight );
}

void Kinect::finalize()
{
    cv::destroyAllWindows();

    // Close Sensor
    if( kinect != nullptr ){
        kinect->Close();
    }
}

void Kinect::update()
{
    updateColor();
    updateDepth();
}

inline void Kinect::updateColor()
{
    // Retrieve Color Frame
    ComPtr<IColorFrame> colorFrame;
    const HRESULT ret = colorFrameReader->AcquireLatestFrame( &colorFrame );
    if( FAILED( ret ) ){
        return;
    }

    // Convert Format ( YUY2 -> BGRA )
    ERROR_CHECK( colorFrame->CopyConvertedFrameDataToArray( static_cast<UINT>( colorBuffer.size() ), &colorBuffer[0], ColorImageFormat::ColorImageFormat_Bgra ) );
}

inline void Kinect::updateDepth()
{
    // Retrieve Depth Frame
    ComPtr<IDepthFrame> depthFrame;
    const HRESULT ret = depthFrameReader->AcquireLatestFrame( &depthFrame );
    if( FAILED( ret ) ){
        return;
    }

    // Retrieve Depth Data
    ERROR_CHECK( depthFrame->CopyFrameDataToArray( static_cast<UINT>( depthBuffer.size() ), &depthBuffer[0] ) );
}

void Kinect::draw()
{
    drawColor();
    drawDepth();
}

inline void Kinect::drawColor()
{
    colorMat = cv::Mat( colorHeight, colorWidth, CV_8UC4, &colorBuffer[0] ).clone();
}

inline void Kinect::drawDepth()
{
    // Retrieve Mapped Coordinates
    std::vector<DepthSpacePoint> depthSpacePoints(colorWidth * colorHeight);
    ERROR_CHECK(coordinateMapper->MapColorFrameToDepthSpace((UINT)depthBuffer.size(),&depthBuffer[0],(UINT)depthSpacePoints.size(),&depthSpacePoints[0]));

    // Mapping Depth to Color Resolution
    std::vector<UINT16> buffer(colorWidth * colorHeight);

    #pragma omp parallel for
    for (int colorY = 0; colorY < colorHeight; colorY++) {
        unsigned int colorOffset = colorY * colorWidth;
        for (int colorX = 0; colorX < colorWidth; colorX++) {
            unsigned int colorIndex = colorOffset + colorX;
            int depthX = static_cast<int>(depthSpacePoints[colorIndex].X + 0.5f);
            int depthY = static_cast<int>(depthSpacePoints[colorIndex].Y + 0.5f);
            if ((0 <= depthX) && (depthX < depthWidth) && (0 <= depthY) && (depthY < depthHeight)) {
                unsigned int depthIndex = depthY * depthWidth + depthX;
                buffer[colorIndex] = depthBuffer[depthIndex];
            }
        }
    }

    // Create cv::Mat from Coordinate Buffer
    depthMat = cv::Mat(colorHeight, colorWidth, CV_16UC1, &buffer[0]).clone();

    // Accumulate the static background depth. Assume the camera doesn't move and the scene doesn't move too much.
    if (!got_background)
    {
        if (n_bg_frames_captured == 0)
            depthMat0 = depthMat.clone();
        else {
            cv::Mat av = 0.99 * depthMat0 + 0.01 * depthMat;
            av.copyTo(depthMat0, (depthMat > 0) & (depthMat0 > 0)); // average where both known
            depthMat.copyTo(depthMat0, depthMat0 == 0); // replace any unknown pixels
        }
        n_bg_frames_captured++;
        if (n_bg_frames_captured > 100)
        {
            got_background = true;
            // fill in the holes
            cv::Mat infilled;
            depthMat0.setTo(8000, depthMat0 == 0);
            cv::erode(depthMat0, infilled, cv::Mat(), cv::Point(-1, -1), 10);
            cv::dilate(infilled, infilled, cv::Mat(), cv::Point(-1, -1), 10);
            infilled.copyTo(depthMat0, depthMat0 == 8000);
        }
    }
}

void Kinect::show()
{
    if (got_background)
        showColor();
    else
        showDepth();
}

inline void Kinect::showColor()
{
    if( colorMat.empty() || depthMat.empty() || depthMat0.empty() ){
        return;
    }

    // mask out the background
    depthMat.setTo(9000, depthMat == 0); // put unknown depths to far away
    cv::Mat mask = depthMat < (depthMat0 - 100); // only keep pixels nearer than the background
    // remove speckle
    cv::erode(mask, mask, cv::Mat(), cv::Point(-1, -1), 1);
    //cv::dilate(mask, mask, cv::Mat(), cv::Point(-1, -1), 1);
    colorMat.setTo(0, mask == 0);

    cv::Mat scaledColor;
    cv::resize(colorMat, scaledColor, cv::Size(), scale, scale);

    cv::imshow("Tango", scaledColor);
}

// Show Depth
inline void Kinect::showDepth()
{
    if( depthMat0.empty() ){
        return;
    }

    // Scaling ( 0-8000 -> 255-0 )
    cv::Mat scaleMat;
    depthMat0.convertTo( scaleMat, CV_8U, -255.0 / 8000.0, 255.0 );

    cv::Mat scaled;
    cv::resize(scaleMat, scaled, cv::Size(), scale, scale);

    cv::imshow("Tango", scaled );
}
