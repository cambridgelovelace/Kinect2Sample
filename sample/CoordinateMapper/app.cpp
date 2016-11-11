#include "app.h"
#include "util.h"

#include <thread>
#include <chrono>

#include <omp.h>

// Choose Streams
#define COLOR
//#define DEPTH

// Constructor
Kinect::Kinect()
    : got_background(false)
    , n_bg_frames_captured(0)
{
    // Initialize
    initialize();
}

// Destructor
Kinect::~Kinect()
{
    // Finalize
    finalize();
}

// Processing
void Kinect::run()
{
    // Main Loop
    while( true ){
        // Update Data
        update();

        // Draw Data
        draw();

        // Show Data
        show();

        // Key Check
        const int key = cv::waitKey( 10 );
        if( key == VK_ESCAPE ){
            break;
        }
    }
}

// Initialize
void Kinect::initialize()
{
    cv::setUseOptimized( true );

    // Initialize Sensor
    initializeSensor();

    // Initialize Color
    initializeColor();

    // Initialize Depth
    initializeDepth();

    // Wait a Few Seconds until begins to Retrieve Data from Sensor ( about 2000-[ms] )
    std::this_thread::sleep_for( std::chrono::seconds( 2 ) );
}

// Initialize Sensor
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

// Initialize Color
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

// Initialize Depth
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

// Finalize
void Kinect::finalize()
{
    cv::destroyAllWindows();

    // Close Sensor
    if( kinect != nullptr ){
        kinect->Close();
    }
}

// Update Data
void Kinect::update()
{
    // Update Color
    updateColor();

    // Update Depth
    updateDepth();
}

// Update Color
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

// Update Depth
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

// Draw Data
void Kinect::draw()
{
    drawColor();
    drawDepth();
}

// Draw Color
inline void Kinect::drawColor()
{
    // Retrieve Mapped Coordinates
    std::vector<ColorSpacePoint> colorSpacePoints( depthWidth * depthHeight );
    ERROR_CHECK( coordinateMapper->MapDepthFrameToColorSpace( depthBuffer.size(), &depthBuffer[0], colorSpacePoints.size(), &colorSpacePoints[0] ) );

    // Mapping Color to Depth Resolution
    std::vector<BYTE> buffer( depthWidth * depthHeight * colorBytesPerPixel );

    #pragma omp parallel for
    for( int depthY = 0; depthY < depthHeight; depthY++ ){
        unsigned int depthOffset = depthY * depthWidth;
        for( int depthX = 0; depthX < depthWidth; depthX++ ){
            unsigned int depthIndex = depthOffset + depthX;
            int colorX = static_cast<int>( colorSpacePoints[depthIndex].X + 0.5f );
            int colorY = static_cast<int>( colorSpacePoints[depthIndex].Y + 0.5f );
            if( ( 0 <= colorX ) && ( colorX < colorWidth ) && ( 0 <= colorY ) && ( colorY < colorHeight ) ){
                unsigned int colorIndex = ( colorY * colorWidth + colorX ) * colorBytesPerPixel;
                depthIndex = depthIndex * colorBytesPerPixel;
                buffer[depthIndex + 0] = colorBuffer[colorIndex + 0];
                buffer[depthIndex + 1] = colorBuffer[colorIndex + 1];
                buffer[depthIndex + 2] = colorBuffer[colorIndex + 2];
                buffer[depthIndex + 3] = colorBuffer[colorIndex + 3];
            }
        }
    }

    // Create cv::Mat from Coordinate Buffer
    colorMat = cv::Mat( depthHeight, depthWidth, CV_8UC4, &buffer[0] ).clone();
}

// Draw Depth
inline void Kinect::drawDepth()
{
    depthMat = cv::Mat(depthHeight, depthWidth, CV_16UC1, &depthBuffer[0]).clone();

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
        if (n_bg_frames_captured > 50)
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

// Show Data
void Kinect::show()
{
#ifdef COLOR
    // Show Color
    showColor();
#endif

#ifdef DEPTH
    // Show Depth
    showDepth();
#endif
}

// Show Color
inline void Kinect::showColor()
{
    if( colorMat.empty() ){
        return;
    }

    if (got_background)
    {
        depthMat.setTo(9000, depthMat == 0); // put unknown depths to far away
        cv::Mat mask = depthMat < (depthMat0 - 100); // only keep pixels nearer than the background
        // remove speckle
        cv::erode(mask, mask, cv::Mat(), cv::Point(-1, -1), 1);
        cv::dilate(mask, mask, cv::Mat(), cv::Point(-1, -1), 1);
        //cv::imshow("Color", colorMat);
        colorMat.setTo(0, mask == 0);
    }
    // Show Image
    cv::imshow("Color", colorMat);
}

// Show Depth
inline void Kinect::showDepth()
{
    if( depthMat.empty() ){
        return;
    }

    // Scaling ( 0-8000 -> 255-0 )
    cv::Mat scaleMat;
    depthMat.convertTo( scaleMat, CV_8U, -255.0 / 8000.0, 255.0 );
    //cv::applyColorMap( scaleMat, scaleMat, cv::COLORMAP_BONE );

    // Resize Image
    cv::Mat resizeMat;
    const double scale = 0.5;
    cv::resize( scaleMat, resizeMat, cv::Size(), scale, scale );

    // Show Image
    cv::imshow( "Depth", resizeMat );
}