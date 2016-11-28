#include "app.h"
#include "util.h"

#include <iomanip>
#include <thread>
#include <chrono>
#include <sys/stat.h>

#include <omp.h>

Kinect::Kinect()
    : crop(240, 0, 1470, 1080)
    , iFrame(0)
    , window_title("Tango")
{
    initializeCapture();
    initializeVideoWriter();
}

Kinect::~Kinect()
{
    finalize();
}

void Kinect::run()
{
    while( true )
    {
        bool ret = readImages();
        if (!ret) continue;
        if (iFrame < n_frames)
        {
            accumulateBackground();
        }
        else
        {
            compositeScene();
        }
        render();
        const int key = cv::waitKey( 10 );
        if( key == VK_ESCAPE )
        {
            break;
        }
        ++iFrame;
    }
}

void Kinect::initializeCapture()
{
    cv::setUseOptimized( true );

    initializeSensor();
    initializeColor();
    initializeDepth();

    // Wait a Few Seconds until begins to Retrieve Data from Sensor ( about 2000-[ms] )
    std::this_thread::sleep_for( std::chrono::seconds( 2 ) );
}

void Kinect::initializeSensor()
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

void Kinect::initializeColor()
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

    int w = (int)(colorWidth * scale);
    int h = (int)(colorHeight * scale);
    w -= w % 4; // video width must be multiple of 4
    h -= h % 2; // video height must be multiple of 2
    colorMatSize = cv::Size(w, h);
}

void Kinect::initializeDepth()
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

    // Retrieve Depth Reliable Range
    ERROR_CHECK(depthFrameSource->get_DepthMinReliableDistance(&minReliableDistance)); // 500
    ERROR_CHECK(depthFrameSource->get_DepthMaxReliableDistance(&maxReliableDistance)); // 4500

    // Allocation Depth Buffer
    depthBuffer.resize( depthWidth * depthHeight );
}

bool fileExists(const std::string& filename)
{
    struct stat buf;
    if (stat(filename.c_str(), &buf) != -1)
    {
        return true;
    }
    return false;
}

void Kinect::initializeVideoWriter()
{
    const int fourcc = cv::VideoWriter::fourcc('M', 'S', 'V', 'C');
    //fourcc = cv::VideoWriter::fourcc('X', '2', '6', '4');

    std::string filename;
    int i_suffix = 1;
    do {
        std::ostringstream oss;
        oss << "output_" << std::setfill('0') << std::setw(4) << i_suffix << ".avi";
        filename = oss.str();
        i_suffix++;
    } while (fileExists(filename));
    video_writer.open(filename.c_str(), fourcc, 15.0, colorMatSize); 

    std::ostringstream oss;
    oss << "Writing to " << filename << "... Hit Esc to stop.";
    window_title = oss.str();
}

void Kinect::finalize()
{
    cv::destroyAllWindows();

    // Close Sensor
    if( kinect != nullptr ){
        kinect->Close();
    }
}

bool Kinect::readImages()
{
    return readColor() && readDepth();
}

bool Kinect::readColor()
{
    // Retrieve Color Frame
    ComPtr<IColorFrame> colorFrame;
    const HRESULT ret = colorFrameReader->AcquireLatestFrame( &colorFrame );
    if( FAILED( ret ) ){
        return false;
    }

    // Convert Format ( YUY2 -> BGRA )
    ERROR_CHECK( colorFrame->CopyConvertedFrameDataToArray( static_cast<UINT>( colorBuffer.size() ), &colorBuffer[0], ColorImageFormat::ColorImageFormat_Bgra ) );

    // copy into OpenCV storage
    colorMat = cv::Mat(colorHeight, colorWidth, CV_8UC4, &colorBuffer[0]).clone();

    // crop and downsize
    cv::resize(colorMat(crop), colorMat, colorMatSize);

    // mirror
    cv::flip(colorMat, colorMat, 1);

    return !colorMat.empty();
}

bool Kinect::readDepth()
{
    // Retrieve Depth Frame
    ComPtr<IDepthFrame> depthFrame;
    const HRESULT ret = depthFrameReader->AcquireLatestFrame(&depthFrame);
    if (FAILED(ret)) {
        return false;
    }

    // Retrieve Depth Data
    ERROR_CHECK(depthFrame->CopyFrameDataToArray(static_cast<UINT>(depthBuffer.size()), &depthBuffer[0]));

    // Retrieve Mapped Coordinates
    std::vector<DepthSpacePoint> depthSpacePoints(colorWidth * colorHeight);
    ERROR_CHECK(coordinateMapper->MapColorFrameToDepthSpace((UINT)depthBuffer.size(), &depthBuffer[0], (UINT)depthSpacePoints.size(), &depthSpacePoints[0]));

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

    // crop and downsize
    cv::resize(depthMat(crop), depthMat, colorMatSize, 0, 0, cv::INTER_NEAREST);

    // mirror
    cv::flip(depthMat, depthMat, 1);

    return !depthMat.empty();
}

void Kinect::accumulateBackground()
{
    // Accumulate the static background depth. Assume the camera and the scene doesn't move too much.
    if (depthMat0.empty())
        depthMat0 = depthMat.clone();
    else {
        cv::Mat av = 0.99 * depthMat0 + 0.01 * depthMat;
        cv::Mat valid_depth_mask = (depthMat > minReliableDistance) & (depthMat < maxReliableDistance);
        cv::Mat valid_depth0_mask = (depthMat0 > minReliableDistance) & (depthMat0 < maxReliableDistance);
        av.copyTo(depthMat0, valid_depth_mask & valid_depth0_mask); // average where both known
        depthMat.copyTo(depthMat0, 255 - valid_depth0_mask); // replace any unknown pixels
        depthMat0.setTo(maxReliableDistance * 2, (depthMat0 < minReliableDistance) | (depthMat0 > maxReliableDistance));
    }
    if (iFrame == n_frames-1)
    {
        fillDepthHoles(depthMat0, minReliableDistance, maxReliableDistance);
        // write into every frame of the video storage
        for (int i = 0; i < n_frames; i++)
            depth_frames[i] = depthMat0.clone();
    }
    color_frames[iFrame] = colorMat;
}

void Kinect::render()
{
    if (false)
    {
        // DEBUG: show the depth
        cv::Mat im;
        depthMat0.convertTo(im, CV_8U, -255.0 / 8000.0, 255.0);  //  [0,8000] -> [255,0]
        cv::imshow(window_title.c_str() , im);
        return;
    }

    if (iFrame < n_frames)
    {
        // show the depth buffer as it accumulates
        cv::Mat im;
        depthMat0.convertTo(im, CV_8U, -255.0 / 8000.0, 255.0);  //  [0,8000] -> [255,0]
        cv::imshow(window_title, im);
    }
    else
    {
        if (false)
        {
            // show the looping depth frames
            cv::Mat im;
            depth_frames[iFrame % n_frames].convertTo(im, CV_8U, -255.0 / 8000.0, 255.0);  //  [0,8000] -> [255,0]
            cv::imshow(window_title, im);
        }
        else 
        {
            // show the looping color images
            cv::imshow(window_title, color_frames[iFrame % n_frames]);
            video_writer.write(color_frames[iFrame % n_frames]);
        }
    }
}

void Kinect::compositeScene()
{
    // write into the scene if a pixel is closer
    cv::Mat& depth_frame = depth_frames[iFrame % n_frames];
    cv::Mat& color_frame = color_frames[iFrame % n_frames];

    // make a mask of the foreground
    depthMat.setTo(maxReliableDistance*2, (depthMat < minReliableDistance) | (depthMat > maxReliableDistance) );
    const double depth_noise_mm = 100;
    cv::Mat mask = depthMat < (depth_frame - depth_noise_mm); // only keep pixels nearer than the background

    if (false)
    {
        // attempt to remove speckle
        cv::erode(mask, mask, cv::Mat(), cv::Point(-1, -1), 2);
        cv::dilate(mask, mask, cv::Mat(), cv::Point(-1, -1), 2);
    }

    colorMat.copyTo(color_frame, mask);
    depthMat.copyTo(depth_frame, mask);
}

// fills holes using a sophisticated interpolation algorithm - SLOW
void Kinect::fillDepthHoles(cv::Mat& im, UINT16 minReliableDistance, UINT16 maxReliableDistance)
{
    cv::Mat holes_mask = (im < minReliableDistance) | (im > maxReliableDistance);
    im.setTo(maxReliableDistance, holes_mask);
    cv::Mat infilled;
    // (inpaint needs CV_8UC1 or CV_8UC3 to work on)
    infilled = 255.0 - 255.0 * (im - minReliableDistance) / (maxReliableDistance - minReliableDistance);
    infilled.convertTo(infilled, CV_8U);
    cv::inpaint(infilled, holes_mask, infilled, 20.0, cv::INPAINT_TELEA);
    infilled.convertTo(im, CV_16UC1);
    im = minReliableDistance + (maxReliableDistance - minReliableDistance) * (255.0 - im) / 255.0;
}
