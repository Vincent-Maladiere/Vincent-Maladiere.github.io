# Video Input & Output

[https://docs.opencv.org/4.5.0/df/d2c/tutorial_table_of_content_videoio.html](https://docs.opencv.org/4.5.0/df/d2c/tutorial_table_of_content_videoio.html)

## Video-Input

- OpenCV processes both real-time image feed (in the case of a webcam) or prerecorded and hard disk drive stored files

```python
captRefrnc = cv.VideoCapture(sourceReference)
refS = (
	int(captRefrnc.get(cv.CAP_PROP_FRAME_WIDTH)),
	int(captRefrnc.get(cv.CAP_PROP_FRAME_HEIGHT))
)
while True: # Show the image captured in the window and repeat
   _, frameReference = captRefrnc.read()
	if frameReference is None:
			print(" < < <  Game over!  > > > ")
      break
  
  # ops ...

  cv.imshow(WIN_RF, frameReference)
```

## Video-Output

```python
string::size_type pAt = source.find_last_of('.');                  // Find extension point
const string NAME = source.substr(0, pAt) + argv[2][0] + ".avi";   // Form the new name with container
int ex = static_cast<int>(inputVideo.get(CAP_PROP_FOURCC));     // Get Codec Type- Int form

Size S = Size(
	(int) inputVideo.get(CAP_PROP_FRAME_WIDTH),    // Acquire input size
  (int) inputVideo.get(CAP_PROP_FRAME_HEIGHT)
);

outputVideo.open(NAME, ex, inputVideo.get(CAP_PROP_FPS), S, true)

for(;;){
		inputVideo >> src;
		
		# ops...
	
		outputVideo << res;
}
```

## Using Kinect and other OpenNI compatible depth sensors

also Using Creative Senz3D and other Intel RealSense SDK compatible depth sensors

[https://docs.opencv.org/4.5.0/d7/d6f/tutorial_kinect_openni.html](https://docs.opencv.org/4.5.0/d7/d6f/tutorial_kinect_openni.html)

VideoCapture can retrieve the following data:

1. data given from depth generator:
    - CAP_OPENNI_DEPTH_MAP - depth values in mm (CV_16UC1)
    - CAP_OPENNI_POINT_CLOUD_MAP - XYZ in meters (CV_32FC3)
    - CAP_OPENNI_DISPARITY_MAP - disparity in pixels (CV_8UC1)
    - CAP_OPENNI_DISPARITY_MAP_32F - disparity in pixels (CV_32FC1)
    - CAP_OPENNI_VALID_DEPTH_MASK - mask of valid pixels (not occluded, not shaded etc.) (CV_8UC1)
2. data given from BGR image generator:
    - CAP_OPENNI_BGR_IMAGE - color image (CV_8UC3)
    - CAP_OPENNI_GRAY_IMAGE - gray image (CV_8UC1)

In order to get depth map from depth sensor use VideoCapture::operator >>, e. g. :

```cpp
VideoCapture capture( CAP_OPENNI );
for(;;)
{
    Mat depthMap;
    capture >> depthMap;
    if( waitKey( 30 ) >= 0 )
        break;
}
```

For getting several data maps use VideoCapture::grab and VideoCapture::retrieve, e.g. :

```cpp
VideoCapture capture(0); // or CAP_OPENNI
for(;;)
{
    Mat depthMap;
    Mat bgrImage;
    capture.grab();
    capture.retrieve( depthMap, CAP_OPENNI_DEPTH_MAP );
    capture.retrieve( bgrImage, CAP_OPENNI_BGR_IMAGE );
    if( waitKey( 30 ) >= 0 )
        break;
}
```

For setting and getting some property of sensor` data generators use VideoCapture::set and VideoCapture::get methods respectively, e.g. :

```cpp
VideoCapture capture( CAP_OPENNI );
capture.set( CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, CAP_OPENNI_VGA_30HZ );
cout << "FPS    " << capture.get( CAP_OPENNI_IMAGE_GENERATOR+CAP_PROP_FPS ) << endl;
```