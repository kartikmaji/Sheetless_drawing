#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <cv.h>
#include <iostream>
#include <cmath>

#include "opencv2/core/core.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/legacy/legacy.hpp"


#define red   CV_RGB(255,0,0)
#define white CV_RGB(255,255,255)
#define black CV_RGB(0,0,0)


/*
    Reference
    Countour detection - http://code.opencv.org/svn/gsoc2012/denoising/trunk/opencv-2.4.2/samples/c/bgfg_codebook.cpp
    http://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html
    http://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html
    http://stackoverflow.com/questions/8895749/cvgetspatialmoment-in-opencv-2-0
*/

IplImage* GetThresholdedImage(IplImage* img, CvScalar& lowerBound, CvScalar& upperBound)
{
    // Convert the image into an HSV image
    IplImage* imgHSV = cvCreateImage(cvGetSize(img), 8, 3);
    cvCvtColor(img, imgHSV, CV_BGR2HSV);

    IplImage* imgThreshed = cvCreateImage(cvGetSize(img), 8, 1);

    cvInRangeS(imgHSV, lowerBound, upperBound, imgThreshed);

    cvReleaseImage(&imgHSV);
    return imgThreshed;
}


//VARIABLES for CODEBOOK METHOD:
CvBGCodeBookModel* model = 0;
const int NCHANNELS = 3;
bool ch[NCHANNELS]={true,true,true}; // This sets what channels should be adjusted for background bounds
int nomdef;
bool flip = false;


int detect(IplImage* img_8uc1,IplImage* img_8uc3);

//
//USAGE:  ch9_background startFrameCollection# endFrameCollection# [movie filename, else from camera]
//If from AVI, then optionally add HighAvg, LowAvg, HighCB_Y LowCB_Y HighCB_U LowCB_U HighCB_V LowCB_V
//
int main(int argc, char** argv)
{

    double area_limit = 700;
    CvScalar lowerBound = cvScalar(20, 100, 100);  // yellow
    CvScalar upperBound = cvScalar(30, 255, 255);

    CvScalar lineColor = black;

    const char* filename = 0;
    IplImage* src = 0, *yuvImage = 0; //yuvImage is for codebook method
    IplImage *ImaskCodeBook = 0,*ImaskCodeBookCC = 0;
    CvCapture* capture = 0;

    int c, n, nframes = 0;
    int nframesToLearnBG = 300;

    model = cvCreateBGCodeBookModel();

    //Set color thresholds to default values
    model->modMin[0] = 3;
    model->modMin[1] = model->modMin[2] = 3;
    model->modMax[0] = 10;
    model->modMax[1] = model->modMax[2] = 10;
    model->cbBounds[0] = model->cbBounds[1] = model->cbBounds[2] = 10;

    printf("Starting capture from camera\n");
    capture = cvCaptureFromCAM(0);

    IplImage* imgDrawing = 0;
    imgDrawing = cvCreateImage( cvSize(cvQueryFrame(capture)->width,cvQueryFrame(capture)->height),
                                cvQueryFrame(capture)->depth,     //Bit depth per channel
                                3  //number of channels
                              );

    cvSet(imgDrawing, white);

    CvFont font, fontbig;
    cvInitFont( &font, CV_FONT_HERSHEY_COMPLEX, 1, .6, 0, 2, CV_AA);
    cvInitFont( &fontbig, CV_FONT_HERSHEY_COMPLEX, 3, .6, 0, 3, CV_AA);

    int confirm_close = 10, confirm_clear = 20; // counters for clear and exit confirmation
    char buffer [50]; // buffer for cvPutText
    int image_num = 0; // to keep track of image numbers for saving
    int posX = 0;
    int posY = 0;


    //MAIN PROCESSING LOOP:
    int numOfFingers = 0;
    while(1)
    {
        IplImage* frame = 0;
    
        src = cvQueryFrame( capture );
        ++nframes;
        if(!src)
            break;


        //First time:
        if( nframes == 1 && src )
        {
            // CODEBOOK METHOD ALLOCATION
            yuvImage = cvCloneImage(src);
            ImaskCodeBook = cvCreateImage( cvGetSize(src), IPL_DEPTH_8U, 1 );
            ImaskCodeBookCC = cvCreateImage( cvGetSize(src), IPL_DEPTH_8U, 1 );
            cvSet(ImaskCodeBook,cvScalar(255));

            printf("1 \n");
        }

        // Median filter to decrease the background noise
        cvSmooth( src, src,
                  CV_MEDIAN,
                  5, 5 //parameters for filter, in this case it is filter size
                );



        // Holds the thresholded image (tracked color -> white, the rest -> black)
        IplImage* imgThresh = GetThresholdedImage(src,lowerBound,upperBound);

        // Calculate the moments to estimate the position of the object
        CvMoments *moments = (CvMoments*)malloc(sizeof(CvMoments));
        cvMoments(imgThresh, moments, 1);


        // The actual moment values
        double moment10 = cvGetSpatialMoment(moments, 1, 0);
        double moment01 = cvGetSpatialMoment(moments, 0, 1);
        double area = cvGetCentralMoment(moments, 0, 0);

        // Holding the last and current positions
        int lastX = posX;
        int lastY = posY;

        posX = 0;
        posY = 0;


        if(moment10/area>=0 && moment10/area < 1280 && moment01/area >=0 && moment01/area < 1280 && area>area_limit)
        {
            posX = moment10/area;
            posY = moment01/area;
        }

        if( src )
        {
            cvCvtColor( src, yuvImage, CV_BGR2YCrCb );//YUV For codebook method
            //This is where we build our background model
            if( nframes-1 < nframesToLearnBG  )
                cvBGCodeBookUpdate( model, yuvImage );

            if( nframes-1 == nframesToLearnBG  )
                cvBGCodeBookClearStale( model, model->t/2 );

            //Find the foreground if any
            if( nframes-1 >= nframesToLearnBG  )
            {
                // Find foreground by codebook method
                cvBGCodeBookDiff( model, yuvImage, ImaskCodeBook );
                // This part just to visualize bounding boxes and centers if desired
                cvCopy(ImaskCodeBook,ImaskCodeBookCC);
                cvSegmentFGMask( ImaskCodeBookCC );
                //bwareaopen_(ImaskCodeBookCC,100);
                cvShowImage( "CodeBook_ConnectComp",ImaskCodeBookCC);
                numOfFingers = detect(ImaskCodeBookCC,src);
                //printf("Num of finger = %d \n", numOfFingers);
                CvPoint cvpoint = cvPoint(150,30); // location of the text
                if(numOfFingers <= 3){
                    lineColor = black;

                    double diff_X = lastX-posX;
                    double diff_Y = lastY-posY;
                    double magnitude = sqrt(pow(diff_X,2) + pow(diff_Y,2));
                    // We want to draw a line only if its a valid position
                    //if(lastX>0 && lastY>0 && posX>0 && posY>0)
                    if(magnitude > 0 && magnitude < 100 && posX > 120 && posX<530)
                    {
                        // Draw a line from the previous point to the current point
                        cvLine(imgDrawing, cvPoint(posX, posY), cvPoint(lastX, lastY), lineColor, 2, CV_AA);
                    }
                }

                else if(numOfFingers > 4){
                    cvPutText( src, "Eraser mode detected.", cvpoint, &font, white );
                    sprintf (buffer, "Clearing the screen in %d",confirm_clear); // count-down for clearing the screen
                    cvPutText( src, buffer, cvPoint(150,70), &font, red );
                    confirm_clear--;
                    if(confirm_clear < 0) // confirm in 10 frames before clearing
                    {
                        confirm_clear = 20;
                        sprintf (buffer, "image%d.jpg",image_num++);
                        cvSaveImage(buffer ,imgDrawing); // save the frame into an image
                        cvSet(imgDrawing, white);
                        cvPutText( src, "Cleared the screen.", cvPoint(150,110), &font, white );
                    }
                }
            }

            if(flip == true)  /* Flips all frame if flip is true*/
            {
                cvFlip(imgThresh, imgThresh, 1);
                cvFlip(imgDrawing, imgDrawing, 1);
                cvFlip(ImaskCodeBook, ImaskCodeBook, 1);
                cvFlip(src, src, 1);
            }

            // Combine everything in frame
            cvAnd(src, imgDrawing, src);

            cvShowImage("Threshold", imgThresh);
            cvShowImage("Drawing", imgDrawing);
            cvShowImage("Raw", src );
            cvShowImage("ForegroundCodeBook", ImaskCodeBook);
        }


        // User input:
        c = cvWaitKey(10)&0xFF;
        c = tolower(c);
        // End processing on ESC, q or Q
        if(c == 27 || c == 'q')
            break;
        if(c == 'f')
        {
            /* Changes flip */
            if(flip==false)
            {
                flip = true;
                printf("flip enabled\n");
            }
            else
            {
                flip = false;
                printf("flip disabled\n");
            }
        }

        // To clear the screen
        if(c == 'c')
        {
            sprintf (buffer, "image%d.jpg",image_num++);
            cvSaveImage(buffer ,imgDrawing); // save the frame into an image
            
            cvSet(imgDrawing, white);
            cvPutText( src, "Cleared the screen.", cvPoint(150,110), &font, white );
        }
    }

    cvReleaseCapture( &capture );
    cvDestroyWindow( "Raw" );
    cvDestroyWindow( "ForegroundCodeBook");
    cvDestroyWindow( "CodeBook_ConnectComp");
    return 0;
}


int  detect(IplImage* img_8uc1,IplImage* img_8uc3) {
    //cvNamedWindow( "aug", 1 );
    //cvThreshold( img_8uc1, img_edge, 128, 255, CV_THRESH_BINARY );
    CvMemStorage* storage = cvCreateMemStorage();
    CvSeq* first_contour = NULL;
    CvSeq* maxitem=NULL;
    double area=0,areamax=0;
    int maxn=0;
    int Nc = cvFindContours(
        img_8uc1,
        storage,
        &first_contour,
        sizeof(CvContour),
        CV_RETR_LIST // Try all four values and see what happens
    );
    int n=0;
    //printf( "Total Contours Detected: %d\n", Nc );

    if(Nc>0)
    {
        for( CvSeq* c=first_contour; c!=NULL; c=c->h_next )
        {

            //cvCvtColor( img_8uc1, img_8uc3, CV_GRAY2BGR );

            area=cvContourArea(c,CV_WHOLE_SEQ );

            if(area>areamax)
                {
                    areamax=area;
                    maxitem=c;
                    maxn=n;
                }
                n++;
            }
            CvMemStorage* storage3 = cvCreateMemStorage(0);
            
            if(areamax>5000)
            {
                maxitem = cvApproxPoly( maxitem, sizeof(CvContour), storage3, CV_POLY_APPROX_DP, 10, 1 );

                CvPoint pt0;

                CvMemStorage* storage1 = cvCreateMemStorage(0);
                CvMemStorage* storage2 = cvCreateMemStorage(0);
                CvSeq* ptseq = cvCreateSeq( CV_SEQ_KIND_GENERIC|CV_32SC2, sizeof(CvContour),
                   sizeof(CvPoint), storage1 );
                CvSeq* hull;
                CvSeq* defects;

                for(int i = 0; i < maxitem->total; i++ )
                {   
                    CvPoint* p = CV_GET_SEQ_ELEM( CvPoint, maxitem, i );
                    pt0.x = p->x;
                    pt0.y = p->y;
                    cvSeqPush( ptseq, &pt0 );
                }

                hull = cvConvexHull2( ptseq, 0, CV_CLOCKWISE, 0 );
                int hullcount = hull->total;

                defects= cvConvexityDefects(ptseq,hull,storage2  );

                //printf(" defect no %d \n",defects->total);

                CvConvexityDefect* defectArray;

                //int m_nomdef=0;
                //cycle marks all defects of convexity of current contours.
                for(;defects;defects = defects->h_next)
                {
                    nomdef = defects->total; // defect amount
                    //outlet_float( m_nomdef, nomdef );

                    //printf(" defect no %d \n",nomdef);

                    if(nomdef == 0)
                        continue;

                    // Alloc memory for defect set.
                    //fprintf(stderr,"malloc\n");
                    defectArray = (CvConvexityDefect*)malloc(sizeof(CvConvexityDefect)*nomdef);

                    // Get defect set.
                    //fprintf(stderr,"cvCvtSeqToArray\n");
                    cvCvtSeqToArray(defects,defectArray, CV_WHOLE_SEQ);

                    // Draw marks for all defects.
                    for(int i=0; i<nomdef; i++)
                    {   
                        //printf(" defect depth for defect %d %f \n",i,defectArray[i].depth);
                        cvLine(img_8uc3, *(defectArray[i].start), *(defectArray[i].depth_point),CV_RGB(255,255,0),1, CV_AA, 0 );
                        cvCircle( img_8uc3, *(defectArray[i].depth_point), 5, CV_RGB(0,0,164), 2, 8,0);
                        cvCircle( img_8uc3, *(defectArray[i].start), 5, CV_RGB(0,0,164), 2, 8,0);
                        cvLine(img_8uc3, *(defectArray[i].depth_point), *(defectArray[i].end),CV_RGB(255,255,0),1, CV_AA, 0 );
                    }

                    char txt[]="0";
                    txt[0]='0'+nomdef-1;
                    // printf(" nomdef=%d, txt=%c \n",nomdef,txt[0]);
                    CvFont font;
                    cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 1.0, 1.0, 0, 5, CV_AA);
                    cvPutText(img_8uc3, txt, cvPoint(50, 50), &font, cvScalar(0, 0, 255, 0));

                    // Free memory.
                    free(defectArray);
                }


            cvReleaseMemStorage( &storage );
            cvReleaseMemStorage( &storage1 );
            cvReleaseMemStorage( &storage2 );
            cvReleaseMemStorage( &storage3 );
            return nomdef-1;
        }
    }
}