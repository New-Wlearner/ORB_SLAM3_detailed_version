/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2020 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/


#include "Tracking.h"

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include"ORBmatcher.h"
#include"FrameDrawer.h"
#include"Converter.h"
#include"Initializer.h"
#include"G2oTypes.h"
#include"Optimizer.h"
#include"PnPsolver.h"

#include<iostream>

#include<mutex>
#include<chrono>
#include <include/CameraModels/Pinhole.h>
#include <include/CameraModels/KannalaBrandt8.h>
#include <include/MLPnPsolver.h>


using namespace std;

namespace ORB_SLAM3
{

// 构造函数
Tracking::Tracking(System *pSys,
                   ORBVocabulary* pVoc, 
                   FrameDrawer *pFrameDrawer, 
                   MapDrawer *pMapDrawer, 
                   Atlas *pAtlas,
                   KeyFrameDatabase* pKFDB, 
                   const string &strSettingPath, 
                   const int sensor, 
                   const string &_nameSeq):
    mState(NO_IMAGES_YET), 
    mSensor(sensor), 
    mTrackedFr(0), 
    mbStep(false),
    mbOnlyTracking(false), 
    mbMapUpdated(false), 
    mbVO(false), 
    mpORBVocabulary(pVoc), 
    mpKeyFrameDB(pKFDB),
    mpInitializer(static_cast<Initializer*>(NULL)), 
    mpSystem(pSys), 
    mpViewer(NULL),
    mpFrameDrawer(pFrameDrawer), 
    mpMapDrawer(pMapDrawer), 
    mpAtlas(pAtlas), 
    mnLastRelocFrameId(0), 
    time_recently_lost(5.0),
    mnInitialFrameId(0), 
    mbCreatedMap(false), 
    mnFirstFrameId(0), 
    mpCamera2(nullptr)
    // mpCamera(nullptr)
{
        // Load camera parameters from settings file
    // 加载配置文件:相机类型,参数及ORB提取器参数
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    // 定义畸变参数k1,k2,p1,p2
    cv::Mat DistCoef = cv::Mat::zeros(4,1,CV_32F);
    // 相机类型
    string sCameraName = fSettings["Camera.type"];
    // 如果是针孔相机的话
    if(sCameraName == "PinHole"){
        // 读取相机内参
        float fx = fSettings["Camera.fx"];
        float fy = fSettings["Camera.fy"];
        float cx = fSettings["Camera.cx"];
        float cy = fSettings["Camera.cy"];
        // 定义一个1*4的向量存储vector
        vector<float> vCamCalib{fx,fy,cx,cy};
        // 利用读取的内参初始化针孔相机模型
        // 设置相机ID和模型
        mpCamera = new Pinhole(vCamCalib);
        // 把相机模型加入到Altas中
        // push到Altas定义的mvpCameras中
        mpAtlas->AddCamera(mpCamera);
        // 读取畸变系数
        DistCoef.at<float>(0) = fSettings["Camera.k1"];
        DistCoef.at<float>(1) = fSettings["Camera.k2"];
        DistCoef.at<float>(2) = fSettings["Camera.p1"];
        DistCoef.at<float>(3) = fSettings["Camera.p2"];
    }
    // review 如果是鱼眼相机
    if(sCameraName == "KannalaBrandt8"){
        float fx = fSettings["Camera.fx"];
        float fy = fSettings["Camera.fy"];
        float cx = fSettings["Camera.cx"];
        float cy = fSettings["Camera.cy"];

        float K1 = fSettings["Camera.k1"];
        float K2 = fSettings["Camera.k2"];
        float K3 = fSettings["Camera.k3"];
        float K4 = fSettings["Camera.k4"];

        vector<float> vCamCalib{fx,fy,cx,cy,K1,K2,K3,K4};

        mpCamera = new KannalaBrandt8(vCamCalib);

        mpAtlas->AddCamera(mpCamera);

        if(sensor==System::STEREO || sensor==System::IMU_STEREO){
            //Right camera
            fx = fSettings["Camera2.fx"];
            fy = fSettings["Camera2.fy"];
            cx = fSettings["Camera2.cx"];
            cy = fSettings["Camera2.cy"];

            K1 = fSettings["Camera2.k1"];
            K2 = fSettings["Camera2.k2"];
            K3 = fSettings["Camera2.k3"];
            K4 = fSettings["Camera2.k4"];

            cout << endl << "Camera2 Parameters: " << endl;
            cout << "- fx: " << fx << endl;
            cout << "- fy: " << fy << endl;
            cout << "- cx: " << cx << endl;
            cout << "- cy: " << cy << endl;

            vector<float> vCamCalib2{fx,fy,cx,cy,K1,K2,K3,K4};

            mpCamera2 = new KannalaBrandt8(vCamCalib2);

            mpAtlas->AddCamera(mpCamera2);

            int leftLappingBegin = fSettings["Camera.lappingBegin"];
            int leftLappingEnd = fSettings["Camera.lappingEnd"];

            int rightLappingBegin = fSettings["Camera2.lappingBegin"];
            int rightLappingEnd = fSettings["Camera2.lappingEnd"];

            static_cast<KannalaBrandt8*>(mpCamera)->mvLappingArea[0] = leftLappingBegin;
            static_cast<KannalaBrandt8*>(mpCamera)->mvLappingArea[1] = leftLappingEnd;

            static_cast<KannalaBrandt8*>(mpCamera2)->mvLappingArea[0] = rightLappingBegin;
            static_cast<KannalaBrandt8*>(mpCamera2)->mvLappingArea[1] = rightLappingEnd;

            fSettings["Tlr"] >> mTlr;
            cout << "- mTlr: \n" << mTlr << endl;
            mpFrameDrawer->both = true;
        }
    }
    // 如果不是针孔也不是鱼眼
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    // 把内参拷贝到成员变量中
    K.copyTo(mK);
    // 如果有k3参数的话
    const float k3 = fSettings["Camera.k3"];

    if(k3!=0)
    {
        // 重新定义畸变系数向量的大小
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    // 把畸变参数拷贝到成员变量中
    DistCoef.copyTo(mDistCoef);
    // 读取基线b和焦距f的乘积
    mbf = fSettings["Camera.bf"];
    // 读取相机帧率
    float fps = fSettings["Camera.fps"];
    // 如果没有定义就设置为默认值为30
    if(fps==0)
        fps=30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    // ? 设置mMinFrames为0和mMaxFrames为帧率(默认30)
    mMinFrames = 0;
    mMaxFrames = fps;

    cout << endl << "Camera Parameters: " << endl;
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- bf: " << mbf << endl;
    cout << "- k1: " << DistCoef.at<float>(0) << endl;
    cout << "- k2: " << DistCoef.at<float>(1) << endl;
    cout << "- p1: " << DistCoef.at<float>(2) << endl;
    cout << "- p2: " << DistCoef.at<float>(3) << endl;
    if(DistCoef.rows==5)
        cout << "- k3: " << DistCoef.at<float>(4) << endl;
    cout << "- fps: " << fps << endl;
    
    // 读取相机图片数据类型
    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if(mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    // Load ORB parameters
    // 读取ORB特征的相关参数
    // 检测的特征点个数
    int nFeatures = fSettings["ORBextractor.nFeatures"];
    // 金字塔的缩放系数
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    // 金字塔层数
    int nLevels = fSettings["ORBextractor.nLevels"];
    // 初始帧检测的特征个数
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    // 最少检测的特征点个数
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];
    // 利用读取的参数初始化ORB特征提取器
    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    // 如果是双目的话需要再初始化一个右目的特征提取器
    if(sensor==System::STEREO || sensor==System::IMU_STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);
    // 单目的初始化时要提取5*nFeatures个特征
    if(sensor==System::MONOCULAR || sensor==System::IMU_MONOCULAR)
        mpIniORBextractor = new ORBextractor(5*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);
    // ?
    initID = 0; lastID = 0;

    cout << endl << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    if(sensor==System::STEREO || sensor==System::RGBD || sensor==System::IMU_STEREO)
    {
        // !  b*ThDepth
        // ?近点和远点的阈值
        mThDepth = mbf*(float)fSettings["ThDepth"]/fx;
        cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
    }
    // review RGBD
    if(sensor==System::RGBD)
    {
        mDepthMapFactor = fSettings["DepthMapFactor"];
        if(fabs(mDepthMapFactor)<1e-5)
            mDepthMapFactor=1;
        else
            mDepthMapFactor = 1.0f/mDepthMapFactor;
    }
    // 系统中有IMU
    if(sensor==System::IMU_MONOCULAR || sensor==System::IMU_STEREO)
    {
        // 获取left camera -> body的外参矩阵
        cv::Mat Tbc;
        fSettings["Tbc"] >> Tbc;
        cout << endl;

        cout << "Left camera to Imu Transform (Tbc): " << endl << Tbc << endl;

        float freq, Ng, Na, Ngw, Naw;
        // IMU频率
        fSettings["IMU.Frequency"] >> freq;
        // 陀螺仪的高斯噪声
        fSettings["IMU.NoiseGyro"] >> Ng;
        // 加速度计误差
        fSettings["IMU.NoiseAcc"] >> Na;
        // 陀螺仪随机游走
        // bw误差
        fSettings["IMU.GyroWalk"] >> Ngw;
        // 加速度计随机游走
        // ba误差
        fSettings["IMU.AccWalk"] >> Naw;

        const float sf = sqrt(freq);
        cout << endl;
        cout << "IMU frequency: " << freq << " Hz" << endl;
        cout << "IMU gyro noise: " << Ng << " rad/s/sqrt(Hz)" << endl;
        cout << "IMU gyro walk: " << Ngw << " rad/s^2/sqrt(Hz)" << endl;
        cout << "IMU accelerometer noise: " << Na << " m/s^2/sqrt(Hz)" << endl;
        cout << "IMU accelerometer walk: " << Naw << " m/s^3/sqrt(Hz)" << endl;
        // 注意标定得到的各个噪声的单位,根据单位×或者÷sf
        // 为成员变量赋值并构建Q协方差矩阵
        mpImuCalib = new IMU::Calib(Tbc,Ng*sf,Na*sf,Ngw/sf,Naw/sf);
        // 上一帧预积分得到帧间位姿
        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);
        // IMU初始化需要的帧
        mnFramesToResetIMU = mMaxFrames;
    }
    // ? 
    mbInitWith3KFs = false;


    //Test Images
    // 如果是双目且相机模型为针孔模型
    if((mSensor == System::STEREO || mSensor == System::IMU_STEREO) && sCameraName == "PinHole")
    {
        cv::Mat K_l, K_r, P_l, P_r, R_l, R_r, D_l, D_r;
        // 读取左目右目的内参矩阵
        fSettings["LEFT.K"] >> K_l;
        fSettings["RIGHT.K"] >> K_r;
        // 读取左目右目的投影矩阵
        fSettings["LEFT.P"] >> P_l;
        fSettings["RIGHT.P"] >> P_r;
        // 左目右目的旋转矩阵
        fSettings["LEFT.R"] >> R_l;
        fSettings["RIGHT.R"] >> R_r;
        // 左目右目的初始位姿
        fSettings["LEFT.D"] >> D_l;
        fSettings["RIGHT.D"] >> D_r;
        // 左目右目的图像宽度和高度
        int rows_l = fSettings["LEFT.height"];
        int cols_l = fSettings["LEFT.width"];
        int rows_r = fSettings["RIGHT.height"];
        int cols_r = fSettings["RIGHT.width"];

        // M1r y M2r son los outputs (igual para l)
        // M1r y M2r son las matrices relativas al mapeo correspondiente a la rectificación de mapa en el eje X e Y respectivamente
        // ?M1r去畸变的相机参数
        cv::initUndistortRectifyMap(K_l,D_l,R_l,P_l.rowRange(0,3).colRange(0,3),cv::Size(cols_l,rows_l),CV_32F,M1l,M2l);
        cv::initUndistortRectifyMap(K_r,D_r,R_r,P_r.rowRange(0,3).colRange(0,3),cv::Size(cols_r,rows_r),CV_32F,M1r,M2r);
    }
    // 单目或者RGB-D直接这只图像的大小,并设置左目为原点
    else
    {
        int cols = 752;
        int rows = 480;
        cv::Mat R_l = cv::Mat::eye(3, 3, CV_32F);
    }

    mnNumDataset = 0;

    //f_track_stats.open("tracking_stats"+ _nameSeq + ".txt");
    /*f_track_stats.open("tracking_stats.txt");
    f_track_stats << "# timestamp, Num KF local, Num MP local, time" << endl;
    f_track_stats << fixed ;*/

#ifdef SAVE_TIMES
    f_track_times.open("tracking_times.txt");
    f_track_times << "# ORB_Ext(ms), Stereo matching(ms), Preintegrate_IMU(ms), Pose pred(ms), LocalMap_track(ms), NewKF_dec(ms)" << endl;
    f_track_times << fixed ;
#endif
}

Tracking::~Tracking()
{
    //f_track_stats.close();
#ifdef SAVE_TIMES
    f_track_times.close();
#endif
}

// 利用其他线程的数据给当前线程成员变量赋值
void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing=pLoopClosing;
}

void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer=pViewer;
}

void Tracking::SetStepByStep(bool bSet)
{
    bStepByStep = bSet;
}

// 获取双目图像
cv::Mat Tracking::GrabImageStereo(const cv::Mat &imRectLeft, 
                                  const cv::Mat &imRectRight, 
                                  const double &timestamp, 
                                  string filename)
{
    mImGray = imRectLeft;
    cv::Mat imGrayRight = imRectRight;
    mImRight = imRectRight;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGB2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGR2GRAY);
        }
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGBA2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGRA2GRAY);
        }
    }

    // ?双目就证明肯定是双目了,没有第二个相机模型证明和左目是一样的
    if (mSensor == System::STEREO && !mpCamera2)
        mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera);
    // 证明左右目相机的参数不同
    else if(mSensor == System::STEREO && mpCamera2)
        mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera,mpCamera2,mTlr);
    // 加上IMU外参
    else if(mSensor == System::IMU_STEREO && !mpCamera2)
        mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera,&mLastFrame,*mpImuCalib);
    else if(mSensor == System::IMU_STEREO && mpCamera2)
        mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera,mpCamera2,mTlr,&mLastFrame,*mpImuCalib);

    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    mCurrentFrame.mNameFile = filename;
    mCurrentFrame.mnDataset = mnNumDataset;
    // 开始跟踪
    Track();

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    double t_track = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t1 - t0).count();

    /*cout << "trracking time: " << t_track << endl;
    f_track_stats << setprecision(0) << mCurrentFrame.mTimeStamp*1e9 << ",";
    f_track_stats << mvpLocalKeyFrames.size() << ",";
    f_track_stats << mvpLocalMapPoints.size() << ",";
    f_track_stats << setprecision(6) << t_track << endl;*/

#ifdef SAVE_TIMES
    f_track_times << mCurrentFrame.mTimeORB_Ext << ",";
    f_track_times << mCurrentFrame.mTimeStereoMatch << ",";
    f_track_times << mTime_PreIntIMU << ",";
    f_track_times << mTime_PosePred << ",";
    f_track_times << mTime_LocalMapTrack << ",";
    f_track_times << mTime_NewKF_Dec << ",";
    f_track_times << t_track << endl;
#endif

    return mCurrentFrame.mTcw.clone();
}

// 获取RGBD图像并开始跟踪
cv::Mat Tracking::GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp, string filename)
{
    mImGray = imRGB;
    cv::Mat imDepth = imD;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    if((fabs(mDepthMapFactor-1.0f)>1e-5) || imDepth.type()!=CV_32F)
        imDepth.convertTo(imDepth,CV_32F,mDepthMapFactor);

    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    mCurrentFrame = Frame(mImGray,imDepth,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    mCurrentFrame.mNameFile = filename;
    mCurrentFrame.mnDataset = mnNumDataset;


    Track();

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    double t_track = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t1 - t0).count();

    /*f_track_stats << setprecision(0) << mCurrentFrame.mTimeStamp*1e9 << ",";
    f_track_stats << mvpLocalKeyFrames.size() << ",";
    f_track_stats << mvpLocalMapPoints.size() << ",";
    f_track_stats << setprecision(6) << t_track << endl;*/

#ifdef SAVE_TIMES
    f_track_times << mCurrentFrame.mTimeORB_Ext << ",";
    f_track_times << mCurrentFrame.mTimeStereoMatch << ",";
    f_track_times << mTime_PreIntIMU << ",";
    f_track_times << mTime_PosePred << ",";
    f_track_times << mTime_LocalMapTrack << ",";
    f_track_times << mTime_NewKF_Dec << ",";
    f_track_times << t_track << endl;
#endif

    return mCurrentFrame.mTcw.clone();
}

// 获取单目图像并开始跟踪
cv::Mat Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp, string filename)
{
    // step1 得到图像信息
    mImGray = im;
    // step2 转换为灰度图
    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }
    // step3 相机模式判断
    if (mSensor == System::MONOCULAR)
    {   
        
        if(mState==NOT_INITIALIZED ||       // 初始化未成功
           mState==NO_IMAGES_YET ||         // 没有接收到图像
           (lastID - initID) < mMaxFrames){ // ?当前帧和上一帧差的小于mMaxFrames
            //! wdx测试
            // cout<<"mImGray"<<mImGray.dims<<endl;
            // cout<<"timestamp"<<timestamp<<endl;
            // cout<<"mpIniORBextractor"<<mpIniORBextractor->GetLevels()<<endl;
            // cout<<"mpORBVocabulary"<<mpORBVocabulary->size()<<endl;
            // cout<<"mDistCoef"<<mDistCoef.dims<<endl;
            // cout<<"mbf"<<mbf<<endl;
            // cout<<"mThDepth"<<mThDepth<<endl;
            // cout<<"mpCamera"<<mpCamera->size()<<endl;
            // 没有完成初始化,生成初始化要用的帧
            //? 单目中的bf是什么
            mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mpCamera,mDistCoef,mbf,mThDepth);
        }
        else{
            // 正常的跟踪的模式产生的帧
            mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mpCamera,mDistCoef,mbf,mThDepth);
        }
    }
    // 如果是单目+IMU模式
    else if(mSensor == System::IMU_MONOCULAR)
    {   
        // 没有完成初始化
        if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET)
        {
            cout << "init extractor" << endl;
            // 需要上一帧和IMU参数的信息
            mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mpCamera,mDistCoef,mbf,mThDepth,&mLastFrame,*mpImuCalib);
        }
        else
            // 正常的跟踪模式
            mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mpCamera,mDistCoef,mbf,mThDepth,&mLastFrame,*mpImuCalib);
    }
    // step4 系统当前状态判断
    // 如果还没有图片
    if (mState==NO_IMAGES_YET)
        // 设置当前帧的时间为t0
        t0=timestamp;
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();

    mCurrentFrame.mNameFile = filename;
    mCurrentFrame.mnDataset = mnNumDataset;
    // 当前帧的ID
    lastID = mCurrentFrame.mnId;
    // step5 开始跟踪
    Track();

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    // 跟踪消耗的时间
    double t_track = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t1 - t0).count();
    cout << "Mono Tracking time: "<< t_track << endl;

    // f_track_stats << setprecision(0) << mCurrentFrame.mTimeStamp*1e9 << ",";
    // f_track_stats << mvpLocalKeyFrames.size() << ",";
    // f_track_stats << mvpLocalMapPoints.size() << ",";
    // f_track_stats << setprecision(6) << t_track << endl;

#ifdef SAVE_TIMES
    f_track_times << mCurrentFrame.mTimeORB_Ext << ",";
    f_track_times << mCurrentFrame.mTimeStereoMatch << ",";
    f_track_times << mTime_PreIntIMU << ",";
    f_track_times << mTime_PosePred << ",";
    f_track_times << mTime_LocalMapTrack << ",";
    f_track_times << mTime_NewKF_Dec << ",";
    f_track_times << t_track << endl;
#endif

    return mCurrentFrame.mTcw.clone();
}


void Tracking::GrabImuData(const IMU::Point &imuMeasurement)
{
    unique_lock<mutex> lock(mMutexImuQueue);
    mlQueueImuData.push_back(imuMeasurement);
}

void Tracking::PreintegrateIMU()
{
    //cout << "start preintegration" << endl;

    // 当前帧没有上一帧积分数据的话
    if(!mCurrentFrame.mpPrevFrame)
    {
        Verbose::PrintMess("non prev frame ", Verbose::VERBOSITY_NORMAL);
        mCurrentFrame.setIntegrated();
        return;
    }

    // cout << "start loop. Total meas:" << mlQueueImuData.size() << endl;
    // 上一帧中IMU的状态
    mvImuFromLastFrame.clear();
    mvImuFromLastFrame.reserve(mlQueueImuData.size());
    // 如果没有数据
    if(mlQueueImuData.size() == 0)
    {
        Verbose::PrintMess("Not IMU data in mlQueueImuData!!", Verbose::VERBOSITY_NORMAL);
        // 从当前帧开始积分
        mCurrentFrame.setIntegrated();
        return;
    }

    while(true)
    {
        bool bSleep = false;
        {
            unique_lock<mutex> lock(mMutexImuQueue);
            if(!mlQueueImuData.empty())
            {
                // 最开始的数据
                IMU::Point* m = &mlQueueImuData.front();
                cout.precision(17);
                // 当前IMU数据的时间戳小于上一帧的时间戳
                if(m->t<mCurrentFrame.mpPrevFrame->mTimeStamp-0.001l)
                {
                    // 就把这个数据删了
                    mlQueueImuData.pop_front();
                }
                // 当前IMU数据的时间戳小于当前帧单大于上一帧
                else if(m->t<mCurrentFrame.mTimeStamp-0.001l)
                {
                    // 就把这个数据从mlQueueImuData中拿出来给mvImuFromLastFrame并删除mlQueueImuData的记录
                    mvImuFromLastFrame.push_back(*m);
                    mlQueueImuData.pop_front();
                }
                else
                {
                    // 第一个大于当前帧的
                    // !mvImuFromLastFrame存储的是两帧中间所有的和一个大于当前帧的
                    mvImuFromLastFrame.push_back(*m);
                    break;
                }
            }
            else
            {
                break;
                bSleep = true;
            }
        }
        if(bSleep)
            usleep(500);
    }


    const int n = mvImuFromLastFrame.size()-1;
    // 初始化一个预积分的对象
    IMU::Preintegrated* pImuPreintegratedFromLastFrame = new IMU::Preintegrated(mLastFrame.mImuBias,mCurrentFrame.mImuCalib);
    // 对于所有的IMU数据
    for(int i=0; i<n; i++)
    {
        float tstep;
        cv::Point3f acc, angVel;
        // 第一次进来
        if((i==0) && (i<(n-1)))
        {
            // 计算第二个IMU数据和第一个IMU数据的时间差
            float tab = mvImuFromLastFrame[i+1].t-mvImuFromLastFrame[i].t;
            // 第一个IMU数据和上一帧图像数据的时间差
            float tini = mvImuFromLastFrame[i].t-mCurrentFrame.mpPrevFrame->mTimeStamp;
            // ?求每个时间段内的平均加速度
            // 开始用a_last+a_2/2
            acc = (mvImuFromLastFrame[i].a+mvImuFromLastFrame[i+1].a-
                    (mvImuFromLastFrame[i+1].a-mvImuFromLastFrame[i].a)*(tini/tab))*0.5f;
            angVel = (mvImuFromLastFrame[i].w+mvImuFromLastFrame[i+1].w-
                    (mvImuFromLastFrame[i+1].w-mvImuFromLastFrame[i].w)*(tini/tab))*0.5f;
            tstep = mvImuFromLastFrame[i+1].t-mCurrentFrame.mpPrevFrame->mTimeStamp;
        }
        else if(i<(n-1))
        {
            acc = (mvImuFromLastFrame[i].a+mvImuFromLastFrame[i+1].a)*0.5f;
            angVel = (mvImuFromLastFrame[i].w+mvImuFromLastFrame[i+1].w)*0.5f;
            tstep = mvImuFromLastFrame[i+1].t-mvImuFromLastFrame[i].t;
        }
        // 最后用a_n-2+a_cur/2
        else if((i>0) && (i==(n-1)))
        {
            float tab = mvImuFromLastFrame[i+1].t-mvImuFromLastFrame[i].t;
            float tend = mvImuFromLastFrame[i+1].t-mCurrentFrame.mTimeStamp;
            acc = (mvImuFromLastFrame[i].a+mvImuFromLastFrame[i+1].a-
                    (mvImuFromLastFrame[i+1].a-mvImuFromLastFrame[i].a)*(tend/tab))*0.5f;
            angVel = (mvImuFromLastFrame[i].w+mvImuFromLastFrame[i+1].w-
                    (mvImuFromLastFrame[i+1].w-mvImuFromLastFrame[i].w)*(tend/tab))*0.5f;
            tstep = mCurrentFrame.mTimeStamp-mvImuFromLastFrame[i].t;
        }
        // 最后一个
        else if((i==0) && (i==(n-1)))
        {
            acc = mvImuFromLastFrame[i].a;
            angVel = mvImuFromLastFrame[i].w;
            tstep = mCurrentFrame.mTimeStamp-mCurrentFrame.mpPrevFrame->mTimeStamp;
        }

        if (!mpImuPreintegratedFromLastKF)
            cout << "mpImuPreintegratedFromLastKF does not exist" << endl;
        // 把每个数据存储在mpImuPreintegratedFromLastKF和mpImuPreintegratedFromLastFrame
        // 每个时间段的平均加速度和到上一帧的时间
        // 积分得到位置和角度
        mpImuPreintegratedFromLastKF->IntegrateNewMeasurement(acc,angVel,tstep);
        pImuPreintegratedFromLastFrame->IntegrateNewMeasurement(acc,angVel,tstep);
    }

    mCurrentFrame.mpImuPreintegratedFrame = pImuPreintegratedFromLastFrame;
    mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;
    mCurrentFrame.mpLastKeyFrame = mpLastKeyFrame;

    if(!mpLastKeyFrame)
    {
        cout << "last KF is empty!" << endl;
    }
    mCurrentFrame.setIntegrated();

    Verbose::PrintMess("Preintegration is finished!! ", Verbose::VERBOSITY_DEBUG);
}


bool Tracking::PredictStateIMU()
{
    // 如果没有上一帧直接返回false
    if(!mCurrentFrame.mpPrevFrame)
    {
        Verbose::PrintMess("No last frame", Verbose::VERBOSITY_NORMAL);
        return false;
    }
    // 地图更新了  且  有上一个关键帧
    if(mbMapUpdated && mpLastKeyFrame)
    {
        //得到上一帧IMU的状态量
        // b1->w
        const cv::Mat twb1 = mpLastKeyFrame->GetImuPosition();
        const cv::Mat Rwb1 = mpLastKeyFrame->GetImuRotation();
        const cv::Mat Vwb1 = mpLastKeyFrame->GetVelocity();
        // 重力方向 (0,0,-9.81)
        const cv::Mat Gz = (cv::Mat_<float>(3,1) << 0,0,-IMU::GRAVITY_VALUE);
        // ?时间
        const float t12 = mpImuPreintegratedFromLastKF->dT;
        // 当前帧的预积分位姿特征值变为1
        cv::Mat Rwb2 = IMU::NormalizeRotation(Rwb1*mpImuPreintegratedFromLastKF->GetDeltaRotation(mpLastKeyFrame->GetImuBias()));
        // p1 = p0 + vt + 0.5(g)t^2 + R*deltaP
        cv::Mat twb2 = twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*mpImuPreintegratedFromLastKF->GetDeltaPosition(mpLastKeyFrame->GetImuBias());
        // v2 = v1 + at  ->   v1 + gt + delta V
        cv::Mat Vwb2 = Vwb1 + t12*Gz + Rwb1*mpImuPreintegratedFromLastKF->GetDeltaVelocity(mpLastKeyFrame->GetImuBias());
        // 得到当前帧的IMU测量
        mCurrentFrame.SetImuPoseVelocity(Rwb2,twb2,Vwb2);
        // 当前预积分得到的值
        mCurrentFrame.mPredRwb = Rwb2.clone();
        mCurrentFrame.mPredtwb = twb2.clone();
        mCurrentFrame.mPredVwb = Vwb2.clone();
        mCurrentFrame.mImuBias = mpLastKeyFrame->GetImuBias();
        mCurrentFrame.mPredBias = mCurrentFrame.mImuBias;
        return true;
    }
    // 证明上一帧不是关键帧
    else if(!mbMapUpdated)
    {
        // 利用普通帧进行预积分
        const cv::Mat twb1 = mLastFrame.GetImuPosition();
        const cv::Mat Rwb1 = mLastFrame.GetImuRotation();
        const cv::Mat Vwb1 = mLastFrame.mVw;
        const cv::Mat Gz = (cv::Mat_<float>(3,1) << 0,0,-IMU::GRAVITY_VALUE);
        const float t12 = mCurrentFrame.mpImuPreintegratedFrame->dT;

        cv::Mat Rwb2 = IMU::NormalizeRotation(Rwb1*mCurrentFrame.mpImuPreintegratedFrame->GetDeltaRotation(mLastFrame.mImuBias));
        cv::Mat twb2 = twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*mCurrentFrame.mpImuPreintegratedFrame->GetDeltaPosition(mLastFrame.mImuBias);
        cv::Mat Vwb2 = Vwb1 + t12*Gz + Rwb1*mCurrentFrame.mpImuPreintegratedFrame->GetDeltaVelocity(mLastFrame.mImuBias);

        mCurrentFrame.SetImuPoseVelocity(Rwb2,twb2,Vwb2);
        mCurrentFrame.mPredRwb = Rwb2.clone();
        mCurrentFrame.mPredtwb = twb2.clone();
        mCurrentFrame.mPredVwb = Vwb2.clone();
        mCurrentFrame.mImuBias =mLastFrame.mImuBias;
        mCurrentFrame.mPredBias = mCurrentFrame.mImuBias;
        return true;
    }
    else
        cout << "not IMU prediction!!" << endl;

    return false;
}


void Tracking::ComputeGyroBias(const vector<Frame*> &vpFs, float &bwx,  float &bwy, float &bwz)
{
    const int N = vpFs.size();
    vector<float> vbx;
    vbx.reserve(N);
    vector<float> vby;
    vby.reserve(N);
    vector<float> vbz;
    vbz.reserve(N);

    cv::Mat H = cv::Mat::zeros(3,3,CV_32F);
    cv::Mat grad  = cv::Mat::zeros(3,1,CV_32F);
    for(int i=1;i<N;i++)
    {
        Frame* pF2 = vpFs[i];
        Frame* pF1 = vpFs[i-1];
        cv::Mat VisionR = pF1->GetImuRotation().t()*pF2->GetImuRotation();
        cv::Mat JRg = pF2->mpImuPreintegratedFrame->JRg;
        cv::Mat E = pF2->mpImuPreintegratedFrame->GetUpdatedDeltaRotation().t()*VisionR;
        cv::Mat e = IMU::LogSO3(E);
        assert(fabs(pF2->mTimeStamp-pF1->mTimeStamp-pF2->mpImuPreintegratedFrame->dT)<0.01);

        cv::Mat J = -IMU::InverseRightJacobianSO3(e)*E.t()*JRg;
        
        grad += J.t()*e;
        H += J.t()*J;
    }

    cv::Mat bg = -H.inv(cv::DECOMP_SVD)*grad;
    
    bwx = bg.at<float>(0);
    bwy = bg.at<float>(1);
    bwz = bg.at<float>(2);

    for(int i=1;i<N;i++)
    {
        Frame* pF = vpFs[i];
        pF->mImuBias.bwx = bwx;
        pF->mImuBias.bwy = bwy;
        pF->mImuBias.bwz = bwz;
        pF->mpImuPreintegratedFrame->SetNewBias(pF->mImuBias);
        pF->mpImuPreintegratedFrame->Reintegrate();
    }
}

void Tracking::ComputeVelocitiesAccBias(const vector<Frame*> &vpFs, float &bax,  float &bay, float &baz)
{
    const int N = vpFs.size();
    const int nVar = 3*N +3; // 3 velocities/frame + acc bias
    const int nEqs = 6*(N-1);

    cv::Mat J(nEqs,nVar,CV_32F,cv::Scalar(0));
    cv::Mat e(nEqs,1,CV_32F,cv::Scalar(0));
    cv::Mat g = (cv::Mat_<float>(3,1)<<0,0,-IMU::GRAVITY_VALUE);

    for(int i=0;i<N-1;i++)
    {
        Frame* pF2 = vpFs[i+1];
        Frame* pF1 = vpFs[i];
        cv::Mat twb1 = pF1->GetImuPosition();
        cv::Mat twb2 = pF2->GetImuPosition();
        cv::Mat Rwb1 = pF1->GetImuRotation();
        cv::Mat dP12 = pF2->mpImuPreintegratedFrame->GetUpdatedDeltaPosition();
        cv::Mat dV12 = pF2->mpImuPreintegratedFrame->GetUpdatedDeltaVelocity();
        cv::Mat JP12 = pF2->mpImuPreintegratedFrame->JPa;
        cv::Mat JV12 = pF2->mpImuPreintegratedFrame->JVa;
        float t12 = pF2->mpImuPreintegratedFrame->dT;
        // Position p2=p1+v1*t+0.5*g*t^2+R1*dP12
        J.rowRange(6*i,6*i+3).colRange(3*i,3*i+3) += cv::Mat::eye(3,3,CV_32F)*t12;
        J.rowRange(6*i,6*i+3).colRange(3*N,3*N+3) += Rwb1*JP12;
        e.rowRange(6*i,6*i+3) = twb2-twb1-0.5f*g*t12*t12-Rwb1*dP12;
        // Velocity v2=v1+g*t+R1*dV12
        J.rowRange(6*i+3,6*i+6).colRange(3*i,3*i+3) += -cv::Mat::eye(3,3,CV_32F);
        J.rowRange(6*i+3,6*i+6).colRange(3*(i+1),3*(i+1)+3) += cv::Mat::eye(3,3,CV_32F);
        J.rowRange(6*i+3,6*i+6).colRange(3*N,3*N+3) -= Rwb1*JV12;
        e.rowRange(6*i+3,6*i+6) = g*t12+Rwb1*dV12;
    }

    cv::Mat H = J.t()*J;
    cv::Mat B = J.t()*e;
    cv::Mat x(nVar,1,CV_32F);
    cv::solve(H,B,x);

    bax = x.at<float>(3*N);
    bay = x.at<float>(3*N+1);
    baz = x.at<float>(3*N+2);

    for(int i=0;i<N;i++)
    {
        Frame* pF = vpFs[i];
        x.rowRange(3*i,3*i+3).copyTo(pF->mVw);
        if(i>0)
        {
            pF->mImuBias.bax = bax;
            pF->mImuBias.bay = bay;
            pF->mImuBias.baz = baz;
            pF->mpImuPreintegratedFrame->SetNewBias(pF->mImuBias);
        }
    }
}

void Tracking::ResetFrameIMU()
{
    // TODO To implement...
}

// 跟踪主函数
void Tracking::Track()
{
#ifdef SAVE_TIMES
    mTime_PreIntIMU = 0;
    mTime_PosePred = 0;
    mTime_LocalMapTrack = 0;
    mTime_NewKF_Dec = 0;
#endif
    // 3代新加
    // 应该是Drawer里面的变量
    if (bStepByStep)
    {
        while(!mbStep)
            usleep(500);
        mbStep = false;
    }

    // 检测到IMUbad
    // ! check
    if(mpLocalMapper->mbBadImu)
    {
        cout << "TRACK: Reset map because local mapper set the bad imu flag " << endl;
        // 证明不行,重新resetactivemap
        // 设置地图reset标志
        mpSystem->ResetActiveMap();
        // 直接返回到system复位
        return;
    }
    // 得到当前地图
    Map* pCurrentMap = mpAtlas->GetCurrentMap();

    // SYSTME_NOT_READY, NO_IMAGE_YET, NOT_INITIALIZED, OK, LOST
    // 如果图像复位过、或者第一次运行，则为NO_IMAGE_YET状态
    // 如果不是NO_IMAGES_YET证明有上一帧
    if(mState!=NO_IMAGES_YET)
    {
        // 上一帧时间戳大于当前帧时间戳
        if(mLastFrame.mTimeStamp>mCurrentFrame.mTimeStamp)
        {
            cerr << "ERROR: Frame with a timestamp older than previous frame detected!" << endl;
            unique_lock<mutex> lock(mMutexImuQueue);
            mlQueueImuData.clear();
            CreateMapInAtlas();
            return;
        }
        // 当前帧时间戳大于上一帧加1s,证明有跳帧.
        else if(mCurrentFrame.mTimeStamp>mLastFrame.mTimeStamp+1.0)
        {
            cout << "id last: " << mLastFrame.mnId << "    id curr: " << mCurrentFrame.mnId << endl;
            // 有没有用IMU
            if(mpAtlas->isInertial())
            {
                // imu已经初始化
                if(mpAtlas->isImuInitialized())
                {
                    cout << "Timestamp jump detected. State set to LOST. Reseting IMU integration..." << endl;
                    // ?闭环和局部建图相关的量
                    if(!pCurrentMap->GetIniertialBA2())
                    {
                        // 直接reset这个地图
                        mpSystem->ResetActiveMap();
                    }
                    
                    else
                    {
                        // 重新创建地图
                        CreateMapInAtlas();
                    }
                }
                // imu没有初始化
                else
                {
                    cout << "Timestamp jump detected, before IMU initialization. Reseting..." << endl;
                    // 直接重置当前地图
                    mpSystem->ResetActiveMap();
                }
            }
            return;
        }
    }

    if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO) && // VIO模式
         mpLastKeyFrame)                                                       // 且上个关键帧存在
        // 利用上个关键帧的Bias设置当前帧的Bias 
        mCurrentFrame.SetNewBias(mpLastKeyFrame->GetImuBias());
    
    if(mState==NO_IMAGES_YET)
    {
        mState = NOT_INITIALIZED;
    }
    // mLastProcessedState 存储了Tracking最新的状态，用于FrameDrawer中的绘制
    mLastProcessedState=mState;

    if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO) && !mbCreatedMap)
    {
#ifdef SAVE_TIMES
        std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
#endif
        // todo 预积分的过程
        PreintegrateIMU();
#ifdef SAVE_TIMES
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        mTime_PreIntIMU = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t1 - t0).count();
#endif


    }
    
    mbCreatedMap = false;

    // Get Map Mutex -> Map cannot be changed
    //? 疑问:这样子不是在大部分的时间中阻止了对地图的更新吗?
    // -- 不过应该注意到ORB-SLAM的主要耗时在特征点的提取和匹配部分,这些部分具体体现在帧的构造函数中,而在那个时候地图是没有被上锁的,
    // -- 还是有足够的时间更新地图的
    unique_lock<mutex> lock(pCurrentMap->mMutexMapUpdate);

    mbMapUpdated = false;
    // ?
    int nCurMapChangeIndex = pCurrentMap->GetMapChangeIndex();
    // ?
    int nMapChangeIndex = pCurrentMap->GetLastMapChange();
    if(nCurMapChangeIndex>nMapChangeIndex)
    {
        // cout << "Map update detected" << endl;
        // 地图是否更新了
        pCurrentMap->SetLastMapChange(nCurMapChangeIndex);
        mbMapUpdated = true;
    }
    // 还没初始化就开始进行初始化
    if(mState==NOT_INITIALIZED)
    {
        if(mSensor==System::STEREO || mSensor==System::RGBD || mSensor==System::IMU_STEREO)
            // 双目初始化
            StereoInitialization();
        else
        {
            MonocularInitialization();
        }
        // 更新draw里面的内容
        mpFrameDrawer->Update(this);
        // 这个状态量在上面的初始化函数中被更新
        // 初始化不成功或者跟踪失败
        if(mState!=OK) // If rightly initialized, mState=OK
        {
            mLastFrame = Frame(mCurrentFrame);
            return;
        }
        // 第一个地图
        if(mpAtlas->GetAllMaps().size() == 1)
        {
            mnFirstFrameId = mCurrentFrame.mnId;
        }
    }
    else
    {
        // System is initialized. Track Frame.
        bool bOK;

        // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
        // 正常的SLAM模式
        if(!mbOnlyTracking)
        {
#ifdef SAVE_TIMES
        std::chrono::steady_clock::time_point timeStartPosePredict = std::chrono::steady_clock::now();
#endif

            // State OK
            // Local Mapping is activated. This is the normal behaviour, unless
            // you explicitly activate the "only tracking" mode.
            // 正常的状态
            if(mState==OK)
            {

                // Local Mapping might have changed some MapPoints tracked in last frame
                // 如果地图点被替换就更新一下这个地图点
                CheckReplacedInLastFrame();
                // 运动模型为空(跟丢了)
                // imu没初始化()
                // 当前帧是重定位帧(作为参考帧初始化?)
                if((mVelocity.empty() && !pCurrentMap->isImuInitialized()) || mCurrentFrame.mnId<mnLastRelocFrameId+2)
                {
                    //Verbose::PrintMess("TRACK: Track with respect to the reference KF ", Verbose::VERBOSITY_DEBUG);
                    // 把上一帧当为参考帧,上一帧位姿作为当前帧位姿初值进行pnp计算位姿,删除错误匹配的地图点
                    // 在上述情况下利用词袋模型进行匹配,利用两帧的匹配优化位姿
                    bOK = TrackReferenceKeyFrame();
                }
                else
                {
                    //Verbose::PrintMess("TRACK: Track with motion model", Verbose::VERBOSITY_DEBUG);
                    // 正常情况下就是用恒速模型进行跟踪
                
                    bOK = TrackWithMotionModel();
                    // 如果跟踪失败了就用紧急的方法
                    if(!bOK)
                        bOK = TrackReferenceKeyFrame();
                }

                // 到这里不是OK,证明TrackReferenceKeyFrame失败了,那就设置状态为LOST
                if (!bOK)
                {
                    if ( mCurrentFrame.mnId<=(mnLastRelocFrameId+mnFramesToResetIMU) &&
                         (mSensor==System::IMU_MONOCULAR || mSensor==System::IMU_STEREO))
                    {
                        mState = LOST;
                    }
                    // ?如果地图中关键帧的个数大于10,就证明是刚丢?可以用IMU撑着
                    else if(pCurrentMap->KeyFramesInMap()>10)
                    {
                        cout << "KF in map: " << pCurrentMap->KeyFramesInMap() << endl;
                        mState = RECENTLY_LOST;
                        mTimeStampLost = mCurrentFrame.mTimeStamp;
                        //mCurrentFrame.SetPose(mLastFrame.mTcw);
                    }
                    else
                    {
                        mState = LOST;
                    }
                }
            }
            // 状态不是OK,证明现在VO遇到了问题
            else
            {
                // 如果关键帧个数大于10,而且丢的时间 < time_recently_lost
                if (mState == RECENTLY_LOST)
                {
                    Verbose::PrintMess("Lost for a short time", Verbose::VERBOSITY_NORMAL);

                    bOK = true;
                    if((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO))
                    {
                        if(pCurrentMap->isImuInitialized())
                            PredictStateIMU();
                        else
                            bOK = false;

                        if (mCurrentFrame.mTimeStamp-mTimeStampLost>time_recently_lost)
                        {
                            mState = LOST;
                            Verbose::PrintMess("Track Lost...", Verbose::VERBOSITY_NORMAL);
                            bOK=false;
                        }
                    }
                    // 没有IMU就不用撑了,直接重定位
                    else
                    {
                        // TODO fix relocalization
                        // 重定位
                        bOK = Relocalization();
                        // 还是没找到
                        if(!bOK)
                        {
                            // 状态就继续是LOST
                            mState = LOST;
                            Verbose::PrintMess("Track Lost...", Verbose::VERBOSITY_NORMAL);
                            bOK=false;
                        }
                    }
                }
                // 如果进来是LOST的状态
                else if (mState == LOST)
                {

                    Verbose::PrintMess("A new map is started...", Verbose::VERBOSITY_NORMAL);
                    // 地图中的关键帧小于10
                    if (pCurrentMap->KeyFramesInMap()<10)
                    {
                        // 重新初始化该地图
                        mpSystem->ResetActiveMap();
                        cout << "Reseting current map..." << endl;
                    }else
                        // 大于10的话,把上一个地图保存,重新生成一个新的地图
                        CreateMapInAtlas();
                    // 没有上一个关键帧了
                    if(mpLastKeyFrame)
                        mpLastKeyFrame = static_cast<KeyFrame*>(NULL);

                    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

                    return;
                }
            }


#ifdef SAVE_TIMES
        std::chrono::steady_clock::time_point timeEndPosePredict = std::chrono::steady_clock::now();

        mTime_PosePred = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(timeEndPosePredict - timeStartPosePredict).count();
#endif

        }
        // 仅定位的模式
        else
        {
            // Localization Mode: Local Mapping is deactivated (TODO Not available in inertial mode)
            // 跟丢了
            if(mState==LOST)
            {   
                // ?有IMU直接重定位
                if(mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO)
                    Verbose::PrintMess("IMU. State LOST", Verbose::VERBOSITY_NORMAL);
                bOK = Relocalization();
            }
            // 没跟丢
            else
            {
                // 没有匹配的时候为true,有匹配的时候为false
                if(!mbVO)
                {
                    // In last frame we tracked enough MapPoints in the map
                    // 上一帧跟踪成功
                    if(!mVelocity.empty())
                    {
                        // 利用恒速模型跟踪
                        bOK = TrackWithMotionModel();
                    }
                    else
                    {
                        // 利用词袋跟踪
                        bOK = TrackReferenceKeyFrame();
                    }
                }
                // 如果上帧没有足够的匹配点
                else
                {
                    // In last frame we tracked mainly "visual odometry" points.

                    // We compute two camera poses, one from motion model and one doing relocalization.
                    // If relocalization is sucessfull we choose that solution, otherwise we retain
                    // the "visual odometry" solution.
                    // 恒速模型标志
                    bool bOKMM = false;
                    // 重定位标志
                    bool bOKReloc = false;
                    // 地图点集合
                    vector<MapPoint*> vpMPsMM;
                    // 利用恒速模型剔除的标志
                    vector<bool> vbOutMM;
                    // 利用恒速模型得到的位姿
                    cv::Mat TcwMM;
                    // 上一帧位姿有的话
                    if(!mVelocity.empty())
                    {
                        //利用恒速模型试试看
                        bOKMM = TrackWithMotionModel();
                        //得到当前帧的地图点
                        vpMPsMM = mCurrentFrame.mvpMapPoints;
                        //当前帧重投影误差大被剔除的点
                        vbOutMM = mCurrentFrame.mvbOutlier;
                        // 得到的位姿
                        TcwMM = mCurrentFrame.mTcw.clone();
                    }
                    // 没有的话直接进行重定位
                    bOKReloc = Relocalization();
                    // 恒速模型成功,重定位不成功
                    if(bOKMM && !bOKReloc)
                    {
                        // 利用恒速模型计算的结果
                        mCurrentFrame.SetPose(TcwMM);
                        mCurrentFrame.mvpMapPoints = vpMPsMM;
                        mCurrentFrame.mvbOutlier = vbOutMM;
                        // 没有足够的匹配点
                        if(mbVO)
                        {
                            for(int i =0; i<mCurrentFrame.N; i++)
                            {
                                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                                {
                                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                                }
                            }
                        }
                    }
                    // 重定位成功
                    else if(bOKReloc)
                    {
                        // 证明有足够的匹配点 > 50
                        mbVO = false;
                    }
                    // 重定位或者恒速模型有一个成功就证明跟踪上了
                    bOK = bOKReloc || bOKMM;
                }
            }
        }
        // 当前帧没有参考关键帧,就给他一个
        if(!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

        // If we have an initial estimation of the camera pose and matching. Track the local map.
        // ? 计算出来位置以后再利用帧到地图的匹配做一个优化
        if(!mbOnlyTracking)
        {
            if(bOK)
            {
#ifdef SAVE_TIMES
                std::chrono::steady_clock::time_point time_StartTrackLocalMap = std::chrono::steady_clock::now();
#endif          // 帧到地图的匹配
                bOK = TrackLocalMap();
#ifdef SAVE_TIMES
                std::chrono::steady_clock::time_point time_EndTrackLocalMap = std::chrono::steady_clock::now();

                mTime_LocalMapTrack = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndTrackLocalMap - time_StartTrackLocalMap).count();
#endif


            }
            // 没有完成帧到地图的跟踪
            if(!bOK)
                cout << "Fail to track local map!" << endl;
        }
        // 仅定位模式
        else
        {
            // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
            // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
            // the camera we will use the local map again.
            // 跟踪点比较少mbVO=true
            // 只有在跟踪成功且跟踪点比较
            if(bOK && !mbVO)
                bOK = TrackLocalMap();
        }

        if(bOK)
            mState = OK;
            // ? bOK和mState有什么关系
        else if (mState == OK)
        {
            // 有IMU就用IMU撑一会
            if(mState == System::IMU_MONOCULAR || mState == System::IMU_STEREO)
            {
                Verbose::PrintMess("Track lost for less than one second",Verbose::VERBOSITY_NORMAL);
                if(!pCurrentMap->isImuInitialized()||!pCurrentMap->GetIniertialBA2())
                {
                    cout << "IMU is not or recently initialized. Reseting active map..." << endl;
                    mpSystem->ResetActiveMap();
                }

                mState=RECENTLY_LOST;

            }
            // 否则就直接设置为失败
            else
                mState=LOST; // visual to lost
            
            //  ?
            if(mCurrentFrame.mnId>mnLastRelocFrameId+mMaxFrames)
            {
                mTimeStampLost = mCurrentFrame.mTimeStamp;
            }
        }

        // Save frame if recent relocalization, since they are used for IMU reset (as we are making copy, it shluld be once mCurrFrame is completely modified)
        // 刚重定位过需要保留当前帧重置IMU
        if((mCurrentFrame.mnId<(mnLastRelocFrameId+mnFramesToResetIMU)) &&  // IMU还没初始化成功
        (mCurrentFrame.mnId > mnFramesToResetIMU) && // 不是第一次初始化
        ((mSensor == System::IMU_MONOCULAR) || (mSensor == System::IMU_STEREO)) && // 有IMU
        pCurrentMap->isImuInitialized()) // 当前地图的IMU初始化成功
        {
            // TODO check this situation
            Verbose::PrintMess("Saving pointer to frame. imu needs reset...", Verbose::VERBOSITY_NORMAL);
            // 保存当前帧和上一帧
            Frame* pF = new Frame(mCurrentFrame);
            pF->mpPrevFrame = new Frame(mLastFrame);

            // Load preintegration
            // 重置当前帧的预积分值
            pF->mpImuPreintegratedFrame = new IMU::Preintegrated(mCurrentFrame.mpImuPreintegratedFrame);
        }
        // ? 当前地图初始化成功
        if(pCurrentMap->isImuInitialized())
        {
            // 跟踪状态正常
            if(bOK)
            {
                // IMU初始化帧够了
                if(mCurrentFrame.mnId==(mnLastRelocFrameId+mnFramesToResetIMU))
                {
                    cout << "RESETING FRAME!!!" << endl;
                    // 初始化IMU
                    //? 这里是空的
                    ResetFrameIMU();
                }
                else if(mCurrentFrame.mnId>(mnLastRelocFrameId+30))
                    // 初始化成功就只传播IMU
                    mLastBias = mCurrentFrame.mImuBias;
            }
        }
        // Update drawer
        mpFrameDrawer->Update(this);
        if(!mCurrentFrame.mTcw.empty())
            // 设置位姿
            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);
        // 还不是LOST
        if(bOK || mState==RECENTLY_LOST)
        {
            // Update motion model
            // 当前帧和上一帧位姿都有
            if(!mLastFrame.mTcw.empty() && !mCurrentFrame.mTcw.empty())
            {
                cv::Mat LastTwc = cv::Mat::eye(4,4,CV_32F);
                mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0,3).colRange(0,3));
                mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0,3).col(3));
                // T_lc
                mVelocity = mCurrentFrame.mTcw*LastTwc;
            }
            // 没有的话就是空的
            else
                mVelocity = cv::Mat();

            if(mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO)
                mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

            // Clean VO matches
            for(int i=0; i<mCurrentFrame.N; i++)
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(pMP)
                    if(pMP->Observations()<1)
                    {
                        mCurrentFrame.mvbOutlier[i] = false;
                        mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                    }
            }

            // Delete temporal MapPoints
            for(list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend =  mlpTemporalPoints.end(); lit!=lend; lit++)
            {
                MapPoint* pMP = *lit;
                delete pMP;
            }
            mlpTemporalPoints.clear();

#ifdef SAVE_TIMES
            std::chrono::steady_clock::time_point timeStartNewKF = std::chrono::steady_clock::now();
#endif
            // 判断是不是需要插入新的关键帧
            bool bNeedKF = NeedNewKeyFrame();
#ifdef SAVE_TIMES
            std::chrono::steady_clock::time_point timeEndNewKF = std::chrono::steady_clock::now();

            mTime_NewKF_Dec = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(timeEndNewKF - timeStartNewKF).count();
#endif



            // Check if we need to insert a new keyframe
            if(bNeedKF &&  // 需要插入
            // 并且跟踪没问题或者带IMU的情况下刚丢
            (bOK|| (mState==RECENTLY_LOST && (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO))))
                // 生成关键帧
                // todo
                CreateNewKeyFrame();

            // We allow points with high innovation (considererd outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position
            // with those points so we discard them in the frame. Only has effect if lastframe is tracked
            for(int i=0; i<mCurrentFrame.N;i++)
            {
                // 存在但是被剔除
                if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
            }
        }

        // Reset if the camera get lost soon after initialization
        // 跟丢了
        if(mState==LOST)
        {
            // 地图里面只有五个关键帧
            if(pCurrentMap->KeyFramesInMap()<=5)
            {
                // 直接重新启动地图
                mpSystem->ResetActiveMap();
                return;
            }
            if ((mSensor == System::IMU_MONOCULAR) || (mSensor == System::IMU_STEREO))
                // IMU还没初始化成功
                if (!pCurrentMap->isImuInitialized())
                {
                    // 直接重新初始化地图
                    Verbose::PrintMess("Track lost before IMU initialisation, reseting...", Verbose::VERBOSITY_QUIET);
                    mpSystem->ResetActiveMap();
                    return;
                }
            // 否则就保存当前的,再重新初始化一个地图
            CreateMapInAtlas();
        }
        // 当前帧没有参考帧,就给他一个参考帧
        // ! 待看这个参考帧是谁
        if(!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;
        // 把当前帧赋值为上一帧
        mLastFrame = Frame(mCurrentFrame);
    }



    // 状态是好的
    if(mState==OK || mState==RECENTLY_LOST)
    {
        // Store frame pose information to retrieve the complete camera trajectory afterwards.
        // 计算当前帧到参考帧的位姿
        if(!mCurrentFrame.mTcw.empty())
        {
            cv::Mat Tcr = mCurrentFrame.mTcw*mCurrentFrame.mpReferenceKF->GetPoseInverse();
            mlRelativeFramePoses.push_back(Tcr);
            mlpReferences.push_back(mCurrentFrame.mpReferenceKF);
            mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
            mlbLost.push_back(mState==LOST);
        }
        else
        {
            // This can happen if tracking is lost
            // 因为是刚丢(小于1s,所以可以直接把上一帧有位姿的帧和参考帧的位姿当成当前帧的)
            mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
            mlpReferences.push_back(mlpReferences.back());
            mlFrameTimes.push_back(mlFrameTimes.back());
            mlbLost.push_back(mState==LOST);
        }

    }
}

// TODO 双目+imu初始化
void Tracking::StereoInitialization()
{
    cout<<"Stereo initial"<<endl;
    // 当前帧的特征点大于500个
    if(mCurrentFrame.N>500)
    {
        // 如果是双目+IMU的系统
        if (mSensor == System::IMU_STEREO)
        {
            // 如果当前帧或者上一帧没有imu预积分的值->直接返回
            if (!mCurrentFrame.mpImuPreintegrated || !mLastFrame.mpImuPreintegrated)
            {
                cout << "not IMU meas" << endl;
                return;
            }
            // ?加速度太小激励不够???
            if (cv::norm(mCurrentFrame.mpImuPreintegratedFrame->avgA-mLastFrame.mpImuPreintegratedFrame->avgA)<0.5)
            {
                cout << cv::norm(mCurrentFrame.mpImuPreintegratedFrame->avgA) << endl;
                cout << "not enough acceleration" << endl;
                return;
            }

            if(mpImuPreintegratedFromLastKF)
                delete mpImuPreintegratedFromLastKF;

            // 初始化imu预积分值
            mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);
            mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;
        }

        // Set Frame pose to the origin (In case of inertial SLAM to imu)
        if (mSensor == System::IMU_STEREO)
        {
            // 深拷贝 得到当前帧的IMU和相机的外参
            cv::Mat Rwb0 = mCurrentFrame.mImuCalib.Tcb.rowRange(0,3).colRange(0,3).clone();
            cv::Mat twb0 = mCurrentFrame.mImuCalib.Tcb.rowRange(0,3).col(3).clone();
            mCurrentFrame.SetImuPoseVelocity(Rwb0, twb0, cv::Mat::zeros(3,1,CV_32F));
        }
        else
            mCurrentFrame.SetPose(cv::Mat::eye(4,4,CV_32F));

        // Create KeyFrame
        // 初始帧
        KeyFrame* pKFini = new KeyFrame(mCurrentFrame,mpAtlas->GetCurrentMap(),mpKeyFrameDB);

        // Insert KeyFrame in the map
        // 把这个关键帧加入到地图中
        mpAtlas->AddKeyFrame(pKFini);
        

        // Create MapPoints and asscoiate to KeyFrame
        // 如果只有一个相机模型
        if(!mpCamera2){
            // 把深度为正的点插入到地图中
            for(int i=0; i<mCurrentFrame.N;i++)
            {
                float z = mCurrentFrame.mvDepth[i];
                if(z>0)
                {
                    // 根据uv计算xyz再投影到世界坐标系
                    cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                    // 新的地图点:3D点,帧,地图
                    MapPoint* pNewMP = new MapPoint(x3D,pKFini,mpAtlas->GetCurrentMap());
                    // 地图添加观测信息 计算该点和共视帧点的距离, 更新方向和深度测量范围
                    pNewMP->AddObservation(pKFini,i);
                    // 帧中添加地图点和ID
                    pKFini->AddMapPoint(pNewMP,i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    // Altas中添加地图点
                    mpAtlas->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                }
            }
        } else{
            for(int i = 0; i < mCurrentFrame.Nleft; i++){
                int rightIndex = mCurrentFrame.mvLeftToRightMatch[i];
                if(rightIndex != -1){
                    cv::Mat x3D = mCurrentFrame.mvStereo3Dpoints[i];

                    MapPoint* pNewMP = new MapPoint(x3D,pKFini,mpAtlas->GetCurrentMap());

                    pNewMP->AddObservation(pKFini,i);
                    pNewMP->AddObservation(pKFini,rightIndex + mCurrentFrame.Nleft);

                    pKFini->AddMapPoint(pNewMP,i);
                    pKFini->AddMapPoint(pNewMP,rightIndex + mCurrentFrame.Nleft);

                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpAtlas->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                    mCurrentFrame.mvpMapPoints[rightIndex + mCurrentFrame.Nleft]=pNewMP;
                }
            }
        }

        Verbose::PrintMess("New Map created with " + to_string(mpAtlas->MapPointsInMap()) + " points", Verbose::VERBOSITY_QUIET);
        // 局部地图插入关键帧
        mpLocalMapper->InsertKeyFrame(pKFini);
        // 更新LastFrame
        mLastFrame = Frame(mCurrentFrame);
        mnLastKeyFrameId=mCurrentFrame.mnId;
        mpLastKeyFrame = pKFini;
        mnLastRelocFrameId = mCurrentFrame.mnId;

        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints=mpAtlas->GetAllMapPoints();
        mpReferenceKF = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;

        mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

        mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.push_back(pKFini);

        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);
        // 初始化成功
        mState=OK;
    }
}


void Tracking::MonocularInitialization()
{

    if(!mpInitializer)
    {
        // Set Reference Frame
        if(mCurrentFrame.mvKeys.size()>100)
        {
            mInitialFrame = Frame(mCurrentFrame);
            mLastFrame = Frame(mCurrentFrame);
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            // 对于当前帧的所有的特征点
            for(size_t i=0; i<mCurrentFrame.mvKeysUn.size(); i++)
            // 把当前帧去畸变的特征存起来
                mvbPrevMatched[i]=mCurrentFrame.mvKeysUn[i].pt;

            if(mpInitializer)
                delete mpInitializer;
            // const Frame &ReferenceFrame, float sigma, int iterations
            mpInitializer =  new Initializer(mCurrentFrame,1.0,200);

            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);

            if (mSensor == System::IMU_MONOCULAR)
            {
                if(mpImuPreintegratedFromLastKF)
                {
                    delete mpImuPreintegratedFromLastKF;
                }
                mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);
                mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;

            }
            return;
        }
    }
    else
    {
        // 这帧不好
        if (((int)mCurrentFrame.mvKeys.size()<=100)||((mSensor == System::IMU_MONOCULAR)&&(mLastFrame.mTimeStamp-mInitialFrame.mTimeStamp>1.0)))
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);

            return;
        }

        // Find correspondences
        ORBmatcher matcher(0.9,true);
        // 更新匹配关系
        int nmatches = matcher.SearchForInitialization(mInitialFrame,mCurrentFrame,mvbPrevMatched,mvIniMatches,100);

        // Check if there are enough correspondences
        // 最后保留的匹配太少的话就叫停这次初始化
        if(nmatches<100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);
            return;
        }

        cv::Mat Rcw; // Current Camera Rotation
        cv::Mat tcw; // Current Camera Translation
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)
        // 其实就是单目初始化,利用F和H解算R,T
        if(mpCamera->ReconstructWithTwoViews(mInitialFrame.mvKeysUn,mCurrentFrame.mvKeysUn,mvIniMatches,Rcw,tcw,mvIniP3D,vbTriangulated))
        {
            // 删除没有三角化的点
            for(size_t i=0, iend=mvIniMatches.size(); i<iend;i++)
            {
                if(mvIniMatches[i]>=0 && !vbTriangulated[i])
                {
                    mvIniMatches[i]=-1;
                    nmatches--;
                }
            }

            // Set Frame Poses
             // 将初始化的第一帧作为世界坐标系，因此第一帧变换矩阵为单位矩阵
            mInitialFrame.SetPose(cv::Mat::eye(4,4,CV_32F));
            // 由Rcw和tcw构造Tcw,并赋值给mTcw，mTcw为世界坐标系到该帧的变换矩阵
            cv::Mat Tcw = cv::Mat::eye(4,4,CV_32F);
            Rcw.copyTo(Tcw.rowRange(0,3).colRange(0,3));
            tcw.copyTo(Tcw.rowRange(0,3).col(3));
            mCurrentFrame.SetPose(Tcw);

            // CreateInitialMapMonocular将3D点包装成MapPoint类型存入KeyFrame和Map中
            CreateInitialMapMonocular();

            // Just for video
            // bStepByStep = true;
        }
    }
}


/**
 * @brief 生成地图点
 * 
 */
void Tracking::CreateInitialMapMonocular()
{
    // Create KeyFrames
    KeyFrame* pKFini = new KeyFrame(mInitialFrame,mpAtlas->GetCurrentMap(),mpKeyFrameDB);
    KeyFrame* pKFcur = new KeyFrame(mCurrentFrame,mpAtlas->GetCurrentMap(),mpKeyFrameDB);

    if(mSensor == System::IMU_MONOCULAR)
        pKFini->mpImuPreintegrated = (IMU::Preintegrated*)(NULL);

    // 计算词袋
    pKFini->ComputeBoW();
    pKFcur->ComputeBoW();

    // Insert KFs in the map
    // 在altas插入关键帧
    mpAtlas->AddKeyFrame(pKFini);
    mpAtlas->AddKeyFrame(pKFcur);

    for(size_t i=0; i<mvIniMatches.size();i++)
    {
        if(mvIniMatches[i]<0)
            continue;
        //Create MapPoint.
        // 插入地图点的套路
        // cv::Mat world_pos(mvIniP3D[i]);
        // MapPoint* pMP = new MapPoint(world_pos,pKFcur,mpAtlas->GetCurrentMap());
        // pKFini->AddMapPoint(pMP,i);
        // pKFcur->AddMapPoint(pMP,mvIniMatches[i]);
        // pMP->AddObservation(pKFini,i);
        // pMP->AddObservation(pKFcur,mvIniMatches[i]);
        // pMP->ComputeDistinctiveDescriptors();
        // pMP->UpdateNormalAndDepth();
        cv::Mat worldPos(mvIniP3D[i]);
        MapPoint* pMP = new MapPoint(worldPos,pKFcur,mpAtlas->GetCurrentMap());
        pKFini->AddMapPoint(pMP,i);
        pKFcur->AddMapPoint(pMP,mvIniMatches[i]);
        pMP->AddObservation(pKFini,i);
        pMP->AddObservation(pKFcur,mvIniMatches[i]);
        pMP->ComputeDistinctiveDescriptors();
        // 更新平均观测方向以及观测距离范围
        pMP->UpdateNormalAndDepth();

        //Fill Current Frame structure
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

        //Add to Map
        mpAtlas->AddMapPoint(pMP);
    }


    // Update Connections
    // 更新共视关系
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    std::set<MapPoint*> sMPs;
    sMPs = pKFini->GetMapPoints();

    // Bundle Adjustment
    Verbose::PrintMess("New Map created with " + to_string(mpAtlas->MapPointsInMap()) + " points", Verbose::VERBOSITY_QUIET);
    Optimizer::GlobalBundleAdjustemnt(mpAtlas->GetCurrentMap(),20);
    // 看看地图点都在哪个帧上
    pKFcur->PrintPointDistribution();
    // 计算当前帧地图点的中值深度
    float medianDepth = pKFini->ComputeSceneMedianDepth(2);
    float invMedianDepth;
    // ?为什么要再乘4??
    if(mSensor == System::IMU_MONOCULAR)
        invMedianDepth = 4.0f/medianDepth; // 4.0f
    else
        invMedianDepth = 1.0f/medianDepth;

    if(medianDepth<0 ||  // 中值深度小于0
       pKFcur->TrackedMapPoints(1)<50) // 小于50个地图点
    {
        Verbose::PrintMess("Wrong initialization, reseting...", Verbose::VERBOSITY_NORMAL);
        mpSystem->ResetActiveMap();
        return;
    }


    //将两帧之间的变换归一化到平均深度1的尺度下
    // Scale initial baseline
    // 得到当前帧的位姿
    cv::Mat Tc2w = pKFcur->GetPose();
    // t/MedianDepth
    Tc2w.col(3).rowRange(0,3) = Tc2w.col(3).rowRange(0,3)*invMedianDepth;
    // 得到新的位姿
    pKFcur->SetPose(Tc2w);

    // Scale points
    // 将地图点也按计算的中值归一化
    vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
    for(size_t iMP=0; iMP<vpAllMapPoints.size(); iMP++)
    {
        if(vpAllMapPoints[iMP])
        {
            MapPoint* pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
            pMP->UpdateNormalAndDepth();
        }
    }

    if (mSensor == System::IMU_MONOCULAR)
    {
        // 当前帧的上一帧是初始帧
        pKFcur->mPrevKF = pKFini;
        // 初始帧的下一帧是当前帧
        pKFini->mNextKF = pKFcur;
        // 当前帧的预积分值
        pKFcur->mpImuPreintegrated = mpImuPreintegratedFromLastKF;
        // 构造函数
        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(pKFcur->mpImuPreintegrated->GetUpdatedBias(), // 偏置
                                                              pKFcur->mImuCalib); // 外参
    }

    // 局部地图中插入关键帧
    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);
    // ?mFirstTs看到局部地图的时候应该就知道了
    mpLocalMapper->mFirstTs=pKFcur->mTimeStamp;
    // 位���
    mCurrentFrame.SetPose(pKFcur->GetPose());
    mnLastKeyFrameId=mCurrentFrame.mnId;
    mpLastKeyFrame = pKFcur;
    mnLastRelocFrameId = mInitialFrame.mnId;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints=mpAtlas->GetAllMapPoints();
    mpReferenceKF = pKFcur;
    mCurrentFrame.mpReferenceKF = pKFcur;

    // Compute here initial velocity
    vector<KeyFrame*> vKFs = mpAtlas->GetAllKeyFrames();
    // 前后帧的位姿变化
    cv::Mat deltaT = vKFs.back()->GetPose()*vKFs.front()->GetPoseInverse();
    mVelocity = cv::Mat();

    // 根据旋转矩阵得到旋转向量
    // todo 不清楚怎么计算
    Eigen::Vector3d phi = LogSO3(Converter::toMatrix3d(deltaT.rowRange(0,3).colRange(0,3)));

    // 当前帧到上一帧占据当前帧到初始帧的时间
    double aux = (mCurrentFrame.mTimeStamp-mLastFrame.mTimeStamp)/(mCurrentFrame.mTimeStamp-mInitialFrame.mTimeStamp);
    // 旋转的变化
    phi *= aux;
    // 把当前帧给上一帧
    mLastFrame = Frame(mCurrentFrame);
    // 设置新的局部地图
    mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);
    // 位姿
    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());
    // 把当前帧地图的初始帧记录下来
    mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.push_back(pKFini);
    // 初始化完成
    mState=OK;
    // 初始化ID
    initID = pKFcur->mnId;
}

// 创建新的地图
void Tracking::CreateMapInAtlas()
{

    mnLastInitFrameId = mCurrentFrame.mnId;
    // 创建一个新的地图
    mpAtlas->CreateNewMap();
    // 用IMU的话就把MAP的mbIsInertial置为1
    if (mSensor==System::IMU_STEREO || mSensor == System::IMU_MONOCULAR)
        // 当前地图有IMU
        mpAtlas->SetInertialSensor();
    // 还没有初始化
    mbSetInit=false;
    // 设置下一个地图初始化帧的ID
    mnInitialFrameId = mCurrentFrame.mnId+1;
    // 重新初始化,设置系统状态为NO_IMAGE_YET
    mState = NO_IMAGES_YET;

    // Restart the variable with information about the last KF
    // 重新给上一帧的量赋值
    // 速度
    mVelocity = cv::Mat();
    //? 上一个重定位的帧是上一个地图的最后一帧(当前帧)
    mnLastRelocFrameId = mnLastInitFrameId; // The last relocation KF_id is the current id, because it is the new starting point for new map
    // 日志输出
    Verbose::PrintMess("First frame id in map: " + to_string(mnLastInitFrameId+1), Verbose::VERBOSITY_NORMAL);
    // 当前帧是不是有足够的地图点的标志
    mbVO = false; // Init value for know if there are enough MapPoints in the last KF
    // 重新初始化,所以把原来的初始化清空
    if(mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR)
    {
        if(mpInitializer)
            delete mpInitializer;
        mpInitializer = static_cast<Initializer*>(NULL);
    }

    if((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO ) && mpImuPreintegratedFromLastKF)
    {
        delete mpImuPreintegratedFromLastKF;
        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);
    }
    // 没有上一帧了
    if(mpLastKeyFrame)
        mpLastKeyFrame = static_cast<KeyFrame*>(NULL);
    // 没有参考帧
    if(mpReferenceKF)
        mpReferenceKF = static_cast<KeyFrame*>(NULL);
    // 默认构造函数
    mLastFrame = Frame();
    mCurrentFrame = Frame();
    // 没有匹配关系
    mvIniMatches.clear();
    // 生成了一个新的map
    mbCreatedMap = true;

}

/**
 * @brief 更换上一帧变换的地图点
 * 
 */
void Tracking::CheckReplacedInLastFrame()
{
    for(int i =0; i<mLastFrame.N; i++)
    {
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];

        if(pMP)
        {
            // 有没有变换
            // ? GetReplaced是在哪里检查的
            MapPoint* pRep = pMP->GetReplaced();
            if(pRep)
            {
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }
}

/**
 * @brief 利用当前帧进行初始化
 * 
 * @return true 初始化成功
 * @return false 初始化失败
 */
// 跟踪失败的时候会进行这个操作
bool Tracking::TrackReferenceKeyFrame()
{
    // Compute Bag of Words vector
    // 当前帧计算词袋
    mCurrentFrame.ComputeBoW();

    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    // !匹配待看
    ORBmatcher matcher(0.7,true);
    // 临时变量,匹配的特征点
    vector<MapPoint*> vpMapPointMatches;
    // 利用词袋信息得到两帧的匹配结果
    int nmatches = matcher.SearchByBoW(mpReferenceKF,mCurrentFrame,vpMapPointMatches);
    // 如果跟踪数量小于15个
    if(nmatches<15)
    {
        // 跟踪失败
        cout << "TRACK_REF_KF: Less than 15 matches!!\n";
        // bOK = false
        return false;
    }
    // 匹配的大于15个
    // 匹配关系
    mCurrentFrame.mvpMapPoints = vpMapPointMatches;
    // 设置位姿
    // todo 这里没有上上帧和上帧的运动模型,所以无法得到恒速模型只有让他们直接相等
    mCurrentFrame.SetPose(mLastFrame.mTcw);

    //mCurrentFrame.PrintPointDistribution();

    cout << " TrackReferenceKeyFrame mLastFrame.mTcw:  " << mLastFrame.mTcw << endl;
    // 利用PNP得到重投影误差最小的位姿,这里只调整位姿
    // 去除误匹配
    // 处理得到mvbOutlier
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    // 利用优化结果剔除误匹配
    int nmatchesMap = 0;
    // 对于当前帧所有的特征点
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        
        if(mCurrentFrame.mvpMapPoints[i])
        {
            // 删除误匹配的地图点
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                // ? Nleft是不是左目的特征点个数
                if(i < mCurrentFrame.Nleft){
                    // 这个特征点不在视角内
                    pMP->mbTrackInView = false;
                }
                else{
                    // 不在右目的视角内
                    pMP->mbTrackInViewR = false;
                }
                // ? 应该已经是了
                pMP->mbTrackInView = false;
                // 记录一下这个特征点在哪个帧中最后一次被观测
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            // 如果这个点没有被剔除且观测次数大于0
            // 匹配的个数++
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }

    // TODO check these conditions
    // 有IMU的时候就直接返回可以了
    if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO)
        return true;
    // 没有IMU的话要保证去除误匹配后匹配的个数大于10
    else
        return nmatchesMap>=10;
}
// 双目或rgbd摄像头根据深度值为上一帧产生新的MapPoints
// 单目直接return
void Tracking::UpdateLastFrame()
{
    // step 1：更新最近一帧的位姿
    // Update pose according to reference keyframe
    // ?localmap的参考帧?
    KeyFrame* pRef = mLastFrame.mpReferenceKF;
    // 参考帧到当前帧的位姿
    // 现在还没有当前帧
    cv::Mat Tlr = mlRelativeFramePoses.back();
    // 计算当前帧到世界坐标系的变换矩阵
    // 单目的时候，相当于只是完成了将上一帧的世界坐标系下的位姿计算出来
    mLastFrame.SetPose(Tlr*pRef->GetPose());
    // 如果上一帧为关键帧，或者单目的情况或仅为跟踪的状态，则退出
    if(mnLastKeyFrameId==mLastFrame.mnId || 
    mSensor==System::MONOCULAR || 
    mSensor==System::IMU_MONOCULAR || 
    !mbOnlyTracking)
        return;

    // step 2：对于双目或rgbd摄像头，为上一帧临时生成新的MapPoints
    // ?注意这些MapPoints不加入到Map中，在tracking的最后会删除
    // ?跟踪过程中需要将将上一帧的MapPoints投影到当前帧可以缩小匹配范围，加快当前帧与上一帧进行特征点匹配
    // Create "visual odometry" MapPoints
    // We sort points according to their measured depth by the stereo/RGB-D sensor
    // step 2.1：得到上一帧有深度值的特征点<深度,索引>
    // 临时变量
    vector<pair<float,int> > vDepthIdx;
    vDepthIdx.reserve(mLastFrame.N);
    for(int i=0; i<mLastFrame.N;i++)
    {
        float z = mLastFrame.mvDepth[i];
        if(z>0)
        {
            vDepthIdx.push_back(make_pair(z,i));
        }
    }

    if(vDepthIdx.empty())
        return;

    // 这里不用重写sort么
    sort(vDepthIdx.begin(),vDepthIdx.end());

    // We insert all close points (depth<mThDepth)
    // If less than 100 close points, we insert the 100 closest ones.
    int nPoints = 0;
    // 对于有深度的所有的点
    for(size_t j=0; j<vDepthIdx.size();j++)
    {
        int i = vDepthIdx[j].second;

        bool bCreateNew = false;

        MapPoint* pMP = mLastFrame.mvpMapPoints[i];
        //地图点是空的
        if(!pMP)
            bCreateNew = true;
        // 或者观测次数是0
        else if(pMP->Observations()<1)
        {
            bCreateNew = true;
        }
        // 生成临时地图点
        if(bCreateNew)
        {
            // 得到世界坐标系的点的坐标
            cv::Mat x3D = mLastFrame.UnprojectStereo(i);
            // 生成一个地图点
            MapPoint* pNewMP = new MapPoint(x3D,mpAtlas->GetCurrentMap(),&mLastFrame,i);

            // ?这里也存了?
            mLastFrame.mvpMapPoints[i]=pNewMP;
            // 存到了这里
            mlpTemporalPoints.push_back(pNewMP);
            nPoints++;
        }
        else
        {
            nPoints++;
        }

        // 找到一百个
        if(vDepthIdx[j].first>mThDepth && nPoints>100)
        {
            break;
        }
    }
}

// 正常状态下利用恒速模型进行跟踪
bool Tracking::TrackWithMotionModel()
{
    // 最好的匹配小于0.9倍的第二好的就算成功
    ORBmatcher matcher(0.9,true);

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points if in Localization Mode
    // 单目计算当前帧位姿
    // 双目或者RGB-D相机添加地图点并投影到世界坐标系 
    UpdateLastFrame();
    
    // ? 待看 IMU相关
    if (mpAtlas->isImuInitialized()  // ! 待看 IMU初始化标志 
    && (mCurrentFrame.mnId>mnLastRelocFrameId+mnFramesToResetIMU)) // ? mnFramesToResetIMU这个是什么帧
    {
        // Predict ste with IMU if it is initialized and it doesnt need reset
        // ?这里ruturn false也没东西要呀
        // 利用IMU得到当前帧的位姿
        PredictStateIMU();
        return true;
    }
    else  // 这个ELSE是不是证明IMU现在不能用
    {
        // 恒速模型
        mCurrentFrame.SetPose(mVelocity*mLastFrame.mTcw);
    }

    // 初始化当前帧地图点的匹配
    fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));

    // Project points seen in previous frame
    int th;

    if(mSensor==System::STEREO)
        th=7;
    else
        th=15;
    // todo 有疑问
    int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR);

    // If few matches, uses a wider window search
    if(nmatches<20)
    {
        Verbose::PrintMess("Not enough matches, wider window search!!", Verbose::VERBOSITY_NORMAL);
        fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
        // 匹配太少的就扩大搜索窗口重新匹配
        nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,2*th,mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR);
        Verbose::PrintMess("Matches with wider search: " + to_string(nmatches), Verbose::VERBOSITY_NORMAL);

    }
    // 重新匹配后匹配点还是小于20
    if(nmatches<20)
    {
        Verbose::PrintMess("Not enough matches!!", Verbose::VERBOSITY_NORMAL);
        // ?有IMU的话可以利用他撑一会?
        if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO)
            return true;
        else
            return false;
    }

    // Optimize frame pose with all matches
    // ?每一帧过来都需要利用重投影误差剔除误匹配优化位姿?
    //?能不能保证实时性
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            // 删除误匹配的地图点
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                if(i < mCurrentFrame.Nleft){
                    pMP->mbTrackInView = false;
                }
                else{
                    pMP->mbTrackInViewR = false;
                }
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }
    // 只是跟踪状态的话
    if(mbOnlyTracking)
    {
        // 判断匹配的特征和地图点
        mbVO = nmatchesMap<10;
        return nmatches>20;
    }

    if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO)
        return true;
    else
        return nmatchesMap>=10;
}

bool Tracking::TrackLocalMap()
{

    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.
    mTrackedFr++;
    // 生成局部关键帧map和局部地图
    UpdateLocalMap();
    // 把局部地图中的地图点投影到当前帧并进行匹配
    SearchLocalPoints();

    // TOO check outliers before PO
    // 这里好像没有用
    int aux1 = 0, aux2=0;
    for(int i=0; i<mCurrentFrame.N; i++)
        if( mCurrentFrame.mvpMapPoints[i])
        {
            aux1++;
            if(mCurrentFrame.mvbOutlier[i])
                aux2++;
        }

    int inliers;
    // imu还没初始化,只进行相机的优化
    if (!mpAtlas->isImuInitialized())
        Optimizer::PoseOptimization(&mCurrentFrame);
    else
    {
        // 刚重定位完,IMU没初始化成功
        if(mCurrentFrame.mnId<=mnLastRelocFrameId+mnFramesToResetIMU)
        {
            Verbose::PrintMess("TLM: PoseOptimization ", Verbose::VERBOSITY_DEBUG);
            Optimizer::PoseOptimization(&mCurrentFrame);
        }
        else
        {
            // if(!mbMapUpdated && mState == OK) //  && (mnMatchesInliers>30))

            if(!mbMapUpdated) //  && (mnMatchesInliers>30))
            {
                Verbose::PrintMess("TLM: PoseInertialOptimizationLastFrame ", Verbose::VERBOSITY_DEBUG);
                inliers = Optimizer::PoseInertialOptimizationLastFrame(&mCurrentFrame); // , !mpLastKeyFrame->GetMap()->GetIniertialBA1());
            }
            else
            {
                Verbose::PrintMess("TLM: PoseInertialOptimizationLastKeyFrame ", Verbose::VERBOSITY_DEBUG);
                inliers = Optimizer::PoseInertialOptimizationLastKeyFrame(&mCurrentFrame); // , !mpLastKeyFrame->GetMap()->GetIniertialBA1());
            }
        }
    }

    // ?这是在做什么
    aux1 = 0, aux2 = 0;
    for(int i=0; i<mCurrentFrame.N; i++)
        if( mCurrentFrame.mvpMapPoints[i])
        {
            aux1++;
            if(mCurrentFrame.mvbOutlier[i])
                aux2++;
        }

    mnMatchesInliers = 0;

    // Update MapPoints Statistics
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(!mCurrentFrame.mvbOutlier[i])
            {
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                if(!mbOnlyTracking)
                {
                    if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                        mnMatchesInliers++;
                }
                else
                    mnMatchesInliers++;
            }
            else if(mSensor==System::STEREO)
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
        }
    }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    mpLocalMapper->mnMatchesInliers=mnMatchesInliers;
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && mnMatchesInliers<50)
        return false;

    if((mnMatchesInliers>10)&&(mState==RECENTLY_LOST))
        return true;


    if (mSensor == System::IMU_MONOCULAR)
    {
        if(mnMatchesInliers<15)
        {
            return false;
        }
        else
            return true;
    }
    else if (mSensor == System::IMU_STEREO)
    {
        if(mnMatchesInliers<15)
        {
            return false;
        }
        else
            return true;
    }
    else
    {
        if(mnMatchesInliers<30)
            return false;
        else
            return true;
    }
}

// 判断要不要插入新的关键帧
// ? 前边只要当前帧和上一个关键帧差0.25s就直接插入了,下边就不用判断是不是大于0.5了
bool Tracking::NeedNewKeyFrame()
{
    if(((mSensor == System::IMU_MONOCULAR) || (mSensor == System::IMU_STEREO)) &&  // 有IMU 
        !mpAtlas->GetCurrentMap()->isImuInitialized()) // 但是没有初始化成功
    {
        if (mSensor == System::IMU_MONOCULAR &&   // 单目+IMU 
           (mCurrentFrame.mTimeStamp-mpLastKeyFrame->mTimeStamp)>=0.25) // 时间已经大于0.25s就插入一个
            return true;
        else if (mSensor == System::IMU_STEREO && (mCurrentFrame.mTimeStamp-mpLastKeyFrame->mTimeStamp)>=0.25) // 双目也是0.25
            return true;
        else
            return false;
    }
    // 仅定位的话就不需要关键帧
    if(mbOnlyTracking)
        return false;

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    // localmapping停的话也不用插入(闭环检测在用)
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
    {
        return false;
    }

    // Return false if IMU is initialazing
    // 正在初始化就返回false
    if (mpLocalMapper->IsInitializing())
        return false;
    // 地图中关键帧的数量
    const int nKFs = mpAtlas->KeyFramesInMap();

    // Do not insert keyframes if not enough frames have passed from last relocalisation

    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames &&  // 上次重定位后还没有足够的帧插入
       nKFs>mMaxFrames) // ?关键帧的个数大于mMaxFrames
    {
        return false;
    }

    // Tracked MapPoints in the reference keyframe
    int nMinObs = 3;
    if(nKFs<=2)
        nMinObs=2;
    // 参考帧中跟踪的地图点数量
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

    // Local Mapping accept keyframes?
    // ?查询局部地图管理器是否繁忙,也就是当前能否接受新的关键帧
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    // Check how many "close" points are being tracked and how many could be potentially created.
    // 被剔除的地图点
    int nNonTrackedClose = 0;
    // 还在用的地图点
    int nTrackedClose= 0;
    // 不是单目的时候
    if(mSensor!=System::MONOCULAR && mSensor!=System::IMU_MONOCULAR)
    {
        // 如果Nleft = -1 就直接用N,否则用Nleft
        // ?什么是N Nleft
        int N = (mCurrentFrame.Nleft == -1) ? mCurrentFrame.N : mCurrentFrame.Nleft;
        for(int i =0; i<N; i++)
        {
            // 深度在可靠范围内
            if(mCurrentFrame.mvDepth[i]>0 && mCurrentFrame.mvDepth[i]<mThDepth)
            {
                // 地图存在且没被剔除
                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                    nTrackedClose++;
                else
                    nNonTrackedClose++;

            }
        }
    }
    // 需不需要插入新的关键帧
    bool bNeedToInsertClose;
    // 剔除的大于70,还在跟踪的小于100
    // track is weak
    bNeedToInsertClose = (nTrackedClose<100) && (nNonTrackedClose>70);

    // Thresholds
     // 设定inlier阈值，和之前帧特征点匹配的inlier比例
    float thRefRatio = 0.75f;
    // 关键帧只有一帧，那么插入关键帧的阈值设置很低
    if(nKFs<2)
        thRefRatio = 0.4f;
    // 单目情况下插入关键帧的阈值很高
    if(mSensor==System::MONOCULAR)
        thRefRatio = 0.9f;

    if(mpCamera2) thRefRatio = 0.75f;

    if(mSensor==System::IMU_MONOCULAR)
    {
        if(mnMatchesInliers>350) // Points tracked from the local map
            thRefRatio = 0.75f;
        else
            thRefRatio = 0.90f;
    }

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    // *情况1a,距离上一个关键帧已经大于mMaxFrames的帧了
    const bool c1a = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    // *localMapper处于空闲状态,也达到插入的最低条件了
    const bool c1b = ((mCurrentFrame.mnId>=mnLastKeyFrameId+mMinFrames) && bLocalMappingIdle);
    //Condition 1c: tracking is weak
    // 仅是双目的时候,和地图的匹配成功率很低
    const bool c1c = mSensor!=System::MONOCULAR && 
                     mSensor!=System::IMU_MONOCULAR && 
                     mSensor!=System::IMU_STEREO && 
                     //? 是不是证明场景变化很大
                     (mnMatchesInliers<nRefMatches*0.25 || bNeedToInsertClose) ;
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    // 和参考帧相比有很少的匹配点
    const bool c2 = (((mnMatchesInliers<nRefMatches*thRefRatio || bNeedToInsertClose)) && mnMatchesInliers>15);

    // Temporal condition for Inertial cases
    // 时间够了
    bool c3 = false;
    if(mpLastKeyFrame)
    {
        // 时间不能超过0.5s,// ? 上边不是只要大于0.25就直接插入么
        if (mSensor==System::IMU_MONOCULAR)
        {
            if ((mCurrentFrame.mTimeStamp-mpLastKeyFrame->mTimeStamp)>=0.5)
                c3 = true;
        }
        else if (mSensor==System::IMU_STEREO)
        {
            if ((mCurrentFrame.mTimeStamp-mpLastKeyFrame->mTimeStamp)>=0.5)
                c3 = true;
        }
    }
    // 匹配点少或者跟丢了
    bool c4 = false;
    // 单目+IMU 匹配在[15 75]中间或状态是 刚丢
    if ((((mnMatchesInliers<75) && (mnMatchesInliers>15)) || mState==RECENTLY_LOST) && ((mSensor == System::IMU_MONOCULAR))) // MODIFICATION_2, originally ((((mnMatchesInliers<75) && (mnMatchesInliers>15)) || mState==RECENTLY_LOST) && ((mSensor == System::IMU_MONOCULAR)))
        c4=true;
    else
        c4=false;

    if(((c1a||c1b||c1c) && c2)||c3 ||c4)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        if(bLocalMappingIdle)
        {
            return true;
        }
        else
        {
            // 把BA打断
            mpLocalMapper->InterruptBA();
            // 不是单目
            if(mSensor!=System::MONOCULAR  && mSensor!=System::IMU_MONOCULAR)
            {
                // 队列里不能阻塞太多关键帧
                // tracking插入关键帧不是直接插入，而且先插入到mlNewKeyFrames中，
                // 然后localmapper再逐个pop出来插入到mspKeyFrames
                if(mpLocalMapper->KeyframesInQueue()<3)
                    return true;
                else
                    return false;
            }
            // ? 单目就直接不插入
            else
                return false;
        }
    }
    // 其他情况也不插入
    else
        return false;
}

// 生成关键帧
void Tracking::CreateNewKeyFrame()
{
    //imu还在初始化
    if(mpLocalMapper->IsInitializing())
        return;
    // localmapping在堵塞状态
    if(!mpLocalMapper->SetNotStop(true))
        return;
    // 生成新的关键帧
    KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpAtlas->GetCurrentMap(),mpKeyFrameDB);
    // IMU初始化成功
    if(mpAtlas->isImuInitialized())
        pKF->bImu = true;
    
    pKF->SetNewBias(mCurrentFrame.mImuBias);
    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;

    if(mpLastKeyFrame)
    {
        pKF->mPrevKF = mpLastKeyFrame;
        mpLastKeyFrame->mNextKF = pKF;
    }
    else
        Verbose::PrintMess("No last KF in KF creation!!", Verbose::VERBOSITY_NORMAL);

    // Reset preintegration from last KF (Create new object)
    if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO)
    {
        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(pKF->GetImuBias(),pKF->mImuCalib);
    }

    if(mSensor!=System::MONOCULAR && mSensor != System::IMU_MONOCULAR) // TODO check if incluide imu_stereo
    {
        mCurrentFrame.UpdatePoseMatrices();
        // cout << "create new MPs" << endl;
        // We sort points by the measured depth by the stereo/RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest.
        int maxPoint = 100;
        if(mSensor == System::IMU_STEREO)
            maxPoint = 100;

        vector<pair<float,int> > vDepthIdx;
        int N = (mCurrentFrame.Nleft != -1) ? mCurrentFrame.Nleft : mCurrentFrame.N;
        vDepthIdx.reserve(mCurrentFrame.N);
        for(int i=0; i<N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                vDepthIdx.push_back(make_pair(z,i));
            }
        }

        if(!vDepthIdx.empty())
        {
            sort(vDepthIdx.begin(),vDepthIdx.end());

            int nPoints = 0;
            for(size_t j=0; j<vDepthIdx.size();j++)
            {
                int i = vDepthIdx[j].second;

                bool bCreateNew = false;

                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(!pMP)
                    bCreateNew = true;
                else if(pMP->Observations()<1)
                {
                    bCreateNew = true;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                }

                if(bCreateNew)
                {
                    cv::Mat x3D;

                    if(mCurrentFrame.Nleft == -1){
                        x3D = mCurrentFrame.UnprojectStereo(i);
                    }
                    else{
                        x3D = mCurrentFrame.UnprojectStereoFishEye(i);
                    }

                    MapPoint* pNewMP = new MapPoint(x3D,pKF,mpAtlas->GetCurrentMap());
                    pNewMP->AddObservation(pKF,i);

                    //Check if it is a stereo observation in order to not
                    //duplicate mappoints
                    if(mCurrentFrame.Nleft != -1 && mCurrentFrame.mvLeftToRightMatch[i] >= 0){
                        mCurrentFrame.mvpMapPoints[mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]]=pNewMP;
                        pNewMP->AddObservation(pKF,mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]);
                        pKF->AddMapPoint(pNewMP,mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]);
                    }

                    pKF->AddMapPoint(pNewMP,i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpAtlas->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                    nPoints++;
                }
                else
                {
                    nPoints++; // TODO check ???
                }

                if(vDepthIdx[j].first>mThDepth && nPoints>maxPoint)
                {
                    break;
                }
            }

            Verbose::PrintMess("new mps for stereo KF: " + to_string(nPoints), Verbose::VERBOSITY_NORMAL);

        }
    }


    mpLocalMapper->InsertKeyFrame(pKF);

    mpLocalMapper->SetNotStop(false);

    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;
    //cout  << "end creating new KF" << endl;
}

void Tracking::SearchLocalPoints()
{
    // Do not search map points already matched
    // 对于当前帧的所有地图点
    for(vector<MapPoint*>::iterator vit=mCurrentFrame.mvpMapPoints.begin(), vend=mCurrentFrame.mvpMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP)
        {
            if(pMP->isBad())
            {
                *vit = static_cast<MapPoint*>(NULL);
            }
            else
            {
                // 增加观测次数
                pMP->IncreaseVisible();
                // 最近一次观测到
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                // ?标记该点不被投影(因为在当前帧中)
                pMP->mbTrackInView = false;
                pMP->mbTrackInViewR = false;
            }
        }
    }

    int nToMatch=0;

    // Project points in frame and check its visibility
    // 对于局部地图中的所有的地图点
    for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        // 在当前帧中的地图点跳过
        if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        if(pMP->isBad())
            continue;
        // Project (this fills MapPoint variables for matching)
        // 判断是不是在视野中
        if(mCurrentFrame.isInFrustum(pMP,0.5))
        {
            pMP->IncreaseVisible();
            nToMatch++;
        }
        // 投影
        if(pMP->mbTrackInView)
        {
            //把地图投影到当前帧
            mCurrentFrame.mmProjectPoints[pMP->mnId] = cv::Point2f(pMP->mTrackProjX, pMP->mTrackProjY);
        }
    }

    if(nToMatch>0)
    {
        ORBmatcher matcher(0.8);
        int th = 1;
        if(mSensor==System::RGBD)
            th=3;
        if(mpAtlas->isImuInitialized())
        {
            if(mpAtlas->GetCurrentMap()->GetIniertialBA2())
                th=2;
            else
                th=3;
        }
        else if(!mpAtlas->isImuInitialized() && (mSensor==System::IMU_MONOCULAR || mSensor==System::IMU_STEREO))
        {
            th=10;
        }

        // If the camera has been relocalised recently, perform a coarser search
        if(mCurrentFrame.mnId<mnLastRelocFrameId+2)
            th=5;

        if(mState==LOST || mState==RECENTLY_LOST) // Lost for less than 1 second
            th=15; // 15
        // 寻找匹配点
        int matches = matcher.SearchByProjection(mCurrentFrame, mvpLocalMapPoints, th, mpLocalMapper->mbFarPoints, mpLocalMapper->mThFarPoints);
    }
}

void Tracking::UpdateLocalMap()
{
    // This is for visualization
    // 这个是为了可视化
    mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

    // Update
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
}

// 根据生成的局部关键帧,把每一帧的地图点添加到局部关键帧中
void Tracking::UpdateLocalPoints()
{
    // 清空
    mvpLocalMapPoints.clear();

    int count_pts = 0;
    // 对于选出来的所有的关键帧
    for(vector<KeyFrame*>::const_reverse_iterator itKF=mvpLocalKeyFrames.rbegin(), itEndKF=mvpLocalKeyFrames.rend(); itKF!=itEndKF; ++itKF)
    {   

        KeyFrame* pKF = *itKF;
        // 得到关键帧中的地图
        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();
        // 把关键帧中的地图点添加到局部地图中
        for(vector<MapPoint*>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
        {

            MapPoint* pMP = *itMP;
            if(!pMP)
                continue;
            if(pMP->mnTrackReferenceForFrame==mCurrentFrame.mnId)
                continue;
            if(!pMP->isBad())
            {
                count_pts++;
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame=mCurrentFrame.mnId;
            }
        }
    }
}

/**
 * @brief 找出所有和当前帧有关系的帧,把当前帧的参考帧改为最佳的共视帧
 * 当前帧的所有共视帧
 * 当前帧共视帧的最佳共视帧
 * 当前帧共视帧的父帧
 * 当前帧共视帧的子帧
 * 和当前帧最近的20帧
 * 
 * 
 */
void Tracking::UpdateLocalKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    // 声明map存储局部地图的关键帧
    map<KeyFrame*,int> keyframeCounter;
    // imu没初始化或者刚重定位完
    if(!mpAtlas->isImuInitialized() || (mCurrentFrame.mnId<mnLastRelocFrameId+2))
    {
        //对当前帧的所有的特征点
        for(int i=0; i<mCurrentFrame.N; i++)
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    // 所有的观测都加1
                    const map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations();
                    for(map<KeyFrame*,tuple<int,int>>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                        keyframeCounter[it->first]++;
                }
                // 否则删除这个地图点
                else
                {
                    mCurrentFrame.mvpMapPoints[i]=NULL;
                }
            }
        }
    }
    // IMU初始化
    else
    {
        for(int i=0; i<mLastFrame.N; i++)
        {
            // Using lastframe since current frame has not matches yet
            if(mLastFrame.mvpMapPoints[i])
            {
                MapPoint* pMP = mLastFrame.mvpMapPoints[i];
                if(!pMP)
                    continue;
                // 一样的操作
                if(!pMP->isBad())
                {
                    const map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations();
                    for(map<KeyFrame*,tuple<int,int>>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                        keyframeCounter[it->first]++;
                }
                else
                {
                    // MODIFICATION
                    mLastFrame.mvpMapPoints[i]=NULL;
                }
            }
        }
    }


    int max=0;
    KeyFrame* pKFmax= static_cast<KeyFrame*>(NULL);

    mvpLocalKeyFrames.clear();
    // ? 为什么是三
    mvpLocalKeyFrames.reserve(3*keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    for(map<KeyFrame*,int>::const_iterator it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++)
    {
        KeyFrame* pKF = it->first;

        if(pKF->isBad())
            continue;

        // 找到观测最多地图点的帧
        if(it->second>max)
        {
            max=it->second;
            pKFmax=pKF;
        }
        
        // 能观测到当前地图点的帧
        mvpLocalKeyFrames.push_back(pKF);
        // 以当前帧为基准构建的局部地图帧
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }

    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    // 与当前帧有共视地图点的帧的共视帧
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        // Limit the number of keyframes
        // ? 在哪里控制数量的
        if(mvpLocalKeyFrames.size()>80) // 80
            break;

        KeyFrame* pKF = *itKF;

        // 得到最共视的十帧,如果数量不足就返回所有的共视帧
        const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

        // 找到一个最共视的帧
        for(vector<KeyFrame*>::const_iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++)
        {
            KeyFrame* pNeighKF = *itNeighKF;
            if(!pNeighKF->isBad())
            {
                if(pNeighKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        // ?共视帧的子帧
        const set<KeyFrame*> spChilds = pKF->GetChilds();
        for(set<KeyFrame*>::const_iterator sit=spChilds.begin(), send=spChilds.end(); sit!=send; sit++)
        {
            KeyFrame* pChildKF = *sit;
            if(!pChildKF->isBad())
            {
                if(pChildKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        // ?共视帧的父帧
        KeyFrame* pParent = pKF->GetParent();
        if(pParent)
        {
            if(pParent->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                break;
            }
        }
    }

    // Add 10 last temporal KFs (mainly for IMU)
    // 添加20个最近的帧(最多20个)
    if((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO) &&mvpLocalKeyFrames.size()<80)
    {
        //cout << "CurrentKF: " << mCurrentFrame.mnId << endl;
        KeyFrame* tempKeyFrame = mCurrentFrame.mpLastKeyFrame;

        const int Nd = 20;
        for(int i=0; i<Nd; i++){
            if (!tempKeyFrame)
                break;
            //cout << "tempKF: " << tempKeyFrame << endl;
            if(tempKeyFrame->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(tempKeyFrame);
                tempKeyFrame->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                tempKeyFrame=tempKeyFrame->mPrevKF;
            }
        }
    }
    // 更新当前帧的参考关键帧，与自己共视程度最高的关键帧作为参考关键帧
    if(pKFmax)
    {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}

bool Tracking::Relocalization()
{
    Verbose::PrintMess("Starting relocalization", Verbose::VERBOSITY_NORMAL);
    // Compute Bag of Words Vector
    // 计算当前帧的词袋
    mCurrentFrame.ComputeBoW();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    // review
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame, mpAtlas->GetCurrentMap());

    // 没有候选帧就直接失败了
    if(vpCandidateKFs.empty()) {
        Verbose::PrintMess("There are not candidates", Verbose::VERBOSITY_NORMAL);
        return false;
    }
    // 找到的闭环候选帧的个数
    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.75,true);

    // review 自定义类MLPnPsolver,存储优化结果
    vector<MLPnPsolver*> vpMLPnPsolvers;
    vpMLPnPsolvers.resize(nKFs);

    // 地图点的个数
    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    // 候选帧不可用标志
    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates=0;

    for(int i=0; i<nKFs; i++)
    {
        KeyFrame* pKF = vpCandidateKFs[i];
        // ? 关键帧是bad的定义
        if(pKF->isBad())
            vbDiscarded[i] = true;
        else
        {
            // 利用词袋搜索到的匹配
            int nmatches = matcher.SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i]);
            // 匹配太少就直接放弃
            if(nmatches<15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            // 设置优化参数
            else
            {
                // todo
                MLPnPsolver* pSolver = new MLPnPsolver(mCurrentFrame,vvpMapPointMatches[i]);
                pSolver->SetRansacParameters(0.99,10,300,6,0.5,5.991);  //This solver needs at least 6 points
                //得到优化的结果
                vpMLPnPsolvers[i] = pSolver;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;
    ORBmatcher matcher2(0.9,true);

    // 当候选帧存在且还没找到合适的时候
    while(nCandidates>0 && !bMatch)
    {
        // 对于每个候选帧
        for(int i=0; i<nKFs; i++)
        {
            if(vbDiscarded[i])
                continue;

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            // ransac达到最大迭代次数
            bool bNoMore;
            // 解算结果
            MLPnPsolver* pSolver = vpMLPnPsolvers[i];
            // todo
            cv::Mat Tcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers);

            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            if(!Tcw.empty())
            {
                Tcw.copyTo(mCurrentFrame.mTcw);

                set<MapPoint*> sFound;

                const int np = vbInliers.size();

                for(int j=0; j<np; j++)
                {
                    if(vbInliers[j])
                    {
                        mCurrentFrame.mvpMapPoints[j]=vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j]=NULL;
                }

                // 返回的是优化后减去误匹配后的匹配个数
                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                // 好的匹配个数太少就继续循环
                if(nGood<10)
                    continue;
                // 把误匹配的点剔除
                for(int io =0; io<mCurrentFrame.N; io++)
                    if(mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io]=static_cast<MapPoint*>(NULL);

                // If few inliers, search by projection in a coarse window and optimize again
                // 匹配点个数大于10小于50
                if(nGood<50)
                {
                    // 再找找有没有其他的匹配点
                    // ?这里返回的不应该是所有的匹配
                    int nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,10,100);
                    // 匹配点够了,就再进行一次优化
                    if(nadditional+nGood>=50)
                    {
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        // 优化后再找一次
                        if(nGood>30 && nGood<50)
                        {
                            sFound.clear();
                            for(int ip =0; ip<mCurrentFrame.N; ip++)
                                if(mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,3,64);

                            // Final optimization
                            // 找够了以后再优化一次
                            if(nGood+nadditional>=50)
                            {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                for(int io =0; io<mCurrentFrame.N; io++)
                                    if(mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io]=NULL;
                            }
                        }
                    }
                }


                // If the pose is supported by enough inliers stop ransacs and continue
                // 如果匹配大于50就认为找到了
                if(nGood>=50)
                {
                    bMatch = true;
                    break;
                }
            }
        }
    }

    if(!bMatch)
    {
        return false;
    }
    else
    {
        mnLastRelocFrameId = mCurrentFrame.mnId;
        cout << "Relocalized!!" << endl;
        return true;
    }

}

void Tracking::Reset(bool bLocMap)
{
    Verbose::PrintMess("System Reseting", Verbose::VERBOSITY_NORMAL);

    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
            usleep(3000);
    }

    // Reset Local Mapping
    if (!bLocMap)
    {
        Verbose::PrintMess("Reseting Local Mapper...", Verbose::VERBOSITY_NORMAL);
        mpLocalMapper->RequestReset();
        Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);
    }


    // Reset Loop Closing
    Verbose::PrintMess("Reseting Loop Closing...", Verbose::VERBOSITY_NORMAL);
    mpLoopClosing->RequestReset();
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear BoW Database
    Verbose::PrintMess("Reseting Database...", Verbose::VERBOSITY_NORMAL);
    mpKeyFrameDB->clear();
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear Map (this erase MapPoints and KeyFrames)
    mpAtlas->clearAtlas();
    mpAtlas->CreateNewMap();
    if (mSensor==System::IMU_STEREO || mSensor == System::IMU_MONOCULAR)
        mpAtlas->SetInertialSensor();
    mnInitialFrameId = 0;

    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    if(mpInitializer)
    {
        delete mpInitializer;
        mpInitializer = static_cast<Initializer*>(NULL);
    }
    mbSetInit=false;

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();
    mCurrentFrame = Frame();
    mnLastRelocFrameId = 0;
    mLastFrame = Frame();
    mpReferenceKF = static_cast<KeyFrame*>(NULL);
    mpLastKeyFrame = static_cast<KeyFrame*>(NULL);
    mvIniMatches.clear();

    if(mpViewer)
        mpViewer->Release();

    Verbose::PrintMess("   End reseting! ", Verbose::VERBOSITY_NORMAL);
}

void Tracking::ResetActiveMap(bool bLocMap)
{
    Verbose::PrintMess("Active map Reseting", Verbose::VERBOSITY_NORMAL);
    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
            usleep(3000);
    }

    Map* pMap = mpAtlas->GetCurrentMap();

    if (!bLocMap)
    {
        Verbose::PrintMess("Reseting Local Mapper...", Verbose::VERBOSITY_NORMAL);
        mpLocalMapper->RequestResetActiveMap(pMap);
        Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);
    }

    // Reset Loop Closing
    Verbose::PrintMess("Reseting Loop Closing...", Verbose::VERBOSITY_NORMAL);
    mpLoopClosing->RequestResetActiveMap(pMap);
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear BoW Database
    Verbose::PrintMess("Reseting Database", Verbose::VERBOSITY_NORMAL);
    mpKeyFrameDB->clearMap(pMap); // Only clear the active map references
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear Map (this erase MapPoints and KeyFrames)
    mpAtlas->clearMap();


    //KeyFrame::nNextId = mpAtlas->GetLastInitKFid();
    //Frame::nNextId = mnLastInitFrameId;
    mnLastInitFrameId = Frame::nNextId;
    mnLastRelocFrameId = mnLastInitFrameId;
    mState = NO_IMAGES_YET; //NOT_INITIALIZED;

    if(mpInitializer)
    {
        delete mpInitializer;
        mpInitializer = static_cast<Initializer*>(NULL);
    }

    list<bool> lbLost;
    // lbLost.reserve(mlbLost.size());
    unsigned int index = mnFirstFrameId;
    cout << "mnFirstFrameId = " << mnFirstFrameId << endl;
    for(Map* pMap : mpAtlas->GetAllMaps())
    {
        if(pMap->GetAllKeyFrames().size() > 0)
        {
            if(index > pMap->GetLowerKFID())
                index = pMap->GetLowerKFID();
        }
    }

    //cout << "First Frame id: " << index << endl;
    int num_lost = 0;
    cout << "mnInitialFrameId = " << mnInitialFrameId << endl;

    for(list<bool>::iterator ilbL = mlbLost.begin(); ilbL != mlbLost.end(); ilbL++)
    {
        if(index < mnInitialFrameId)
            lbLost.push_back(*ilbL);
        else
        {
            lbLost.push_back(true);
            num_lost += 1;
        }

        index++;
    }
    cout << num_lost << " Frames had been set to lost" << endl;

    mlbLost = lbLost;

    mnInitialFrameId = mCurrentFrame.mnId;
    mnLastRelocFrameId = mCurrentFrame.mnId;

    mCurrentFrame = Frame();
    mLastFrame = Frame();
    mpReferenceKF = static_cast<KeyFrame*>(NULL);
    mpLastKeyFrame = static_cast<KeyFrame*>(NULL);
    mvIniMatches.clear();

    if(mpViewer)
        mpViewer->Release();

    Verbose::PrintMess("   End reseting! ", Verbose::VERBOSITY_NORMAL);
}

vector<MapPoint*> Tracking::GetLocalMapMPS()
{
    return mvpLocalMapPoints;
}

void Tracking::ChangeCalibration(const string &strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    Frame::mbInitialComputations = true;
}

// 设置模式: true 定位模式; false SLAM模式
void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
}

void Tracking::UpdateFrameIMU(const float s, const IMU::Bias &b, KeyFrame* pCurrentKeyFrame)
{
    Map * pMap = pCurrentKeyFrame->GetMap();
    unsigned int index = mnFirstFrameId;
    list<ORB_SLAM3::KeyFrame*>::iterator lRit = mlpReferences.begin();
    list<bool>::iterator lbL = mlbLost.begin();
    for(list<cv::Mat>::iterator lit=mlRelativeFramePoses.begin(),lend=mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lbL++)
    {
        if(*lbL)
            continue;

        KeyFrame* pKF = *lRit;

        while(pKF->isBad())
        {
            pKF = pKF->GetParent();
        }

        if(pKF->GetMap() == pMap)
        {
            (*lit).rowRange(0,3).col(3)=(*lit).rowRange(0,3).col(3)*s;
        }
    }

    mLastBias = b;

    mpLastKeyFrame = pCurrentKeyFrame;

    mLastFrame.SetNewBias(mLastBias);
    mCurrentFrame.SetNewBias(mLastBias);

    cv::Mat Gz = (cv::Mat_<float>(3,1) << 0, 0, -IMU::GRAVITY_VALUE);

    cv::Mat twb1;
    cv::Mat Rwb1;
    cv::Mat Vwb1;
    float t12;

    while(!mCurrentFrame.imuIsPreintegrated())
    {
        usleep(500);
    }


    if(mLastFrame.mnId == mLastFrame.mpLastKeyFrame->mnFrameId)
    {
        mLastFrame.SetImuPoseVelocity(mLastFrame.mpLastKeyFrame->GetImuRotation(),
                                      mLastFrame.mpLastKeyFrame->GetImuPosition(),
                                      mLastFrame.mpLastKeyFrame->GetVelocity());
    }
    else
    {
        twb1 = mLastFrame.mpLastKeyFrame->GetImuPosition();
        Rwb1 = mLastFrame.mpLastKeyFrame->GetImuRotation();
        Vwb1 = mLastFrame.mpLastKeyFrame->GetVelocity();
        t12 = mLastFrame.mpImuPreintegrated->dT;

        mLastFrame.SetImuPoseVelocity(Rwb1*mLastFrame.mpImuPreintegrated->GetUpdatedDeltaRotation(),
                                      twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*mLastFrame.mpImuPreintegrated->GetUpdatedDeltaPosition(),
                                      Vwb1 + Gz*t12 + Rwb1*mLastFrame.mpImuPreintegrated->GetUpdatedDeltaVelocity());
    }

    if (mCurrentFrame.mpImuPreintegrated)
    {
        twb1 = mCurrentFrame.mpLastKeyFrame->GetImuPosition();
        Rwb1 = mCurrentFrame.mpLastKeyFrame->GetImuRotation();
        Vwb1 = mCurrentFrame.mpLastKeyFrame->GetVelocity();
        t12 = mCurrentFrame.mpImuPreintegrated->dT;

        mCurrentFrame.SetImuPoseVelocity(Rwb1*mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaRotation(),
                                      twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaPosition(),
                                      Vwb1 + Gz*t12 + Rwb1*mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaVelocity());
    }

    mnFirstImuFrameId = mCurrentFrame.mnId;
}


cv::Mat Tracking::ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2)
{
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    cv::Mat R12 = R1w*R2w.t();
    cv::Mat t12 = -R1w*R2w.t()*t2w+t1w;

    cv::Mat t12x = Converter::tocvSkewMatrix(t12);

    const cv::Mat &K1 = pKF1->mK;
    const cv::Mat &K2 = pKF2->mK;


    return K1.t().inv()*t12x*R12*K2.inv();
}


void Tracking::CreateNewMapPoints()
{
    // Retrieve neighbor keyframes in covisibility graph
    const vector<KeyFrame*> vpKFs = mpAtlas->GetAllKeyFrames();

    ORBmatcher matcher(0.6,false);

    cv::Mat Rcw1 = mpLastKeyFrame->GetRotation();
    cv::Mat Rwc1 = Rcw1.t();
    cv::Mat tcw1 = mpLastKeyFrame->GetTranslation();
    cv::Mat Tcw1(3,4,CV_32F);
    Rcw1.copyTo(Tcw1.colRange(0,3));
    tcw1.copyTo(Tcw1.col(3));
    cv::Mat Ow1 = mpLastKeyFrame->GetCameraCenter();

    const float &fx1 = mpLastKeyFrame->fx;
    const float &fy1 = mpLastKeyFrame->fy;
    const float &cx1 = mpLastKeyFrame->cx;
    const float &cy1 = mpLastKeyFrame->cy;
    const float &invfx1 = mpLastKeyFrame->invfx;
    const float &invfy1 = mpLastKeyFrame->invfy;

    const float ratioFactor = 1.5f*mpLastKeyFrame->mfScaleFactor;

    int nnew=0;

    // Search matches with epipolar restriction and triangulate
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF2 = vpKFs[i];
        if(pKF2==mpLastKeyFrame)
            continue;

        // Check first that baseline is not too short
        cv::Mat Ow2 = pKF2->GetCameraCenter();
        cv::Mat vBaseline = Ow2-Ow1;
        const float baseline = cv::norm(vBaseline);

        if((mSensor!=System::MONOCULAR)||(mSensor!=System::IMU_MONOCULAR))
        {
            if(baseline<pKF2->mb)
            continue;
        }
        else
        {
            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
            const float ratioBaselineDepth = baseline/medianDepthKF2;

            if(ratioBaselineDepth<0.01)
                continue;
        }

        // Compute Fundamental Matrix
        cv::Mat F12 = ComputeF12(mpLastKeyFrame,pKF2);

        // Search matches that fullfil epipolar constraint
        vector<pair<size_t,size_t> > vMatchedIndices;
        matcher.SearchForTriangulation(mpLastKeyFrame,pKF2,F12,vMatchedIndices,false);

        cv::Mat Rcw2 = pKF2->GetRotation();
        cv::Mat Rwc2 = Rcw2.t();
        cv::Mat tcw2 = pKF2->GetTranslation();
        cv::Mat Tcw2(3,4,CV_32F);
        Rcw2.copyTo(Tcw2.colRange(0,3));
        tcw2.copyTo(Tcw2.col(3));

        const float &fx2 = pKF2->fx;
        const float &fy2 = pKF2->fy;
        const float &cx2 = pKF2->cx;
        const float &cy2 = pKF2->cy;
        const float &invfx2 = pKF2->invfx;
        const float &invfy2 = pKF2->invfy;

        // Triangulate each match
        const int nmatches = vMatchedIndices.size();
        for(int ikp=0; ikp<nmatches; ikp++)
        {
            const int &idx1 = vMatchedIndices[ikp].first;
            const int &idx2 = vMatchedIndices[ikp].second;

            const cv::KeyPoint &kp1 = mpLastKeyFrame->mvKeysUn[idx1];
            const float kp1_ur=mpLastKeyFrame->mvuRight[idx1];
            bool bStereo1 = kp1_ur>=0;

            const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];
            const float kp2_ur = pKF2->mvuRight[idx2];
            bool bStereo2 = kp2_ur>=0;

            // Check parallax between rays
            cv::Mat xn1 = (cv::Mat_<float>(3,1) << (kp1.pt.x-cx1)*invfx1, (kp1.pt.y-cy1)*invfy1, 1.0);
            cv::Mat xn2 = (cv::Mat_<float>(3,1) << (kp2.pt.x-cx2)*invfx2, (kp2.pt.y-cy2)*invfy2, 1.0);

            cv::Mat ray1 = Rwc1*xn1;
            cv::Mat ray2 = Rwc2*xn2;
            const float cosParallaxRays = ray1.dot(ray2)/(cv::norm(ray1)*cv::norm(ray2));

            float cosParallaxStereo = cosParallaxRays+1;
            float cosParallaxStereo1 = cosParallaxStereo;
            float cosParallaxStereo2 = cosParallaxStereo;

            if(bStereo1)
                cosParallaxStereo1 = cos(2*atan2(mpLastKeyFrame->mb/2,mpLastKeyFrame->mvDepth[idx1]));
            else if(bStereo2)
                cosParallaxStereo2 = cos(2*atan2(pKF2->mb/2,pKF2->mvDepth[idx2]));

            cosParallaxStereo = min(cosParallaxStereo1,cosParallaxStereo2);

            cv::Mat x3D;
            if(cosParallaxRays<cosParallaxStereo && cosParallaxRays>0 && (bStereo1 || bStereo2 || cosParallaxRays<0.9998))
            {
                // Linear Triangulation Method
                cv::Mat A(4,4,CV_32F);
                A.row(0) = xn1.at<float>(0)*Tcw1.row(2)-Tcw1.row(0);
                A.row(1) = xn1.at<float>(1)*Tcw1.row(2)-Tcw1.row(1);
                A.row(2) = xn2.at<float>(0)*Tcw2.row(2)-Tcw2.row(0);
                A.row(3) = xn2.at<float>(1)*Tcw2.row(2)-Tcw2.row(1);

                cv::Mat w,u,vt;
                cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);

                x3D = vt.row(3).t();

                if(x3D.at<float>(3)==0)
                    continue;

                // Euclidean coordinates
                x3D = x3D.rowRange(0,3)/x3D.at<float>(3);

            }
            else if(bStereo1 && cosParallaxStereo1<cosParallaxStereo2)
            {
                x3D = mpLastKeyFrame->UnprojectStereo(idx1);
            }
            else if(bStereo2 && cosParallaxStereo2<cosParallaxStereo1)
            {
                x3D = pKF2->UnprojectStereo(idx2);
            }
            else
                continue; //No stereo and very low parallax

            cv::Mat x3Dt = x3D.t();

            //Check triangulation in front of cameras
            float z1 = Rcw1.row(2).dot(x3Dt)+tcw1.at<float>(2);
            if(z1<=0)
                continue;

            float z2 = Rcw2.row(2).dot(x3Dt)+tcw2.at<float>(2);
            if(z2<=0)
                continue;

            //Check reprojection error in first keyframe
            const float &sigmaSquare1 = mpLastKeyFrame->mvLevelSigma2[kp1.octave];
            const float x1 = Rcw1.row(0).dot(x3Dt)+tcw1.at<float>(0);
            const float y1 = Rcw1.row(1).dot(x3Dt)+tcw1.at<float>(1);
            const float invz1 = 1.0/z1;

            if(!bStereo1)
            {
                float u1 = fx1*x1*invz1+cx1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                if((errX1*errX1+errY1*errY1)>5.991*sigmaSquare1)
                    continue;
            }
            else
            {
                float u1 = fx1*x1*invz1+cx1;
                float u1_r = u1 - mpLastKeyFrame->mbf*invz1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                float errX1_r = u1_r - kp1_ur;
                if((errX1*errX1+errY1*errY1+errX1_r*errX1_r)>7.8*sigmaSquare1)
                    continue;
            }

            //Check reprojection error in second keyframe
            const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
            const float x2 = Rcw2.row(0).dot(x3Dt)+tcw2.at<float>(0);
            const float y2 = Rcw2.row(1).dot(x3Dt)+tcw2.at<float>(1);
            const float invz2 = 1.0/z2;
            if(!bStereo2)
            {
                float u2 = fx2*x2*invz2+cx2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                if((errX2*errX2+errY2*errY2)>5.991*sigmaSquare2)
                    continue;
            }
            else
            {
                float u2 = fx2*x2*invz2+cx2;
                float u2_r = u2 - mpLastKeyFrame->mbf*invz2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                float errX2_r = u2_r - kp2_ur;
                if((errX2*errX2+errY2*errY2+errX2_r*errX2_r)>7.8*sigmaSquare2)
                    continue;
            }

            //Check scale consistency
            cv::Mat normal1 = x3D-Ow1;
            float dist1 = cv::norm(normal1);

            cv::Mat normal2 = x3D-Ow2;
            float dist2 = cv::norm(normal2);

            if(dist1==0 || dist2==0)
                continue;

            const float ratioDist = dist2/dist1;
            const float ratioOctave = mpLastKeyFrame->mvScaleFactors[kp1.octave]/pKF2->mvScaleFactors[kp2.octave];

            if(ratioDist*ratioFactor<ratioOctave || ratioDist>ratioOctave*ratioFactor)
                continue;

            // Triangulation is succesfull
            MapPoint* pMP = new MapPoint(x3D,mpLastKeyFrame,mpAtlas->GetCurrentMap());

            pMP->AddObservation(mpLastKeyFrame,idx1);
            pMP->AddObservation(pKF2,idx2);

            mpLastKeyFrame->AddMapPoint(pMP,idx1);
            pKF2->AddMapPoint(pMP,idx2);

            pMP->ComputeDistinctiveDescriptors();

            pMP->UpdateNormalAndDepth();

            mpAtlas->AddMapPoint(pMP);
            nnew++;
        }
    }
    TrackReferenceKeyFrame();
}

void Tracking::NewDataset()
{
    mnNumDataset++;
}

int Tracking::GetNumberDataset()
{
    return mnNumDataset;
}

int Tracking::GetMatchesInliers()
{
    return mnMatchesInliers;
}

} //namespace ORB_SLAM
