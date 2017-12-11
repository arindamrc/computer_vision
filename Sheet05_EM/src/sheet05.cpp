
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

#define PI 3.14159
std::string PATH_Image   = "./images/gnome.png";
cv::Rect bb_Image(92,65,105,296);

void part1__1(const cv::Mat& img, const cv::Mat& mask_fg, const cv::Mat& mask_bg);
void part1__2(const cv::Mat& img, const cv::Mat& mask_fg, const cv::Mat& mask_bg);

void show_segments(cv::Mat& img_segmented, int nclusters);

////////////////////////////////////
// class declaration for task 1_1 //
////////////////////////////////////

class GMM_opencv{
private:
    cv::Ptr<cv::ml::EM> em;
    int num_clus;
    cv::Mat samples;               // add more variables if necessary
public:
    GMM_opencv();
    ~GMM_opencv();
    void init(const int nmix, const cv::Mat& img, const cv::Mat& mask);
    void learnGMM();
    cv::Mat return_posterior(const cv::Mat& img);
};

GMM_opencv::GMM_opencv(){

}

GMM_opencv::~GMM_opencv(){

}

void GMM_opencv::init(const int nmix, const cv::Mat &img_orig, const cv::Mat &mask){
    this->num_clus = nmix;
    this->em = cv::ml::EM::create();
    this->em->setClustersNumber(nmix);
    cv::Mat img;
    img_orig.convertTo(img, CV_32FC1);
    samples = cv::Mat(img.rows * img.cols, img.channels(), CV_32FC1);

    int id=0;
    for(int y = 0; y < img.rows; ++y)
    {
        for(int x = 0; x < img.cols; ++x)
        {
            cv::Vec3f pix = img.at<cv::Vec3f>(y,x);

            if(mask.at<uchar>(y,x) > 0){
                this->samples.at<cv::Vec3f>(id, 0) = pix;
                id++;
            }
        }
    }
}

void GMM_opencv::learnGMM(){
    this->em->trainEM(samples);
}

cv::Mat GMM_opencv::return_posterior(const cv::Mat &img_orig){
    cv::Mat predict_img(img_orig.rows, img_orig.cols, CV_64FC1);
    cv::Mat img;
    img_orig.convertTo(img, CV_32FC1);
    std::cout << "orig: " << img_orig.size() << " predict: " << predict_img.size() << " img: " << img.size() << std::endl;
    cv::Mat sample(1, 3, CV_32FC1);
    cv::Mat probs(1, this->num_clus, CV_64FC1);
    for(int y = 0; y< img.rows; y++){
        for(int x = 0; x< img.cols; x++){
            cv::Vec3f pixel = img.at<cv::Vec3f>(y,x);
            sample.at<float>(0) = pixel[0];
            sample.at<float>(1) = pixel[1];
            sample.at<float>(2) = pixel[2];
            cv::Vec2d predicted = this->em->predict2(sample, probs);
            predict_img.at<double>(y,x) = exp(predicted[0]);
        }
    }
    std::cout << "created mat" << std::endl;
    return predict_img;
}

////////////////////////////////////
// class declaration for task 1_2 //
////////////////////////////////////

class GMM_custom{
private:
    int num_clus;
    std::vector<float> wt;             // cache for E step + final model
    std::vector<cv::Mat_<float> > mu;
    std::vector<cv::Mat_<float> > cov;
    cv::Mat samples;           // training pixel samples
    cv::Mat posterior;         // posterior probability for M step
    int maxIter;

    bool performEM();                   // iteratively called by learnGMM()
public:
    GMM_custom();
    ~GMM_custom();
    void init(const int nmix, const cv::Mat& img, const cv::Mat& mask, int iterations); // call this once per image
    void learnGMM();    // call this to learn GMM
    cv::Mat return_posterior(cv::Mat& img);     // call this to generate probability map
};




////////////////////////////////////
// 2_* and 3 are theoretical work //
////////////////////////////////////

int main()
{

    // Uncomment the part of the exercise that you wish to implement.
    // For the final submission all implemented parts should be uncommented.
    cv::Mat img=cv::imread(PATH_Image);
    assert(img.rows*img.cols>0);
    cv::Mat mask_fg(img.rows,img.cols,CV_8U); mask_fg.setTo(0); mask_fg(bb_Image).setTo(255);
    cv::Mat mask_bg(img.rows,img.cols,CV_8U); mask_bg.setTo(255); mask_bg(bb_Image).setTo(0);
    cv::Mat show=img.clone();
    cv::rectangle(show,bb_Image,cv::Scalar(0,0,255),1);
    cv::imshow("Image",show);
    cv::imshow("mask_fg",mask_fg);
    cv::imshow("mask_bg",mask_bg);
    cv::waitKey(0);

    part1__1(img,mask_fg,mask_bg);
//    part1__2(img,mask_fg,mask_bg);

    std::cout <<                                                                                                   std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////    END    /////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout <<                                                                                                   std::endl;

}


void part1__1(const cv::Mat& img, const cv::Mat& mask_fg, const cv::Mat& mask_bg)
{
    std::cout <<                                                                                                   std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////    Part 1__1  /////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout <<                                                                                                   std::endl;


    int nmix=10;

    GMM_opencv gmm_fg;
    gmm_fg.init(nmix,img,mask_fg);
    std::cout << "init done..." <<std::endl;
    gmm_fg.learnGMM();
    std::cout << "model created..." <<std::endl;
    cv::Mat fg=gmm_fg.return_posterior(img);
    std::cout << "posterior created..." <<std::endl;

    GMM_opencv gmm_bg;
    gmm_bg.init(nmix,img,mask_bg);
    std::cout << "init done..." <<std::endl;
    gmm_bg.learnGMM();
    std::cout << "model created..." <<std::endl;
    cv::Mat bg=gmm_bg.return_posterior(img);
    std::cout << "posterior created..." <<std::endl;

    cv::Mat show=bg+fg;
    cv::divide(fg,show,show);
    show.convertTo(show,CV_8U,255);
    cv::imshow("gmm_opencv",show);
    cv::waitKey(0);


    cv::destroyAllWindows();
}


void part1__2(const cv::Mat& img, const cv::Mat& mask_fg, const cv::Mat& mask_bg)
{
    std::cout <<                                                                                                   std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////    Part 1__2 //////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout <<                                                                                                   std::endl;

    /**
    int nmix=

    GMM_custom gmm_fg;
    gmm_fg.init(nmix,img,mask_fg);
    gmm_fg.learnGMM();
    Mat fg=gmm_fg.return_posterior(img);

    GMM_custom gmm_bg;
    gmm_bg.init(nmix,img,mask_bg);
    gmm_bg.learnGMM();
    Mat bg=gmm_bg.return_posterior(img);

    Mat show=bg+fg;
    cv::divide(bg,show,show);
    show.convertTo(show,CV_8U,255);
    cv::imshow("gmm_custom",show);
    cv::waitKey(0);
    **/

    cv::destroyAllWindows();
}

