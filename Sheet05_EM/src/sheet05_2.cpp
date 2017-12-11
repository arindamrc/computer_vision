#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <math.h>

#define PI 3.14159
std::string PATH_Image   = "./images/gnome.png";
cv::Rect bb_Image(92,65,105,296);

void part1__2(const cv::Mat& img, const cv::Mat& mask_fg, const cv::Mat& mask_bg);

////////////////////////////////////
// class declaration for task 1_1 //
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
    const float EPSILON = 1e-6;
    float gaussian_const;

    float gaussian(cv::Mat_<float> x, cv::Mat_<float> mean, cv::Mat_<float> dev);
    float predict(cv::Mat_<float> pix);     // call this to generate probability map
    void stop_overflow(float& val);
    
public:
    GMM_custom();
    ~GMM_custom();
    void init(const int nmix, const cv::Mat& img, const cv::Mat& mask, int iterations); // call this once per image
    void learnGMM();    // call this to learn GMM
    cv::Mat return_posterior(const cv::Mat &img_orig);
};

GMM_custom::GMM_custom()
{
}

GMM_custom::~GMM_custom()
{
}

void GMM_custom::stop_overflow(float& val)
{
        if(std::isnan(val) || val < std::numeric_limits<float>::lowest() || fabs(val) < this->EPSILON){
                val = this->EPSILON;
        }else if(val > std::numeric_limits<float>::max()){
                val = std::numeric_limits<float>::max();
        }
}


float GMM_custom::gaussian(cv::Mat_<float> x, cv::Mat_<float> mean, cv::Mat_<float> covMat)
{
        cv::Mat_<float> y = (x - mean);
        cv::Mat_<float> yT(x.cols, x.rows);
        cv::Mat_<float> covMat_Inv(covMat.rows, covMat.cols);
        
        // transpose the difference
        cv::transpose(y, yT);
        
        // invert the covariance matrix
        if(!cv::invert(covMat, covMat_Inv, cv::DECOMP_CHOLESKY)){
                cv::invert(covMat, covMat_Inv, cv::DECOMP_SVD);
        }
        
        cv::Mat quadForm = (yT * covMat_Inv * y);
        float exponential = std::exp(-0.5 * quadForm.at<float>(0,0));
        float det = fabs(cv::determinant(covMat));
        if(det <= 0.0){
                det = this->EPSILON;
        }
        float normalizer = this->gaussian_const * (1.0 / std::sqrt(det));
        if(!std::isfinite(normalizer)){
                normalizer = std::numeric_limits<float>::max();
        }
        float retVal = normalizer * exponential;
        this->stop_overflow(retVal);
        return retVal;

}


void GMM_custom::init(const int nmix, const cv::Mat& img_orig, const cv::Mat& mask, int iterations)
{
        this->num_clus = nmix;
        this->maxIter = iterations;
        cv::Mat img, labels, centers;
        img_orig.convertTo(img, CV_32FC1);
        this->samples = cv::Mat(img.rows * img.cols, img.channels(), CV_32FC1);
        this->posterior = cv::Mat::zeros(this->samples.rows, 1, CV_32FC1);
        
        this->gaussian_const = 1.0 / pow(2.0 * M_PI, 0.5 * this->samples.cols);

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
        std::cout << "samples created" << std::endl;
        
        // start with k-means; this provides a good initialization for EM
        cv::kmeans(samples, this->num_clus, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, this->maxIter, 1.0), 3, cv::KMEANS_PP_CENTERS, centers);
        
        for(int row = 0; row < centers.rows; row++){
                
                // initialize the centers
                cv::Mat_<float> c(img.channels(), 1);
                for(int col = 0; col < centers.cols; col++){
                        float tmp = centers.at<float>(row,col);
                        c(col, 0) = tmp;
                }
                this->mu.push_back(c);
                
                // initialize the covariances with identity matrices 
                cv::Mat_<float> covar = cv::Mat::ones(img.channels(), img.channels(), CV_32FC1);
                this->cov.push_back(covar);
                
                // initialize the mixture weights
                wt.push_back(1.0/this->num_clus);
        }
}

void GMM_custom::learnGMM()
{
        for(int i = 0; i <= this->maxIter; i++){
                
                
                std::cout << "E-step" << std::endl;
                //*********************************E-step*************************************
                cv::Mat_<float> gamma(this->samples.rows, this->num_clus); 
                cv::Mat_<float> x(this->samples.cols, 1);
                
                // for all samples
                for(int n = 0; n < this->samples.rows; n++){
                        cv::transpose(this->samples.row(n), x); // copy row to another Mat
                        
                        // for all gaussians in the mixture
                        float normalizer = 0.0f;
                        for(int k = 0; k < this->num_clus; k++){
                                float tmp = wt[k] * this->gaussian(x, mu[k], cov[k]);
                                this->stop_overflow(tmp);
                                gamma.at<float>(n,k) = tmp;
                                normalizer += tmp;
                        }
                        
                        // the likelihood value for this sample
                        this->stop_overflow(normalizer);
                        this->posterior.at<float>(n, 0) = normalizer;
                        
                        // normalize the responsibilities
                        if(fabs(normalizer) > this->EPSILON){ // prevent division by zero
                                cv::Mat normalized = (1.0 / normalizer) * gamma.row(n);
                                normalized.copyTo(gamma.row(n));
                        }
                }
                
                // All iterations done
                if(i == this->maxIter){
                        break; // No need for the last M-step
                }
                // E-step done
                
                
                std::cout << "M-step" << std::endl;
                //*********************************M-step*************************************
                cv::Mat N_k;
                cv::reduce(gamma, N_k, 0, CV_REDUCE_SUM, -1); 
                
                // for all gaussians in the mixture
                for(int k = 0; k < this->num_clus; k++){
                        
                        // compute new mean
                        cv::Mat_<float> mu_k(this->samples.cols, 1);
                        // for all samples
                        for(int n = 0; n < this->samples.rows; n++){
                                cv::transpose(this->samples.row(n), x);
                                float tmp = gamma.at<float>(n,k);
                                this->stop_overflow(tmp);
                                mu_k += (tmp * x);
                        }
                        
                        float sumWt = N_k.at<float>(0,k);
                        this->stop_overflow(sumWt);
                        
                        if(fabs(sumWt) > this->EPSILON){
                                mu_k /= sumWt;
                        }
                        this->mu[k] = mu_k;
                        
                        // compute new covariance
                        cv::Mat_<float> sigma_k(this->samples.cols, this->samples.cols);
                        // for all samples 
                        for(int n = 0; n < this->samples.rows; n++){
                                cv::transpose(this->samples.row(n), x);
                                cv::Mat y = x - this->mu[k];
                                cv::Mat yT;
                                cv::transpose(y, yT);
                                float tmp = gamma.at<float>(n,k);
                                this->stop_overflow(tmp);
                                sigma_k += tmp * y * yT;
                        }
                        
                        if(fabs(sumWt) > this->EPSILON){ // prevent division by zero
                                sigma_k /= sumWt; 
                        }
                        this->cov[k] = sigma_k;
                        
                        this->wt[k] = sumWt / this->samples.rows;
                } // M-step done
                
        }
}

float GMM_custom::predict(cv::Mat_<float> pix)
{
        float likelihood = 0.0f;
        for(int i = 0; i < this->num_clus; i++){
                likelihood += this->gaussian(pix, this->mu[i], this->cov[i]);
        }
        return likelihood;
}

cv::Mat GMM_custom::return_posterior(const cv::Mat& img_orig)
{
        cv::Mat predict_img(img_orig.rows, img_orig.cols, CV_32FC1);
        cv::Mat img;
        img_orig.convertTo(img, CV_32FC1);
        std::cout << "orig: " << img_orig.size() << " predict: " << predict_img.size() << " img: " << img.size() << std::endl;
        cv::Mat_<float> sample(3, 1, CV_32FC1);
        for(int y = 0; y< img.rows; y++){
                for(int x = 0; x< img.cols; x++){
                        cv::Vec3f pixel = img.at<cv::Vec3f>(y,x);
                        sample(0,0) = pixel[0];
                        sample(1,0) = pixel[1];
                        sample(2,0) = pixel[2];
                        predict_img.at<float>(y,x) = this->predict(sample);
                }
        }
        std::cout << "created mat" << std::endl;
        return predict_img;
}


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

        part1__2(img,mask_fg,mask_bg);

        std::cout <<                                                                                                   std::endl;
        std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
        std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
        std::cout << "////    END    /////////////////////////////////////////////////////////////////////////////" << std::endl;
        std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
        std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
        std::cout <<                                                                                                   std::endl;

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

        int nmix=10;

        GMM_custom gmm_fg;
        gmm_fg.init(nmix,img,mask_fg,10);
        gmm_fg.learnGMM();
        cv::Mat fg=gmm_fg.return_posterior(img);

        GMM_custom gmm_bg;
        gmm_bg.init(nmix,img,mask_bg,10);
        gmm_bg.learnGMM();
        cv::Mat bg=gmm_bg.return_posterior(img);

        cv::Mat show=bg+fg;
        cv::divide(fg,show,show);
        show.convertTo(show,CV_8U,255);
        cv::imshow("gmm_custom",show);
        cv::waitKey(0);

        cv::destroyAllWindows();
}

