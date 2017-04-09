#include "feedforward.h"

#define NGYUEN_WIDROW 0.7

std::vector<Eigen::MatrixXd>Feedforward::get_weights()const{return this->weights;}
std::vector<Eigen::VectorXd>Feedforward::get_deltas()const{return this->deltas;}
std::vector<Eigen::VectorXd>Feedforward::get_out()const{return this->out;}
std::vector<Eigen::VectorXd>Feedforward::get_bias()const{return this->bias;}
float Feedforward::get_learning_rate()const{return this->learning_rate;}
void Feedforward::set_learning_rate(float& new_rate){this->learning_rate=new_rate;}

Feedforward::Feedforward(std::vector<int>& net_structure){
    //Create weights matrices following NguyenWidrow
    std::vector<float> beta_value;
    Eigen::VectorXd filler=Eigen::VectorXd::Zero(1);
    Eigen::MatrixXd filler2=Eigen::MatrixXd::Zero(1,1);
    this->deltas.push_back(filler);
    this->bias.push_back(filler);
    this->weights.push_back(filler2);
    for(unsigned int i=1;i<net_structure.size();i++){
        //beta_value=(0.7*pow(num_hidden,(1.0/num_inputs)))
        beta_value.push_back(NGYUEN_WIDROW*pow(net_structure[i],(1.0/net_structure[0])));
        //Initialize OUT vectors
        Eigen::VectorXd out_i(net_structure[i-1]); // We subtract 1 because the input layer has output
        this->out.push_back(out_i);
        //Initialize delta and bias vectors
        Eigen::VectorXd delta_and_bias_i=Eigen::VectorXd::Zero(net_structure[i]);
        this->deltas.push_back(delta_and_bias_i);
        this->bias.push_back(delta_and_bias_i);
    }
    Eigen::VectorXd results(net_structure[net_structure.size()-1]);
    this->out.push_back(results);
    //weights=weights/weights.norm()*beta_value
    /*
    for(unsigned int i=1;i<net_structure.size();i++){
        // Initialize weight matrices with random variables
        Eigen::MatrixXd weight_i=Eigen::MatrixXd::Random(net_structure[i],net_structure[i-1]); // We add +1 column for the bias
        for(int j=0;j<weight_i.rows();j++){
            //weights=weights/weights.norm()*beta_value
            weight_i.row(j)=weight_i.row(j)/((weight_i.row(j)).norm()*beta_value[i]);
        }
        this->weights.push_back(weight_i);
    }
    */
    for(unsigned int i=1;i<net_structure.size();i++){
        Eigen::MatrixXd weight_i=Eigen::MatrixXd::Random(net_structure[i],net_structure[i-1]); // We add +1 column for the bias
        this->weights.push_back(weight_i);
    }
}

void Feedforward::forward_pass(Eigen::VectorXd& input){
    try{
        if(input.size()!=this->out[0].size()){
            throw "[ERROR] Input vector not equal to input layer size!!";
        }
    }catch(const char* msg){
        std::cerr<<msg<<std::endl;
    }
    this->out[0]=input;
    for(unsigned int i=1;i<this->out.size();i++){
        //std::cout<<"\nOUT "<<i<<" = "<<this->out[i]<<"\n";
        this->out[i]=((((this->weights[i]*this->out[i-1])+this->bias[i]).array()).tanh()).matrix();
        //std::cout<<"OUT "<<i<<" = "<<this->out[i]<<"\n";
        //std::cout<<"------------------------";
    }
}

void Feedforward::backpropagation(Eigen::VectorXd& target){
    try{
        if(target.size()!=this->out.back().size()){
            throw "[ERROR] Target vector and output vector have different sizes!!";
        }
    }
    catch(const char* msg){
        std::cerr<<msg<<std::endl;
    }
    // Calculate delta for last layer and update weights
    this->deltas.back()=((this->out.back()-target).array()*(1.0-pow(((this->weights.back()*this->out[this->out.size()-2]).array()).tanh(),2.0))).matrix();
    this->weights.back()=this->weights.back()-this->learning_rate*this->deltas.back()*(this->out[this->out.size()-2].transpose());
    this->bias.back()=this->bias.back()-this->learning_rate*this->deltas.back();
    // Start weight updates
    for(int i=this->out.size()-2;i>0;i--){
        // Calculate delta
        this->deltas[i]=((this->weights[i+1].transpose()*this->deltas[i+1]).array()*(1.0-pow(((this->weights[i]*this->out[i-1]).array()).tanh(),2.0))).matrix();
        // Update weight
        this->weights[i]=this->weights[i]-this->learning_rate*(this->deltas[i]*(this->out[i-1].transpose()));
        // Update bias
        this->bias[i]=this->bias[i]-this->learning_rate*this->deltas[i];
    }
}

float Feedforward::learn(Eigen::VectorXd& target, Eigen::VectorXd& input){
    this->forward_pass(input);
    this->backpropagation(target);
    return this->error(target);
}

float Feedforward::error(Eigen::VectorXd& target)const{
    return pow((this->out[this->out.size()-1]-target).norm(),2.0);
}

