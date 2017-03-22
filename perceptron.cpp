#include "perceptron.h"
#include <math.h>       // tanh
#include <iostream>

Eigen::Vector3d Perceptron::get_weights() const{return this->weights;}

Eigen::Vector2d Perceptron::get_line() const{
    Eigen::Vector2d A(-this->weights(0)/this->weights(2),-this->weights(1)/this->weights(2)); //y = -w1/w3 -(w2/w3)*x
    return A;
}

void Perceptron::set_weights(Eigen::Vector3d new_weights){
    for(char i=0;i<3;i++){
        this->weights[i]=new_weights[i];
    }
}

Perceptron::Perceptron(){
    this->weights<<0,0,0;
}

void Perceptron::set_eta(float& new_eta){this->eta=new_eta;}

virtual char Perceptron::activate(Eigen::Vector3d input)const{
    if(this->weights.dot(input)>0){
        return 1;
    }
    else{
        return -1;
    }
}

unsigned long int Perceptron::learn(std::vector<std::vector<float> >data){
    unsigned long int epoch=0;
    unsigned int correct=0;
    unsigned long int i=0;
    bool done=false;
    Eigen::Vector3d input;
    char y; // 1 or -1
    while(!done){
        epoch++;
        input<<1,data[i][0],data[i][1];
        y=this->activate(input);
        if(!(y==data[i][2])){
            this->weights+=this->eta*data[i][2]*input; // If the answer is incorrect, update weights
            correct=0;
        }
        else{
            correct++;
        }
        i++;
        if(i==data.size())i=0;
        if(correct==data.size())done=true;
    }
    return epoch;
}
