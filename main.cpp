#include "perceptron.h"
#include "parser.cpp"
#include "feedforward.h"
#include <iostream>
#include <fstream>
#include <vector>
//DELETE SOME OF THESE (IF NO ALL)
#include <math.h>


int perceptron_test (){
    std::cout<<"Please input filename with learning data\n";
    std::string filename;
    std::cin>>filename;
    std::vector<std::vector<float> >database=parseCSV(filename,3);
    std::random_shuffle ( database.begin(), database.end() );
    Perceptron my_perceptron;
    int epoch=my_perceptron.learn(database);
    std::cout<<"epoch: "<<epoch<<"\n";
    std::cout<<"function: "<<my_perceptron.get_line()(1)<<"x+"<<my_perceptron.get_line()(0)<<"\n";
    return 0;
}

int eigenTest(){
    Eigen::VectorXd A(3);
    std::cout<<"Vector A is off size: "<<A.size()<<"\n";
    A<<1,2,3,4;
    std::cout<<A;
    return 0;
}

void size_test(){
    std::vector<int>z;
    z.push_back(3);
    z.push_back(4);
    z.push_back(3);
    z.push_back(2);
    std::cout<<"FOR FFNN WITH STRUCTURE:\n";
    for(int i=0;i<z.size();i++){
        std::cout<<z[i];
    }
    Feedforward my_ff(z);
    for(int i=0;i<z.size();i++){
        std::cout<<"\nWEIGHT MATRIX "<<i<<" size: "<<my_ff.get_weights()[i].rows()<<"x"<<my_ff.get_weights()[i].cols();
        std::cout<<"\nBIAS VECTOR "<<i<<" size: "<<my_ff.get_bias()[i].size();
        std::cout<<"\nDELTA VECTOR "<<i<<" size: "<<my_ff.get_deltas()[i].size();
        std::cout<<"\nOUT VECTOR "<<i<<" size: "<<my_ff.get_out()[i].size();
    }
}


int main(){
    std::vector<int>z;
    z.push_back(2);
    z.push_back(5);
    z.push_back(1);
    Feedforward ff(z);
    std::vector<Eigen::VectorXd>inputVector;
    std::vector<Eigen::VectorXd>targetVector;
    Eigen::VectorXd out0(1);
    Eigen::VectorXd out1(1);
    out0(0)=0.0;
    out1(0)=1.0;
    Eigen::VectorXd in00(2);
    Eigen::VectorXd in01(2);
    Eigen::VectorXd in10(2);
    Eigen::VectorXd in11(2);
    in00<<0.0,0.0;
    in01<<0.0,1.0;
    in10<<1.0,0.0;
    in11<<1.0,1.0;
    inputVector.push_back(in00);
    inputVector.push_back(in01);
    inputVector.push_back(in10);
    inputVector.push_back(in11);
    targetVector.push_back(out0);
    targetVector.push_back(out1);
    targetVector.push_back(out1);
    targetVector.push_back(out0);
    int epoch=1;
    std::cout<<Eigen::VectorXd::Zero(z.size())<<"\n";
    while(epoch<10000){
        for(unsigned int i=1;i<4;i++){
            std::cout<<"\ni ="<<i<<"\n";
            std::cout<<"ERROR: "<<ff.learn(targetVector[i],inputVector[i])<<"\n";
            std::cout<<"EPOCH: "<<epoch++<<"\n";
            for(unsigned int j=0;j<z.size();j++){
                //std::cout<<"\nWEIGHTS:\n"<<ff.get_weights()[j];
                std::cout<<"\nBias:\n"<<ff.get_bias()[j]<<"\n";
                //std::cout<<"\nDeltas:\n"<<ff.get_deltas()[j]<<"\n";
                //std::cout<<"\nWeights:\n"<<ff.get_weights()[j]<<"\n";
                //std::cout<<"\nOut:\n"<<ff.get_out()[j]<<"\n";
            }
        }
    }

    for(unsigned int i=1;i<4;i++){
            std::cout<<"\n\ntesting...\n";
            ff.forward_pass(inputVector[i]);
            std::cout<<"input\n"<<inputVector[i]<<"\ngives: "<<ff.get_out().back();
            std::cout<<"\n when it should give: "<<targetVector[i]<<"\n";
    }
    for(int i=1;i<z.size();i++){
        std::cout<<"\nFor "<<i<<":\n";
        std::cout<<"\nBias:\n"<<ff.get_bias()[i]<<"\n";
        std::cout<<"\nDeltas:\n"<<ff.get_deltas()[i]<<"\n";
        std::cout<<"\nWeights:\n"<<ff.get_weights()[i]<<"\n";
        std::cout<<"\nOut:\n"<<ff.get_out()[i]<<"\n";
    }
    return 0;
}


