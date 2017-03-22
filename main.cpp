#include "perceptron.h"
#include "parser.cpp"
#include <iostream>
#include <fstream>
#include <vector>

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

int main(){
    perceptron_test();
    return 0;
}
