#include <vector>
#include <Eigen/Dense>
#include <iostream>

class Feedforward{
private:
    std::vector<Eigen::MatrixXd>weights;
    std::vector<Eigen::VectorXd>deltas;
    std::vector<Eigen::VectorXd>out;
    std::vector<Eigen::VectorXd>bias;
    float learning_rate;
public:
    std::vector<Eigen::MatrixXd>get_weights()const;
    std::vector<Eigen::VectorXd>get_deltas()const;
    std::vector<Eigen::VectorXd>get_out()const;
    std::vector<Eigen::VectorXd>get_bias()const;
    float get_learning_rate()const;
    void set_learning_rate(float& new_rate);
    Feedforward(std::vector<int>& net_structure);
    void forward_pass(Eigen::VectorXd& input);
    void backpropagation(Eigen::VectorXd& target);
    float error(Eigen::VectorXd& target)const;
    float learn(Eigen::VectorXd& target, Eigen::VectorXd& input);
};

