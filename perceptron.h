#include <Eigen/Dense>
#include <vector>

class Perceptron {
protected:
    Eigen::Vector3d weights;
    bool shuffle=true;
    float eta=1.0;
    char type=0; //0 STEP; 1 TANH;
    virtual char activate(Eigen::Vector3d input) const;
public:
    Perceptron();
    Eigen::Vector3d get_weights() const;
    void set_weights(Eigen::Vector3d new_weights);
    void set_eta(float& new_eta);
    unsigned long int learn(std::vector<std::vector<float> >data); //data MUST be [x y class]
    Eigen::Vector2d get_line() const;
};

class Perceptron_tanh public Perceptron {
private:
    double activate(Eigen::Vectorxd x)const;
};

