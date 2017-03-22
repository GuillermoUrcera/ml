#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <sstream>

inline std::vector<std::vector<float> > parseCSV(std::string filename,unsigned char num_items){
    class FileError{};
    std::ifstream input(filename.c_str());
    std::vector<std::vector<float> >database;
    std::vector<float> row;
    std::string element;
    std::stringstream currentLine;
    // Open file
    try{
        if(!input.is_open())throw(FileError());
    }
    catch(FileError){
        std::cout<<"[WARN]::Can't open file!\n";
    }
    while(std::getline(input,element)){ // Get a line
        currentLine<<element;
        while(std::getline(currentLine,element,',')){
            row.push_back(atof(element.c_str())); // Add elements (columns to row)
        }
        // Clear flags of currentLine
        currentLine.clear();
        currentLine.seekg(0, std::ios::beg);
        database.push_back(row); // Add rows to database
        row.clear(); //Flush the row
        currentLine.str(std::string()); // Flush the stream
    }
    // Close file
    input.close();
    return database;
}

