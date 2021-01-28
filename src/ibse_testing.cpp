#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>


///////////////////////////////
// Reading data from text files
//
static
std::vector<std::string> split(const std::string& subject)
{
   std::istringstream ss{subject};
   using StrIt = std::istream_iterator<std::string>;
   std::vector<std::string> container{StrIt{ss}, StrIt{}};
   return container;
}


std::vector<Eigen::Vector3d>
readPosition(const std::string &fn)
{
   std::vector<Eigen::Vector3d> pos;

   std::string line;
   std::ifstream data_file(fn);

   if (data_file.is_open())
   {
      while (std::getline(data_file,line))
      {
         std::vector<std::string> strings = split(line);
         pos.push_back( Eigen::Vector3d( ::atof(strings[1].c_str()),
                                         ::atof(strings[2].c_str()),
                                         ::atof(strings[3].c_str()) ) );
      }
      data_file.close();
   }
   
   return pos;
}


std::vector<double>
readTime(const std::string &fn)
{
   std::vector<double> times;

   std::string line;
   std::ifstream data_file(fn);

   if (data_file.is_open())
   {
      while (std::getline(data_file,line))
      {
         std::vector<std::string> strings = split(line);
         times.push_back( 1e-9 * ::atof(strings[0].c_str()) );
      }
      data_file.close();
   }

   double start_time = times[0];
   for (auto& tt : times)
      tt = tt - start_time;
   
   return times;
}

