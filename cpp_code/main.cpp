/*
*   Copyright (c) 2010-2013, MIT Probabilistic Computing Project
*
*   Lead Developers: Dan Lovell and Jay Baxter
*   Authors: Dan Lovell, Baxter Eaves, Jay Baxter, Vikash Mansinghka
*   Research Leads: Vikash Mansinghka, Patrick Shafto
*
*   Licensed under the Apache License, Version 2.0 (the "License");
*   you may not use this file except in compliance with the License.
*   You may obtain a copy of the License at
*
*       http://www.apache.org/licenses/LICENSE-2.0
*
*   Unless required by applicable law or agreed to in writing, software
*   distributed under the License is distributed on an "AS IS" BASIS,
*   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*   See the License for the specific language governing permissions and
*   limitations under the License.
*/
#include <iostream>
//
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/program_options.hpp>
//
#include "State.h"
#include "DateTime.h"

typedef boost::numeric::ublas::matrix<double> matrixD;

using namespace std;

int main(int argc, char** argv) {
    cout << endl << "Hello World: begin main" << endl;

    std::time_t default_seed = 0;
    string default_data_filename = "SynData2.csv";
    string default_samples_filename = "samples.out";
    string col_initialization = "from_the_prior";
    string row_initialization = "from_the_prior";
    int default_nChains = 1;
    int default_nSamples = 4;
    int default_burnIn = 5;
    int default_lag = 2;
    int default_nGrid = 31;
    //
    std::time_t seed;
    string data_filename;
    string samples_filename;
    int nChains;
    int nSamples;
    int burnIn;
    int lag;
    int nGrid;

    //parse some arguments
    namespace po = boost::program_options;
    po::options_description desc("Options");
    desc.add_options()
    ("help,h", "produce help message")
    ("seed", po::value<std::time_t>(&seed)->default_value(default_seed),
     "set inference seed")
    ("data_filename", po::value<string>(&data_filename)->default_value(
         default_data_filename), "data to run inference on")
    ("samples_filename", po::value<string>(&samples_filename)->default_value(
         default_samples_filename), "filename to save samples in")
    ("nChains", po::value<int>(&nChains)->default_value(default_nChains),
     "set number of inference chains to run")
    ("nSamples", po::value<int>(&nSamples)->default_value(default_nSamples),
     "set number of samples to draw (@lag)")
    ("burnIn", po::value<int>(&burnIn)->default_value(default_burnIn),
     "set number of burn in iterations")
    ("lag", po::value<int>(&lag)->default_value(default_lag),
     "set number of iterations per sample")
    ("nGrid", po::value<int>(&nGrid)->default_value(default_nGrid),
     "set number of bins in hyper inference")
    ;
    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);
        if (vm.count("help")) {
            std::cout << desc << "\n";
            exit(0);
        }
    } catch (const boost::program_options::error& e) {
        std::cerr << e.what() << std::endl;
    }

    matrixD data;
    LoadData(data_filename, data);
    vector<int> global_row_indices = create_sequence(data.size1());
    vector<int> global_column_indices = create_sequence(data.size2());
    int n_cols = data.size2();
    vector<string> GLOBAL_COL_DATATYPES(n_cols, "normal_inverse_gamma");
    vector<int> GLOBAL_COL_MULTINOMIAL_COUNTS(n_cols, 0);

    ofstream out(samples_filename.c_str(), ios_base::app);
    if (!out) {
        cout << "Cannot open file: " << samples_filename << endl;
        return -1;
    }
    out << "nChains = " << nChains << endl;
    out << "nSamples = " << nSamples << endl;
    out << "burnIn = " << burnIn << endl;
    out << "lag = " << lag << endl;
    out << endl;
    out.close();

    int raw_sample_idx = 0;
    for (int nc = 0; nc < nChains; nc++) {
        cout << "nc = " << nc << endl;
        int SEED = nc;
        // need to randomize initialization of State
        State s = State(data, GLOBAL_COL_DATATYPES, GLOBAL_COL_MULTINOMIAL_COUNTS,
                        global_row_indices, global_column_indices, col_initialization,
                        row_initialization, nGrid, SEED);
        Timer T(true);
        for (int nb = 0; nb < burnIn; nb++) {
            s.transition(data);
            cout << "Done raw sample idx: " << raw_sample_idx++ << endl;
        }

        //s.SaveResult(samples_filename, 0);
        cout << " t1 = " << T.GetElapsed() << endl;
        //
        for (int ns = 1; ns < nSamples; ns++) {
            cout << "ns = " << ns << endl;
            for (int nl = 0; nl < lag; nl++) {
                s.transition(data);
                cout << "Done raw sample idx: " << raw_sample_idx++ << endl;
            }
            //s.SaveResult(samples_filename, ns);
            cout << "t2 = " <<  T.GetElapsed() << endl;
        }
    }
    cout << endl << "Goodbye World: end main" << endl;
}
