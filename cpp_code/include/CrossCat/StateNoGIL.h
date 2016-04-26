#ifndef GUARD_stateNoGIL_h
#define GUARD_stateNoGIL_h

#include <set>
#include <vector>
#include "View.h"
#include "utils.h"
#include "constants.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>   // for log()
#include <limits>
#include "Matrix.h"

#include "Python.h"
#include "State.h"

// Wraps CrossCat State methods with logic releasing the Python GIL in cases
// where significant computation could be done. See State.h for descriptions of
// wrapped functions.

// Only functions referenced by State.pyx have been wrapped

class StateNoGIL {
 public:
    StateNoGIL(const MatrixD& data,
               const std::vector<std::string>& GLOBAL_COL_DATATYPES,
               const std::vector<int>& GLOBAL_COL_MULTINOMIAL_COUNTS,
               const std::vector<int>& global_row_indices,
               const std::vector<int>& global_col_indices,
               const std::map<int, CM_Hypers>& HYPERS_M,
               const std::vector<std::vector<int> >& column_partition,
               const std::map<int, std::set<int> >& col_ensure_dep,
               const std::map<int, std::set<int> >& col_ensure_ind,
               double COLUMN_CRP_ALPHA,
               const std::vector<std::vector<std::vector<int> > >& row_partition_v,
               const std::vector<double>& row_crp_alpha_v,
               const std::vector<double>& ROW_CRP_ALPHA_GRID = empty_vector_double,
               const std::vector<double>& COLUMN_CRP_ALPHA_GRID = empty_vector_double,
               const std::vector<double>& S_GRID = empty_vector_double,
               const std::vector<double>& MU_GRID = empty_vector_double,
               int N_GRID=31, int SEED=0, int CT_KERNEL=0);
    StateNoGIL(const MatrixD& data,
               const std::vector<std::string>& GLOBAL_COL_DATATYPES,
               const std::vector<int>& GLOBAL_COL_MULTINOMIAL_COUNTS,
               const std::vector<int>& global_row_indices,
               const std::vector<int>& global_col_indices,
               const std::string& col_initialization = FROM_THE_PRIOR,
               std::string row_initialization = "",
               const std::vector<double>& ROW_CRP_ALPHA_GRID = empty_vector_double,
               const std::vector<double>& COLUMN_CRP_ALPHA_GRID = empty_vector_double,
               const std::vector<double>& S_GRID = empty_vector_double,
               const std::vector<double>& MU_GRID = empty_vector_double,
               int N_GRID=31, int SEED=0, int CT_KERNEL=0);
    ~StateNoGIL();
    int get_num_views() const;
    double insert_row(const std::vector<double>& row_data, int matching_row_idx,
                      int row_idx=-1);
    double get_column_crp_alpha() const;
    double get_column_crp_score() const;
    double transition_features(const MatrixD& data,
                               std::vector<int> which_features);
     double get_data_score() const;
    double get_marginal_logp() const;
    std::map<std::string, double> get_row_partition_model_hypers_i(int view_idx ) const;
    std::vector<int> get_row_partition_model_counts_i(int view_idx) const;
    std::vector<std::vector<std::map<std::string, double> > >
        get_column_component_suffstats_i(int view_idx) const;
    std::vector<CM_Hypers> get_column_hypers() const;
    std::map<std::string, double> get_column_partition_hypers() const;
    std::vector<int> get_column_partition_assignments() const;
    std::vector<int> get_column_partition_counts() const;
    std::map<int, std::set<int> > get_column_dependencies() const;
    std::map<int, std::set<int> > get_column_independencies() const;
    std::vector<std::vector<int> > get_X_D() const;
    std::vector<double> get_draw(int row_idx, int random_seed) const;
    std::map<int, std::vector<int> > get_column_groups() const;
    double transition_view_i(int which_view, const MatrixD& data);
    double transition_views(const MatrixD& data);
    double transition_row_partition_assignments(const MatrixD& data,
                                                std::vector<int> which_rows);
    double transition_views_zs(const MatrixD& data);
    double transition_views_row_partition_hyper();
    double transition_row_partition_hyperparameters(const std::vector<int>& which_cols);
    double transition_column_hyperparameters(std::vector<int> which_cols);
    double transition_views_col_hypers();
    double calc_row_predictive_logp(const std::vector<double>& in_vd);
    double transition_column_crp_alpha();
    double transition(const MatrixD& data);
    double draw_rand_u();
    double draw_rand_i();
    std::string to_string(const std::string& join_str = "\n",
                          bool top_level = false) const;
    // XXX: These must be initialized in cython. They need to be class-level,
    // therefore "static", but to be useful they can't be "const".
    static bool release_GIL;
 private:
    State *state;
    static void start_thread();
    static void end_thread();
};

#endif  // GUARD_stateNoGIL_h
