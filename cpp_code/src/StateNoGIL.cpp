#include "StateNoGIL.h"

using namespace std;

////////////////////////////////////////////////////////////////////////
// GIL/thread logic

static void StateNoGIL::start_thread() {
    if release_GIL {
            Py_BEGIN_ALLOW_THREADS;
        }
}

static void StateNoGIL::end_thread() {
    if release_GIL {
            Py_END_ALLOW_THREADS;
        }
}

////////////////////////////////////////////////////////////////////////
// Method wraps

StateNoGIL::StateNoGIL(const MatrixD& data,
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
           const std::vector<double>& ROW_CRP_ALPHA_GRID,
           const std::vector<double>& COLUMN_CRP_ALPHA_GRID,
           const std::vector<double>& S_GRID,
           const std::vector<double>& MU_GRID,
           int N_GRID, int SEED, int CT_KERNEL) {
    start_thread();
    state = new State::State(data,
                             GLOBAL_COL_DATATYPES, GLOBAL_COL_MULTINOMIAL_COUNTS,
                             global_row_indices, global_col_indices, HYPERS_M,
                             column_partition ,col_ensure_dep, col_ensure_ind,
                             COLUMN_CRP_ALPHA, row_partition_v, row_crp_alpha_v,
                             ROW_CRP_ALPHA_GRID, COLUMN_CRP_ALPHA_GRID,
                             S_GRID, MU_GRID, N_GRID, SEED, CT_KERNEL);
    end_thread();
}

StateNoGIL::StateNoGIL(const MatrixD& data,
           const std::vector<std::string>& GLOBAL_COL_DATATYPES,
           const std::vector<int>& GLOBAL_COL_MULTINOMIAL_COUNTS,
           const std::vector<int>& global_row_indices,
           const std::vector<int>& global_col_indices,
           const std::string& col_initialization,
           std::string row_initialization,
           const std::vector<double>& ROW_CRP_ALPHA_GRID,
           const std::vector<double>& COLUMN_CRP_ALPHA_GRID,
           const std::vector<double>& S_GRID,
           const std::vector<double>& MU_GRID,
           int N_GRID, int SEED, int CT_KERNEL) {
    start_thread();
    state = new State::State(data, GLOBAL_COL_DATATYPES,
                             GLOBAL_COL_MULTINOMIAL_COUNTS, global_row_indices,
                             global_col_indices, col_initialization,
                             row_initialization, ROW_CRP_ALPHA_GRID,
                             COLUMN_CRP_ALPHA_GRID, S_GRID, MU_GRID, N_GRID,
                             SEED, CT_KERNEL);
    end_thread();
}

StateNoGIL::~StateNoGIL() {
    delete state;
}

int StateNoGIL::get_num_views() const {
    return state->get_num_views();
}

double StateNoGIL::insert_row(const vector<double>& row_data,
                              int matching_row_idx,
                              int row_idx) {
    double rv;
    start_thread();
    rv = state->insert_row(row_data, matching_row_idx, row_idx);
    end_thread();
    return rv;
}

double StateNoGIL::get_column_crp_alpha() const {
    return state->get_column_crp_alpha();
}

double StateNoGIL::get_column_crp_score() const {
    return state->get_column_crp_score();
}

double StateNoGIL::transition_features(const MatrixD &data,
                                  vector<int> which_features) {
    double rv;
    start_thread();
    rv = state->transition_features(data, which_features);
    end_thread();
    return rv;
}

double StateNoGIL::get_data_score() const {
    double rv;
    start_thread();
    rv = state->get_data_score();
    end_thread();
    return rv;
}

double StateNoGIL::get_marginal_logp() const {
    double rv;
    start_thread();
    rv = state->get_marginal_logp();
    end_thread();
    return rv;
}

map<string, double>
    StateNoGIL::get_row_partition_model_hypers_i(int view_idx) const {
    map<string, double> rv;
    start_thread();
    rv = state->get_row_partition_model_hypers_i(view_idx);
    end_thread();
    return rv;
}

vector<int>
    StateNoGIL::get_row_partition_model_counts_i(int view_idx) const {
    vector<int> rv;
    start_thread();
    rv = state->get_row_partition_model_counts_i(view_idx);
    end_thread();
    return rv;
}

vector<vector<map<string, double> > >
    StateNoGIL::get_column_component_suffstats_i(
    int view_idx) const {
    vector<vector<map<string, double> > > rv;
    start_thread();
    rv = state->get_column_component_suffstats_i(view_idx);
    end_thread();
    return rv;
}

vector<CM_Hypers> StateNoGIL::get_column_hypers() const {
    vector<CM_Hypers> rv;
    start_thread();
    rv = state->get_column_hypers();
    end_thread();
    return rv;
}

map<string, double> StateNoGIL::get_column_partition_hypers() const {
    map<string, double> rv;
    start_thread();
    rv = state->get_column_partition_hypers();
    end_thread();
    return rv;
}

vector<int> StateNoGIL::get_column_partition_assignments() const {
    vector<int> rv;
    start_thread();
    rv = state->get_column_partition_assignments();
    end_thread();
    return rv;
}

vector<int> StateNoGIL::get_column_partition_counts() const {
    vector<int> rv;
    start_thread();
    rv = state->get_column_partition_counts();
    end_thread();
    return rv;
}

std::map<int, std::set<int> > StateNoGIL::get_column_dependencies() const {
    return state->get_column_dependencies();
}

std::map<int, std::set<int> > StateNoGIL::get_column_independencies() const {
    return state->get_column_independencies();
}

vector<vector<int> > StateNoGIL::get_X_D() const {
    vector<vector<int> > rv;
    start_thread();
    rv = state->get_X_D();
    end_thread();
    return rv;
}

vector<double> StateNoGIL::get_draw(int row_idx, int random_seed) const {
    vector<double> rv;
    start_thread();
    rv = state->get_draw(row_idx, random_seed);
    end_thread();
    return rv;
}

map<int, vector<int> > StateNoGIL::get_column_groups() const {
    map<int, vector<int> > rv;
    start_thread();
    rv = state->get_column_groups();
    end_thread();
    return rv;
}

double StateNoGIL::transition_view_i(int which_view, const MatrixD& data) {
    double rv;
    start_thread();
    rv = state->transition_view_i(which_view, data);
    end_thread();
    return rv;
}

double StateNoGIL::transition_views(const MatrixD& data) {
    double rv;
    start_thread();
    rv = state->transition_views(data);
    end_thread();
    return rv;
}

double StateNoGIL::transition_row_partition_assignments(const MatrixD& data,
                                                   vector<int> which_rows) {
    double rv;
    start_thread();
    rv = state->transition_row_partition_assignments(data, which_rows);
    end_thread();
    return rv;
}

double StateNoGIL::transition_views_zs(const MatrixD& data) {
    double rv;
    start_thread();
    rv = state->transition_views_zs(data);
    end_thread();
    return rv;
}

double StateNoGIL::transition_views_row_partition_hyper() {
    double rv;
    start_thread();
    rv = state->transition_views_row_partition_hyper();
    end_thread();
    return rv;
}

double StateNoGIL::transition_row_partition_hyperparameters(const vector<int>&
                                                       which_cols) {
    double rv;
    start_thread();
    rv = state->transition_row_partition_hyperparameters(which_cols);
    end_thread();
    return rv;
}

double StateNoGIL::transition_column_hyperparameters(vector<int> which_cols) {
    double rv;
    start_thread();
    rv = state->transition_column_hyperparameters(which_cols);
    end_thread();
    return rv;
}

double StateNoGIL::transition_views_col_hypers() {
    double rv;
    start_thread();
    rv = state->transition_views_col_hypers();
    end_thread();
    return rv;
}

double StateNoGIL::calc_row_predictive_logp(const vector<double>& in_vd) {
    double rv;
    start_thread();
    rv = state->calc_row_predictive_logp(in_vd);
    end_thread();
    return rv;
}

double StateNoGIL::transition_column_crp_alpha() {
    double rv;
    start_thread();
    rv = state->transition_column_crp_alpha();
    end_thread();
    return rv;
}

double StateNoGIL::transition(const MatrixD& data) {
    double rv;
    start_thread();
    rv = state->transition(data);
    end_thread();
    return rv;
}

double StateNoGIL::draw_rand_u() {
    double rv;
    start_thread();
    rv = state->draw_rand_u();
    end_thread();
    return rv;
}

double StateNoGIL::draw_rand_i() {
    double rv;
    start_thread();
    rv = state->draw_rand_i();
    end_thread();
    return rv;
}

string StateNoGIL::to_string(const string& join_str, bool top_level) const {
    return state->to_string(join_str, top_level);
}
