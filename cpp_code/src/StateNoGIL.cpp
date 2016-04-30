#include "StateNoGIL.h"

using namespace std;

// RAII-style GIL release. Release and acquisition logic taken from the
// Py_{BEGIN,END}_ALLOW_THREADS macros:
// https://github.com/python/cpython/blob/ee9c03765/Include/ceval.h#L187
class ReleaseGIL {
public:
    ReleaseGIL(bool release_GIL) {
        _release_GIL = release_GIL;
        if (_release_GIL) {
            _save = PyEval_SaveThread();
        }
    };
    ~ReleaseGIL() {
        if (_release_GIL) {
            PyEval_RestoreThread(_save);
        }
    };
private:
    bool _release_GIL;
    PyThreadState *_save;
};

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
    ReleaseGIL _r = ReleaseGIL(release_GIL);
    state = new State::State(data,
                             GLOBAL_COL_DATATYPES, GLOBAL_COL_MULTINOMIAL_COUNTS,
                             global_row_indices, global_col_indices, HYPERS_M,
                             column_partition ,col_ensure_dep, col_ensure_ind,
                             COLUMN_CRP_ALPHA, row_partition_v, row_crp_alpha_v,
                             ROW_CRP_ALPHA_GRID, COLUMN_CRP_ALPHA_GRID,
                             S_GRID, MU_GRID, N_GRID, SEED, CT_KERNEL);
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
    ReleaseGIL _r = ReleaseGIL(release_GIL);
    state = new State::State(data, GLOBAL_COL_DATATYPES,
                             GLOBAL_COL_MULTINOMIAL_COUNTS, global_row_indices,
                             global_col_indices, col_initialization,
                             row_initialization, ROW_CRP_ALPHA_GRID,
                             COLUMN_CRP_ALPHA_GRID, S_GRID, MU_GRID, N_GRID,
                             SEED, CT_KERNEL);
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
    ReleaseGIL _r = ReleaseGIL(release_GIL);
    return state->insert_row(row_data, matching_row_idx, row_idx);
}

double StateNoGIL::get_column_crp_alpha() const {
    return state->get_column_crp_alpha();
}

double StateNoGIL::get_column_crp_score() const {
    return state->get_column_crp_score();
}

double StateNoGIL::transition_features(const MatrixD &data,
                                       vector<int> which_features) {
    ReleaseGIL _r = ReleaseGIL(release_GIL);
    return state->transition_features(data, which_features);
}

double StateNoGIL::get_data_score() const {
    ReleaseGIL _r = ReleaseGIL(release_GIL);
    return state->get_data_score();
}

double StateNoGIL::get_marginal_logp() const {
    ReleaseGIL _r = ReleaseGIL(release_GIL);
    return state->get_marginal_logp();
}

map<string, double>
    StateNoGIL::get_row_partition_model_hypers_i(int view_idx) const {
    ReleaseGIL _r = ReleaseGIL(release_GIL);
    return state->get_row_partition_model_hypers_i(view_idx);
}

vector<int>
    StateNoGIL::get_row_partition_model_counts_i(int view_idx) const {
    ReleaseGIL _r = ReleaseGIL(release_GIL);
    return state->get_row_partition_model_counts_i(view_idx);
}

vector<vector<map<string, double> > >
    StateNoGIL::get_column_component_suffstats_i(int view_idx) const {
    ReleaseGIL _r = ReleaseGIL(release_GIL);
    return state->get_column_component_suffstats_i(view_idx);
}

vector<CM_Hypers> StateNoGIL::get_column_hypers() const {
    ReleaseGIL _r = ReleaseGIL(release_GIL);
    return state->get_column_hypers();
}

map<string, double> StateNoGIL::get_column_partition_hypers() const {
    ReleaseGIL _r = ReleaseGIL(release_GIL);
    return state->get_column_partition_hypers();
}

vector<int> StateNoGIL::get_column_partition_assignments() const {
    ReleaseGIL _r = ReleaseGIL(release_GIL);
    return state->get_column_partition_assignments();
}

vector<int> StateNoGIL::get_column_partition_counts() const {
    ReleaseGIL _r = ReleaseGIL(release_GIL);
    return state->get_column_partition_counts();
}

std::map<int, std::set<int> > StateNoGIL::get_column_dependencies() const {
    return state->get_column_dependencies();
}

std::map<int, std::set<int> > StateNoGIL::get_column_independencies() const {
    return state->get_column_independencies();
}

vector<vector<int> > StateNoGIL::get_X_D() const {
    ReleaseGIL _r = ReleaseGIL(release_GIL);
    return state->get_X_D();
}

vector<double> StateNoGIL::get_draw(int row_idx, int random_seed) const {
    ReleaseGIL _r = ReleaseGIL(release_GIL);
    return state->get_draw(row_idx, random_seed);
}

map<int, vector<int> > StateNoGIL::get_column_groups() const {
    ReleaseGIL _r = ReleaseGIL(release_GIL);
    return state->get_column_groups();
}

double StateNoGIL::transition_view_i(int which_view, const MatrixD& data) {
    ReleaseGIL _r = ReleaseGIL(release_GIL);
    return state->transition_view_i(which_view, data);
}

double StateNoGIL::transition_views(const MatrixD& data) {
    ReleaseGIL _r = ReleaseGIL(release_GIL);
    return state->transition_views(data);
}

double StateNoGIL::transition_row_partition_assignments(const MatrixD& data,
                                                   vector<int> which_rows) {
    ReleaseGIL _r = ReleaseGIL(release_GIL);
    return state->transition_row_partition_assignments(data, which_rows);
}

double StateNoGIL::transition_views_zs(const MatrixD& data) {
    ReleaseGIL _r = ReleaseGIL(release_GIL);
    return state->transition_views_zs(data);
}

double StateNoGIL::transition_views_row_partition_hyper() {
    ReleaseGIL _r = ReleaseGIL(release_GIL);
    return state->transition_views_row_partition_hyper();
}

double StateNoGIL::transition_row_partition_hyperparameters(const vector<int>&
                                                       which_cols) {
    ReleaseGIL _r = ReleaseGIL(release_GIL);
    return state->transition_row_partition_hyperparameters(which_cols);
}

double StateNoGIL::transition_column_hyperparameters(vector<int> which_cols) {
    ReleaseGIL _r = ReleaseGIL(release_GIL);
    return state->transition_column_hyperparameters(which_cols);
}

double StateNoGIL::transition_views_col_hypers() {
    ReleaseGIL _r = ReleaseGIL(release_GIL);
    return state->transition_views_col_hypers();
}

double StateNoGIL::calc_row_predictive_logp(const vector<double>& in_vd) {
    ReleaseGIL _r = ReleaseGIL(release_GIL);
    return state->calc_row_predictive_logp(in_vd);
}

double StateNoGIL::transition_column_crp_alpha() {
    ReleaseGIL _r = ReleaseGIL(release_GIL);
    return state->transition_column_crp_alpha();
}

double StateNoGIL::transition(const MatrixD& data) {
    ReleaseGIL _r = ReleaseGIL(release_GIL);
    return state->transition(data);
}

double StateNoGIL::draw_rand_u() {
    ReleaseGIL _r = ReleaseGIL(release_GIL);
    return state->draw_rand_u();
}

double StateNoGIL::draw_rand_i() {
    ReleaseGIL _r = ReleaseGIL(release_GIL);
    return state->draw_rand_i();
}

string StateNoGIL::to_string(const string& join_str, bool top_level) const {
    return state->to_string(join_str, top_level);
}
