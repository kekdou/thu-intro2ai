#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "Point.h"
#include "Strategy.h"

namespace {

constexpr int kEmpty = 0;
constexpr int kUserPiece = 1;
constexpr int kSelfPiece = 2;
constexpr int kDraw = 0;

struct StrategyConfig {
	int time_limit_ms = 2700;
	int max_iterations = 220000;
	double uct_c = 1.38;
	int rollout_depth = 96;
	double center_bias = 0.18;
	int root_candidate_cap = 9;
	int rollout_candidate_cap = 7;
};

struct MoveResult {
	bool ok = false;
	int row = -1;
	int col = -1;
	int winner = -1;
	bool is_tie = false;
};

struct BoardState {
	int rows = 0;
	int cols = 0;
	std::vector<int> board_flat;
	std::vector<int> top;
	int no_x = -1;
	int no_y = -1;
	int current_player = kSelfPiece;
	int last_x = -1;
	int last_y = -1;

	BoardState() = default;
	BoardState(int m, int n, const int *top_ptr, const int *board_ptr, int nox, int noy, int player, int lx, int ly)
		: rows(m), cols(n), board_flat(m * n), top(top_ptr, top_ptr + n), no_x(nox), no_y(noy), current_player(player), last_x(lx), last_y(ly) {
		for (int i = 0; i < m * n; ++i) board_flat[i] = board_ptr[i];
	}

	inline int index(int x, int y) const { return x * cols + y; }
	inline int cell(int x, int y) const { return board_flat[index(x, y)]; }
	inline void set_cell(int x, int y, int v) { board_flat[index(x, y)] = v; }

	bool in_bounds(int x, int y) const {
		return x >= 0 && x < rows && y >= 0 && y < cols;
	}

	bool is_tie() const {
		for (int c = 0; c < cols; ++c) {
			if (top[c] > 0) return false;
		}
		return true;
	}

	int winner_from_last_move(int x, int y) const {
		if (!in_bounds(x, y)) return -1;
		int piece = cell(x, y);
		if (piece == kEmpty) return -1;
		static const int dirs[4][2] = {{1, 0}, {0, 1}, {1, 1}, {1, -1}};
		for (int d = 0; d < 4; ++d) {
			int cnt = 1;
			for (int sign = -1; sign <= 1; sign += 2) {
				int dx = dirs[d][0] * sign;
				int dy = dirs[d][1] * sign;
				int nx = x + dx;
				int ny = y + dy;
				while (in_bounds(nx, ny) && cell(nx, ny) == piece) {
					++cnt;
					nx += dx;
					ny += dy;
				}
			}
			if (cnt >= 4) return piece;
		}
		return -1;
	}
};

struct MctsNode {
	MctsNode *parent = nullptr;
	std::vector<std::unique_ptr<MctsNode> > children;
	std::vector<int> untried_columns;
	BoardState state;
	int move_column = -1;
	int visits = 0;
	double value_sum = 0.0;

	MctsNode(MctsNode *p, const BoardState &s, const std::vector<int> &untried, int move)
		: parent(p), untried_columns(untried), state(s), move_column(move) {}
};

struct TreeCache {
	bool valid = false;
	std::unique_ptr<MctsNode> root;
};

int infer_current_player(const int *board_flat, int total_cells) {
	int self_cnt = 0;
	int user_cnt = 0;
	for (int i = 0; i < total_cells; ++i) {
		if (board_flat[i] == kSelfPiece) ++self_cnt;
		else if (board_flat[i] == kUserPiece) ++user_cnt;
	}
	return (self_cnt <= user_cnt) ? kSelfPiece : kUserPiece;
}

std::vector<int> build_center_order(int col_count) {
	std::vector<int> order;
	order.reserve(col_count);
	int center = (col_count - 1) / 2;
	order.push_back(center);
	for (int d = 1; static_cast<int>(order.size()) < col_count; ++d) {
		if (center - d >= 0) order.push_back(center - d);
		if (center + d < col_count) order.push_back(center + d);
	}
	return order;
}

std::vector<int> collect_legal_columns(const BoardState &board, const std::vector<int> &order) {
	std::vector<int> cols;
	cols.reserve(board.cols);
	for (int c : order) {
		if (c >= 0 && c < board.cols && board.top[c] > 0) cols.push_back(c);
	}
	return cols;
}

bool apply_column_move(BoardState &board, int column, MoveResult &out) {
	out = MoveResult();
	if (column < 0 || column >= board.cols || board.top[column] <= 0) return false;
	int row = board.top[column] - 1;
	int piece = board.current_player;
	board.set_cell(row, column, piece);
	board.top[column] = row;
	if (board.top[column] == board.no_x && column == board.no_y) --board.top[column];
	board.last_x = row;
	board.last_y = column;
	out.ok = true;
	out.row = row;
	out.col = column;
	out.winner = board.winner_from_last_move(row, column);
	out.is_tie = (out.winner == -1 && board.is_tie());
	board.current_player = (board.current_player == kSelfPiece) ? kUserPiece : kSelfPiece;
	return true;
}

bool detect_terminal(const BoardState &board, int &winner) {
	if (board.last_x != -1 && board.last_y != -1) {
		winner = board.winner_from_last_move(board.last_x, board.last_y);
		if (winner != -1) return true;
	}
	if (board.is_tie()) {
		winner = kDraw;
		return true;
	}
	winner = -1;
	return false;
}

bool find_immediate_winning_column(const BoardState &state, int player, const std::vector<int> &order, int &out_col) {
	for (int c : order) {
		if (state.top[c] <= 0) continue;
		BoardState probe = state;
		probe.current_player = player;
		MoveResult mv;
		if (!apply_column_move(probe, c, mv)) continue;
		if (mv.winner == player) {
			out_col = c;
			return true;
		}
	}
	return false;
}

bool is_safe_move_against_immediate_loss(const BoardState &state, int column, const std::vector<int> &order) {
	if (column < 0 || column >= state.cols || state.top[column] <= 0) return false;
	BoardState next = state;
	MoveResult mv;
	if (!apply_column_move(next, column, mv)) return false;
	if (mv.winner != -1 || mv.is_tie) return true;
	int opp = next.current_player;
	int opp_win_col = -1;
	return !find_immediate_winning_column(next, opp, order, opp_win_col);
}

std::vector<int> build_ranked_candidates(const BoardState &state, const std::vector<int> &order, int cap) {
	std::vector<int> legal = collect_legal_columns(state, order);
	if (legal.empty()) return legal;
	int win_col = -1;
	if (find_immediate_winning_column(state, state.current_player, order, win_col)) return std::vector<int>(1, win_col);

	std::vector<int> safe;
	safe.reserve(legal.size());
	for (int c : legal) {
		if (is_safe_move_against_immediate_loss(state, c, order)) safe.push_back(c);
	}
	std::vector<int> &src = safe.empty() ? legal : safe;
	if (cap > 0 && static_cast<int>(src.size()) > cap) src.resize(static_cast<size_t>(cap));
	return src;
}

bool same_position(const BoardState &a, const BoardState &b) {
	return a.rows == b.rows && a.cols == b.cols && a.no_x == b.no_x && a.no_y == b.no_y && a.current_player == b.current_player &&
		a.top == b.top && a.board_flat == b.board_flat;
}

bool promote_root_by_move(TreeCache &cache, int move_col) {
	if (!cache.valid || !cache.root) return false;
	for (size_t i = 0; i < cache.root->children.size(); ++i) {
		if (cache.root->children[i] && cache.root->children[i]->move_column == move_col) {
			std::unique_ptr<MctsNode> next = std::move(cache.root->children[i]);
			next->parent = nullptr;
			cache.root = std::move(next);
			return true;
		}
	}
	return false;
}

void rebuild_or_advance_cache(TreeCache &cache, const BoardState &current, int last_x, int last_y, const std::vector<int> &order, const StrategyConfig &cfg) {
	if (cache.valid && cache.root && last_x != -1 && last_y != -1) {
		if (!promote_root_by_move(cache, last_y)) {
			cache.valid = false;
			cache.root.reset();
		}
	}
	if (cache.valid && cache.root && !same_position(cache.root->state, current)) {
		cache.valid = false;
		cache.root.reset();
	}
	if (!cache.valid || !cache.root) {
		std::vector<int> root_moves = build_ranked_candidates(current, order, cfg.root_candidate_cap);
		cache.root.reset(new MctsNode(nullptr, current, root_moves, -1));
		cache.valid = true;
	}
}

class MctsEngine {
public:
	MctsEngine(const StrategyConfig &cfg, std::mt19937_64 &rng, const std::vector<int> &order)
		: cfg_(cfg), rng_(rng), center_order_(order) {}

	int search(TreeCache &cache) {
		if (!cache.valid || !cache.root) return -1;
		MctsNode *root = cache.root.get();
		root_player_ = root->state.current_player;
		auto start = std::chrono::steady_clock::now();

		for (int it = 0; it < cfg_.max_iterations; ++it) {
			auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
				std::chrono::steady_clock::now() - start).count();
			if (elapsed_ms >= cfg_.time_limit_ms) break;

			MctsNode *node = root;
			BoardState sim = root->state;
			int winner = -1;

			while (!detect_terminal(sim, winner) && node->untried_columns.empty() && !node->children.empty()) {
				node = select_child_uct(node);
				sim = node->state;
			}

			if (!detect_terminal(sim, winner) && !node->untried_columns.empty()) {
				MctsNode *expanded = expand_one(node);
				if (expanded) {
					node = expanded;
					sim = node->state;
				}
			}

			double value = rollout(sim);
			backpropagate(node, value);
		}
		return select_final_column(*root);
	}

private:
	const StrategyConfig &cfg_;
	std::mt19937_64 &rng_;
	const std::vector<int> &center_order_;
	int root_player_ = kSelfPiece;

	MctsNode *select_child_uct(MctsNode *node) {
		MctsNode *best = nullptr;
		double best_score = -std::numeric_limits<double>::infinity();
		for (size_t i = 0; i < node->children.size(); ++i) {
			MctsNode *child = node->children[i].get();
			double score = std::numeric_limits<double>::infinity();
			if (child->visits > 0) {
				double avg = child->value_sum / child->visits;
				double exploit = (node->state.current_player == root_player_) ? avg : (1.0 - avg);
				double explore = cfg_.uct_c * std::sqrt(std::log(std::max(1, node->visits)) / child->visits);
				score = exploit + explore;
			}
			if (score > best_score) {
				best_score = score;
				best = child;
			}
		}
		return best;
	}

	MctsNode *expand_one(MctsNode *node) {
		int col = node->untried_columns.back();
		node->untried_columns.pop_back();
		BoardState child_state = node->state;
		MoveResult mv;
		if (!apply_column_move(child_state, col, mv)) return nullptr;
		std::vector<int> child_moves = build_ranked_candidates(child_state, center_order_, cfg_.rollout_candidate_cap);
		std::unique_ptr<MctsNode> child(new MctsNode(node, child_state, child_moves, col));
		MctsNode *raw = child.get();
		node->children.push_back(std::move(child));
		return raw;
	}

	int rollout_policy(const BoardState &state) {
		int win_col = -1;
		if (find_immediate_winning_column(state, state.current_player, center_order_, win_col)) return win_col;
		int opp = (state.current_player == kSelfPiece) ? kUserPiece : kSelfPiece;
		if (find_immediate_winning_column(state, opp, center_order_, win_col)) return win_col;
		std::vector<int> cands = build_ranked_candidates(state, center_order_, cfg_.rollout_candidate_cap);
		if (cands.empty()) return -1;
		std::vector<double> weights;
		weights.reserve(cands.size());
		double center = (state.cols - 1) * 0.5;
		for (int c : cands) {
			double dist = std::abs(c - center);
			double norm = (center > 0.0) ? (dist / center) : 0.0;
			weights.push_back(1.0 + cfg_.center_bias * (1.0 - norm));
		}
		std::discrete_distribution<int> pick(weights.begin(), weights.end());
		return cands[pick(rng_)];
	}

	double evaluate_static(const BoardState &state) {
		int center = (state.cols - 1) / 2;
		double score = 0.5;
		for (int r = 0; r < state.rows; ++r) {
			int v = state.cell(r, center);
			if (v == root_player_) score += 0.01;
			else if (v != kEmpty) score -= 0.01;
		}
		if (score < 0.0) score = 0.0;
		if (score > 1.0) score = 1.0;
		return score;
	}

	double rollout(BoardState state) {
		int winner = -1;
		if (detect_terminal(state, winner)) {
			if (winner == kDraw) return 0.5;
			return winner == root_player_ ? 1.0 : 0.0;
		}
		for (int d = 0; d < cfg_.rollout_depth; ++d) {
			int col = rollout_policy(state);
			if (col < 0) return 0.5;
			MoveResult mv;
			if (!apply_column_move(state, col, mv)) return 0.0;
			if (mv.winner != -1) return mv.winner == root_player_ ? 1.0 : 0.0;
			if (mv.is_tie) return 0.5;
		}
		return evaluate_static(state);
	}

	void backpropagate(MctsNode *node, double value) {
		while (node) {
			node->visits += 1;
			node->value_sum += value;
			node = node->parent;
		}
	}

	int select_final_column(const MctsNode &root) {
		if (root.children.empty()) return -1;
		int best_col = -1;
		int best_visits = -1;
		double best_avg = -1.0;
		double center = (root.state.cols - 1) * 0.5;
		for (size_t i = 0; i < root.children.size(); ++i) {
			const MctsNode *child = root.children[i].get();
			if (child->visits <= 0) continue;
			double avg = child->value_sum / child->visits;
			bool better = false;
			if (child->visits > best_visits) better = true;
			else if (child->visits == best_visits && avg > best_avg + 1e-12) better = true;
			else if (child->visits == best_visits && std::abs(avg - best_avg) <= 1e-12 && best_col != -1) {
				double cur_dist = std::abs(child->move_column - center);
				double best_dist = std::abs(best_col - center);
				if (cur_dist < best_dist) better = true;
			}
			if (better) {
				best_col = child->move_column;
				best_visits = child->visits;
				best_avg = avg;
			}
		}
		return best_col;
	}
};

int choose_fallback_column(const BoardState &board, const std::vector<int> &preferred, const std::vector<int> &order) {
	for (int c : preferred) {
		if (c >= 0 && c < board.cols && board.top[c] > 0) return c;
	}
	for (int c : order) {
		if (c >= 0 && c < board.cols && board.top[c] > 0) return c;
	}
	for (int c = 0; c < board.cols; ++c) {
		if (board.top[c] > 0) return c;
	}
	return -1;
}

TreeCache &cache_instance() {
	static TreeCache cache;
	return cache;
}

} // namespace

extern "C" Point *getPoint(const int M, const int N, const int *top, const int *_board,
						   const int lastX, const int lastY, const int noX, const int noY)
{
	/*
		不要更改这段代码
	*/
	int x = -1, y = -1;
	int **board = new int *[M];
	for (int i = 0; i < M; i++) {
		board[i] = new int[N];
		for (int j = 0; j < N; j++) {
			board[i][j] = _board[i * N + j];
		}
	}

	/*
		根据你自己的策略来返回落子点,也就是根据你的策略完成对x,y的赋值
		该部分对参数使用没有限制，为了方便实现，你可以定义自己新的类、.h文件、.cpp文件
	*/
	StrategyConfig config;
	std::vector<int> center_order = build_center_order(N);
	int current_player = infer_current_player(_board, M * N);
	BoardState state(M, N, top, _board, noX, noY, current_player, lastX, lastY);

	TreeCache &cache = cache_instance();
	rebuild_or_advance_cache(cache, state, lastX, lastY, center_order, config);

	int selected_col = -1;
	int tactical_col = -1;
	if (find_immediate_winning_column(state, state.current_player, center_order, tactical_col)) {
		selected_col = tactical_col;
	} else {
		int opp = (state.current_player == kSelfPiece) ? kUserPiece : kSelfPiece;
		if (find_immediate_winning_column(state, opp, center_order, tactical_col)) {
			selected_col = tactical_col;
		}
	}

	if (selected_col == -1) {
		std::mt19937_64 rng(static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
		MctsEngine engine(config, rng, center_order);
		selected_col = engine.search(cache);
	}

	std::vector<int> preferred;
	preferred.push_back(selected_col);
	int best_col = choose_fallback_column(state, preferred, center_order);
	if (best_col != -1) {
		y = best_col;
		x = top[best_col] - 1;
		promote_root_by_move(cache, best_col);
	} else {
		cache.valid = false;
		cache.root.reset();
	}

	/*
		不要更改这段代码
	*/
	clearArray(M, N, board);
	return new Point(x, y);
}

/*
	getPoint函数返回的Point指针是在本so模块中声明的，为避免产生堆错误，应在外部调用本so中的
	函数来释放空间，而不应该在外部直接delete
*/
extern "C" void clearPoint(Point *p)
{
	delete p;
	return;
}

/*
	清除top和board数组
*/
void clearArray(int M, int N, int **board)
{
	(void)N;
	for (int i = 0; i < M; i++) {
		delete[] board[i];
	}
	delete[] board;
}

/*
	添加你自己的辅助函数，你可以声明自己的类、函数，添加新的.h .cpp文件来辅助实现你的想法
*/
