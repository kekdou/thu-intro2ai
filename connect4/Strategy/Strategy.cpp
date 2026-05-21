#include <iostream>
#include <chrono>
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <cmath>

#include "Point.h"
#include "Strategy.h"

const int MAX_M = 12;
const int MAX_N = 12;

const int USER = 1;
const int SELF = 2;

const int MAX_PATH = 160;
const int MAX_NODE_POOL = 450000;
const double UCT_C = 0.95;

const int TIME_LIMIT_MS = 2850;

using namespace std;

struct FastRng {
    uint64_t s;
    FastRng() {
        const uint64_t t = static_cast<uint64_t>(chrono::high_resolution_clock::now().time_since_epoch().count());
        s = 0x9e3779b97f4a7c15ULL ^ (t + 0x243f6a8885a308d3ULL);
    }
    inline int nextInt(int n) {
        /*返回 n 以内的一个随机数*/
        s ^= s << 7;
        s ^= s >> 9;
        if (n <= 1) return 0;
        return static_cast<int>(static_cast<uint32_t>(s) % static_cast<uint32_t>(n));
    }
};

struct State {
    /*完整棋局状态*/
    int M, N;
    int noX, noY;
    uint8_t board[MAX_M][MAX_N];
    int8_t top[MAX_N];
    uint8_t toMove;    // 当前轮到谁下
};

struct Node {
    /*MCST 中的一个节点*/
    uint64_t key;    // 当前局面的哈希值，用来在搜索图里快速复用节点
    int child_col[MAX_N];
    int child_idx[MAX_N];
    uint8_t child_count;
    uint8_t untried[MAX_N];
    uint8_t untried_count;
    uint8_t player;
    uint8_t terminal;
    uint8_t winner;
    int visits;
    float value;    // win: +1, tie: +0.5, lost: +0
};

struct Graph {
    vector<Node> nodes;
    unordered_map<uint64_t, int> key_to_idx;
    FastRng rng;
    int root;
    Graph(): root(-1) {
        nodes.reserve(MAX_NODE_POOL);
        key_to_idx.reserve(MAX_NODE_POOL * 2);
    }
    void clear() {
        nodes.clear();
        key_to_idx.clear();
        root = -1;
    }
};

struct SearchCache {
    Graph graph;
    State root_state;
};

SearchCache g_cache;

uint64_t g_zobrist[MAX_M][MAX_N][2];
uint64_t g_misc[64];
bool g_hash_init = false;

inline uint64_t splitMix64(uint64_t x) {
    /*将连续整数变成 64 位随机数*/
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

inline void initHash() {
    if (g_hash_init) return;
    uint64_t seed = 0x123456789abcdef0ULL;
    for (int i = 0; i < MAX_M; i++) {
        for (int j = 0; j < MAX_N; j++) {
            g_zobrist[i][j][0] = splitMix64(seed++);
            g_zobrist[i][j][1] = splitMix64(seed++);
        }
    }
    for (int i = 0; i < 64; i++) g_misc[i] = splitMix64(seed++);
    g_hash_init = true;
}

inline uint64_t hashState(State& st) {
    /*将 State 转成哈希值，用于判断是否搜索过*/
    uint64_t h = g_misc[0];    // 融入 State 中的所有信息
    h ^= g_misc[(st.M & 63)];
    h ^= g_misc[(st.N & 63)];
    h ^= g_misc[(st.noX + 17) & 63];
    h ^= g_misc[(st.noY + 29) & 63];
    h ^= g_misc[(st.toMove == SELF) ? 43 : 11];
    for (int c = 0; c < st.N; c++) {
        h ^= splitMix64(static_cast<uint64_t>(st.top[c] + 37 * c + 101));
    }
    for (int i = 0; i < st.M; i++) {
        for (int j = 0; j < st.N; j++) {
            int p = st.board[i][j];
            if (p == USER) h ^= g_zobrist[i][j][0];
            else if (p == SELF) h ^= g_zobrist[i][j][1];
        }
    }
    return h;
}

inline bool hasAnyMove(State& st) {
    /*检查是否有空余位置落子*/
    for (int c = 0; c < st.N; c++) {
        if (st.top[c] > 0) return 1;
    }
    return 0;
}

inline bool checkWinAt(State& st, int x, int y, int piece) {
    /*判断在 x, y 落子是否构成连成 4*/
    static int dx[4] = {0, 1, 1, 1};
    static int dy[4] = {1, 0, 1, -1};
    for (int d = 0; d < 4; d++) {
        int count = 1;
        int tx = x + dx[d];
        int ty = y + dy[d];
        while (tx >= 0 && tx < st.M && ty >= 0 && ty < st.N && st.board[tx][ty] == piece) {
            count++;
            tx += dx[d];
            ty += dy[d];
        }
        tx = x - dx[d];
        ty = y - dy[d];
        while (tx >= 0 && tx < st.M && ty >= 0 && ty < st.N && st.board[tx][ty] == piece) {
            count++;
            tx -= dx[d];
            ty -= dy[d];
        }
        if (count >= 4) return 1;
    }
    return 0;
}

inline int turnPlayer(int p) {
    return (p == SELF) ? USER : SELF;
}

inline int applyMove(State& st, int col) {
    /*在 col 列落子，轮换回合*/
    int x = st.top[col] - 1;
    st.board[x][col] = st.toMove;
    st.top[col]--;
    if (x == st.noX + 1 && col == st.noY) {    // 外部已经处理过，但是保险先加上
        st.top[col]--;
    }
    st.toMove = static_cast<uint8_t>(turnPlayer(st.toMove));
    return x;
}

inline int orderedLegalCols(State& st, int out_cols[MAX_N]) {
    /*按靠近中间排序合法列，并返回数量*/
    int count = 0;
    int center_l = (st.N - 1) / 2;
    int center_r = st.N / 2;
    for (int d = 0; d <= st.N / 2; d++) {
        int c1 = center_l - d;
        if (c1 >= 0 && st.top[c1] > 0) out_cols[count++] = c1;
        int c2 = center_r + d;
        if (c2 != c1 && c2 < st.N && st.top[c2] > 0) out_cols[count++] = c2;
    }
    return count;
}

inline int collectWinningMoves(State& st, int piece, int out_cols[MAX_N]) {
    /*找出 ordered 序列中必胜的列，返回数量*/
    int legal[MAX_N];
    int count = orderedLegalCols(st, legal);
    int win_count = 0;
    for (int i = 0; i < count; i++) {
        int col = legal[i];
        int x = st.top[col] - 1;
        st.board[x][col] = static_cast<uint8_t>(piece);
        bool win = checkWinAt(st, x, col, piece);
        st.board[x][col] = 0;
        if (win) out_cols[win_count++] = legal[i];
    }
    return win_count;
}

inline int createNode(Graph& g, State& st, bool terminal, int winner, uint64_t key) {
    /*创建一个新节点，返回其在图中的 idx*/
    if (static_cast<int>(g.nodes.size()) >= MAX_NODE_POOL) return -1;
    Node new_node;
    new_node.key = key;
    new_node.child_count = 0;
    new_node.untried_count = 0;
    new_node.player = st.toMove;
    new_node.terminal = terminal ? 1 : 0;
    new_node.winner = static_cast<uint8_t>(winner);
    new_node.visits = 0;
    new_node.value = 0.0f;
    if (!terminal) {
        int cols[MAX_N];
        int count = orderedLegalCols(st, cols);
        for (int i = 0; i < count; i++) new_node.untried[new_node.untried_count++] = static_cast<uint8_t>(cols[i]);
    }
    int idx = static_cast<int>(g.nodes.size());
    g.nodes.push_back(new_node);
    g.key_to_idx[key] = idx;
    return idx;
}

inline void resetCacheToState(State& root_state) {
    /*清零 cache，并将 root 设置为 root_state*/
    g_cache.graph.clear();
    uint64_t key = hashState(root_state);
    g_cache.graph.root = createNode(g_cache.graph, root_state, 0, 0, key);
    g_cache.root_state = root_state;
}

inline bool equalState(State& a, State& b) {
    /*判断两个状态是否相同*/
    if (a.M != b.M || a.N != b.N || a.noX != b.noX || a.noY != b.noY || a.toMove != b.toMove) return 0;
    for (int c = 0; c < a.N; c++) {
        if (a.top[c] != b.top[c]) return 0;
    }
    for (int i = 0; i < a.M; i++) {
        for (int j = 0; j < a.N; j++) {
            if (a.board[i][j] != b.board[i][j]) return 0;
        }
    }
    return 1;
}

inline int copyNodeRec(Graph& src, Graph& dst, unordered_map<int, int>& remap, int old_idx) {
    /*递归拷贝 old_idx 的子图*/
    unordered_map<int, int>::iterator it = remap.find(old_idx);
    if (it != remap.end()) return it->second;
    int new_idx = static_cast<int>(dst.nodes.size());
    remap[old_idx] = new_idx;
    dst.nodes.push_back(src.nodes[old_idx]);
    Node& new_node = dst.nodes.back();
    dst.key_to_idx[new_node.key] = new_idx;
    Node& old_node = src.nodes[old_idx];
    for (int i = 0; i < old_node.child_count; i++) {
        int old_child = old_node.child_idx[i];
        int new_child = copyNodeRec(src, dst, remap, old_child);
        new_node.child_idx[i] = new_child;
    }
    return new_idx;
}

inline Graph extractSubgraph(Graph& src, int src_root) {
    /*提取旧图中以 src_root 为根的子图*/
    Graph dst;
    dst.clear();
    dst.nodes.reserve(min(MAX_NODE_POOL, static_cast<int>(src.nodes.size())));
    dst.key_to_idx.reserve(min(MAX_NODE_POOL * 2, static_cast<int>(src.key_to_idx.size() * 2 + 1)));
    unordered_map<int, int> remap;
    remap.reserve(src.nodes.size() / 2 + 1);
    dst.root = copyNodeRec(src, dst, remap, src_root);
    return dst;
}

inline void prepareCacheRoot(State& root_state) {
    /*将 root_state 作为根节点初始化 cache 或者设为根节点，cache 过大提取子图*/
    if (g_cache.graph.root < 0) {
        resetCacheToState(root_state);
        return;
    }
    if (equalState(g_cache.root_state, root_state)) return;
    uint64_t key = hashState(root_state);
    unordered_map<uint64_t, int>::iterator it = g_cache.graph.key_to_idx.find(key);
    if (it == g_cache.graph.key_to_idx.end()) {
        resetCacheToState(root_state);
        return;
    }
    int new_root = it->second;
    if (static_cast<int>(g_cache.graph.nodes.size()) > (MAX_NODE_POOL * 7) / 10) {
        g_cache.graph = extractSubgraph(g_cache.graph, new_root);
    } else {
        g_cache.graph.root = new_root;
    }
    g_cache.root_state = root_state;
}

inline int getOrCreateNode(Graph& g, State& st, bool terminal, int winner) {
    /*通过哈希值判断节点是否存在，存在直接复用，不存在则创建*/
    uint64_t key = hashState(st);
    unordered_map<uint64_t, int>::iterator it = g.key_to_idx.find(key);
    if (it != g.key_to_idx.end()) {
        Node& node = g.nodes[it->second];
        if (terminal && !node.terminal) {
            node.terminal = 1;
            node.winner = static_cast<uint8_t>(winner);
            node.untried_count = 0;
        }
        return it->second;
    }
    return createNode(g, st, terminal, winner, key);
}

inline int findChildByCol(Node& node, int col) {
    for (int i = 0; i < node.child_count; i++) {
        if (node.child_col[i] == col) return i;
    }
    return -1;
}

inline int rollout(State& st, FastRng& rng) {
    /*重复落子，直到决出胜负或平局，返回赢家*/
    while (1) {
        if (!hasAnyMove(st)) return 0;
        int col = -1;
        int cur = st.toMove;
        int opp = turnPlayer(cur);
        int wins[MAX_N];
        int win_count = collectWinningMoves(st, cur, wins);
        int threats[MAX_N];
        int threat_count = collectWinningMoves(st, opp, threats);
        int legal[MAX_N];
        int legal_count = orderedLegalCols(st, legal);
        if (win_count > 0) col = wins[rng.nextInt(win_count)];
        else if (threat_count > 0) col = threats[rng.nextInt(threat_count)];
        else if (legal_count > 0) col = legal[rng.nextInt(min(4, legal_count))];
        if (col < 0) return 0;
        int x = applyMove(st, col);
        if (checkWinAt(st, x, col, cur)) return cur;
    }
}

inline double scoreForPlayer(uint8_t player, int winner) {
    if (winner == 0) return 0.5;
    return (winner == player) ? 1.0 : 0.0;
}

inline int pickRootMoveFallback(State& st) {
    int cols[MAX_N];
    int count = orderedLegalCols(st, cols);
    return (count > 0) ? cols[0] : 0;
};

inline bool moveAllowOppWin(State& st, int col, int self_player) {
    /*判断是否为死手*/
    int x = applyMove(st, col);
    bool self_win = checkWinAt(st, x, col, self_player);
    bool opp_win = 0;
    if (!self_win) {
        int wins[MAX_N];
        opp_win = (collectWinningMoves(st, st.toMove, wins) > 0);
    }
    st.toMove = static_cast<uint8_t>(self_player);
    if (x == st.noX + 1 && col == st.noY) {
        st.top[col]++;
    }
    st.top[col]++;
    st.board[x][col] = 0;
    return (!self_win && opp_win);
}

inline int chooseBestMove(Graph& g, State& root_state) {
    /*从根节点选择下一步，首选返回制胜，然后比较访问次数和胜率，全为死手返回访问次数最多*/
    Node& root = g.nodes[g.root];
    int best_col = -1, safe_col = -1;
    int best_visit = -1, safe_visit = -1;
    double best_val = -1.0, safe_val = -1.0;
    State tmp = root_state;
    for (int i = 0; i < root.child_count; i++) {
        int child_idx = root.child_idx[i];
        Node& child = g.nodes[child_idx];
        int col = root.child_col[i];
        if (child.terminal && child.winner == SELF) return col;
        int v = child.visits;
        double mean = (v > 0) ? (child.value / static_cast<double>(v)) : 0.5;
        double root_val = 1.0 - mean;
        if (v > best_visit || (v == best_visit && root_val > best_val)) {
            best_visit = v;
            best_val = root_val;
            best_col = col;
        }
        if (!moveAllowOppWin(tmp, col, SELF)) {
            if (v > safe_visit || (v == safe_visit && root_val > safe_val)) {
                safe_visit = v;
                safe_val = root_val;
                safe_col = col;
            }
        }
    }
    if (safe_col >= 0) return safe_col;
    if (best_col >= 0) return best_col;
    return pickRootMoveFallback(root_state);
}

inline int runMCTS(Graph& g, State& root_state, chrono::steady_clock::time_point& deadline) {
    /*蒙特卡洛搜索，选择，拓展，模拟，回传*/
    int path[MAX_PATH];
    int iterations = 0;   // 每循环 16 次检查一次时间，减少 chrono 的资源
    while (static_cast<int>(g.nodes.size()) + 1 < MAX_NODE_POOL) {
        if ((iterations & 15) == 0 && chrono::steady_clock::now() >= deadline) break;
        iterations++;
        State st = root_state;
        int cur = g.root;
        int path_len = 0;
        path[path_len++] = cur;
        while (1) {
            Node& node = g.nodes[cur];
            if (node.terminal) break;
            if (node.untried_count > 0) {
                int pick = g.rng.nextInt(node.untried_count);
                int col = node.untried[pick];
                node.untried_count--;
                node.untried[pick] = node.untried[node.untried_count];
                int player = st.toMove;
                int row = applyMove(st, col);
                bool win = checkWinAt(st, row, col, player);
                bool tie = (!win && !hasAnyMove(st));
                int child_idx = getOrCreateNode(g, st, win || tie, win ? player : 0);
                if (findChildByCol(node, col) < 0 && node.child_count < MAX_N) {
                    node.child_col[node.child_count] = col;
                    node.child_idx[node.child_count] = child_idx;
                    node.child_count++;
                }
                cur = child_idx;
                path[path_len++] = cur;
                break;
            }
            double log_n = log(static_cast<double>(node.visits + 1));
            int best_edge = -1;
            double best_score = -1e100;
            for (int i = 0; i < node.child_count; i++) {
                int child_idx = node.child_idx[i];
                Node& child = g.nodes[child_idx];
                double s = 0.0;
                if (child.visits == 0) {
                    s = 1e50;
                } else {
                    double mean = child.value / static_cast<double>(child.visits);
                    double exploit = 1.0 - mean;
                    double explore = UCT_C * sqrt(log_n / static_cast<double>(child.visits));
                    s = exploit + explore;
                }
                if (s > best_score) {
                    best_score = s;
                    best_edge = i;
                }
            }
            int col = node.child_col[best_edge];
            applyMove(st, col);
            cur = node.child_idx[best_edge];
            path[path_len++] = cur;
            if (path_len >= MAX_PATH) break;
        }
        Node& leaf = g.nodes[cur];
        int winner = leaf.terminal ? leaf.winner : rollout(st, g.rng);
        for (int i = 0; i < path_len; i++) {
            Node& n = g.nodes[path[i]];
            n.visits++;
            n.value += static_cast<float>(scoreForPlayer(n.player, winner));
        }
    }
    return chooseBestMove(g, root_state);
}

inline void removeUntriedMove(Node& node, int col) {
    for (int i = 00; i < node.untried_count; i++) {
        if (node.untried[i] == col) {
            node.untried_count--;
            node.untried[i] = node.untried[node.untried_count];
            return;
        }
    }
}

inline void ensureRootChild(Graph& g, State root_state, int col) {
    /*保证该步落子在搜索图的根节点有对应子节点*/
    Node& root = g.nodes[g.root];
    if (findChildByCol(root, col) >= 0) {
        removeUntriedMove(root, col);
        return;
    }
    State st = root_state;
    int player = st.toMove;
    int x = applyMove(st, col);
    bool win = checkWinAt(st, x, col, player);
    bool tie = (!win && !hasAnyMove(st));
    int child_idx = getOrCreateNode(g, st, win || tie, win ? player : 0);
    root.child_col[root.child_count] = col;
    root.child_idx[root.child_count] = child_idx;
    root.child_count++;
    removeUntriedMove(root, col);
}

extern "C" Point *getPoint(const int M, const int N, const int *top, const int *_board,
						   const int lastX, const int lastY, const int noX, const int noY)
{
    auto t0 = chrono::steady_clock::now();
    auto deadline = t0 + chrono::milliseconds(TIME_LIMIT_MS);
    initHash();
    State root_state;
    root_state.M = M;
    root_state.N = N;
    root_state.noX = noX;
    root_state.noY = noY;
    root_state.toMove = SELF;
	for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            root_state.board[i][j] = static_cast<uint8_t>(_board[i * N + j]);
        }
    }
    for (int c = 0; c < N; c++) {
        root_state.top[c] = static_cast<int8_t>(top[c]);
    }
    prepareCacheRoot(root_state);
    int wins[MAX_N];
    State tmp = root_state;
    int y = -1;
    int win_count = collectWinningMoves(tmp, SELF, wins);
    if (win_count > 0) {
        y = wins[0];
    } else {
        tmp = root_state;
        int threats[MAX_N];
        int threat_count = collectWinningMoves(tmp, USER, threats);
        if (threat_count == 1) {    // 只有一个拦截点就立马拦截
            y = threats[0];
        } else {
            y = runMCTS(g_cache.graph, root_state, deadline);
        }
    }
    ensureRootChild(g_cache.graph, root_state, y);
    g_cache.root_state = root_state;
    return new Point(top[y] - 1, y);
}

extern "C" void clearPoint(Point *p)
{
	delete p;
	return;
}
