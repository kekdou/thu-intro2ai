#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <vector>

#include "Point.h"
#include "Strategy.h"

namespace {

constexpr int MAX_M = 12;
constexpr int MAX_N = 12;
constexpr int USER = 1;
constexpr int SELF = 2;
constexpr int MAX_PATH = 160;
constexpr int MAX_NODE_POOL = 450000;
constexpr double UCT_C = 0.95;
constexpr int THINK_TIME_MS = 2850;

struct FastRng {
    uint64_t s;

    FastRng() {
        const uint64_t t = static_cast<uint64_t>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count());
        s = 0x9e3779b97f4a7c15ULL ^ (t + 0x243f6a8885a308d3ULL);
    }

    inline uint32_t nextU32() {
        s ^= s << 7;
        s ^= s >> 9;
        return static_cast<uint32_t>(s);
    }

    inline int nextInt(int n) {
        if (n <= 1) return 0;
        return static_cast<int>(nextU32() % static_cast<uint32_t>(n));
    }
};

struct State {
    int M, N;
    int noX, noY;
    uint8_t board[MAX_M][MAX_N];
    int8_t top[MAX_N];
    uint8_t toMove;
};

struct Node {
    uint64_t key;
    int childCol[MAX_N];
    int childIdx[MAX_N];
    uint8_t childCount;
    uint8_t untried[MAX_N];
    uint8_t untriedCount;
    uint8_t playerToMove;
    uint8_t terminal;
    uint8_t terminalWinner;  // 0 draw, 1 user, 2 self
    int visits;
    float winSum;            // score from playerToMove perspective
};

struct Graph {
    std::vector<Node> nodes;
    std::unordered_map<uint64_t, int> keyToIdx;
    FastRng rng;
    int root;

    Graph() : root(-1) {
        nodes.reserve(MAX_NODE_POOL);
        keyToIdx.reserve(MAX_NODE_POOL * 2);
    }

    void clear() {
        nodes.clear();
        keyToIdx.clear();
        root = -1;
    }
};

struct SearchCache {
    Graph graph;
    State rootState;
    bool valid;

    SearchCache() : valid(false) {}
};

SearchCache g_cache;

uint64_t g_zobrist[MAX_M][MAX_N][2];
uint64_t g_misc[64];
bool g_hashInit = false;

inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

inline void initHash() {
    if (g_hashInit) return;
    uint64_t seed = 0x123456789abcdef0ULL;
    for (int i = 0; i < MAX_M; ++i) {
        for (int j = 0; j < MAX_N; ++j) {
            g_zobrist[i][j][0] = splitmix64(seed++);
            g_zobrist[i][j][1] = splitmix64(seed++);
        }
    }
    for (int i = 0; i < 64; ++i) g_misc[i] = splitmix64(seed++);
    g_hashInit = true;
}

inline uint64_t hashState(const State& st) {
    uint64_t h = g_misc[0];
    h ^= g_misc[(st.M & 63)];
    h ^= g_misc[(st.N & 63)];
    h ^= g_misc[(st.noX + 17) & 63];
    h ^= g_misc[(st.noY + 29) & 63];
    h ^= g_misc[(st.toMove == SELF) ? 43 : 11];
    for (int c = 0; c < st.N; ++c) {
        h ^= splitmix64(static_cast<uint64_t>(st.top[c] + 37 * c + 101));
    }
    for (int i = 0; i < st.M; ++i) {
        for (int j = 0; j < st.N; ++j) {
            const int p = st.board[i][j];
            if (p == USER) h ^= g_zobrist[i][j][0];
            else if (p == SELF) h ^= g_zobrist[i][j][1];
        }
    }
    return h;
}

inline bool equalState(const State& a, const State& b) {
    if (a.M != b.M || a.N != b.N || a.noX != b.noX || a.noY != b.noY || a.toMove != b.toMove) {
        return false;
    }
    for (int c = 0; c < a.N; ++c) {
        if (a.top[c] != b.top[c]) return false;
    }
    for (int i = 0; i < a.M; ++i) {
        for (int j = 0; j < a.N; ++j) {
            if (a.board[i][j] != b.board[i][j]) return false;
        }
    }
    return true;
}

inline bool hasAnyMove(const State& st) {
    for (int c = 0; c < st.N; ++c) {
        if (st.top[c] > 0) return true;
    }
    return false;
}

inline int playerOpponent(int p) {
    return (p == SELF) ? USER : SELF;
}

inline bool checkWinAt(const State& st, int x, int y, int piece) {
    static const int dx[4] = {0, 1, 1, 1};
    static const int dy[4] = {1, 0, 1, -1};
    for (int d = 0; d < 4; ++d) {
        int cnt = 1;
        int tx = x + dx[d], ty = y + dy[d];
        while (tx >= 0 && tx < st.M && ty >= 0 && ty < st.N && st.board[tx][ty] == piece) {
            ++cnt;
            tx += dx[d];
            ty += dy[d];
        }
        tx = x - dx[d];
        ty = y - dy[d];
        while (tx >= 0 && tx < st.M && ty >= 0 && ty < st.N && st.board[tx][ty] == piece) {
            ++cnt;
            tx -= dx[d];
            ty -= dy[d];
        }
        if (cnt >= 4) return true;
    }
    return false;
}

inline int applyMove(State& st, int col) {
    const int x = st.top[col] - 1;
    st.board[x][col] = st.toMove;
    --st.top[col];
    if (x == st.noX + 1 && col == st.noY) {
        --st.top[col];
    }
    st.toMove = static_cast<uint8_t>(playerOpponent(st.toMove));
    return x;
}

inline void undoMove(State& st, int col, int x, int piece) {
    st.toMove = static_cast<uint8_t>(piece);
    if (x == st.noX + 1 && col == st.noY) {
        ++st.top[col];
    }
    ++st.top[col];
    st.board[x][col] = 0;
}

inline int orderedLegalCols(const State& st, int outCols[MAX_N]) {
    int cnt = 0;
    const int centerL = (st.N - 1) / 2;
    const int centerR = st.N / 2;
    for (int d = 0; d < st.N; ++d) {
        const int c1 = centerL - d;
        if (c1 >= 0 && st.top[c1] > 0) outCols[cnt++] = c1;
        const int c2 = centerR + d;
        if (c2 != c1 && c2 < st.N && st.top[c2] > 0) outCols[cnt++] = c2;
    }
    return cnt;
}

inline bool isWinningMove(State& st, int col, int piece) {
    if (st.top[col] <= 0) return false;
    const int oldTop = st.top[col];
    const int x = oldTop - 1;
    st.board[x][col] = static_cast<uint8_t>(piece);
    --st.top[col];
    if (x == st.noX + 1 && col == st.noY) {
        --st.top[col];
    }
    const bool win = checkWinAt(st, x, col, piece);
    st.top[col] = static_cast<int8_t>(oldTop);
    st.board[x][col] = 0;
    return win;
}

inline int collectWinningMoves(State& st, int piece, int outCols[MAX_N]) {
    int legal[MAX_N];
    const int cnt = orderedLegalCols(st, legal);
    int wcnt = 0;
    for (int i = 0; i < cnt; ++i) {
        if (isWinningMove(st, legal[i], piece)) outCols[wcnt++] = legal[i];
    }
    return wcnt;
}

inline int pickPolicyMove(State& st, FastRng& rng) {
    const int cur = st.toMove;
    const int opp = playerOpponent(cur);

    int wins[MAX_N];
    const int wcnt = collectWinningMoves(st, cur, wins);
    if (wcnt > 0) return wins[rng.nextInt(wcnt)];

    int threats[MAX_N];
    const int tcnt = collectWinningMoves(st, opp, threats);
    if (tcnt > 0) return threats[rng.nextInt(tcnt)];

    int legal[MAX_N];
    const int lcnt = orderedLegalCols(st, legal);
    if (lcnt <= 0) return -1;
    const int k = std::min(4, lcnt);
    return legal[rng.nextInt(k)];
}

inline int rollout(State& st, FastRng& rng) {
    while (true) {
        if (!hasAnyMove(st)) return 0;
        const int cur = st.toMove;
        const int col = pickPolicyMove(st, rng);
        if (col < 0) return 0;
        const int x = applyMove(st, col);
        if (checkWinAt(st, x, col, cur)) return cur;
    }
}

inline double scoreForPlayer(uint8_t playerToMove, int winner) {
    if (winner == 0) return 0.5;
    return (winner == playerToMove) ? 1.0 : 0.0;
}

inline int createNode(Graph& g, const State& st, bool terminal, int winner, uint64_t key) {
    if (static_cast<int>(g.nodes.size()) >= MAX_NODE_POOL) return -1;

    Node node;
    node.key = key;
    node.childCount = 0;
    node.untriedCount = 0;
    node.playerToMove = st.toMove;
    node.terminal = terminal ? 1 : 0;
    node.terminalWinner = static_cast<uint8_t>(winner);
    node.visits = 0;
    node.winSum = 0.0f;

    if (!terminal) {
        int cols[MAX_N];
        const int cnt = orderedLegalCols(st, cols);
        for (int i = 0; i < cnt; ++i) node.untried[node.untriedCount++] = static_cast<uint8_t>(cols[i]);
    }

    const int idx = static_cast<int>(g.nodes.size());
    g.nodes.push_back(node);
    g.keyToIdx[key] = idx;
    return idx;
}

inline int getOrCreateNode(Graph& g, const State& st, bool terminal, int winner) {
    const uint64_t key = hashState(st);
    std::unordered_map<uint64_t, int>::iterator it = g.keyToIdx.find(key);
    if (it != g.keyToIdx.end()) {
        Node& n = g.nodes[it->second];
        if (terminal && !n.terminal) {
            n.terminal = 1;
            n.terminalWinner = static_cast<uint8_t>(winner);
            n.untriedCount = 0;
        }
        return it->second;
    }
    return createNode(g, st, terminal, winner, key);
}

inline void removeUntriedMove(Node& node, int col) {
    for (int i = 0; i < node.untriedCount; ++i) {
        if (node.untried[i] == col) {
            node.untried[i] = node.untried[node.untriedCount - 1];
            --node.untriedCount;
            return;
        }
    }
}

inline int findChildByCol(const Node& node, int col) {
    for (int i = 0; i < node.childCount; ++i) {
        if (node.childCol[i] == col) return i;
    }
    return -1;
}

inline int copyNodeRec(const Graph& src, Graph& dst, std::unordered_map<int, int>& remap, int oldIdx) {
    std::unordered_map<int, int>::iterator it = remap.find(oldIdx);
    if (it != remap.end()) return it->second;

    const int newIdx = static_cast<int>(dst.nodes.size());
    remap[oldIdx] = newIdx;
    dst.nodes.push_back(src.nodes[oldIdx]);
    Node& nn = dst.nodes.back();
    dst.keyToIdx[nn.key] = newIdx;

    const Node& old = src.nodes[oldIdx];
    for (int i = 0; i < old.childCount; ++i) {
        const int oldCh = old.childIdx[i];
        const int newCh = copyNodeRec(src, dst, remap, oldCh);
        nn.childIdx[i] = newCh;
    }
    return newIdx;
}

inline Graph extractSubgraph(const Graph& src, int srcRoot) {
    Graph dst;
    if (srcRoot < 0 || srcRoot >= static_cast<int>(src.nodes.size())) return dst;
    dst.clear();
    dst.nodes.reserve(std::min(MAX_NODE_POOL, static_cast<int>(src.nodes.size())));
    dst.keyToIdx.reserve(std::min(MAX_NODE_POOL * 2, static_cast<int>(src.keyToIdx.size() * 2 + 1)));

    std::unordered_map<int, int> remap;
    remap.reserve(src.nodes.size() / 2 + 1);
    dst.root = copyNodeRec(src, dst, remap, srcRoot);
    return dst;
}

inline void resetCacheToState(const State& rootState) {
    g_cache.graph.clear();
    const uint64_t key = hashState(rootState);
    g_cache.graph.root = createNode(g_cache.graph, rootState, false, 0, key);
    g_cache.rootState = rootState;
    g_cache.valid = (g_cache.graph.root >= 0);
}

inline void prepareCacheRoot(const State& rootState) {
    if (!g_cache.valid || g_cache.graph.root < 0) {
        resetCacheToState(rootState);
        return;
    }

    if (equalState(g_cache.rootState, rootState)) {
        return;
    }

    const uint64_t key = hashState(rootState);
    std::unordered_map<uint64_t, int>::iterator it = g_cache.graph.keyToIdx.find(key);
    if (it == g_cache.graph.keyToIdx.end()) {
        resetCacheToState(rootState);
        return;
    }
    const int newRoot = it->second;
    if (newRoot < 0 || newRoot >= static_cast<int>(g_cache.graph.nodes.size())) {
        resetCacheToState(rootState);
        return;
    }

    if (static_cast<int>(g_cache.graph.nodes.size()) > (MAX_NODE_POOL * 7) / 10) {
        Graph next = extractSubgraph(g_cache.graph, newRoot);
        if (next.root < 0) {
            resetCacheToState(rootState);
            return;
        }
        g_cache.graph = std::move(next);
    } else {
        g_cache.graph.root = newRoot;
    }
    g_cache.rootState = rootState;
    g_cache.valid = true;
}

inline bool moveAllowsImmediateOppWin(State& st, int col, int selfPlayer) {
    const int x = applyMove(st, col);
    const bool selfWin = checkWinAt(st, x, col, selfPlayer);
    bool oppHasWin = false;
    if (!selfWin) {
        int wins[MAX_N];
        oppHasWin = (collectWinningMoves(st, st.toMove, wins) > 0);
    }
    undoMove(st, col, x, selfPlayer);
    return (!selfWin && oppHasWin);
}

inline int pickRootMoveFallback(const State& st) {
    int cols[MAX_N];
    const int cnt = orderedLegalCols(st, cols);
    if (cnt > 0) return cols[0];
    for (int c = 0; c < st.N; ++c) {
        if (st.top[c] > 0) return c;
    }
    return 0;
}

inline int chooseBestRootMove(const Graph& g, const State& rootState) {
    if (g.root < 0 || g.root >= static_cast<int>(g.nodes.size())) return pickRootMoveFallback(rootState);
    const Node& root = g.nodes[g.root];
    if (root.childCount <= 0) return pickRootMoveFallback(rootState);

    int bestCol = -1;
    int bestVisit = -1;
    double bestVal = -1.0;

    int safeCol = -1;
    int safeVisit = -1;
    double safeVal = -1.0;

    for (int i = 0; i < root.childCount; ++i) {
        const int chIdx = root.childIdx[i];
        if (chIdx < 0 || chIdx >= static_cast<int>(g.nodes.size())) continue;
        const Node& ch = g.nodes[chIdx];
        if (ch.terminal && ch.terminalWinner == SELF) return root.childCol[i];

        const int v = ch.visits;
        const double mean = (v > 0) ? (ch.winSum / static_cast<double>(v)) : 0.5;
        const double rootVal = 1.0 - mean;
        if (v > bestVisit || (v == bestVisit && rootVal > bestVal)) {
            bestVisit = v;
            bestVal = rootVal;
            bestCol = root.childCol[i];
        }
    }

    State tmp = rootState;
    for (int i = 0; i < root.childCount; ++i) {
        const int col = root.childCol[i];
        const int chIdx = root.childIdx[i];
        if (chIdx < 0 || chIdx >= static_cast<int>(g.nodes.size())) continue;
        const Node& ch = g.nodes[chIdx];
        const int v = ch.visits;
        const double mean = (v > 0) ? (ch.winSum / static_cast<double>(v)) : 0.5;
        const double rootVal = 1.0 - mean;
        if (!moveAllowsImmediateOppWin(tmp, col, SELF)) {
            if (v > safeVisit || (v == safeVisit && rootVal > safeVal)) {
                safeVisit = v;
                safeVal = rootVal;
                safeCol = col;
            }
        }
    }

    if (safeCol >= 0) return safeCol;
    if (bestCol >= 0) return bestCol;
    return pickRootMoveFallback(rootState);
}

inline int runMCTS(Graph& g, const State& rootState,
                   const std::chrono::steady_clock::time_point& deadline) {
    if (g.root < 0 || g.root >= static_cast<int>(g.nodes.size())) return pickRootMoveFallback(rootState);

    int path[MAX_PATH];
    int iterations = 0;
    while (static_cast<int>(g.nodes.size()) + 1 < MAX_NODE_POOL) {
        if ((iterations & 15) == 0 && std::chrono::steady_clock::now() >= deadline) break;
        ++iterations;

        State st = rootState;
        int cur = g.root;
        int pathLen = 0;
        path[pathLen++] = cur;

        while (true) {
            Node& node = g.nodes[cur];
            if (node.terminal) break;

            if (node.untriedCount > 0) {
                const int pick = g.rng.nextInt(node.untriedCount);
                const int col = node.untried[pick];
                node.untried[pick] = node.untried[node.untriedCount - 1];
                --node.untriedCount;

                const int player = st.toMove;
                const int row = applyMove(st, col);
                const bool win = checkWinAt(st, row, col, player);
                const bool tie = (!win && !hasAnyMove(st));

                int childIdx = getOrCreateNode(g, st, win || tie, win ? player : 0);
                if (childIdx < 0) break;

                if (findChildByCol(node, col) < 0 && node.childCount < MAX_N) {
                    node.childCol[node.childCount] = col;
                    node.childIdx[node.childCount] = childIdx;
                    ++node.childCount;
                }
                cur = childIdx;
                path[pathLen++] = cur;
                break;
            }

            if (node.childCount <= 0) {
                node.terminal = 1;
                node.terminalWinner = 0;
                break;
            }

            const double logN = std::log(static_cast<double>(node.visits + 1));
            int bestEdge = -1;
            double bestScore = -1e100;
            for (int i = 0; i < node.childCount; ++i) {
                const int chIdx = node.childIdx[i];
                if (chIdx < 0 || chIdx >= static_cast<int>(g.nodes.size())) continue;
                const Node& ch = g.nodes[chIdx];
                double s = 0.0;
                if (ch.visits == 0) {
                    s = 1e50;
                } else {
                    const double chMean = ch.winSum / static_cast<double>(ch.visits);
                    const double exploit = 1.0 - chMean;
                    const double explore = UCT_C * std::sqrt(logN / static_cast<double>(ch.visits));
                    s = exploit + explore;
                }
                if (s > bestScore) {
                    bestScore = s;
                    bestEdge = i;
                }
            }

            if (bestEdge < 0) {
                node.terminal = 1;
                node.terminalWinner = 0;
                break;
            }

            const int col = node.childCol[bestEdge];
            const int x = applyMove(st, col);
            (void)x;
            cur = node.childIdx[bestEdge];
            path[pathLen++] = cur;
            if (pathLen >= MAX_PATH) break;
        }

        const Node& leaf = g.nodes[cur];
        const int winner = leaf.terminal ? leaf.terminalWinner : rollout(st, g.rng);

        for (int i = 0; i < pathLen; ++i) {
            Node& n = g.nodes[path[i]];
            n.visits += 1;
            n.winSum += static_cast<float>(scoreForPlayer(n.playerToMove, winner));
        }
    }

    return chooseBestRootMove(g, rootState);
}

inline void ensureRootChildForMove(Graph& g, const State& rootState, int col) {
    if (g.root < 0 || g.root >= static_cast<int>(g.nodes.size())) return;
    if (col < 0 || col >= rootState.N || rootState.top[col] <= 0) return;

    Node& root = g.nodes[g.root];
    if (findChildByCol(root, col) >= 0) {
        removeUntriedMove(root, col);
        return;
    }

    State st = rootState;
    const int player = st.toMove;
    const int x = applyMove(st, col);
    const bool win = checkWinAt(st, x, col, player);
    const bool tie = (!win && !hasAnyMove(st));

    int chIdx = getOrCreateNode(g, st, win || tie, win ? player : 0);
    if (chIdx < 0) return;
    if (root.childCount < MAX_N) {
        root.childCol[root.childCount] = col;
        root.childIdx[root.childCount] = chIdx;
        ++root.childCount;
    }
    removeUntriedMove(root, col);
}

inline State buildState(const int M, const int N, const int* top, const int* board,
                        const int noX, const int noY) {
    State st;
    st.M = M;
    st.N = N;
    st.noX = noX;
    st.noY = noY;
    st.toMove = SELF;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            st.board[i][j] = static_cast<uint8_t>(board[i * N + j]);
        }
    }
    for (int c = 0; c < N; ++c) {
        st.top[c] = static_cast<int8_t>(top[c]);
    }
    return st;
}

}  // namespace

extern "C" Point *getPoint(const int M, const int N, const int *top, const int *_board,
                           const int lastX, const int lastY, const int noX, const int noY)
{
    (void)lastX;
    (void)lastY;
    const auto t0 = std::chrono::steady_clock::now();
    const auto deadline = t0 + std::chrono::milliseconds(THINK_TIME_MS);
    initHash();

    const State rootState = buildState(M, N, top, _board, noX, noY);
    prepareCacheRoot(rootState);
    if (!g_cache.valid) {
        return new Point(top[0] - 1, 0);
    }

    int wins[MAX_N];
    State tmp = rootState;
    int y = -1;

    const int winCnt = collectWinningMoves(tmp, SELF, wins);
    if (winCnt > 0) {
        y = wins[0];
    } else {
        tmp = rootState;
        int threats[MAX_N];
        const int threatCnt = collectWinningMoves(tmp, USER, threats);
        if (threatCnt == 1) {
            y = threats[0];
        } else {
            y = runMCTS(g_cache.graph, rootState, deadline);
            if (y < 0 || y >= N || top[y] <= 0) {
                y = pickRootMoveFallback(rootState);
            }
        }
    }

    ensureRootChildForMove(g_cache.graph, rootState, y);
    g_cache.rootState = rootState;
    return new Point(top[y] - 1, y);
}

extern "C" void clearPoint(Point *p)
{
    delete p;
}

void clearArray(int M, int N, int **board)
{
    (void)N;
    for (int i = 0; i < M; i++)
    {
        delete[] board[i];
    }
    delete[] board;
}
