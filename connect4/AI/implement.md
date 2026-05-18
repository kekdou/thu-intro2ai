# 从零实现 `Strategy/Strategy.cpp`：MCTS + UCT + 跨回合复用（教学版）

本文只基于 `Strategy/Strategy.cpp` 当前实现进行讲解，不依赖其他目录代码。

这份文档按“**最容易上手**”的顺序，带你从 0 写出当前版本的 `Strategy/Strategy.cpp`。目标是：

1. 先有一个完全正确、可运行的四子棋 AI。
2. 再提升到 MCTS + UCT。
3. 最后加上跨回合复用搜索树（准确说是复用搜索图），做到强度和速度都更好。

文档会覆盖：
- 最佳实现顺序（分阶段）
- 每个核心函数的作用
- 为什么要这样实现
- 实现要点与常见坑

---

## 0. 先理解接口和棋盘语义

实验框架要求你实现两个导出函数：

- `extern "C" Point* getPoint(...)`
- `extern "C" void clearPoint(Point* p)`

`getPoint` 输入里最关键的是：
- `M, N`：棋盘行列。
- `top[c]`：第 `c` 列下一个可落子的位置是 `top[c]-1` 行。
- `_board[i*N+j]`：棋盘内容，`0` 空，`1` 用户，`2` 机器。
- `noX, noY`：禁手点（该位置不可放），并且会影响对应列的 `top` 跳过逻辑。

你返回 `new Point(x, y)`，框架会执行这个落子。

---

## 1. 推荐实现顺序（先跑通，再变强）

### 阶段 A：先把“状态 + 基础规则”写完整（不做搜索）

先实现这些函数：
- `buildState`
- `playerOpponent`
- `hasAnyMove`
- `checkWinAt`
- `applyMove`
- `undoMove`
- `orderedLegalCols`
- `isWinningMove`
- `collectWinningMoves`
- `pickRootMoveFallback`

这一步完成后，你已经有“正确模拟棋局”的底座。

### 阶段 B：做一个能下棋的“轻策略”

实现：
- `pickPolicyMove`
- `rollout`
- 在 `getPoint` 中加“先手杀/堵对手单杀”的硬规则

即使还没 MCTS，这一步也能保证 AI 基本不犯低级错误。

### 阶段 C：实现 MCTS + UCT（单回合）

实现：
- `Node / Graph` 数据结构
- `createNode`
- `getOrCreateNode`
- `findChildByCol`
- `removeUntriedMove`
- `scoreForPlayer`
- `runMCTS`
- `chooseBestRootMove`
- `ensureRootChildForMove`

到这里就已经是“标准可用”的强 AI 版本。

### 阶段 D：做跨回合复用（关键优化）

实现：
- `splitmix64`
- `initHash`
- `hashState`
- `equalState`
- `SearchCache`
- `resetCacheToState`
- `prepareCacheRoot`
- `copyNodeRec`
- `extractSubgraph`

这一步是性能核心，能显著增加有效模拟次数。

### 阶段 E：参数调优 + 稳定性

围绕以下常量调参：
- `THINK_TIME_MS = 2850`
- `UCT_C = 0.95`
- `MAX_NODE_POOL = 450000`
- rollout 候选宽度 `k = min(4, lcnt)`
- 时间检查频率 `(iterations & 15)`

---

## 2. 代码结构总览（先建立全局心智）

当前实现可看成 6 层：

1. **基础棋盘层**：`State` + 落子/回退/胜负判断。
2. **默认策略层**：rollout policy（优先即胜/防即败）。
3. **搜索节点层**：`Node`，保存统计值（`visits`, `winSum` 等）。
4. **搜索容器层**：`Graph`，存所有节点 + `keyToIdx`。
5. **复用缓存层**：`SearchCache`，跨回合保留图并切换根。
6. **接口层**：`getPoint` 串起来完成一次决策。

---

## 3. 逐函数讲解：作用 + 原因 + 实现要点

下面按你真正写代码的顺序解释。

## 3.1 常量与数据结构

### `constexpr` 常量

- `MAX_M, MAX_N`: 棋盘上界（12）
- `USER = 1, SELF = 2`: 双方标识
- `MAX_PATH`: 一次模拟路径最大长度（防止异常死循环）
- `MAX_NODE_POOL`: 节点池上限，防内存爆
- `UCT_C`: UCT 探索系数
- `THINK_TIME_MS`: 单步思考时间预算（2850ms）

为什么：
- 把“策略参数”和“安全边界”集中管理，方便调优。

### `struct State`

保存一个局面：
- `M, N, noX, noY`
- `board[MAX_M][MAX_N]`
- `top[MAX_N]`
- `toMove`

为什么：
- MCTS 的每次模拟都要复制并推进局面，`State` 必须是紧凑且可快速拷贝。

### `struct Node`

一个搜索图节点（一个局面），核心字段：
- `key`：局面 hash
- `childCol[] / childIdx[] / childCount`
- `untried[] / untriedCount`
- `playerToMove`
- `terminal / terminalWinner`
- `visits / winSum`

为什么：
- MCTS 的 UCT 值来源于统计量；同时需要增量扩展（`untried`）。

### `struct Graph`

全局搜索容器：
- `nodes`
- `keyToIdx`（hash 到节点下标）
- `rng`
- `root`

为什么：
- 用“图”而不是“树”，同一局面可以共享节点，避免重复搜索。

### `struct SearchCache`

- `graph`
- `rootState`
- `valid`

为什么：
- 跨回合把上一步搜索成果接着用，提升每回合有效深度。

---

## 3.2 哈希与复用基础

### `splitmix64`

作用：生成高质量伪随机 64 位值。

为什么：
- Zobrist hash 要求随机位分布好，碰撞概率低。

### `initHash`

作用：初始化 `g_zobrist` 和 `g_misc`。

为什么：
- 只初始化一次，避免每步重复构建随机表。

### `hashState(const State&)`

作用：把整个局面编码成 `uint64_t key`。

包含信息：
- 棋盘内容
- `top[]`
- `toMove`
- `M,N,noX,noY`

为什么：
- 跨回合定位根节点、图节点复用都依赖它。

### `equalState(const State&, const State&)`

作用：完整逐项比较两个状态是否一致。

为什么：
- 哈希可能极小概率碰撞；以及快速判断“当前状态是否与缓存根完全相同”。

---

## 3.3 基础规则与局面推进

### `hasAnyMove`

作用：检查是否还有可下列。

### `playerOpponent`

作用：返回对手编号（`1 <-> 2`）。

### `checkWinAt`

作用：只围绕最后落子点 `(x,y)` 在 4 个方向计数，判断是否连成 4。

为什么这么做：
- 比全盘扫描快得多，且逻辑正确（胜负一定由最后一步触发）。

### `applyMove`

作用：在 `State` 上执行一手。

关键细节：
- 落点是 `x = top[col]-1`。
- `top[col]--` 后，若遇到禁手跳跃条件 `x == noX+1 && col == noY`，再 `--top[col]`。
- 最后切换 `toMove`。

### `undoMove`

作用：回退一手，完全逆转 `applyMove`。

为什么必要：
- 用于“试探落子”类逻辑（例如是否会给对手立即致胜）。

### `orderedLegalCols`

作用：按“从中心向两边”输出合法列。

为什么：
- 中心列在四子棋通常更强，这个顺序让搜索更快命中好分支。

### `isWinningMove`

作用：临时在某列为指定 `piece` 落子，判断是否立刻赢。

技巧：
- 直接原地改 `State` 再恢复，不分配新内存，速度更高。

### `collectWinningMoves`

作用：收集当前方所有“一步致胜”的列。

为什么：
- 在 `getPoint` 和 rollout policy 中都非常有价值。

---

## 3.4 rollout 策略层

### `pickPolicyMove`

策略顺序：
1. 自己有一步赢就下赢。
2. 否则若对手有一步赢就堵。
3. 否则在中心优先的合法列前 `k` 个中随机。

为什么：
- 比纯随机 rollout 强很多；在同样时间下显著提升 MCTS 收敛速度。

### `rollout`

作用：从当前状态快速模拟到终局，返回赢家（`0/1/2`）。

为什么：
- 给叶子节点估值，供反向传播。

---

## 3.5 节点创建与图管理

### `scoreForPlayer(playerToMove, winner)`

作用：把终局赢家转换成某个节点视角分数：
- 赢 `1.0`
- 和 `0.5`
- 输 `0.0`

为什么：
- `winSum` 是“节点方视角”的累计得分，便于 UCT 统一处理。

### `createNode`

作用：创建新节点并初始化可扩展动作 (`untried`)。

关键：
- 终局节点 `untriedCount=0`。
- 非终局时根据 `orderedLegalCols` 填充。

### `getOrCreateNode`

作用：按 hash 复用已有节点，否则创建。

为什么：
- 图搜索的核心入口，避免重复建树。

### `findChildByCol`

作用：在父节点里查某列对应 child 是否已挂接。

### `removeUntriedMove`

作用：某列已经扩展后，从 `untried` 删除（swap-pop）。

为什么：
- `O(1)` 近似删除，避免 vector 擦除开销。

---

## 3.6 跨回合复用（核心性能点）

### `resetCacheToState`

作用：当无法复用时，清空图并以当前局面重建根。

### `prepareCacheRoot`

作用：每回合开始时准备根：
1. 若缓存无效，重置。
2. 若 `rootState` 完全相同，直接继续。
3. 否则用 `hashState` 在图中找对应节点作为新根。
4. 图过大时，提取新根可达子图做压缩。

为什么显著提升：
- 前几回合投入的搜索不会丢掉，后续回合直接站在“已有统计”上继续搜索。

### `copyNodeRec` + `extractSubgraph`

作用：把新根可达区域复制到新图，回收无关旧节点。

为什么：
- 防止图无限膨胀，保持局部性和内存可控。

---

## 3.7 根节点落子决策

### `moveAllowsImmediateOppWin`

作用：检测“我下这手后，对手是否马上有一步赢”。

### `chooseBestRootMove`

流程：
1. 若某子节点是“我方必胜终局”，立刻返回。
2. 否则按访问次数优先，`rootVal=1-chMean` 作为 tie-break。
3. 再过滤一遍“不会立刻送对手必胜”的安全手，优先安全手。

为什么：
- 标准“最高访问数”有时会选到战术陷阱；加安全过滤可明显减少低级送杀。

### `pickRootMoveFallback`

作用：任何异常下保证返回合法列，防止接口崩。

---

## 3.8 MCTS 主循环

### `runMCTS(Graph&, const State&, deadline)`

四阶段经典流程：

1. **Selection**
- 在已扩展子节点中按 UCT 选择：
  - 未访问 child：分数设极大（强制先探索）
  - 已访问 child：`exploit + UCT_C * sqrt(logN / n)`

2. **Expansion**
- 若节点有 `untried`，随机拿一个动作扩展 child。

3. **Simulation**
- 对新叶子调用 `rollout` 得到 winner。

4. **Backpropagation**
- 沿路径更新 `visits` 与 `winSum`。

稳定性细节：
- 每 16 次迭代检查一次时间，超时立刻退出。
- 路径长度受 `MAX_PATH` 保护。
- 节点数受 `MAX_NODE_POOL` 限制。

---

## 3.9 外部接口串接

### `buildState`

作用：把框架参数 (`top`, `_board`, `M/N/noX/noY`) 转成内部 `State`。

### `getPoint`

完整决策顺序（推荐照抄）：
1. 记录开始时间和 `deadline`。
2. `initHash()`。
3. `buildState`。
4. `prepareCacheRoot`。
5. 先做硬规则：
   - 我方一步赢 -> 直接下。
   - 对手唯一一步赢 -> 必堵。
6. 否则 `runMCTS` 选点。
7. 容错合法性检查，必要时 fallback。
8. `ensureRootChildForMove`：保证这步在根上有 child 关联（便于后续复用）。
9. 返回 `new Point(top[y]-1, y)`。

### `clearPoint`

作用：释放 `getPoint` 返回的堆对象。

### `clearArray`

作用：框架遗留接口，当前策略逻辑不依赖它。

---

## 4. 为什么这套顺序是“最佳学习路径”

先阶段 A/B 再 C/D 的原因：

1. `State` 与规则层先对，后续任何 bug 都少一半。
2. rollout policy 先写，可以先验证“棋感”而不是盲调 UCT。
3. 先做单回合 MCTS，确认算法正确后再加复用，定位问题更容易。
4. 跨回合复用最后加，能清晰看到性能提升来源，避免把正确性问题和缓存问题混在一起。

---

## 5. 最小实现检查清单（每阶段都能自测）

### A 阶段自测
- `applyMove/undoMove` 前后状态完全一致。
- `checkWinAt` 横竖斜都能触发。
- 禁手列 `top` 跳跃正确。

### B 阶段自测
- 有一步赢时一定下赢。
- 对手一步赢时能堵住。

### C 阶段自测
- `runMCTS` 能在时间内稳定返回合法列。
- `visits` 会增长，且根 child 访问分布有区分。

### D 阶段自测
- 连续回合中，图大小和根命中行为符合预期。
- 图太大时可触发子图提取，不崩溃。

### E 阶段自测
- 2850ms 下无超时。
- 对固定对手胜率稳定，不因随机种子剧烈波动。

---

## 6. 调参优先级（2850ms 版本）

建议顺序：

1. `UCT_C`：`0.75 / 0.90 / 1.05 / 1.20`
2. rollout `k`：`2 / 3 / 4 / 5`
3. 子图压缩阈值：`55% / 65% / 75%`
4. `MAX_NODE_POOL`：`300k / 400k / 500k`
5. 时间检查频率：`7 / 15 / 31`

经验：
- 先做 `UCT_C + k` 二维网格，通常收益最大。

---

## 7. 常见坑（你最容易踩的）

1. 忘记在 `undoMove` 恢复禁手列 `top`。
2. `winSum` 视角搞错（必须按 `node.playerToMove` 记分）。
3. `getOrCreateNode` 复用到旧节点后未处理终局状态升级。
4. 根切换后没有同步 `g_cache.rootState`。
5. 时间检查不够频繁，导致 3s 附近超时。
6. `top[y]-1` 与内部选择列 `y` 不一致。

---

## 8. 最终建议：如何“完整重写一遍”

按下面顺序在 `Strategy/Strategy.cpp` 从空文件实现：

1. 写常量、`State/Node/Graph/SearchCache`。
2. 写哈希：`splitmix64/initHash/hashState/equalState`。
3. 写规则基础：`playerOpponent/hasAnyMove/checkWinAt/applyMove/undoMove`。
4. 写动作集合：`orderedLegalCols/isWinningMove/collectWinningMoves`。
5. 写 rollout：`pickPolicyMove/rollout`。
6. 写节点管理：`createNode/getOrCreateNode/findChildByCol/removeUntriedMove`。
7. 写复用：`resetCacheToState/prepareCacheRoot/copyNodeRec/extractSubgraph`。
8. 写决策：`chooseBestRootMove/moveAllowsImmediateOppWin/pickRootMoveFallback`。
9. 写 `runMCTS`。
10. 写接口：`buildState/getPoint/clearPoint/clearArray`。
11. 编译 + 对战测试，再调参。

这就是一条“从能跑到能打”的最短路径。
