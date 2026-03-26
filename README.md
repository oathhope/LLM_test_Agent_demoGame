# MsPacman LLM Agent

基于 **HTTP JSON-RPC** 的大模型控制 MsPacman 吃豆人游戏框架。通过 RGB 帧像素分析提取结构化游戏状态，由 LLM（Claude / GPT 等）实时决策操控 Pacman 吃豆子、躲鬼魂。

---

## 项目文件结构

```
demo_game/
├─ pacman/                            # 吃豆人核心
│   ├─ game/
│   │   ├─ game_main.py              #   游戏主进程：ALE + pygame 渲染 + Flask RPC 服务器
│   │   └─ state_extractor.py        #   RGB 帧像素分析 → 结构化状态提取
│   ├─ agent/
│   │   └─ pacman_agent.py           #   LLM Agent：Prompt 构建 + 决策循环 + 历史记忆
│   ├─ logger.py                     #   JSONL + 文本双日志记录器
│   ├─ logs/                         #   运行日志输出目录
│   ├─ run_pacman_game.py            #   游戏服务器启动入口
│   └─ run_pacman_agent.py           #   Agent 启动入口
├─ requirements.txt
└─ README.md
```

---

## 整体架构

采用 **Game Server ↔ Agent 分离架构**，通过 HTTP JSON-RPC 通信：

```
┌─────────────────────────────┐        HTTP         ┌─────────────────────────────┐
│       Game Server           │  ◄───────────────►  │         LLM Agent           │
│  (game_main.py)             │                      │  (pacman_agent.py)          │
│                             │   GET  /state        │                             │
│  ALE MsPacman-v5 游戏引擎   │   POST /action       │  1. 获取 state              │
│  pygame 窗口实时渲染         │   POST /reset        │  2. 构建 Prompt             │
│  RGB 帧 → 像素分析          │                      │  3. 调用 LLM 决策           │
│  Flask RPC 端口 5001        │                      │  4. 解析 JSON → 发送 action │
└─────────────────────────────┘                      └─────────────────────────────┘
```

**Agent 运作循环：**

```
初始化连接 → 轮询循环:
  1. GET /state          获取结构化游戏状态（位置、鬼魂、可通行方向、豆子分布）
  2. Build Prompt        将状态 + 历史轨迹 + 决策规则拼为自然语言 prompt
  3. Call LLM            调用 Claude / GPT 等大模型
  4. Parse Response      从 LLM 输出中解析 {"action":"LEFT","steps":5,"reason":"..."}
  5. POST /action        发送动作到游戏服务器，执行 N 帧
  6. 等待 interval → 回到 1
```

---

## State（状态结构）

状态通过 **RGB 帧像素分析** 实时提取，核心逻辑在 `pacman/game/state_extractor.py`：

```python
{
    "pacman_x": 79, "pacman_y": 103,      # 像素坐标（由颜色质心计算）
    "score": 1320,
    "lives": 2,
    "total_reward": 1320.0,
    "dots_pixels": 9184,                   # 剩余豆子像素面积估算

    "ghosts": [                            # 四只鬼魂信息
        {"name": "Blinky", "x": 80, "y": 55,
         "frightened": False,              # 是否受惊（可吃）
         "distance": 48.0,
         "direction_from_pacman": "右上方"},
        # ... Pinky, Inky, Clyde
    ],
    "nearest_ghost": {...},                # 最近鬼魂
    "dangerous_ghost": {...},              # <50px 的危险鬼魂（或 null）

    "passable_directions": ["UP", "LEFT", "RIGHT"],  # 可通行方向
    "direction_info": {                    # 每方向深度扫描
        "UP":    {"corridor": 22, "dots": 5,  "passable": True},
        "DOWN":  {"corridor": 0,  "dots": 0,  "passable": False},
        "LEFT":  {"corridor": 17, "dots": 12, "passable": True},
        "RIGHT": {"corridor": 8,  "dots": 4,  "passable": True},
    },
    "safe_directions": ["UP", "LEFT"],     # 远离鬼魂的安全方向

    "last_move_success": True,             # 上次动作是否移动成功
    "last_blocked_direction": null,        # 上次撞墙方向
}
```

### 像素分析技术细节

状态提取基于 RGB 颜色匹配，核心颜色常量：

| 实体 | RGB 颜色 | 用途 |
|------|----------|------|
| Pacman | (210, 164, 74) 黄色 | 定位玩家坐标（质心） |
| Blinky | (200, 72, 72) 红色 | 定位红鬼 |
| Pinky | (198, 89, 179) 粉色 | 定位粉鬼 |
| Inky | (84, 184, 153) 青色 | 定位蓝鬼 |
| Clyde | (198, 108, 58) 橙色 | 定位橙鬼 |
| 受惊鬼魂 | (66, 72, 200) 深蓝 | 判断鬼魂是否可吃 |
| 豆子 | (228, 111, 111) 粉红 | 统计剩余豆子 |
| 墙壁 | b>100, b>r×2, b>g×2 | 通行性检测 |

**通行性检测 —— 宽截面扫描算法** (`_get_passable_directions`)：
1. 沿目标方向步进 4\~15px
2. 每步在**正交方向扫描 ±4px 截面**（共 9 个采样点）
3. 只要截面上有任一非墙像素 → 该步算"可通行"
4. ≥50% 的步可通行 → 该方向判定为可通行
5. 解决了 Pacman 中心偏移时单点检测撞墙壁边缘的问题

**通道深度扫描** (`_scan_direction_info`)：
1. 沿方向步进 4\~60px，同样使用宽截面检测
2. 允许最多 2px 连续墙壁（容错 sprite 边缘，`wall_streak < 3`）
3. 同时统计沿途豆子像素数（正交 ±2px 范围内检测豆子颜色）

---

## Action（动作空间）

共 **9 种动作**，映射到 ALE 动作 ID，定义在 `pacman/game/game_main.py`：

| 动作 | ALE ID | 说明 |
|------|--------|------|
| `NOOP` | 0 | 不动 |
| `UP` | 1 | 向上 |
| `RIGHT` | 2 | 向右 |
| `LEFT` | 3 | 向左 |
| `DOWN` | 4 | 向下 |
| `UPRIGHT` | 5 | 右上斜移 |
| `UPLEFT` | 6 | 左上斜移 |
| `DOWNRIGHT` | 7 | 右下斜移 |
| `DOWNLEFT` | 8 | 左下斜移 |

> Agent 实际只使用 UP / DOWN / LEFT / RIGHT / NOOP 五种基础动作。

**动作请求格式（POST /action）：**
```json
{"action": "LEFT", "steps": 5}
```
`steps` 表示连续执行帧数（1\~10），游戏服务器渲染中间帧后返回最终状态。

---

## Prompt 工程

Agent 每步将结构化状态转为如下 prompt 发给 LLM（`pacman_agent.py._build_prompt()`）：

```
你是控制 Ms.Pac-Man 的AI。目标：吃豆子得高分，躲鬼魂。

=== 当前状态 ===
位置: (79,103)  分数:1320  生命:2  剩余豆子像素:9184
上一步: LEFT ✓成功

=== 最近轨迹（旧→新） ===
  RIGHT→(109,29) | LEFT→(94,31) | UP→(82,31) | ...
  ⚠ 检测到来回走！请选一个方向持续走下去，不要反复切换！

=== 四方向详情 ===
  UP: ✓通行 通道22px 🟡5颗豆子
  DOWN: ✗墙壁
  LEFT: ✓通行 通道17px 🟡12颗豆子
  RIGHT: ✓通行 通道8px 🟡4颗豆子

=== 鬼魂 ===
  Blinky: 右上方 48px 【危险】
⚠ 紧急: Blinky 仅48px！
  逃跑方向: UP, LEFT

=== 决策规则(按优先级) ===
1. 撞墙→必须换方向
2. 鬼魂<50px且非受惊→逃跑(选远离鬼的可通行方向)
3. 受惊鬼魂→追着吃
4. 优先选有豆子的方向，且选最近没走过的路
5. ❗不要来回走！选定方向就持续走到底（至少3-5步）

输出JSON: {"action":"方向","steps":连续步数(1-10),"reason":"理由"}
```

关键增强特性：

| 特性 | 说明 |
|------|------|
| **轨迹记忆** | 记录最近 8 步的动作和位置，避免重复路径 |
| **振荡检测** | 方向切换率 >70% 时提示 LLM 保持方向 |
| **卡住检测** | 连续 3 步位置不变 → 强制切换方向（不依赖 LLM） |
| **撞墙禁止** | 上次撞墙方向标记为 ⛔ 禁止 |
| **探索建议** | 统计近期未走过的方向，提示 LLM 探索新区域 |

---

## Agent 配置

### LLM Provider 支持

| Provider | 说明 | 配置方式 |
|----------|------|---------|
| `anthropic` | Anthropic Claude 系列 | 环境变量 `ANTHROPIC_AUTH_TOKEN` + `ANTHROPIC_BASE_URL` |
| `openai` | OpenAI GPT 系列 | 环境变量 `OPENAI_API_KEY` |
| `ollama` | 本地 Ollama（如 qwen2.5） | `--base-url http://localhost:11434/v1` |
| `mock` | 规则引擎（无需 API Key） | 内置规则决策，用于测试和调试 |

### 命令行参数

```
python pacman/run_pacman_agent.py [OPTIONS]

  --provider    LLM 提供方: anthropic / mock (默认: anthropic)
  --model       模型名 (默认: claude-sonnet-4-5)
  --server-url  游戏服务器地址 (默认: http://127.0.0.1:5001)
  --max-turns   最大回合数, 0=无限 (默认: 0)
  --delay       回合间隔秒数 (默认: 0.5)
  --mock        使用 mock 规则引擎（无需 API Key）
```

### 决策流水线

```
ALE MsPacman-v5 游戏帧 (210×160 RGB)
    ↓ state_extractor.extract_state()
结构化状态 (位置/鬼魂/通行性/豆子分布)
    ↓ pacman_agent._build_prompt()
自然语言 Prompt (含轨迹记忆/振荡警告/方向详情)
    ↓ Claude / GPT / Mock
JSON 响应 {"action":"LEFT","steps":5,"reason":"..."}
    ↓ POST /action → game_main.py
ALE 执行 5 帧动作 → pygame 渲染 → 返回新状态
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install flask requests ale-py gymnasium pygame anthropic
```

### 2. 启动游戏（两个终端）

```bash
# 终端 1：启动游戏服务器（弹出 pygame 窗口）
cd demo_game
python pacman/run_pacman_game.py

# 终端 2：启动 Agent
python pacman/run_pacman_agent.py --mock                    # mock 模式（测试用）
python pacman/run_pacman_agent.py --provider anthropic \    # Claude AI
    --model claude-sonnet-4-5 --delay 0.3
```

### 3. 配置 LLM API

```bash
# Anthropic Claude
set ANTHROPIC_AUTH_TOKEN=sk-xxx
set ANTHROPIC_BASE_URL=https://api.anthropic.com
python pacman/run_pacman_agent.py --provider anthropic --model claude-sonnet-4-5
```

---

## RPC 接口

游戏服务器默认运行在 `http://127.0.0.1:5001`：

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/state` | 获取结构化游戏状态 |
| POST | `/action` | 执行动作 `{"action": "LEFT", "steps": 5}` |
| POST | `/reset` | 重置游戏 |
| GET | `/health` | 健康检查 |

---

## 性能记录（Claude claude-sonnet-4-5）

| 版本 | 分数 | NOOP 率 | 改动 |
|------|------|---------|------|
| v1 基础版 | 320 | — | 无记忆、无方向感知 |
| v2 +记忆+方向 | 750 | 39% (86/218) | 加入轨迹记忆 + 方向豆子计数 |
| v3 +宽截面检测 | **1320** | **2.7%** (3/111) | 修复像素检测算法，消除通行性误判 |

---

## 日志系统

运行日志保存在 `pacman/logs/` 目录，每次运行生成三个文件：

| 文件 | 格式 | 用途 |
|------|------|------|
| `run_YYYYMMDD_HHMMSS.jsonl` | JSONL | 每步详细记录（机器可分析） |
| `run_YYYYMMDD_HHMMSS.log` | 文本 | 人类可读的运行日志 |
| `run_YYYYMMDD_HHMMSS_stats.json` | JSON | 总分、步数、动作分布统计 |
