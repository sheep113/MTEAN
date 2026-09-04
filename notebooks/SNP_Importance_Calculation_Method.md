# SNP重要性计算方法说明

## 版本信息
- **文档版本**: 2.0
- **日期**: 2026年1月17日
- **方法**: 
  - **旧方案**: 简单平均聚合（drawattentionweights.ipynb）
  - **新方案**: Utility-Correlation综合评分 + 局部富集 + 经验P值推断（drawattentionweights_enrichment.ipynb）

---

## 目录

1. [方法对比总览](#方法对比总览)
2. [旧方案：简单聚合方法](#旧方案简单聚合方法)
3. [新方案：富集增强方法](#新方案富集增强方法)
4. [输入数据](#输入数据)
5. [新方案详细计算流程](#新方案详细计算流程)
6. [数学公式详解](#数学公式详解)
7. [输出结果](#输出结果)
8. [可视化说明](#可视化说明)

---

## 方法对比总览

### 核心差异

| 维度 | 旧方案 (drawattentionweights.ipynb) | 新方案 (drawattentionweights_enrichment.ipynb) |
|------|-------------------------------------|-----------------------------------------------|
| **数据来源** | 直接从权重文件聚合 | 从预计算的权重和表型标签计算 |
| **多样本处理** | 跨样本平均/最大值/加权平均 | 考虑每个样本与表型的关系 |
| **评分维度** | 单一：权重大小 | 双维度：普遍性(Utility) + 相关性(Correlation) |
| **空间信息** | 不考虑 | 考虑局部染色体聚集（窗口富集） |
| **统计推断** | 直接使用分数/归一化 | 基于秩次的经验P值 |
| **表型关联** | 无 | 显式计算权重-表型相关性 |

### 适用场景

- **旧方案**：快速探索性分析，不需要表型关联验证
- **新方案**（推荐）：需要统计显著性评估和表型关联分析的场景

---

---

## 旧方案：简单聚合方法

### 方法描述

直接从注意力权重矩阵聚合得到重要性分数，**不考虑表型信息**。

### 输入

$$\mathbf{W} \in \mathbb{R}^{B \times E \times S}$$

其中：
- $B$：批次样本数
- $E$：专家数
- $S$：SNP位点数

### 计算步骤

#### Step 1: 跨样本聚合

**方法1：简单平均**
$$\mathbf{W}_{\text{avg}} = \frac{1}{B}\sum_{b=1}^{B} \mathbf{W}_{b,:,:} \in \mathbb{R}^{E \times S}$$

**方法2：最大值**
$$\mathbf{W}_{\text{max}} = \max_{b \in [1,B]} \mathbf{W}_{b,:,:} \in \mathbb{R}^{E \times S}$$

**方法3：加权平均**（根据样本权重强度加权）
$$\alpha_b = \frac{\text{mean}(|\mathbf{W}_b|)}{\sum_{b'=1}^{B} \text{mean}(|\mathbf{W}_{b'}|)}$$
$$\mathbf{W}_{\text{weighted}} = \sum_{b=1}^{B} \alpha_b \mathbf{W}_{b,:,:}$$

#### Step 2: 跨专家聚合（可选）

$$S_i^{\text{overall}} = \frac{1}{E}\sum_{e=1}^{E} W_{e,i}$$

或保留每个专家的独立分数：
$$S_i^{(e)} = W_{e,i}, \quad e \in [1, E]$$

#### Step 3: 归一化

$$S_i^{\text{norm}} = \frac{S_i - \min(S)}{\max(S) - \min(S)}$$

### 优点
- 计算简单快速
- 不需要表型标签
- 可用于无监督探索

### 缺点
- **无法验证与表型的关联**
- 不考虑样本间表型变异
- 无统计显著性评估
- 忽略空间聚集信息

---

## 新方案：富集增强方法

### 方法概述

### 方法概述

本方法通过结合**位点普遍性**（Utility）和**位点-表型相关性**（Correlation）两个维度，并考虑**局部染色体富集**，计算每个SNP位点的重要性，最后使用**无分布假设的经验P值**进行统计推断。

### 核心思想

1. **普遍性（Utility）**：如果某个位点在多个样本中都获得较高的注意力权重，说明该位点具有普遍重要性
2. **相关性（Correlation）**：如果某个位点的注意力权重变化与表型变化高度相关，说明该位点与表型关联
3. **局部富集**：考虑染色体上相邻SNP的局部聚集效应，增强区域信号
4. **综合评分**：两者结合，既要"普遍"又要"相关"，并考虑空间聚集
5. **统计推断**：使用秩次统计转换为经验P值，无需假设数据分布

### 关键创新点

- ✅ **表型驱动**：显式计算权重-表型相关性
- ✅ **空间感知**：通过滑动窗口捕捉染色体局部富集
- ✅ **多层评分**：提供5种评分方法（Enrichment、Smoothed Corr、Enriched Corr、Combined v1、Combined v2）
- ✅ **统计严格**：基于秩次的无分布假设经验P值

---

## 输入数据

### 1. 注意力权重矩阵（新方案专用）
$$\mathbf{W} \in \mathbb{R}^{B \times E \times S}$$

其中：
- $B$：样本数量（**所有样本，而非批次**）
- $E$：专家/表型数量
- $S$：SNP位点数量

**关键说明**：
- 注意力权重已在SNP维度归一化，即对每个样本每个专家：
  $$\sum_{i=1}^{S} W_{b,e,i} = 1, \quad \forall b \in [1,B], e \in [1,E]$$
- 权重通过 `evaluation/calSNPweights.py` 从模型输出中提取
- 保存为 `.npy` 格式：`snp_weights_{split}.npy` (shape: [B, E, S])

### 2. 表型真实值矩阵（新方案专用）
$$\mathbf{Y} \in \mathbb{R}^{B \times E}$$

其中：
- $Y_{b,e}$：第 $b$ 个样本在第 $e$ 个专家对应表型的真实值
- 专家与表型一一对应（如：Expert 0 ↔ DTA, Expert 1 ↔ DTS, Expert 2 ↔ DTT）

**关键说明**：
- 表型标签从原始数据集中提取
- 保存为 `.npy` 格式：`predictions_phenotype_labels_{split}.npy` (shape: [B, E])
- **样本顺序一致性**：通过 SequentialSampler 确保权重和表型标签的样本顺序完全对齐

### 3. SNP元数据
- **SNP_ID**：位点标识符
- **Chromosome**：染色体编号
- **Position**：染色体上的物理位置（bp）

来源：从 BIM 文件中根据保留的SNP索引提取

---

## 新方案详细计算流程

### Step 0: 专家选择与数据准备

### Step 0: 专家选择与数据准备

对每个专家 $e$ 独立计算（多专家并行分析）：
$$\mathbf{W}_e = \mathbf{W}[:, e, :] \in \mathbb{R}^{B \times S}$$
$$\mathbf{Y}_e = \mathbf{Y}[:, e] \in \mathbb{R}^{B}$$

**重要**：每个样本 $b$ 的权重向量 $\mathbf{W}_{b,e,:}$ 与其表型值 $Y_{b,e}$ 一一对应。

---

### Step 1: 计算普遍性指标（Utility）—— 多样本信息综合的核心

#### 1.1 多样本权重数据的组织

对于专家 $e$（对应某个表型），每个SNP位点 $i$ 的权重数据为一个**长度为B的向量**：

$$\mathbf{w}_i = [W_{1,e,i}, W_{2,e,i}, \ldots, W_{B,e,i}]^T \in \mathbb{R}^{B}$$

其中：
- $W_{b,e,i}$：第 $b$ 个样本对位点 $i$ 的注意力权重
- 每个样本的权重向量已归一化：$\sum_{i=1}^{S} W_{b,e,i} = 1$

**示例**（假设B=5，某个SNP的权重向量）：
```
样本1对该SNP的权重: 0.0012
样本2对该SNP的权重: 0.0008
样本3对该SNP的权重: 0.0015
样本4对该SNP的权重: 0.0003
样本5对该SNP的权重: 0.0011

→ 权重向量: [0.0012, 0.0008, 0.0015, 0.0003, 0.0011]
```

#### 1.2 原始LFC（Log-Fold-Change）—— 跨样本统计量

对每个位点 $i$，计算其**跨所有样本**的相对重要性：

$$U_i^{\text{raw}} = \log_{10}\left(\frac{\mu_i + \epsilon}{\text{median}_i + \epsilon}\right)$$

其中：
- $\mu_i = \frac{1}{B}\sum_{b=1}^{B} W_{b,e,i}$：位点 $i$ 在**所有 B 个样本**上的**平均**权重
- $\text{median}_i = \text{median}(\{W_{1,e,i}, W_{2,e,i}, \ldots, W_{B,e,i}\})$：位点 $i$ 的**中位数**权重（跨所有样本）
- $\epsilon = 10^{-12}$：数值稳定性小常数

**多样本信息综合方式**：

**第一步：计算群体中心趋势**
- **平均值** $\mu_i$：反映该SNP在群体中的**典型权重水平**
- **中位数** $\text{median}_i$：群体权重的**鲁棒中心**（不受极端值影响）

**第二步：计算相对重要性**
- $\frac{\mu_i}{\text{median}_i}$：平均值相对于中位数的比值
  - 比值 > 1：该SNP的平均权重超过中位数（正偏态，部分样本给予高权重）
  - 比值 ≈ 1：权重分布对称，平均值接近中位数
  - 比值 < 1：该SNP的平均权重低于中位数（负偏态，少数样本给予高权重）

**第三步：对数变换**
- $\log_{10}$：将比值映射到线性尺度
- 使得倍数变化更容易比较（2倍 vs 10倍 vs 100倍）

**物理意义**：
- $U_i^{\text{raw}} > 0$：该位点的平均权重高于中位数（普遍重要）
- $U_i^{\text{raw}} \approx 0$：该位点权重接近中位数（普通位点）
- $U_i^{\text{raw}} < 0$：该位点的平均权重低于中位数（不普遍）

**示例计算**：
```
样本权重向量: [0.0012, 0.0008, 0.0015, 0.0003, 0.0011]
平均值: 0.00098
中位数: 0.0011
比值: 0.00098 / 0.0011 = 0.891
LFC: log10(0.891) = -0.050

解释：该SNP的平均权重略低于中位数，Utility为负
```

#### 1.3 为什么使用 mean/median 而非其他统计量？

**设计理由**：

1. **平均值（mean）**：
   - 综合所有样本的贡献（敏感于所有样本）
   - 如果某个SNP在多数样本中获得高权重，mean会体现这一趋势

2. **中位数（median）**：
   - 作为归一化基准（鲁棒统计量）
   - 不受极端值影响（少数样本的极高或极低权重不会扭曲基准）

3. **mean/median 比值**：
   - **检测偏态分布**：如果某个SNP的权重在群体中呈现正偏态（大多数样本高，少数样本低），mean > median，Utility为正
   - **区分普遍性与特异性**：
     - 普遍重要：多数样本都给高权重 → mean ≈ median（都高） → Utility高
     - 特异重要：少数样本给极高权重，多数给低权重 → mean > median（但median低） → Utility中等
     - 不重要：多数样本都给低权重 → mean ≈ median（都低） → Utility低

**对比其他可能的选择**：

| 统计量 | 优点 | 缺点 | 为何不用 |
|--------|------|------|----------|
| **mean/median** ✓ | 综合群体趋势，鲁棒 | - | 当前方案 |
| max | 捕捉最强信号 | 只看单个样本，不反映普遍性 | 旧方案可选，易受噪声影响 |
| sum | 综合总贡献 | 已被归一化抵消，无意义 | 权重已归一化 |
| std/mean (CV) | 反映变异程度 | 不直接反映重要性 | 适合变异分析，非重要性评估 |
| percentile(75) | 鲁棒的高分位数 | 丢失部分样本信息 | 不如mean全面 |

**关键理解**：
- Utility 衡量的是某个SNP在**群体层面**的普遍重要性
- 通过 **mean/median** 比值，同时考虑了：
  - **群体平均表现**（mean）
  - **群体基准水平**（median）
  - **分布偏态**（比值偏离1的程度）
- 这是一个**跨样本统计量**，不依赖于表型（与Correlation形成互补）

#### 1.2 归一化到[0, 1]

由于LFC可能为负，需要先转正值再归一化：

$$U_i^{\exp} = \exp(U_i^{\text{raw}})$$

$$U_i = \frac{U_i^{\exp} - \min(U^{\exp})}{\max(U^{\exp}) - \min(U^{\exp}) + \epsilon}$$

其中 $U^{\exp} = \{U_1^{\exp}, U_2^{\exp}, \ldots, U_S^{\exp}\}$

**结果**：$U_i \in [0, 1]$，越接近1表示该位点普遍性越强

---

### Step 2: 计算相关性指标（Correlation）

#### 2.1 Pearson相关系数

对每个位点 $i$，计算其**权重向量**与**表型向量**的相关性：

$$C_i = \left|\text{Pearson}\left(\mathbf{W}_{:,e,i}, \mathbf{Y}_e\right)\right|$$

展开形式：

$$C_i = \left|\frac{\sum_{b=1}^{B} (W_{b,e,i} - \bar{W}_i)(Y_{b,e} - \bar{Y}_e)}{\sqrt{\sum_{b=1}^{B}(W_{b,e,i} - \bar{W}_i)^2} \cdot \sqrt{\sum_{b=1}^{B}(Y_{b,e} - \bar{Y}_e)^2}}\right|$$

其中：
- $\bar{W}_i = \frac{1}{B}\sum_{b=1}^{B} W_{b,e,i}$：位点 $i$ 的平均权重
- $\bar{Y}_e = \frac{1}{B}\sum_{b=1}^{B} Y_{b,e}$：表型的平均值
- 取**绝对值**：我们关心相关性的强度，不关心方向（正相关或负相关都重要）

**结果**：$C_i \in [0, 1]$，越接近1表示该位点与表型相关性越强

**关键理解**：
- Correlation 衡量的是某个SNP的权重变化与表型变化的**协同性**
- 如果某个SNP在表型高的样本中权重高，在表型低的样本中权重低（或反之），其相关性就高
- 这是一个**跨样本关联统计量**，完全依赖于表型信息
- **样本对应关系至关重要**：$W_{b,e,i}$ 和 $Y_{b,e}$ 必须来自同一个样本 $b$

**为什么取绝对值？**
- 正相关：SNP权重↑ → 表型↑（促进作用）
- 负相关：SNP权重↑ → 表型↓（抑制作用）
- 在基因型-表型关联分析中，两种方向的关联都是有意义的

---

### Step 3: 基础综合评分

将两个指标相乘得到基础综合分数：

$$S_i^{\text{base}} = U_i \times C_i$$

**含义**：
- $S_i^{\text{base}}$ 高：该位点既**普遍重要**（多数样本权重高），又与**表型强相关**
- $S_i^{\text{base}}$ 低：该位点要么不普遍，要么与表型无关，或两者都不满足

**范围**：$S_i^{\text{base}} \in [0, 1]$

---

### Step 4: 局部富集增强（新方案核心创新）

#### 4.1 窗口配置

- **WINDOW_SIZE**（默认50）：每侧考虑的SNP数量
- 对每个染色体独立处理，避免跨染色体污染

#### 4.2 Smoothed Correlation（平滑相关性）

对原始相关性进行局部归一化：

$$C_i^{\text{smooth}} = \frac{C_i}{\text{median}(\{|C_j| : j \in \text{window}(i)\}) + \epsilon}$$

其中：
- $\text{window}(i) = [i-w, i+w]$：以位点 $i$ 为中心的窗口（$w$ = WINDOW_SIZE）
- 分母是窗口内相关性绝对值的中位数

**意义**：
- 高于局部背景的相关性被放大
- 低于局部背景的相关性被缩小
- 归一化到 $[-1, 1]$

#### 4.3 Enrichment Score（富集分数）

对 Utility 应用局部密度加权：

$$\text{Enrichment}_i = U_i \times (1 + \rho_i^{\text{utility}})$$

其中：
$$\rho_i^{\text{utility}} = \text{mean}(\text{top50\%}(\{U_j : j \in \text{window}(i)\}))$$

**意义**：
- 如果某个SNP周围有很多高Utility的SNP（局部聚集），其enrichment分数会被增强
- $\rho_i$ 是局部密度指标，范围 $[0, 1]$
- 归一化到 $[0, 1]$

#### 4.4 Enriched Correlation（富集相关性）

**修正公式**（两步法）：

**第一步**：先计算 Smoothed Correlation（如 4.2）

**第二步**：对 Smoothed Correlation 应用局部密度加权

$$C_i^{\text{enriched}} = C_i^{\text{smooth}} \times (1 + \rho_i^{\text{smooth}})$$

其中：
$$\rho_i^{\text{smooth}} = \text{mean}(\text{top50\%}(\{|C_j^{\text{smooth}}| : j \in \text{window}(i)\}))$$

**意义**：
- 先进行局部归一化（相对于局部背景）
- 再应用局部聚集增强
- 归一化到 $[-1, 1]$（保留方向）

#### 4.5 Combined v1

$$\text{Combined v1}_i = \text{Enrichment}_i \times |C_i^{\text{smooth}}|$$

- Utility维度考虑富集，Correlation维度仅平滑

#### 4.6 Combined v2（推荐）

$$\text{Combined v2}_i = \text{Enrichment}_i \times (1 + |C_i^{\text{enriched}}|)$$

- Utility和Correlation两个维度都考虑富集
- 使用 $(1 + |\cdot|)$ 形式保证不会缩小utility的贡献
- **这是新方案推荐的最终评分**

**为什么使用 $(1 + |\cdot|)$ 而非直接乘？**
- 直接乘法：如果 $C_i^{\text{enriched}}$ 很小，会压制 Enrichment 信号
- $(1 + |\cdot|)$ 形式：保证至少保留 Enrichment 的全部贡献，相关性作为额外的增强因子

---

### Step 5: 经验P值计算（核心统计推断）

### Step 5: 经验P值计算（核心统计推断）

**适用于任何评分**（基础分、Enrichment、Combined v1/v2等）

#### 5.1 升序排名

使用scipy的`rankdata`函数计算升序排名：

$$R_i^{\text{asc}} = \text{rankdata}(\{S_1, S_2, \ldots, S_S\}, \text{method='average'})$$

**说明**：
- 最小的分数 → `rank = 1`
- 最大的分数 → `rank = S`
- `method='average'`：处理ties（相同分数取平均秩次）

#### 5.2 转换为降序排名

$$R_i = S + 1 - R_i^{\text{asc}}$$

**结果**：
- 最高的分数 → `rank = 1`（最重要）
- 最低的分数 → `rank = S`（最不重要）

#### 5.3 经验P值

$$P_i^{\text{empirical}} = \frac{R_i}{S + 1}$$

**统计含义**：
- $P_i$ 是观测到 $\geq S_i$ 分数的位点比例（基于秩次的经验累积分布）
- 等价于**排列检验**的零分布P值，但无需显式生成零模型
- **分布自由**：不假设分数服从任何参数分布（如高斯分布）

**为什么除以 $(S+1)$ 而不是 $S$？**
- 避免P值为0（最小P值为 $\frac{1}{S+1}$）
- 统计学惯例，更保守的估计

**物理意义**：
- $P_i = 0.01$：该SNP的分数排在前1%
- $P_i = 0.05$：该SNP的分数排在前5%
- $P_i = 0.50$：该SNP的分数处于中位数

#### 5.4 -log10转换（用于可视化）

$$\text{-log10}(P_i) = -\log_{10}\left(\frac{R_i}{S + 1}\right)$$

**GWAS曼哈顿图标准**：
- 值越大，P值越小，信号越显著
- 例如：
  - $\text{-log10}(P) = 2$：对应 $P = 0.01$（前1%）
  - $\text{-log10}(P) = 3$：对应 $P = 0.001$（前0.1%）
  - $\text{-log10}(P) = 5$：对应 $P = 0.00001$（前0.001%）

---

## 数学公式汇总

### 旧方案完整管道

对专家 $e$：

$$
\begin{align}
\text{跨样本聚合:} \quad & W_{e,i} = \frac{1}{B}\sum_{b=1}^{B} W_{b,e,i} \quad \text{或其他聚合方式} \\
\\
\text{归一化:} \quad & S_i = \frac{W_{e,i} - \min(W_e)}{\max(W_e) - \min(W_e)} \in [0,1]
\end{align}
$$

### 新方案完整管道

对专家 $e$，位点 $i$：

**基础评分：**
$$
\begin{align}
\text{Utility:} \quad & U_i^{\text{raw}} = \log_{10}\left(\frac{\text{mean}(W_{:,e,i})}{\text{median}(W_{:,e,i}) + \epsilon}\right) \\
& U_i = \text{MinMaxScale}(\exp(U_i^{\text{raw}})) \in [0,1] \\
\\
\text{Correlation:} \quad & C_i = \left|\text{Pearson}(W_{:,e,i}, Y_{:,e})\right| \in [0,1] \\
\\
\text{Base Score:} \quad & S_i^{\text{base}} = U_i \times C_i \in [0,1]
\end{align}
$$

**局部富集增强：**
$$
\begin{align}
\text{Enrichment:} \quad & \text{Enrich}_i = U_i \times (1 + \rho_i^{\text{utility}}) \\
\\
\text{Smoothed Corr:} \quad & C_i^{\text{smooth}} = \frac{C_i}{\text{median}_{\text{window}(i)}(|C|) + \epsilon} \\
\\
\text{Enriched Corr:} \quad & C_i^{\text{enriched}} = C_i^{\text{smooth}} \times (1 + \rho_i^{\text{smooth}}) \\
\\
\text{Combined v2:} \quad & S_i = \text{Enrich}_i \times (1 + |C_i^{\text{enriched}}|) \quad \textbf{(推荐)}
\end{align}
$$

**经验P值：**
$$
\begin{align}
\text{Rank (descending):} \quad & R_i = S + 1 - \text{rankdata}(S)_i \\
\\
\text{Empirical P-value:} \quad & P_i = \frac{R_i}{S + 1} \\
\\
\text{Manhattan Y-axis:} \quad & Y_i = -\log_{10}(P_i)
\end{align}
$$

---

## 输出结果

### 旧方案输出

对每个专家 $e$：

1. **`importance_score`** ($S_i$)：归一化后的重要性分数 [0,1]

**文件**：
- 通常不单独保存，直接用于可视化

---

### 新方案输出

对每个专家 $e$，输出以下数组（长度均为 $S$）：

1. **`utility`** ($U_i^{\text{raw}}$)：原始Log-Fold-Change
2. **`utility_normalized`** ($U_i$)：归一化后的普遍性指标 [0,1]
3. **`correlation`** ($C_i$)：与表型的相关性 [0,1]
4. **`score`** ($S_i^{\text{base}}$)：基础综合评分 [0,1]
5. **`enrichment_score`**：Utility富集分数 [0,1]
6. **`smoothed_correlation`**：平滑相关性 [-1,1]
7. **`enriched_correlation`**：富集相关性 [-1,1]
8. **`combined_v1`**：组合评分v1 [0,1]
9. **`combined_v2`**（推荐）：组合评分v2 [0,1]
10. **`empirical_p`** ($P_i$)：经验P值（对任意评分）
11. **`neg_log10_p`** ($-\log_{10}(P_i)$)：用于曼哈顿图的Y轴

**文件保存**：
- **曼哈顿图（每个专家×5种方法）**：
  - `manhattan_enrichment_{ExpertName}.png`
  - `manhattan_smoothed_corr_{ExpertName}.png`
  - `manhattan_enriched_corr_{ExpertName}.png`
  - `manhattan_combined_v1_{ExpertName}.png`
  - `manhattan_combined_v2_{ExpertName}.png`（推荐）
- **Top SNPs列表（每个专家×5种方法）**：
  - `top_snps_enrichment_{ExpertName}.csv`
  - `top_snps_smoothed_corr_{ExpertName}.csv`
  - `top_snps_enriched_corr_{ExpertName}.csv`
  - `top_snps_combined_v1_{ExpertName}.csv`
  - `top_snps_combined_v2_{ExpertName}.csv`（推荐）

---

## 可视化说明

### 旧方案可视化

**1. 重要性分数分布直方图**
- X轴：归一化重要性分数
- Y轴：频数
- 展示整体分数分布

**2. 专家对比箱线图**
- 比较不同专家的分数分布差异

**3. Top K SNPs条形图**
- 展示得分最高的K个SNP

---

### 新方案可视化

#### 1. 曼哈顿图（Manhattan Plot）

**X轴**：染色体位置（genome-wide）  
**Y轴**：$-\log_{10}(\text{Empirical P-value})$  
**点的颜色**：按染色体交替（标准GWAS风格）  
**点的大小**：
- 显著点（$-\log_{10}(P) > 3$）：大点 + 黑色边框
- 非显著点：小点，半透明

**富集趋势线叠加**：
- 灰色平滑曲线，展示局部富集强度
- 通过高斯平滑实现（sigma = 染色体SNP数 × 0.05）

**阈值线**：
- **灰色虚线**：$-\log_{10}(P) = 3$（经验显著性阈值，对应前0.1%）

**解读**：
- 越高的点 = 越小的P值 = 越显著的关联
- 超过灰线的点：经验显著（前0.1%）
- 富集趋势线高的区域：该区域存在多个重要SNP聚集

#### 2. Top SNPs表格

按 $-\log_{10}(P)$ 降序排列，显示：
- **SNP ID**：位点标识符
- **Chromosome**：染色体
- **Position**：物理位置
- **-log10(P)**：显著性分数

---

## 新旧方案对比总结

### 多样本信息综合方式的本质差异

#### 旧方案：直接聚合，丢失样本间关系

**数据流**：
```
输入：权重矩阵 [B, E, S]

处理方式1（平均）:
  样本1: [w1_snp1, w1_snp2, ..., w1_snpS]
  样本2: [w2_snp1, w2_snp2, ..., w2_snpS]
  ...
  样本B: [wB_snp1, wB_snp2, ..., wB_snpS]
  ↓ 平均
  结果: [mean(w_snp1), mean(w_snp2), ..., mean(w_snpS)]

处理方式2（最大值）:
  结果: [max(w_snp1), max(w_snp2), ..., max(w_snpS)]
  
输出：每个SNP一个分数（纯粹是权重的聚合）
```

**特点**：
- ❌ **丢失样本间变异信息**：平均/最大值后无法知道各样本的权重分布
- ❌ **无法验证表型关联**：不知道权重高的SNP是否与表型真正相关
- ✓ **计算简单**：无需表型标签
- ✓ **适合无监督探索**

**假设**：权重大的SNP就是重要的（缺乏验证）

---

#### 新方案：保留样本信息，双维度验证

**数据流**：
```
输入：权重矩阵 [B, E, S] + 表型向量 [B, E]

对每个SNP i:
  权重向量: wi = [W1_i, W2_i, ..., WB_i]  ← 保留所有样本
  表型向量: y = [y1, y2, ..., yB]        ← 样本对应的表型值
  
  维度1 - Utility（群体统计）:
    计算 wi 的统计特征（mean, median）
    → Utility_i = log10(mean(wi) / median(wi))
    → 反映该SNP在群体中的普遍性
  
  维度2 - Correlation（样本间关联）:
    计算 wi 与 y 的相关性
    → Corr_i = |Pearson(wi, y)|
    → 反映该SNP的权重变化是否与表型变化协同
  
  综合评分:
    Score_i = Utility_i × Corr_i
    → 既普遍又相关的SNP得分最高
```

**特点**：
- ✓ **保留所有样本信息**：每个样本的权重都参与Utility和Correlation计算
- ✓ **双维度验证**：
  - Utility：该SNP在群体中是否普遍重要？
  - Correlation：该SNP的权重变化是否与表型变化一致？
- ✓ **统计严格**：基于相关性和经验P值，有统计学支撑
- ✓ **可解释性强**：可以区分"普遍重要但与表型无关"vs"与表型相关但不普遍"

**假设**：重要的SNP应该同时满足群体普遍性和表型关联性（有验证）

---

#### 关键对比表

| 方面 | 旧方案 | 新方案 |
|------|--------|--------|
| **样本利用方式** | 跨样本聚合后丢弃原始数据 | 保留所有样本数据用于统计计算 |
| **多样本信息综合** | 简单算术平均/最大值 | mean/median比值 + Pearson相关性 |
| **样本间变异** | 丢失（聚合后无法恢复） | 保留（用于计算相关性） |
| **表型信息** | 不使用 | 核心驱动因素（Correlation维度） |
| **评分维度** | 单一（权重大小） | 双维度（Utility + Correlation） |
| **统计基础** | 描述性统计 | 推断性统计（Pearson + 经验P值） |
| **空间信息** | 忽略 | 考虑染色体局部富集 |
| **可验证性** | 无法验证与表型关联 | 可通过相关性验证 |

---

#### 样本信息利用的深度对比

**场景假设**：某个SNP在5个样本中的权重和表型

| 样本 | SNP权重 | 表型值 | 旧方案处理 | 新方案处理 |
|------|---------|--------|-----------|-----------|
| 1 | 0.0015 | 50 | ↓ | ↓ (保留) |
| 2 | 0.0008 | 30 | ↓ | ↓ (保留) |
| 3 | 0.0012 | 45 | ↓ | ↓ (保留) |
| 4 | 0.0003 | 25 | ↓ | ↓ (保留) |
| 5 | 0.0011 | 40 | ↓ | ↓ (保留) |
| **聚合结果** | mean=0.00098 | - | **0.00098** | **Utility + Corr** |

**旧方案分析**：
```
只看权重聚合值: 0.00098
无法知道：
- 这个SNP的权重分布如何？
- 权重高的样本表型是否也高？
- 这个SNP与表型有关联吗？
```

**新方案分析**：
```
Utility计算:
  mean = 0.00098
  median = 0.0011
  Utility = log10(0.00098/0.0011) = -0.050
  → 该SNP略低于群体中位数水平

Correlation计算:
  权重向量: [0.0015, 0.0008, 0.0012, 0.0003, 0.0011]
  表型向量: [50, 30, 45, 25, 40]
  Pearson(权重, 表型) = 0.98
  → 权重与表型高度正相关！
  
综合分析:
  - Utility虽然为负，但相关性极高
  - 说明该SNP虽然整体权重不高，但与表型变化强相关
  - 这是一个重要的发现，旧方案无法捕捉！
```

**结论**：新方案通过保留所有样本信息并计算样本间的相关性，能够发现旧方案遗漏的表型关联信号。

### SNP重要性分数的提取逻辑

**旧方案**：
```
多个样本的权重 → 聚合(平均/最大值) → 单个分数
```
- 假设：权重大的SNP就是重要的
- 问题：无法验证该SNP是否真的与表型相关

**新方案**：
```
多个样本的权重 + 对应的表型 → 
  计算Utility（跨样本统计）+ 
  计算Correlation（权重-表型相关性）+ 
  局部富集增强 → 
  综合评分 → 
  经验P值
```
- 假设：重要的SNP应该同时满足：
  1. 在群体中普遍获得高权重（Utility）
  2. 权重变化与表型变化相关（Correlation）
  3. 在染色体上局部聚集（Enrichment）
- 优势：统计严格，可解释性强

### 经验P值的计算基础

**关键理解**：经验P值是基于**秩次统计**的，与具体的分数计算方法无关。

**通用流程**：
1. 对所有SNP计算某种评分（可以是任意评分方法）
2. 将所有SNP的分数从小到大排序，得到秩次
3. 秩次转换为经验P值：$P_i = \frac{S+1-\text{rank}_i}{S+1}$

**在新方案中的应用**：
- 可以对5种不同的评分（Enrichment、Smoothed Corr、Enriched Corr、Combined v1、Combined v2）分别计算经验P值
- 每种评分方法会产生不同的排序，因此P值也不同
- **Combined v2通常产生最可靠的P值**，因为它综合考虑了最多的信息维度

---

## 方法优势

### 旧方案优势与局限

#### ✓ 优势
1. **计算简单快速** - 无需表型标签
2. **适合探索性分析** - 快速筛选候选位点
3. **无需样本对应** - 可以处理不完整的数据
4. **实现直观** - 容易理解和实现

#### ✗ 局限
1. **无法验证表型关联** - 不知道权重大的SNP是否真的与表型相关
2. **丢失样本间变异** - 聚合后无法恢复各样本的权重分布
3. **易受噪声影响** - 如果某个样本有极端权重，会影响mean/max
4. **缺乏统计推断** - 无P值，无显著性评估
5. **忽略空间信息** - 不考虑染色体上SNP的聚集效应

---

### 新方案优势与创新

#### ✓ 核心优势

**1. 双维度验证机制**
- **Utility（普遍性）**：该SNP在群体中是否普遍获得高权重？
- **Correlation（关联性）**：该SNP的权重变化是否与表型变化协同？
- **综合评分**：只有同时满足两个条件的SNP才获得高分

**优势**：避免"假阳性"（权重高但与表型无关的SNP）

**示例**：
```
场景A: 高Utility, 低Correlation
  → 该SNP在所有样本中都获得高权重（普遍）
  → 但权重变化与表型无关（可能是模型偏好的背景信号）
  → 新方案评分: 低（Utility × 0.1 = 低）
  → 旧方案评分: 高（只看权重大小）
  → 结论: 新方案避免了假阳性

场景B: 中Utility, 高Correlation  
  → 该SNP的权重中等（不算很普遍）
  → 但权重变化与表型高度相关（特异性关联）
  → 新方案评分: 中高（0.5 × 0.9 = 0.45）
  → 旧方案评分: 中（只看平均权重）
  → 结论: 新方案发现了隐藏的表型关联

场景C: 高Utility, 高Correlation
  → 该SNP既普遍重要又与表型相关
  → 新方案评分: 很高（0.9 × 0.9 = 0.81）
  → 旧方案评分: 高（只看权重大小）
  → 结论: 两者都能发现，但新方案更有信心
```

**2. 保留完整样本信息**
- 不聚合，保留每个样本的权重向量
- 利用样本间变异计算相关性
- 可以追溯每个样本对SNP重要性的贡献

**优势**：信息无损，可以进行更深入的样本级分析

**示例**：
```
旧方案：
  SNP_456的分数 = 0.0012（无法知道来源）

新方案：
  SNP_456的权重分布:
    样本1: 0.0015 (表型=50, 高贡献)
    样本2: 0.0008 (表型=30, 低贡献)
    样本3: 0.0013 (表型=48, 高贡献)
  
  可以分析:
    - 高表型样本对该SNP的权重更高（验证关联）
    - 可以排查异常样本（样本2权重异常低）
    - 可以进行亚群分析（不同表型范围）
```

**3. 统计严格性**
- **无分布假设**：基于秩次统计，不假设正态分布
- **经验P值**：提供显著性评估，避免主观判断
- **鲁棒性**：对异常值和数据分布不敏感

**优势**：科学可靠，适合发表

**4. 空间感知能力**
- **局部富集**：考虑染色体上相邻SNP的聚集
- **区域信号增强**：如果某个区域有多个相关SNP，都会被增强
- **孤立噪声抑制**：孤立的高分SNP会被相对抑制

**优势**：符合真实基因组的连锁不平衡特性

**示例**：
```
染色体区域1: 
  SNP_100: Utility=0.8, Corr=0.7, 附近有5个高分SNP
  → Enrichment增强: 0.8 × (1+0.6) = 1.28 → 归一化=1.0
  → 富集后评分更高

染色体区域2:
  SNP_500: Utility=0.8, Corr=0.7, 附近都是低分SNP
  → Enrichment增强: 0.8 × (1+0.2) = 0.96 → 归一化=0.75
  → 富集后评分较低（可能是噪声）
```

**5. 多层评分体系**
- 提供5种评分方法（Enrichment、Smoothed Corr、Enriched Corr、Combined v1、Combined v2）
- 可以根据研究目的选择合适的评分
- Combined v2综合所有信息，推荐使用

**优势**：灵活性高，适应不同研究需求

---

### 新旧方案的能力对比

| 能力维度 | 旧方案 | 新方案 | 新方案优势体现 |
|---------|--------|--------|--------------|
| **发现普遍重要的SNP** | ✓ | ✓✓ | 新方案更准确（考虑分布偏态） |
| **发现表型相关的SNP** | ✗ | ✓✓ | 新方案独有（计算相关性） |
| **区分假阳性** | ✗ | ✓✓ | 新方案可区分"权重高但无关"的SNP |
| **发现特异性关联** | ✗ | ✓✓ | 新方案可发现"权重中等但高度相关"的SNP |
| **统计显著性** | ✗ | ✓✓ | 新方案提供经验P值 |
| **空间聚集感知** | ✗ | ✓✓ | 新方案考虑染色体局部富集 |
| **样本级分析** | ✗ | ✓✓ | 新方案保留所有样本信息 |
| **计算效率** | ✓✓ | ✓ | 旧方案更快 |
| **数据要求** | ✓✓ | ✓ | 旧方案无需表型标签 |

---

### 实际应用场景推荐

#### 何时使用旧方案？
1. **快速探索**：初步筛选候选区域
2. **无表型数据**：只有权重数据时
3. **计算资源受限**：需要快速处理大规模数据
4. **非研究目的**：演示、教学

#### 何时使用新方案？（推荐）
1. **正式研究**：需要发表或报告的结果
2. **表型关联分析**：研究基因型-表型关系
3. **统计验证**：需要P值和显著性评估
4. **深入分析**：需要理解SNP的作用机制
5. **多维度评估**：需要区分普遍性和特异性

---

### 新方案的科学贡献

**1. 方法学创新**
- 首次将Transformer注意力权重与表型关联分析结合
- 引入局部富集增强到基因组学分析
- 提出多维度综合评分框架

**2. 生物学洞察**
- 可以区分"普遍重要"vs"特异关联"的SNP
- 可以发现连锁不平衡区域的协同效应
- 可以追溯样本级的权重-表型关系

**3. 统计学严谨**
- 无分布假设的经验P值
- 鲁棒的秩次统计
- 考虑空间相关性的富集分析



## 实现细节

### 新方案Python代码片段

```python
from scipy.stats import rankdata, pearsonr
import numpy as np
#### 旧方案（简单聚合）：

**步骤**：
```
输入数据:
  样本1的权重: [w1_snp1, w1_snp2, ..., w1_snpS]
  样本2的权重: [w2_snp1, w2_snp2, ..., w2_snpS]
  ...
  样本B的权重: [wB_snp1, wB_snp2, ..., wB_snpS]

聚合方式:
  对每个SNP j:
    score_j = mean([w1_snpj, w2_snpj, ..., wB_snpj])
    或
    score_j = max([w1_snpj, w2_snpj, ..., wB_snpj])

输出: [score_snp1, score_snp2, ..., score_snpS]
```

**信息损失示例**：
```
SNP_123 的权重分布:
  样本1: 0.0015  (高表型: 50)
  样本2: 0.0008  (低表型: 30)
  样本3: 0.0012  (中表型: 45)
  样本4: 0.0003  (低表型: 25)
  样本5: 0.0011  (中表型: 40)

旧方案只保留:
  mean = 0.00098 或 max = 0.0015
  
丢失的信息:
  ❌ 不知道这5个权重值的分布
  ❌ 不知道权重与表型的关系
  ❌ 无法验证该SNP是否与表型相关
```

---

#### 新方案（双维度综合）：

**步骤**：
```
输入数据:
  权重矩阵: W[B, S] - 每行是一个样本的所有SNP权重
  表型向量: y[B] - 每个样本的表型值

对每个SNP i:
  
  步骤1 - 提取该SNP的跨样本权重向量:
    wi = [W[1,i], W[2,i], ..., W[B,i]]  ← B个样本对该SNP的权重
  
  步骤2 - 计算Utility（群体统计特征）:
    mean_i = mean(wi)                    ← 平均权重
    median_i = median(wi)                ← 中位数权重
    Utility_i = log10(mean_i / median_i) ← 相对重要性
  
  步骤3 - 计算Correlation（与表型的关联）:
    Corr_i = |Pearson(wi, y)|            ← 权重-表型相关性
    其中 y = [y1, y2, ..., yB] 是表型向量
  
  步骤4 - 综合评分:
    Score_i = Utility_i × Corr_i         ← 双维度乘积

输出: 
  - utility[S]: 每个SNP的普遍性
  - correlation[S]: 每个SNP与表型的相关性
  - score[S]: 综合评分
```

**信息利用示例**（同样的SNP_123）：
```
SNP_123 的完整数据:
  权重向量: wi = [0.0015, 0.0008, 0.0012, 0.0003, 0.0011]
  表型向量: y  = [50,     30,     45,     25,     40]

新方案分析:
  
  1. Utility计算:
     mean = 0.00098
     median = 0.0011
     Utility = log10(0.00098/0.0011) = -0.050
     → 该SNP略低于群体中位数
  
  2. Correlation计算:
     排序观察:
       权重 ↑: 0.0003 < 0.0008 < 0.0011 < 0.0012 < 0.0015
       表型 ↑: 25     < 30     < 40     < 45     < 50
       → 明显正相关！
     
     Pearson(wi, y) = 0.98
     → 权重与表型高度正相关
  
  3. 综合评分:
     虽然Utility为负（不算很普遍），
     但相关性极高（0.98），
     说明该SNP是表型的重要预测因子！
  
  4. 结论:
     ✓ 这是一个重要的SNP
     ✓ 它的重要性来自于"与表型的关联"而非"普遍性"
     ✓ 旧方案只看mean=0.00098会认为不重要
     ✓ 新方案发现了隐藏的表型关联信号
```

---

#### 关键差异总结

| 维度 | 旧方案 | 新方案 |
|------|--------|--------|
| **多样本信息利用** | 聚合成单个值 | 保留完整向量 |
| **统计深度** | 一阶统计（mean/max） | 二阶统计（mean, median, correlation） |
| **表型关联** | 无法检测 | 显式计算 |
| **信息损失** | 丢失样本间变异 | 保留所有信息 |
| **发现能力** | 只能发现"权重大"的SNP | 能发现"权重-表型相关"的SNP |

**核心优势**：新方案通过保留所有样本的权重向量和表型向量，能够计算它们之间的**样本间相关性**（across-sample correlation），这是旧方案通过聚合完全丢失的信息。 / (S + 1)
    neg_log10_p = -np.log10(empirical_p)
    
    return {
        'utility': utility,
        'correlation': correlation,
        'score': score,
        'empirical_p': empirical_p,
        'neg_log10_p': neg_log10_p
    }


def compute_enrichment_score(meta_df, utility_scores, window_size=50):
    """
    局部富集增强（针对Utility）
    
    meta_df: DataFrame包含['Chromosome', 'Position']
    utility_scores: [S] - Utility分数
    window_size: 窗口大小（每侧SNP数量）
    """
    df = meta_df.copy()
    df['utility'] = utility_scores
    df['chr_num'] = pd.to_numeric(df['Chromosome'], errors='coerce').fillna(1).astype(int)
    df = df.sort_values(['chr_num', 'Position']).reset_index(drop=True)
    
    n = len(df)
    enrichment = np.zeros(n)
    
    for chr_val in df['chr_num'].unique():
        mask = df['chr_num'] == chr_val
        idxs = np.where(mask)[0]
        chr_utility = df.loc[mask, 'utility'].values
        
        for i, idx in enumerate(idxs):
            # 窗口范围
            start = max(0, i - window_size)
            end = min(len(idxs), i + window_size + 1)
            window_utility = chr_utility[start:end]
            
            # 计算局部密度（top 50%的平均值）
            top_k = max(1, len(window_utility) // 2)
            top_vals = np.partition(window_utility, -top_k)[-top_k:]
            local_density = np.mean(top_vals)
            
            # 应用富集
            enrichment[idx] = chr_utility[i] * (1 + local_density)
    
    # 归一化
    if enrichment.max() > 0:
        enrichment /= enrichment.max()
    
    return enrichment
```

---

## 参考文献

本方法结合了以下领域的思想：
1. **GWAS曼哈顿图**：经验P值和可视化标准
2. **RNA-seq差异表达分析**：Log-Fold-Change normalization
3. **注意力机制解释**：Transformer attention weights as feature importance
4. **非参数统计**：Rank-based inference without distribution assumptions
5. **空间基因组学**：局部富集分析（enrichment analysis）

---

## 附录：术语表

| 术语 | 英文 | 含义 |
|------|------|------|
| 普遍性 | Utility | 位点在多数样本中的重要程度（群体统计量） |
| 相关性 | Correlation | 位点权重与表型的关联强度（权重-表型协同性） |
| 富集 | Enrichment | 考虑局部染色体聚集的增强分数 |
| 综合分数 | Combined Score | 多维度综合评分（Utility × Correlation × Enrichment） |
| 经验P值 | Empirical P-value | 基于秩次的分布自由P值估计 |
| 秩次 | Rank | 分数的排序位置 |
| LFC | Log-Fold-Change | 对数倍数变化，相对于中位数 |
| 窗口大小 | Window Size | 局部富集计算时考虑的相邻SNP数量 |
| 平滑相关性 | Smoothed Correlation | 相对于局部背景归一化的相关性 |
| 富集相关性 | Enriched Correlation | 考虑局部聚集增强的相关性 |

---

## 关键FAQ

### Q1: 为什么新方案需要表型标签，旧方案不需要？

**A**: 
- **旧方案**：仅依赖权重大小，假设"权重大=重要"，无需验证
- **新方案**：通过计算权重-表型相关性，验证权重变化是否真的与表型相关

### Q2: 多个样本的SNP重要性分数是如何提取的？

**旧方案**：
```
样本1的权重: [w1_snp1, w1_snp2, ..., w1_snpS]
样本2的权重: [w2_snp1, w2_snp2, ..., w2_snpS]
...
样本B的权重: [wB_snp1, wB_snp2, ..., wB_snpS]

聚合 → 每个SNP一个分数: [score_snp1, score_snp2, ..., score_snpS]
```

**新方案**：
```
样本1: 权重向量 + 表型值y1
样本2: 权重向量 + 表型值y2
...
样本B: 权重向量 + 表型值yB

对每个SNP i:
  - Utility_i = log(mean(所有样本的w_i) / median(所有样本的w_i))
  - Corr_i = Pearson([w1_i, w2_i, ..., wB_i], [y1, y2, ..., yB])
  - Score_i = Utility_i × Corr_i
```

**关键差异**：
- 旧方案：跨样本聚合后丢失样本间变异信息
- 新方案：保留所有样本信息，利用样本间变异计算相关性

### Q3: 经验P值是如何从分数转换的？

**步骤**：
1. 所有SNP的分数：`[score_1, score_2, ..., score_S]`
2. 排序得到秩次：分数最大的SNP → rank=1，分数最小的SNP → rank=S
3. 转换为P值：`P_i = rank_i / (S+1)`

**解释**：
- P值表示"有多少比例的SNP分数≥当前SNP"
- 例如：rank=10, S=1000 → P=10/1001≈0.01 → 该SNP排在前1%

### Q4: 为什么Combined v2推荐？

**Combined v2** = Enrichment × (1 + |Enriched Corr|)

**优势**：
1. **Enrichment**：考虑Utility的局部富集
2. **Enriched Corr**：考虑Correlation的局部富集
3. **乘法形式**：$(1 + |\cdot|)$ 保证不会压制Enrichment信号
4. **多维度**：同时考虑普遍性、相关性和空间聚集

**对比其他方法**：
- Base Score：不考虑空间信息
- Enrichment：不考虑相关性
- Combined v1：只对Utility做富集，Correlation只做平滑
- Combined v2：两个维度都做富集（最全面）

---

**文档版本**: 2.0  
**最后更新**: 2026年1月17日  
**作者**: DNA Whisper团队  

如有疑问，请参考代码实现：
- 旧方案：`notebooks/drawattentionweights.ipynb`
- 新方案：`notebooks/drawattentionweights_enrichment.ipynb`
