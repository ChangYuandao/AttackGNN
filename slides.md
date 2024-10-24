---
theme: academic
coverAuthor: "常守豪"
---



## AttackGNN:  Red-Teaming GNNs in Hardware Security Using Reinforcement Learning

<span class="subtitle-footer">2024 USENIX</span>

<div style="position: absolute; right: 20px; bottom: 20px;">
  <img src="/src/HNU.svg" style="width: 100px; height: 100px;">
</div>

<style>
.subtitle-footer {
  display: block;
  text-align: right;
  font-size: 12px;
  margin-top: -10px;
  color: #666;
  padding-right: 70px; /* 调整这个值来控制距离右侧的距离 */
}
</style>
---
layout: center
---


# Introduction

---


<v-clicks depth="2">

- ### Threats Due to Globalized IC Supply Chain
  - Modern computing systems heavily rely on integrated circuits (ICs), which serve as their foundation
  - The cost of employing such advanced foundries is exorbitant
  - These companies (such as NVIDIA and Apple) rely on external, overseas foundries for the production of their integrated circuits, which can raise issues related to trust and security.

- ### Impact of Hardware Security Problems 
  -  IP piracy
  -  HTs
  -  Reverse engineering
  -  Hardware obfuscation

- ### Graph Neural Networks in Hardware Security
  -  Demonstrating cutting-edge performance in IP piracy identification, HT detection and localization, circuit reverse engineering, and hardware obfuscation breaking, among other areas.
  -   <span style="color:red;">Problem: These techniques have not been evaluated thoroughly!</span>
</v-clicks>

<!--
现代计算系统严重依赖于集成电路（IC）。为了实现IC的高性能和低功耗，必须能够使用更小更快的晶体管，它们是IC的基本组成部分。持续不断地缩小晶体管的需求要求使用最先进的制造设施，通常称为晶圆厂。
然而，使用这样先进的晶圆厂的成本极其高昂。
为了解决设计成本的挑战并克服严格的上市时间限制，像英伟达和苹果这样的著名IC设计公司采取了无工厂模式运营。他们将IC制造外包给海外第三方晶圆厂，这引入了潜在的信任问题。
比如说这种分布式的供应链安排导致了许多安全问题，包括知识产权盗版和被称为硬件木马的恶意逻辑插入等。

为了确保硬件安全的过程，研究人员最近利用图神经网络进行了一系列与硬件安全相关的任务，展示出了在识别知识产权盗版、检测和定位硬件特洛伊木马、反向工程电路[11]以及破解硬件混淆技术等方面的最先进性能。然而，在使用基于GNN的技术保障硬件安全方面存在一个重要的差距：这些技术尚未得到彻底评估。特别是，针对基于机器学习系统的对抗性攻击的威胁极其恶劣，必须得到有效理解和缓解。

例如，如果一个未经过对抗性鲁棒性彻底评估的GNN用于检测知识产权盗版，它可能会错误地将一个盗版电路分类为非盗版，这对知识产权设计公司来说可能造成巨大损失
-->
--- 
layout: figure-side
figureUrl: "/src/figure1.png"
figureX: "l"
figureCaption: "High-level overview of the proposed RL-based adversarial example attack against GNNs in hardware security"
---

### AttackGNN 

<v-clicks depth="2">

- Which performs a thorough evaluation of the GNN-based techniques in hardware security
- Devise adversarial examples for problems 
  - ##### detecting IP Piracy
  - ##### detecting/localizing HTs
  - ##### reverse engineering circuits
  - ##### breaking hardware obfuscation techniques for protecting circuit functionality
- Challenges
  - ##### simply brute-forcing all combinations of perturbations is clearly impossible
  - ##### testing adversarially-perturbed large circuits is also expensive
  - ##### different circuits have vastly different structures

</v-clicks>

<!--

所以这篇论文的作者提出了一种叫做AttackGNN的方法来填补上述研究空白，AttackGNN对基于GNN的技术在硬件安全中的应用进行了全面评估。为此，针对硬件安全中的各种问题设计了对抗性实例，即电路，这些问题包括(i) 检测知识产权盗版，(ii) 检测/定位硬件特洛伊木马（HTs），(iii) 反向工程电路，以及 (iv) 破解用于保护电路功能的硬件混淆技术。

由于应用领域是硬件安全，所以对对抗性实例生成的挑战有进一步的要求。处理的是需要遵守设计规则约束的电路，而不是任意图形，传统的基于扰动的对抗性实例技术，如添加/删除边、注入节点或修改特征，在我们的案例中并不适用。此外，典型的电路包含数千个门，即节点，甚至更多的线，即边。如此庞大的电路设计空间使得问题更加具有挑战性。简单地暴力尝试所有扰动组合显然是不可能的。例如，如果我们仅限于删除两个边的扰动，在一个有1000条边的小电路中，可能的组合就有1000选2 = 499,500种。

另一个在处理如此大规模电路时需要考虑的实际问题是，对它们进行操作的成本很高。例如，编译大型电路可能需要几分钟的时间。因此，为了确保一种实用的技术，对于此类电路而言，需要在运行时间和有效性之间找到平衡。同样，测试对抗性扰动后的大型电路也是昂贵的，因为基于GNN的工具需要更多时间来分析它们。

此外，不同的电路结构差异很大。例如，加密电路与加法器电路相比，将拥有非常不同的门和它们之间的连接。这意味着对一种电路有效的扰动可能在其他电路上的表现不佳。
-->
---
layout: figure-side
figureUrl: "/src/figure1.png"
figureX: "l"
figureCaption: "High-level overview of the proposed RL-based adversarial example attack against GNNs in hardware security"
---

### AttackGNN 

<v-clicks depth="2">

- Solution
  - ##### We address these hurdles by modeling the adversarial example generation problem as a Markov decision process (MDP) and solving it using reinforcement learning (RL).
- Improvement aboout RL agent
  - ##### designing effective and generalizable actions
  - ##### sparse rewards for faster training
  - ##### enabling multi-task learning using contextual MDPs
- <span style="color:red;">A first-of-its-kind RL-based adversarial example generation technique, AttackGNN for GNNs used in hardware security </span>

</v-clicks>

<!--
我们通过将对抗性实例生成问题建模为马尔可夫决策过程（MDP）并使用强化学习（RL）来解决这一问题，以应对上述挑战。

强化学习在大型设计空间探索中展现出了巨大的潜力，在未知和不确定的问题空间里通过导航，找到最优或接近最优的解决方案。然而，直接应用强化学习不足以生成高质量的对抗性实例。因此，我们在三个方面对RL代理进行了和优化：1）设计有效且可泛化的动作，即对电路的功能保持扰动；2）稀疏奖励以加速训练，即提高对更大规模电路的适应能力；3）使用上下文MDP启用多任务学习，即单一RL代理能够生成对抗所有GNN的成功对抗性实例。

开发了一种基于强化学习（RL）的对抗性实例生成技术，AttackGNN，这是首次专门针对用于硬件安全的图神经网络（GNNs）而设计的技术。
-->

---
layout: center
---

# Background

---

# Graph Neural Networks 
<v-clicks depth="1">

- GNNs used in classification tasks learn representations of nodes in a graph by repeatedly aggregating and transforming the information (i.e., features) from their neighbor nodes
- After a fixed number of aggregation iterations, the aggregated features are reduced by taking  their sums, averages, or maximums
- The reduced outputs are passed to a classifier (e.g., a two-layer fully-connected network) for final classification

</v-clicks>

<!--
图神经网络（GNNs）作为一种强大的框架，已经崭露头角，用于分析和建模由图表示的结构化数据。通常，用于分类任务的GNNs通过重复聚合和转换来自邻居节点的信息（即特征）来学习图中节点的表示。经过固定次数的聚合迭代后，聚合的特征通过求和、平均或取最大值等方式进行简化。简化的输出被传递给分类器（例如，两层全连接网络）以完成最终的分类。
-->
---
layout: two-cols
---
# Reinforcement Learning


<v-clicks depth="1">

- RL is a powerful framework in the field of artificial intelligence that enables an agent to learn and make sequential decisions in dynamic environments through interaction and feedback
- Rooted in the concept of learning from rewards, RL employs an iterative process where an agent interacts with an environment, receives feedback in the form of rewards, and adjusts its behavior to maximize cumulative rewards over time

</v-clicks>


::right::

<v-clicks depth="1">

- By learning an optimal policy (a function that maps state-action pairs to probabilities of selecting a particular action in a given state), the RL agent aims to make informed decisions in different states to maximize its long-term rewards
- This learning paradigm is particularly well-suited for solving Markov decision processes (MDPs), which are mathematical models used to represent decision-making problems with sequential interactions

</v-clicks>

<!--
强化学习（RL）是人工智能领域的一个强大框架，它使代理能够在动态环境中通过交互和反馈学习并作出序列决策。基于从奖励中学习的概念，RL采用了一个迭代过程，其中智能体与环境互动，以奖励的形式接收反馈，并调整其行为以随时间最大化累积奖励。通过学习一个最优策略，RL代理旨在在不同状态下作出明智的决策，以最大化其长期奖励。这种学习范式特别适合解决马尔可夫决策过程（MDPs）。
-->

---
layout: center
---

# Threat Model

---

<v-clicks depth="2">

- Attacker's Capacity.
  - ##### The adversarial attack happens after the model has been trained
  - ##### The model is fixed and the adversary cannot change the model parameters or structure
- Attacker's Abilities
  - ##### The attacker can introduce arbitrary perturbations, albeit those perturbations cannot change the functionality of the circuit, and he/she cannot violate circuit design rules
- Attacker's Knowledge
  - ##### The attacker does not have access to the model's parameters or training labels 
  - ##### He/she can only perform black-box queries for output scores or labels
- Attacker's Goal
  - ##### The attacker aims to generate input samples (i.e., circuits) that result in misclassification by the target GNN model

</v-clicks>

<!--
我们考虑了一种标准且广泛使用的对抗性实例生成威胁模型。为此做出以下假设：

攻击者的容量：对抗性攻击发生在模型训练之后。模型是固定的，攻击者不能改变模型参数或结构。具体来说，攻击者不能毒害模型或在模型中注入后门。

攻击者的能力：攻击者可以引入任意的扰动，但这些扰动不能改变电路的功能，也不能违反电路设计规则。这些扰动包括但不限于任何组合的添加/删除边、注入节点等，只要最终的扰动电路保持原始功能并且不违反电路设计规则即可。

攻击者的知识：攻击者的知识指的是攻击者对其打算攻击的模型所知的信息量。我们假设一个黑盒设置。攻击者无法访问模型的参数或训练标签。他/她只能执行黑盒查询以获取输出分数或标签。

攻击者的目标：攻击者旨在生成导致目标GNN模型误分类的输入样本（即电路）。例如，当目标模型是一种用于检测两个输入电路之间是否存在知识产权盗版的基于GNN的技术时，对于任何原始电路，通过对其进行扰动,攻击者的目标是创建一个被盗版版本的电路，从而使误将其分类为“未被盗版”。

本工作的目标不是提出新的技术用于插入/检测硬件木马（HTs）、检测/规避知识产权盗版或反向工程。换句话说，AttackGNN并不是针对硬件安全技术的攻击或防御。相反，它是对用于硬件安全的GNNs的攻击，无论是用于恶意行为还是善意行为。
-->
---
layout: center
---

# Methodology

---
layout: figure-side
figureUrl: "/src/figure2.png"
figureX: "l"
figureCaption: "Illustration of why existing adversarial example generation techniques are inappropriate for our case"
---

### Limitations of Existing Adversarial Example Generation Techniques

- Perturbation techniques use one or more of the following four approaches
  - ##### adding edges
  - ##### deleting edges
  - ##### injecting nodes
  - ##### modifying features
- Cannot be used for hardware security problems that operate on Boolean circuits
  - ##### these perturbations affect the functionality of the circuit
  - ##### they may also lead to violations of circuit design rules
  - ##### they violate our threat model

<!--
在GNN的背景下，对抗性实例指的是故意制作以欺骗GNN预测的输入。通常，GNN中的对抗性实例是通过向输入图数据中引入扰动生成的。这些扰动技术使用了以下四种方法中的一种或多种：添加边、删除边、注入节点和修改特征。

然而，这样的扰动不能用于操作布尔电路的硬件安全问题，因为(i)这些扰动会影响电路的功能，(ii)它们也可能导致违反电路设计规则，和/或(iii)它们违反了我们的威胁模型。

例如，图2使用了一个全加器电路及其图表示.扰动在图中添加了一条边（用红色显示）。但是这样做(i)改变了电路的功能，(ii)导致一个门输入有两个驱动器，这违反了电路设计规则。同样，删除边或注入节点的扰动技术也不能直接应用于硬件安全中使用的GNN。此外，修改特征不适用，suiran 这种扰动技术既不会改变电路的功能也不会导致设计规则违规，但我们的威胁模型阻止我们直接控制节点特征。

由于这些广泛使用的针对GNN的扰动技术在我们的案例中不适用，我们设计了一种新的方法来生成满足不改变电路功能、不违反设计规则约束以及不违反我们的模型的对抗性实例。
-->

---
layout: two-cols
---

# Preliminary Formulation

- State $\mathcal{S}$
  - ##### the set of all possible values of the state vector
  - ##### $S_t$ = _[# inputs, # outputs, # gates, # wires, # AND gates, # OR gates, # NAND gates, # NOR gates, # INV gates, # BUF gates, # XOR gates, # XNOR gates, # other gates]_ 
- Action $\mathcal{A}$
  - ##### the set of the following functionalitypreserving transformations in ABC
  - ##### _{refactor, rewrite, resub, balance, refactor -z, rewrite -z, resub -z}_

::right::

- State transition $P(s_{t+1}|a_t,s_t)$
  - ##### the probability that action $a_t$ in state $s_t$ leads to the state $s_{t+1}$
  - ##### $P(s_{t+1}|a_t,s_t)=\begin{cases} 1, if ABC(s_t,a_t)= s_{t+1} \\0,otherwise \\\end{cases}$
- Reward function $R(s_t,a_t)=r_t$
  - ##### $R(s_t,a_t)=r_t=\begin{cases} \alpha, if \quad GNN4IP(s_t,s_{t+1}) = not \quad pirated \\0,otherwise \\\end{cases}$
- Discount factor $\gamma(0 \le \gamma \le 1)$
  - ##### indicates the importance of future rewards relative to the current reward

<!--
我们必须设计一系列不会改变电路功能但仍然会导致误分类的扰动。
这里，由于我们使用开源综合工具ABC，我们使用其基本电路变换，如重构（refactor）、重构-z（refactor -z）、重子替换（resub）、平衡（balance）等，来引起功能保持型的扰动。当这些变换应用于电路时，它们会改变电路的结构但不会改变其功能。

为了找到导致GNN4IP误分类的最佳变换序列，我们设计了一个RL代理，该代理可以找到解决MDP（马尔可夫决策过程）的最佳策略。

奖励的设计使代理尝试以最少的扰动次数成功逃避 GNN4IP 的检测。
-->

---
layout: two-cols
---

# Effective and Generalizable Actions

- Challenge:Ineffective and Specific Actions
  - ##### they do not change the state significantly for several of the circuits
  - ##### they are specific to the ABC synthesis tool, resulting in virtually zero compatibility with other open-source

::right::

- Solution
  - ##### devise novel, more effective actions
  <img src="/src/figure3.png" />

<!--
挑战：无效和特定的动作。

初步的形式化仅依赖于ABC综合工具提供的变换来扰动电路。这些变换（即上面定义的动作）存在两个问题：

1、对于许多电路而言，它们不会显著改变状态。因此，GNN4IP很容易检测到原始电路与盗版电路之间的结构相似性。
2、它们特定于ABC综合工具，导致与其他开源及工业标准的商业综合工具几乎完全不兼容。

例如，如果选择了动作a1，则综合工具被允许使用2输入的AND（表中的AND2）、OR、NAND和NOR标准单元，以及XOR、XNOR、INV和BUF标准单元，但不允许使用3或更高输入的AND、OR、NAND和NOR标准单元。因此，如果在状态中选择了动作a1，那么将不会包含3输入的AND、OR、NAND和NOR标准单元。

简单地将这10个新动作添加到之前提到的8个动作（重构、重写...等）中，将会导致代理的动作空间变得极其庞大。因此，为了减少动作空间，对于每个剧集中的T步，我们只让代理在10个标准单元策略中选择一个，并应用三个固定的变换（如果使用ABC的话）：重写、平衡、重构（按此顺序）。
-->

---
layout: two-cols
---

# Sparse Rewards for Faster Training

- Challenge:Unnecessary Reward Computations
  - ##### this involves loading the trained model, parsing the original and current circuits, and performing a forward pass of the GNN
  - ##### this time-intensive reward computation slows the RL training process dramatically

::right::

- Solution
  - ##### the strategy of computing rewards only at the end of the episode instead of at each step of the episode
  <img src="/src/figure4.png" />

<!--
挑战：不必要的奖励计算

初步形式化面临的另一个挑战是它涉及到在每一步都进行奖励计算。由于这涉及到加载训练模型、解析原始和当前电路以及执行GNN的前向传播，因此至少需要几秒钟的时间。由于RL代理通常需要几千步，甚至几万步才能学会，这种耗时的奖励计算极大地减慢了RL训练过程。

为了减少训练时间，我们采用了仅在每个episode结束时而非每个步骤中计算奖励的策略。这样做减少了奖励计算的频率，从而在训练期间每集所需的时间大大减少。需要注意的是，在episode结束时计算奖励而不是在每一步中计算，可能会影响代理的性能，即可能导致次优收敛。然而，我们的结果显示，我们的代理仍然能够收敛到一个有效的策略，即它学会了生成成功的对抗性实例。
-->
---
layout: two-cols
---

# Multi-Task Learning

- Challenge:MDP Specific to one GNN
  - ##### if we wish to target other GNN techniques, we would need to devise separate MDPs, each with their separate RL agents
  - ##### training separate RL agents for different tasks is not ideal because each RL agent would be independent and would require training from scratch instead of learning knowledge common among different tasks
  - ##### this would result in a large runtime to generate adversarial examples against all the GNNs, limiting the scalability of our technique

::right::

- Solution
  - ##### devise a contextual Markov decision process (CMDP) formulation that can enable multi-task learning by a single RL agent
  - ##### A CMDP is denoted as a tuple  $(\mathcal{C},\mathcal{S},\mathcal{A},\mathcal{M(c)})$
    - ##### $\mathcal{C}$ is the set of one-hot encoded binary strings, one for each GNN we target. four GNNs (GNN4IP, TrojanSAINT, GNN-RE, and OMLA). $\mathcal{C}$ = {1000, 0100, 0010, 0001}
    - ##### $\mathcal{S}$ is the state space
    - ##### $\mathcal{A}$ is the action space
    - ##### $\mathcal{M(c)}=(\mathcal{S},\mathcal{A},\mathcal{P}^c(s_{t+1}|a_t,s_t),R^c(s_t,a_t),\gamma^c)$ 
      - ##### TrojanSAINT:$R(s_T,a_T)=r_T=1-\alpha_{TS}(s_{T+1})$
      - ##### GNN-RE:$R(s_T,a_T)=r_T=1-\alpha_{RE}(s_{T+1})$
      - ##### OMLA:$R(s_T,a_T)=r_T=e^{-5|0.5-\alpha_{OMLA}(s_{T+1})|}$

<!--
挑战：特定于单个GNN的MDP

到目前为止，我们已经构建了一个MDP，当由RL代理解决时，可以生成针对GNN4IP的对抗性实例。然而，这个MDP是特定于GNN4IP的。如果我们希望针对其他GNN技术，我们需要为每个GNN设计单独的MDP，每个MDP都需要独立的RL代理。换句话说，我们将有不同的RL代理来学习不同的任务，即生成针对不同GNN的对抗性实例。然而，为不同的任务分别训练RL代理并不是理想的情况，因为每个RL代理都是独立的，需要从零开始训练，而不是学习不同任务之间的共通知识。这将导致生成针对所有GNN的对抗性实例的运行时间大幅增加，限制了我们技术的可扩展性。为了克服这一挑战，我们需要设计一个能够学习不同任务的单一RL代理，即生成针对所有GNN的成功对抗性实例。


αTS(N) 是TrojanSAINT在输入的HT感染电路N上的性能，根据[52]测量为真阳性和真阴性率的平均值。

αRE(N) 是训练好的GNN-RE分类器的准确率，该分类器接受电路N作为输入，并返回N中节点的标签（“加法器”，“减法器”，“比较器”，“乘法器”，或“控制逻辑”）。


αOMLA(N) 是训练好的OMLA GNN的关键预测准确率，
-->>

---
layout: figure
figureUrl: "/src/figure5.jpg"
figureCaption: "Final AttackGNN architecture"
---
# Final Formulation

<!--

展示了我们针对硬件安全中GNN的RL代理的最终架构。对于每个episode，代理从随机挑选的电路和随机挑选的目标GNN（图中为GNN4IP）开始，并根据由神经网络参数化的策略采取行动。根据所采取的行动,使用适当的综合工具（ABC/Synopsys Design Compiler/Cadence Genus）编译以生成代理的下一个状态。然后，代理选择另一个行动，如此循环。这个循环重复T次，构成了一个episode。在episode结束时，使用该episode选择的GNN评估最终状态，以产生代理的奖励。在固定大小的一批episode之后，近端策略优化（PPO）算法将奖励转化为损失，这些损失由Adam优化器用于更新构成代理的神经网络的参数。经过多次这样的参数更新批次后，奖励饱和，神经网络收敛，导致代理学习到一个最优或接近最优的策略，以生成针对所有目标GNN的成功对抗性实例。
-->

---
layout: center
---

# Results

---
layout: figure
figureUrl: "/src/figure6.jpg"
figureCaption: "Number of successful AttackGNN-generated adversarial circuits against GNN4IP (higher values: better attack)"
---
# Success Against GNN4IP

<!--
展示了我们的RL代理为GNN4IP训练集中每个31个电路找到的成功对抗性电路的数量。正如结果所示，AttackGNN轻松生成了许多针对GNN4IP的成功对抗性电路。
-->

---
layout: figure
figureUrl: "/src/figure7.png"
figureCaption: "Top: Number of successful AttackGNN-generated adversarial circuits against TrojanSAINT (higher values: better attack). Bottom: Distribution of TrojanSAINT’s scores for those adversarial circuits (lower values: better attack)"
---
# Success Against TrojanSAINT

<!--
展示了由AttackGNN生成的成功对抗性电路数量（顶部）以及这些电路的TrojanSAINT评分分布（底部）。从图中可以观察到，即使AttackGNN没有针对17种不同的TrojanSAINT GNN分别进行训练，它仍然能够轻松生成大量的成功对抗性电路来对抗TrojanSAINT。
-->

---
layout: figure
figureUrl: "/src/figure8.png"
figureCaption: "Top: Number of AttackGNN-generated successful adversarial circuits against GNN-RE (higher values: better attack) Bottom: Distribution of GNN-RE’s accuracy for those adversarial circuits (lower values: better attack)"
---
# Success Against GNN-RE

<!--
展示了由AttackGNN生成的针对GNN-RE的成功对抗性实例数量（顶部）以及这些对抗性电路的GNN-RE准确率分布（底部）。电路标签的命名规则在表4中进行了说明。尽管AttackGNN的单一RL代理在训练过程中对GNN-RE看到的电路进行了扰动，但它成功地欺骗了GNN-RE，导致GNN-RE的准确率显著下降
-->

---
layout: figure
figureUrl: "/src/figure9.png"
figureCaption: "Top: Number of successful adversarial circuits against OMLA (higher values: better attack). Bottom: OMLA’s accuracy for them (values near 0.5: better attack)"
---
# Success Against OMLA

<!--
展示了由AttackGNN生成的针对OMLA的成功对抗性电路数量（顶部）以及这些对抗性电路的OMLA密钥预测准确率（KPA）分布（底部）。图中包括了四个电路集：c1355、c1908、c2670和c3540。这两个图表清楚地展示了AttackGNN在扰动混淆电路方面的成功，使得OMLA的性能不比随机猜测更好。
-->

---
layout: figure
figureUrl: "/src/figure10.png"
figureCaption: "Confusion matrix for GNN4TJ predictions"
---
# Success Against GNN4TJ

<!--
仔细观察图中的混淆矩阵可以发现一个有趣的见解：GNN4TJ的假阳性率为15/(15+0)=100%。换句话说，GNN4TJ将所有电路都分类为含有硬件木马。
-->

