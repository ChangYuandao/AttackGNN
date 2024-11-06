---
theme: academic
coverAuthor: "常守豪，S241000742"
---



## AttackGNN:  Red-Teaming GNNs in Hardware Security Using Reinforcement Learning

Vasudev Gohil$^1$,, Satwik Patnaik$^2$, Dileep Kalathil$^1$, Jeyavijayan Rajendran$^1$

$^1$Texas A&M University, USA, $^2$University of Delaware, USA

$^1${gohil.vasudev, dileep.kalathil, jv.rajendran}@tamu.edu, $^2$satwik@udel.edu

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
layout: figure
figureUrl: "/src/Zen_2_Matisse_Ryzen-5-3600.jpg"
figureCaption: "Integrated Circuit"
---

<!--
首先呢，在经济全球化的大背景下，我们知道，像集成电路这种产业，是在全球各个地方进行供应链整合的，
那么基于此呢，万一我在某个环节的运输途中，我对芯片植入木马，或者说我把芯片抄袭了，总而言之呢，这种分布式的供应链就会带来许多的安全问题。

那么我该如何解决这种安全问题呢？

然后就有一部分研究人员，他们发现，使用图神经网络，也就是GNN，在一些硬件安全问题上，比方说识别盗版电路、检测和定位木马等问题上具有不错的表现。
那我现在其实就是可以用GNN来解决这种问题了。

但是呢，GNN在这些安全问题上是不是彻底可靠的，我们不知道，之前的所有研究也都没这么做过。那万一GNN不可靠，我把盗版电路识别成非盗版的，那我的知识产权就非常受损。

所以自然而然，那我这篇文章是干什么的呢，就是开足马力攻击GNN，看看GNN是否表现得还是很稳定。
-->
---
layout: center
---

# Background

---
layout: figure
figureUrl: "/src/Graph_Example.png"
figureCaption: "Trump VS Harris"
---

# What is graph?

<!--
首先呢，我们要介绍一下什么是图神经网络。在此之前，我相信大家都或多或少地了解过什么是神经网络，但是对GNN可能不太熟悉，其实无非就是名字上多了个图。

我们看PPT，PPT上有点，也有线，什么数据可以抽象成图呢，比方说现在Trump和Harris竞选，一个个节点呢表示人，中间这个John假设是trump，Mr.Hi代表哈哈姐，那这些线段就可以表示投票的人更倾向谁这种特征，我们可以看到，更靠近哈哈姐的线段所连接的点是红色的，所以实际生活中，人际关系可以抽象成图。
-->

---
layout: figure-side
figureUrl: "/src/Simple_GNN.png"
figureX: "l"
figureCaption: "A single layer of a simple GNN"
---

# Graph Neural Networks 

- 输入是一个图
- 每个组件（V、E、U）由多层感知机更新生成新图形
- GNN的核心思想是通过信息传播机制，让每个节点在多个迭代中聚合其邻居节点的信息，从而更新自身的特征表示

<!--
那我们既然理解了图，就再讲解一下图神经网络。

图神经网络一般有三个要素，一个是V（代表着节点），一个是E（代表着边），一个是U（也就是这个整体）。

那图和神经网络是如何联系在一起的呢，我们看PPT上左边的图，左边呢是第N层，右边是N+1层，每一层上它的几个基本量也就是U、V、E经过多次感知机，从而完成到一下曾的更新。

然后将单层GNN堆叠在一起，就生成了图神经网络。为了帮助大家更好地理解，给大家演示一下。
-->
---
layout: figure
figureUrl: "/src/GNN_Hardware.png"
figureCaption: "GNN used in hardware security"
---

# GNN used in hardware security(GNN4IP)

<!--
大家可以思考一下，那图神经网络，是如何应用在硬件安全领域上呢，比方说集成芯片。

你想，我们刚刚讲了图，如果某种物体，可以抽象成图，是不是就可以用图神经网络来完成一些事情呢。那电路有什么样的特征呢，

比方说，我现在有个电池和电阻，分别由电压属性和电阻值属性，他们之间有导线。那就很清晰了。图神经网络的工作就是通过边传播信息，例如电流从电池流向电阻，同时根据电阻的阻值更新电流的强度。

所以电路可以被抽象为图，节点代表电路元件，边代表元件之间的连接。

那大家理解了电路怎么变成图的，那图神经网络又是如何实际应用的呢？

简单来说，我们看PPT上的例子，如果电路之间有两个环，我就把它分为一类，而只有一个环的是另一个类别，那么如果原始电路有两个环的话，你再检测到其他电路也有两个环，特征相似，那么我是不是就可以认为它有可能抄袭了我的电路。

到了这里，我相信大家对什么是图神经网络，以及图神经网络是如何在硬件安全上应用有了一个直观的感受。

到这里，不要忘了我们的意图是什么，对，就是疯狂地攻击GNN，也就是生成对抗性样本。

怎么生成对抗性样本呢对于图神经网络而言，一般就是比如说我增加节点或者删除节点，或者改变线的特征。

那么问题又来了，电路规模小还好，但是实际上集成芯片的电路非常精密复杂和庞大，比方说现在有1000条边，我要删除两条边，那么就有奖金50万种组合，显然，这些不能靠人为实现。

所以问题又变成，如何解决这种大型空间探索问题呢？这也就是我认为这篇论文的作者引入强化学习的原因。
-->


---
layout: figure-side
figureUrl: "/src/frozen_lake.gif"
figureX: "l"
figureCaption: "A simple example about reinforcement learning"
---

# Reinforcement Learning

- 自主探索
- 设置目标
- 设置奖励
- 自我学习
- 适用于大型探索问题

<!--

啥是强化学习呢，说白了就是你在地上放个机器人，它自己对周围空间进行探索，它的任务是找到这个盒子，但不过你也能看到，我肯定是不想让它掉进冰窟窿的。所以我就设置一个奖励，或者说惩罚，掉进冰窟窿加10分，到达终点加100分。那现在它就可以随机探索了，人类就不用管了，也就是说到最后，他会学习到一条路径，既不掉进冰窟窿，还能到达终点，人类也不用管。

强化学习简单地理解即使这样子的。

-->
---
layout: figure-side
figureUrl: "/src/Simple_RL.png"
figureX: "l"
figureCaption: "A simple example about reinforcement learning"
---

# Reinforcement Learning


- $\mathcal{S}$ ：状态空间
- $\mathcal{A}$ ：动作空间
- $\mathcal{R}$ ：获得的奖励
- $P(s_{t+1}|a_t,s_t)$ ：状态转移概率
- $\gamma$ ： 折扣因子
- episode ：可以理解为一次训练完成


---
layout: center
---

# Threat Model


---
layout: figure-side
figureUrl: "/src/figure1.png"
figureX: "l"
figureCaption: "High-level overview of the proposed RL-based adversarial example attack against GNNs in hardware security"
---

# What I Want to do?

- AttackGNN

<!--

因为电路可以被抽象为图，所以可以用图神经网络，因为问题空间庞大，所以引入了强化学习。

那我们现在就可以攻击GNN了，正如图上所示，这是本文工作的一个简单概述。原有的图经过一些扰动，当然这个扰动你可以认为是机器人帮我们人类进行改造的，然后生成了一个对抗性样本，这个对抗性样本是盗版电路名单时可以生成同样的图神经网络。

导致本来我该把这个对抗性样本分类为盗版，但是现在不能正确分类了。

那我该怎么具体实现呢？

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

# Limitations

- 现有的GNN扰动技术不适用于布尔电路
  - 添加node、删除node、注入node、修改feature
- 如何既扰动电路，又不违背电路设计规则？
  - 开源工具ABC（A System for Sequential Synthesis and Verification）

<!--

首先呢传统的GNN扰动技术，是没办法应用在布尔电路的，比方说在左边的图中，我在cout后再拉一条线，也就是这个红线，让他变成sum的输入。但这就违背了电路设计规则，因为现在有两个驱动器应用在门输入上。

所以怎么办呢，正好，有一种开源工具ABC可以实现这种功能。
-->

---
layout: figure-side
figureUrl: "/src/figure3.png"
figureX: "l"
figureCaption: "New actions based on allowed/prohibited standard cells"
---

# ABC is not enough

- 无法显著改变电路状态
- 不能与其他开源工具兼容

- Solution ： 图中的十个新动作  
  - 可以明显改变电路状态
  - 基于标准单元的操作，与合成工具无关

<!--
但这就够了吗，还不够。

因为ABC无法显著改变电路状态，也就是说，通过ABC，改变后的电路生成的GNN太容易被识别出来了，但你要知道我们的工作是进行彻底地评估，也就是说我想要找出来那些电路能变得面目全非，但是依然会生成相同的GNN的电路。所以我就必须显著改变电路状态。

另一个问题就是无法与其他开源兼容，这个就比较好理解。

所以本文的作者又设计了10个新的动作，比夫说我在执行a1的时候，实心的点代表使用，空心代表不使用。

所以解决了上述问题，那么我们现在，对电路采取动作这一关，就过去了。
-->

---
layout: figure-side
figureUrl: "/src/bandit.png"
figureX: "l"
---

# How to improve RL agent

- 不必要的reward计算
  - 在每个episode结束之后计算reward
  - 次优收敛，仍然有效
- 只针对特定任务
  - 改变为多任务学习
  - 如何改变？增加上下文空间
  - $\mathcal{C}=\{1000,0100,0010,0001\}$,GNNs={GNN4IP, TrojanSAINT, GNN-RE, and OMLA}
  - 选择任务不同，奖励标准不同

<!--
那么使用强化学习会不会也有一些问题呢？答案当然是肯定的。

首先呢我们每一次会经历非常多的状态，如果我在每个状态结束后都计算奖励，那么每次任务都要浪费一大堆时间进行计算了。那我能怎么改变呢，自然而然我们能想到，我只在每次任务结束后再计算。那你肯定会想，那我最后的学到的策略是最优的吗？作者证明策略是次优的，但是仍然收敛，并且有效，也就是做了一个取舍。

哪还有一个问题，就是本文的工作，肯定不是只针对盗版电路这个硬件安全，GNN还应用在硬件木马、逆转电路、混淆电路等方面，我肯定要对其他方面的任务进行评估。所以强化学习的奖励函数不能只针对IP盗版呀。那怎么办呢？

答案很简单，那我就让机器人，学会多任务呗。

作者用了一个很巧妙的方法，就是在传统的强化学习上加了一个上下文。比方说我当前的C选到的编码为1000，我就执行IP盗版的任务，那我就用该任务下的奖励函数。
-->

---
layout: figure-side
figureUrl: "/src/bandit.png"
figureX: "l"
---


- GNN4IP 

$r_T= 
\begin{cases} 
\alpha, & \text{if GNN4IP}(s_T, s_{T+1}) = not pirated \\
0, & \text{otherwise} 
\end{cases}
$

- TrojanSAINT

$r_T = 1 - \alpha_{TS}(s_{T+1})$

- GNN-RE

$r_T = 1 - \alpha_{RE}(s_{T+1})$

- OMLA

$r_T = e^{-5|0.5 - \alpha_{OMLA}(s_{T+1})|}$


<!--
那我们看一下这个奖励函数。

对于GNN4IP这个任务来说的话，它是知识产权盗版检测的问题，而我所作的事情是想让它的GNN分辨不出来，所以当它无法识别盗版的时候，我给机器人一个奖励。

而木马这个任务，a表示的是真阳性率与真阴性率的比值，而我既然注射了木马，我只关注我能不能正确识别出来，也就是识别木马越厉害，真阳性率越高，a越高，因此奖励越少。

对于逆转电路，这个a是分类器的准确率，自然而然，他能分辨的准确率越高，说明我攻击效果越不好，所以奖励越低。

而OMLA，我们可以看到，当a接近0.5的时候，我们的奖励越高，为什么呢，因为对于混淆电路来说，当它能够正确识别关键位为0或1的时候，那说明我们工作不起效果，所以奖励低。
-->


---
layout: figure
figureUrl: "/src/Final_model.jpg"
figureCaption: "Final AttackGNN architecture"
---

# The final model

---
layout: center
---

# Experimental Results

---
layout: figure-side
figureUrl: "/src/figure6.jpg"
figureX: "l"
---

# Success Against GNN4IP

- 31 个电路中的每一个生成了对抗性示例
- 值越高 -> 攻击越好

---
layout: figure-side
figureUrl: "/src/figure5.png"
figureX: "l"
---

# Success Against GNN4IP

- 对于大多数成功的对抗性电路，GNN4IP的相似性得分明显小于 0

---
layout: figure-side
figureUrl: "/src/figure7.png"
figureX: "l"
---

# Success Against TrojanSAINT

- 标准：$\frac{真阳性率}{真阴性率}$ 低于0.5则为成功
- 顶部：值越高 -> 攻击效果越好
- 底部：值越低 -> 攻击效果越好

---
layout: figure-side
figureUrl: "/src/image.png"
figureX: "l"
---

# Success Against GNN-RE

- 标准：密钥预测精度 (KPA) 来衡量 OMLA 的成功
- 顶部：值越高 -> 攻击效果越好
- 底部：值越接近0.5 -> 攻击效果越好

---
layout: center
---

# Conclusion


---
layout: figure-side
figureUrl: "/src/figure1.png"
figureX: "l"

---

# Some thoughts


- 首个对应用于硬件安全的GNN分类系统进行对抗性样本攻击
- 未来需要更多的工作来避免我们这种攻击