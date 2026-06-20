# Learning Latent Dynamics for Planning from Pixels (PlaNet)

- **Authors:** Hafner, Lillicrap, Fischer, Villegas, Ha, Lee, Davidson
- **arXiv:** 1811.04551 (2019)
- **Source PDF:** bibliography/planet.pdf
- **Extracted:** 2026-06-20 via `pymupdf4llm.to_markdown`
- **Why here:** The latent world-model + CEM-MPC planning shape this repo follows.

---

**Learning Latent Dynamics for Planning from Pixels** 

**Danijar Hafner**[1 2] **Timothy Lillicrap**[3] **Ian Fischer**[4] **Ruben Villegas**[1 5] **David Ha**[1] **Honglak Lee**[1] **James Davidson**[1] 

## **Abstract** 

Planning has been very successful for control tasks with known environment dynamics. To leverage planning in unknown environments, the agent needs to learn the dynamics from interactions with the world. However, learning dynamics models that are accurate enough for planning has been a long-standing challenge, especially in image-based domains. We propose the Deep Planning Network (PlaNet), a purely model-based agent that learns the environment dynamics from images and chooses actions through fast online planning in latent space. To achieve high performance, the dynamics model must accurately predict the rewards ahead for multiple time steps. We approach this using a latent dynamics model with both deterministic and stochastic transition components. Moreover, we propose a multi-step variational inference objective that we name latent overshooting. Using only pixel observations, our agent solves continuous control tasks with contact dynamics, partial observability, and sparse rewards, which exceed the difficulty of tasks that were previously solved by planning with learned models. PlaNet uses substantially fewer episodes and reaches final performance close to and sometimes higher than strong model-free algorithms. 

## **1. Introduction** 

Planning is a natural and powerful approach to decision making problems with known dynamics, such as game playing and simulated robot control (Tassa et al., 2012; Silver et al., 2017; Moravˇcík et al., 2017). To plan in unknown environments, the agent needs to learn the dynamics from experience. Learning dynamics models that are accurate 

> 1Google Brain 2University of Toronto 3DeepMind 4Google Research[5] University of Michigan. Correspondence to: Danijar Hafner <mail@danijar.com>. 

_Proceedings of the 36[th] International Conference on Machine Learning_ , Long Beach, California, PMLR 97, 2019. Copyright 2019 by the author(s). 

enough for planning has been a long-standing challenge. Key difficulties include model inaccuracies, accumulating errors of multi-step predictions, failure to capture multiple possible futures, and overconfident predictions outside of the training distribution. 

Planning using learned models offers several benefits over model-free reinforcement learning. First, model-based planning can be more data efficient because it leverages a richer training signal and does not require propagating rewards through Bellman backups. Moreover, planning carries the promise of increasing performance just by increasing the computational budget for searching for actions, as shown by Silver et al. (2017). Finally, learned dynamics can be independent of any specific task and thus have the potential to transfer well to other tasks in the environment. 

Recent work has shown promise in learning the dynamics of simple low-dimensional environments (Deisenroth & Rasmussen, 2011; Gal et al., 2016; Amos et al., 2018; Chua et al., 2018; Henaff et al., 2018). However, these approaches typically assume access to the underlying state of the world and the reward function, which may not be available in practice. In high-dimensional environments, we would like to learn the dynamics in a compact latent space to enable fast planning. The success of such latent models has previously been limited to simple tasks such as balancing cartpoles and controlling 2-link arms from dense rewards (Watter et al., 2015; Banijamali et al., 2017). 

In this paper, we propose the Deep Planning Network (PlaNet), a model-based agent that learns the environment dynamics from pixels and chooses actions through online planning in a compact latent space. To learn the dynamics, we use a transition model with both stochastic and deterministic components. Moreover, we experiment with a novel generalized variational objective that encourages multi-step predictions. PlaNet solves continuous control tasks from pixels that are more difficult than those previously solved by planning with learned models. 

Key contributions of this work are summarized as follows: 

- **Planning in latent spaces** We solve a variety of tasks from the DeepMind control suite, shown in Figure 1, by learning a dynamics model and efficiently planning in 

**Learning Latent Dynamics for Planning from Pixels** 

**==> picture [74 x 73] intentionally omitted <==**

**==> picture [73 x 73] intentionally omitted <==**

**==> picture [74 x 73] intentionally omitted <==**

**==> picture [74 x 73] intentionally omitted <==**

**==> picture [74 x 73] intentionally omitted <==**

**==> picture [74 x 73] intentionally omitted <==**

**==> picture [454 x 9] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) Cartpole (b) Reacher (c) Cheetah (d) Finger (e) Cup (f) Walker<br>**----- End of picture text -----**<br>


Figure 1: Image-based control domains used in our experiments. The images show agent observations before downscaling to 64 _×_ 64 _×_ 3 pixels. (a) The cartpole swingup task has a fixed camera so the cart can move out of sight. (b) The reacher task has only a sparse reward. (c) The cheetah running task includes both contacts and a larger number of joints. (d) The finger spinning task includes contacts between the finger and the object. (e) The cup task has a sparse reward that is only given once the ball is caught. (f) The walker task requires balance and predicting difficult interactions with the ground when the robot is lying down. 

its latent space. Our agent substantially outperforms the model-free A3C and in some cases D4PG algorithm in final performance, with on average 200 _×_ less environment interaction and similar computation time. 

## **Algorithm 1:** Deep Planning Network (PlaNet) 

|**Input :**<br>_R_ Action repeat<br>_S_ Seed episodes|_p_(_st | st−_1_, at−_1) <br>_p_(_ot | st_)|Transition model<br>Observation model|
|---|---|---|
|_C_ Collect interval<br>_B_ Batch size|_p_(_rt | st_)<br>_q_(_st | o≤t, a<t_)|Reward model<br>Encoder|
|_L_ Chunk length|_p_(_ϵ_)|Exploration noise|
|_α_ Learning rate|||



- **Recurrent state space model** We design a latent dynamics model with both deterministic and stochastic components (Buesing et al., 2018; Chung et al., 2015). Our experiments indicate having both components to be crucial for high planning performance. 

**1** Initialize dataset _D_ with _S_ random seed episodes. 

   - **2** Initialize model parameters _θ_ randomly. 

- **Latent overshooting** We generalize the standard variational bound to include multi-step predictions. Using **3** only terms in latent space results in a fast regularizer that can improve long-term predictions and is compati- **4** ble with any latent sequence model. **5** 

   - **3 while** _not converged_ **do** 

      - // Model fitting 

      - **for** _update step s_ = 1 _..C_ **do** 

         - Draw sequence chunks _{_ ( _ot, at, rt_ ) _[L] t_ =[+] _k[k][}] i[B]_ =1 _[∼D]_ uniformly at random from the dataset. 

         - Compute loss _L_ ( _θ_ ) from Equation 3. Update model parameters _θ ← θ − α∇θL_ ( _θ_ ). 

**6 7** 

## **2. Latent Space Planning** 

To solve unknown environments via planning, we need to model the environment dynamics from experience. PlaNet does so by iteratively collecting data using planning and training the dynamics model on the gathered data. In this section, we introduce notation for the environment and describe the general implementation of our model-based agent. In this section, we assume access to a learned dynamics model. Our design and training objective for this model are detailed in Section 3. Section 3.. 

- // Data collection 

- 1 _←_ env.reset() 

**8** does so by iteratively collecting data using planning and training the dynamics model on the gathered data. In this **9** section, we introduce notation for the environment and de- **10** scribe the general implementation of our model-based agent. In this section, we assume access to a learned dynamics **11** model. Our design and training objective for this model are **12** detailed in Section 3. Section 3.. **13 Problem setup** Since individual image observations gen- **14** erally do not reveal the full state of the environment, we **15** consider a partially observable Markov decision process **16** (POMDP). We define a discrete time step _t_ , hidden states _st_ , image observations _ot_ , continuous action vectors _at_ , and scalar rewards _rt_ , that follow the stochastic dynamics 

- **for** _time step t_ = 1 _.._ � _RT_ � **do** Infer belief over current state _q_ ( _st | o≤t, a<t_ ) from the history. 

_at ←_ planner( _q_ ( _st | o≤t, a<t_ ) _, p_ ), see Algorithm 2 in the appendix for details. Add exploration noise _ϵ ∼ p_ ( _ϵ_ ) to the action. **for** _action repeat k_ = 1 _..R_ **do** _rt[k][, o][k] t_ +1 _[←]_[env.step(] _[a][t]_[)] _rt, ot_ +1 _←_[�] _[R] k_ =1 _[r] t[k][, o][R] t_ +1 _D ←D ∪{_ ( _ot, at, rt_ ) _[T] t_ =1 _[}]_ 

where we assume a fixed initial state _s_ 0 without loss of generality. The goal is to implement a policy p( _at | o≤t, a<t_ ) that maximizes the expected sum of rewards Ep�� _Tt_ =1 _[r][t]_ �, where the expectation is over the distributions of the environment and the policy. 

Transition function: _st ∼_ p( _st | st−_ 1 _, at−_ 1) Observation function: _ot ∼_ p( _ot | st_ ) (1) Reward function: _rt ∼_ p( _rt | st_ ) Policy: _at ∼_ p( _at | o≤t, a<t_ ) _,_ 

2 

**Learning Latent Dynamics for Planning from Pixels** 

**Model-based planning** PlaNet learns a transition model _p_ ( _st | st−_ 1 _, at−_ 1), observation model _p_ ( _ot | st_ ), and reward model _p_ ( _rt | st_ ) from previously experienced episodes (note italic letters for the model compared to upright letters for the true dynamics). The observation model provides a rich training signal but is not used for planning. We also learn an encoder _q_ ( _st | o≤t, a<t_ ) to infer an approximate belief over the current hidden state from the history using filtering. Given these components, we implement the policy as a planning algorithm that searches for the best sequence of future actions. We use model-predictive control (MPC; Richards, 2005) to allow the agent to adapt its plan based on new observations, meaning we replan at each step. In contrast to model-free and hybrid reinforcement learning algorithms, we do not use a policy or value network. 

**Experience collection** Since the agent may not initially visit all parts of the environment, we need to iteratively collect new experience and refine the dynamics model. We do so by planning with the partially trained model, as shown in Algorithm 1. Starting from a small amount of _S_ seed episodes collected under random actions, we train the model and add one additional episode to the data set every _C_ update steps. When collecting episodes for the data set, we add small Gaussian exploration noise to the action. To reduce the planning horizon and provide a clearer learning signal to the model, we repeat each action _R_ times, as common in reinforcement learning (Mnih et al., 2015; 2016). 

**Planning algorithm** We use the cross entropy method (CEM; Rubinstein, 1997; Chua et al., 2018) to search for the best action sequence under the model, as outlined in Algorithm 2. We decided on this algorithm because of its robustness and because it solved all considered tasks when given the true dynamics for planning. CEM is a populationbased optimization algorithm that infers a distribution over action sequences that maximize the objective. As detailed in Algorithm 2 in the appendix, we initialize a time-dependent diagonal Gaussian belief over optimal action sequences _at_ : _t_ + _H ∼_ Normal( _µt_ : _t_ + _H , σt_[2] : _t_ + _H_[I][)][, where] _[ t]_[ is the current] time step of the agent and _H_ is the length of the planning horizon. Starting from zero mean and unit variance, we repeatedly sample _J_ candidate action sequences, evaluate them under the model, and re-fit the belief to the top _K_ action sequences. After _I_ iterations, the planner returns the mean of the belief for the current time step, _µt_ . Importantly, after receiving the next observation, the belief over action sequences starts from zero mean and unit variance again to avoid local optima. 

To evaluate a candidate action sequence under the learned model, we sample a state trajectory starting from the current state belief, and sum the mean rewards predicted along the sequence. Since we use a population-based optimizer, 

we found it sufficient to consider a single trajectory per action sequence and thus focus the computational budget on evaluating a larger number of different sequences. Because the reward is modeled as a function of the latent state, the planner can operate purely in latent space without generating images, which allows for fast evaluation of large batches of action sequences. The next section introduces the latent dynamics model that the planner uses. 

## **3. Recurrent State Space Model** 

For planning, we need to evaluate thousands of action sequences at every time step of the agent. Therefore, we use a recurrent state-space model (RSSM) that can predict forward purely in latent space, similar to recently proposed models (Karl et al., 2016; Buesing et al., 2018; Doerr et al., 2018). This model can be thought of as a non-linear Kalman filter or sequential VAE. Instead of an extensive comparison to prior architectures, we highlight two findings that can guide future designs of dynamics models: our experiments show that both stochastic and deterministic paths in the transition model are crucial for successful planning. In this section, we remind the reader of latent state-space models and then describe our dynamics model. 

> **Latent dynamics** We consider sequences _{ot, at, rt}[T] t_ =1 with discrete time step _t_ , image observations _ot_ , continuous action vectors _at_ , and scalar rewards _rt_ . A typical latent state-space model is shown in Figure 2b and resembles the structure of a partially observable Markov decision process. It defines the generative process of the images and rewards using a hidden state sequence _{st}[T] t_ =1[,] 

**==> picture [228 x 41] intentionally omitted <==**

where we assume a fixed initial state _s_ 0 without loss of generality. The transition model is Gaussian with mean and variance parameterized by a feed-forward neural network, the observation model is Gaussian with mean parameterized by a deconvolutional neural network and identity covariance, and the reward model is a scalar Gaussian with mean parameterized by a feed-forward neural network and unit variance. Note that the log-likelihood under a Gaussian distribution with unit variance equals the mean squared error up to a constant. 

**Variational encoder** Since the model is non-linear, we cannot directly compute the state posteriors that are needed for parameter learning. Instead, we use an encoder _q_ ( _s_ 1: _T | o_ 1: _T , a_ 1: _T_ ) =[�] _[T] t_ =1 _[q]_[(] _[s][t][|][s][t][−]_[1] _[, a][t][−]_[1] _[, o][t]_[)][ to infer approx-] imate state posteriors from past observations and actions, where _q_ ( _st | st−_ 1 _, at−_ 1 _, ot_ ) is a diagonal Gaussian with mean and variance parameterized by a convolutional neural 

3 

**Learning Latent Dynamics for Planning from Pixels** 

**==> picture [436 x 129] intentionally omitted <==**

**----- Start of picture text -----**<br>
a 1 a 2<br>a 1 a 2 a 1 a 2 h 1 h 2 h 3<br>h 1 h 2 h 3 s 1 s 2 s 3 s 1 s 2 s 3<br>o 1 , r 1 o 2 , r 2 o 3 , r 3 o 1 , r 1 o 2 , r 2 o 3 , r 3 o 1 , r 1 o 2 , r 2 o 3 , r 3<br>(a) Deterministic model (RNN) (b) Stochastic model (SSM) (c) Recurrent state-space model (RSSM)<br>**----- End of picture text -----**<br>


Figure 2: Latent dynamics model designs. In this example, the model observes the first two time steps and predicts the third. Circles represent stochastic variables and squares deterministic variables. Solid lines denote the generative process and dashed lines the inference model. (a) Transitions in a recurrent neural network are purely deterministic. This prevents the model from capturing multiple futures and makes it easy for the planner to exploit inaccuracies. (b) Transitions in a state-space model are purely stochastic. This makes it difficult to remember information over multiple time steps. (c) We split the state into stochastic and deterministic parts, allowing the model to robustly learn to predict multiple futures. 

network followed by a feed-forward neural network. We use the filtering posterior that conditions on past observations since we are ultimately interested in using the model for planning, but one may also use the full smoothing posterior during training (Babaeizadeh et al., 2017; Gregor & Besse, 2018). 

**Training objective** Using the encoder, we construct a variational bound on the data log-likelihood. For simplicity, we write losses for predicting only the observations — the reward losses follow by analogy. The variational bound obtained using Jensen’s inequality is 

**==> picture [229 x 79] intentionally omitted <==**

For the derivation, please see Equation 8 in the appendix. Estimating the outer expectations using a single reparameterized sample yields an efficient objective for inference and learning in non-linear latent variable models that can be optimized using gradient ascent (Kingma & Welling, 2013; Rezende et al., 2014; Krishnan et al., 2017). 

**Deterministic path** Despite its generality, the purely stochastic transitions make it difficult for the transition model to reliably remember information for multiple time steps. In theory, this model could learn to set the variance to zero for some state components, but the optimization procedure may not find this solution. This motivates including a deterministic sequence of activation vectors _{ht}[T] t_ =1[that] allow the model to access not just the last state but all previous states deterministically (Chung et al., 2015; Buesing et al., 2018). We use such a model, shown in Figure 2c, that 

we name recurrent state-space model (RSSM), 

**==> picture [228 x 55] intentionally omitted <==**

where _f_ ( _ht−_ 1 _, st−_ 1 _, at−_ 1) is implemented as a recurrent neural network (RNN). Intuitively, we can understand this model as splitting the state into a stochastic part _st_ and a deterministic part _ht_ , which depend on the stochastic and deterministic parts at the previous time step through the RNN. We use the encoder _q_ ( _s_ 1: _T | o_ 1: _T , a_ 1: _T_ ) =[�] _[T] t_ =1 _[q]_[(] _[s][t][|][h][t][, o][t]_[)] to parameterize the approximate state posteriors. Importantly, all information about the observations must pass through the sampling step of the encoder to avoid a deterministic shortcut from inputs to reconstructions. 

In the next section, we identify a limitation of the standard objective for latent sequence models and propose a generalization of it that improves long-term predictions. 

## **4. Latent Overshooting** 

In the previous section, we derived the typical variational bound for learning and inference in latent sequence models (Equation 3). As show in Figure 3a, this objective function contains reconstruction terms for the observations and KLdivergence regularizers for the approximate posteriors. A limitation of this objective is that the stochastic path of the transition function _p_ ( _st | st−_ 1 _, at−_ 1) is only trained via the KL-divergence regularizers for one-step predictions: the gradient flows through _p_ ( _st | st−_ 1 _, at−_ 1) directly into _q_ ( _st−_ 1) but never traverses a chain of multiple _p_ ( _st | st−_ 1 _, at−_ 1). In this section, we generalize this variational bound to _latent overshooting_ , which trains all multi-step predictions in latent space. We found that several dynamics models benefit 

4 

**Learning Latent Dynamics for Planning from Pixels** 

**==> picture [387 x 142] intentionally omitted <==**

**----- Start of picture text -----**<br>
s 3 | 1 s 3 | 1<br>s 2 | 1 s 3 | 2 s 2 | 1 s 3 | 2 s 2 | 1 s 3 | 2<br>s 1 | 1 s 2 | 2 s 3 | 3 s 1 | 1 s 2 | 2 s 3 | 3 s 1 | 1 s 2 | 2 s 3 | 3<br>o 1 , r 1 o 2 , r 2 o 3 , r 3 o 1 , r 1 o 2 , r 2 o 3 , r 3 o 1 , r 1 o 2 , r 2 o 3 , r 3<br>(a) Standard variational bound (b) Observation overshooting (c) Latent overshooting<br>**----- End of picture text -----**<br>


Figure 3: Unrolling schemes. The labels _si|j_ are short for the state at time _i_ conditioned on observations up to time _j_ . Arrows pointing at shaded circles indicate log-likelihood loss terms. Wavy arrows indicate KL-divergence loss terms. (a) The standard variational objectives decodes the posterior at every step to compute the reconstruction loss. It also places a KL on the prior and posterior at every step, which trains the transition function for one-step predictions. (b) Observation overshooting (Amos et al., 2018) decodes all multi-step predictions to apply additional reconstruction losses. This is typically too expensive in image domains. (c) Latent overshooting predicts all multi-step priors. These state beliefs are trained towards their corresponding posteriors in latent space to encourage accurate multi-step predictions. 

from latent overshooting, although our final agent using the RSSM model does not require it (see Appendix D). 

**Limited capacity** If we could train our model to make perfect one-step predictions, it would also make perfect multistep predictions, so this would not be a problem. However, when using a model with limited capacity and restricted distributional family, training the model only on one-step predictions until convergence does in general not coincide with the model that is best at multi-step predictions. For successful planning, we need accurate multi-step predictions. Therefore, we take inspiration from Amos et al. (2018) and earlier related ideas (Krishnan et al., 2015; Lamb et al., 2016; Chiappa et al., 2017), and train the model on multistep predictions of all distances. We develop this idea for latent sequence models, showing that multi-step predictions can be improved by a loss in latent space, without having to generate additional images. 

**Multi-step prediction** We start by generalizing the standard variational bound (Equation 3) from training one-step predictions to training multi-step predictions of a fixed distance _d_ . For ease of notation, we omit actions in the conditioning set here; every distribution over _st_ is conditioned upon _a<t_ . We first define multi-step predictions, which are computed by repeatedly applying the transition model and integrating out the intermediate states, 

**==> picture [228 x 47] intentionally omitted <==**

The case _d_ = 1 recovers the one-step transitions used in the original model. Given this definition of a multi-step predic- 

tion, we generalize Equation 3 to the variational bound on the multi-step predictive distribution _pd_ , 

**==> picture [224 x 90] intentionally omitted <==**

For the derivation, please see Equation 9 in the appendix. Maximizing this objective trains the multi-step predictive distribution. This reflects the fact that during planning, the model makes predictions without having access to all the preceding observations. 

We conjecture that Equation 6 is also a lower bound on ln _p_ ( _o_ 1: _T_ ) based on the data processing inequality. Since the latent state sequence is Markovian, for _d ≥_ 1 we have I( _st_ ; _st−d_ ) _≤_ I( _st_ ; _st−_ 1) and thus E[ln _pd_ ( _o_ 1: _T_ )] _≤_ E[ln _p_ ( _o_ 1: _T_ )]. Hence, every bound on the multi-step predictive distribution is also a bound on the one-step predictive distribution in expectation over the data set. For details, please see Equation 10 in the appendix. In the next paragraph, we alleviate the limitation that a particular _pd_ only trains predictions of one distance and arrive at our final objective. 

**Latent overshooting** We introduced a bound on predictions of a given distance _d_ . However, for planning we need accurate predictions not just for a fixed distance but for all distances up to the planning horizon. We introduce latent overshooting for this, an objective function for latent sequence models that generalizes the standard variational bound (Equation 3) to train the model on multi-step predic- 

5 

**Learning Latent Dynamics for Planning from Pixels** 

tions of all distances 1 _≤ d ≤ D_ , 

**==> picture [224 x 65] intentionally omitted <==**

Latent overshooting can be interpreted as a regularizer in latent space that encourages consistency between one-step and multi-step predictions, which we know should be equivalent in expectation over the data set. We include weighting factors _{βd}[D] d_ =1[analogously to the] _[ β]_[-VAE (][Higgins et al.][,] 2016). While we set all _β>_ 1 to the same value for simplicity, they could be chosen to let the model focus more on long-term or short-term predictions. In practice, we stop gradients of the posterior distributions for overshooting distances _d >_ 1, so that the multi-step predictions are trained towards the informed posteriors, but not the other way around. 

## **5. Experiments** 

We evaluate PlaNet on six continuous control tasks from pixels. We explore multiple design axes of the agent: the stochastic and deterministic paths in the dynamics model, iterative planning, and online experience collection. We refer to the appendix for hyper parameters (Appendix A) and additional experiments (Appendices C to E). Besides the action repeat, we use the same hyper parameters for all tasks. Within less than one hundredth the episodes, PlaNet outperforms A3C (Mnih et al., 2016) and achieves similar performance to the top model-free algorithm D4PG (Barth-Maron et al., 2018). The training time of 10 to 20 hours (depending on the task) on a single Nvidia V100 GPU compares favorably to that of A3C and D4PG. Our implementation uses TensorFlow Probability (Dillon et al., 2017). Please visit https://danijar.com/planet for access to the code and videos of the trained agent. 

For our evaluation, we consider six image-based continuous control tasks of the DeepMind control suite (Tassa et al., 2018), shown in Figure 1. These environments provide qualitatively different challenges. The cartpole swingup task requires a long planning horizon and to memorize the cart when it is out of view, reacher has a sparse reward given when the hand and goal area overlap, finger spinning includes contact dynamics between the finger and the object, cheetah exhibits larger state and action spaces, the cup task only has a sparse reward for when the ball is caught, and the walker is challenging because the robot first has to stand up and then walk, resulting in collisions with the ground that are difficult to predict. In all tasks, the only observations are third-person camera images of size 64 _×_ 64 _×_ 3 pixels. 

**Comparison to model-free methods** Figure 4 compares the performance of PlaNet to the model-free algorithms reported by Tassa et al. (2018). Within 100 episodes, PlaNet outperforms the policy-gradient method A3C trained from proprioceptive states for 100,000 episodes, on all tasks. After 500 episodes, it achieves performance similar to D4PG, trained from images for 100,000 episodes, except for the finger task. PlaNet surpasses the final performance of D4PG with a relative improvement of 26% on the cheetah running task. We refer to Table 1 for numerical results, which also includes the performance of CEM planning with the true dynamics of the simulator. 

**Model designs** Figure 4 additionally compares design choices of the dynamics model. We train PlaNet using our recurrent state-space model (RSSM), as well as versions with purely deterministic GRU (Cho et al., 2014), and purely stochastic state-space model (SSM). We observe the importance of both stochastic and deterministic elements in the transition function on all tasks. The deterministic part allows the model to remember information over many time steps. The stochastic component is even more important – the agent does not learn without it. This could be because the tasks are stochastic from the agent’s perspective due to partial observability of the initial states. The noise might also add a safety margin to the planning objective that results in more robust action sequences. 

**Agent designs** Figure 5 compares PlaNet, a version collecting episodes under random actions rather than by planning, and a version that at each environment step selects the best action out of 1000 sequences rather than iteratively refining plans via CEM. We observe that online data collection helps for all tasks and is necessary for the cartpole, finger, and walker tasks. Iterative search for action sequences using CEM improves performance on all tasks. 

**One agent all tasks** Figure 7 in the appendix shows the performance of a single agent trained on all six tasks. The agent is not told which task it is facing; it needs to infer this from the image observations. We pad the action spaces with unused elements to make them compatible and adapt Algorithm 1 to collect one episode of each task every _C_ update steps. We use the same hyper parameters as for the main experiments above. The agent solves all tasks while learning slower compared to individually trained agents. This indicates that the model can learn to predict multiple domains, regardless of the conceptually different visuals. 

## **6. Related Work** 

Previous work in model-based reinforcement learning has focused on planning in low-dimensional state spaces (Gal et al., 2016; Higuera et al., 2018; Henaff et al., 2018; 

6 

**==> picture [488 x 269] intentionally omitted <==**

**----- Start of picture text -----**<br>
Learning Latent Dynamics for Planning from Pixels<br>Cartpole Swing Up Reacher Easy Cheetah Run<br>1000 1000 1000<br>800 800 800<br>600 600 600<br>400 400 400<br>200 200 200<br>0 0 0<br>5 250 500 750 1000 5 250 500 750 1000 5 250 500 750 1000<br>Finger Spin Cup Catch Walker Walk<br>1000 1000 1000<br>800 800 800<br>600 600 600<br>400 400 400<br>200 200 200<br>0 0 0<br>5 250 500 750 1000 5 250 500 750 1000 5 250 500 750 1000<br>PlaNet (RSSM) Stochastic (SSM) D4PG (100k episodes)<br>Deterministic (GRU) A3C (100k episodes, proprio)<br>**----- End of picture text -----**<br>


Figure 4: Comparison of PlaNet to model-free algorithms and other model designs. Plots show test performance over the number of collected episodes. We compare PlaNet using our RSSM (Section 3) to purely deterministic (GRU) and purely stochastic models (SSM). The RNN does not use latent overshooting, as it does not have stochastic latents. The lines show medians and the areas show percentiles 5 to 95 over 5 seeds and 10 trajectories. The shaded areas are large on two of the tasks due to the sparse rewards. 

Table 1: Comparison of PlaNet to the model-free algorithms A3C and D4PG reported by Tassa et al. (2018). The training curves for these are shown as orange lines in Figure 4 and as solid green lines in Figure 6 in their paper. From these, we estimate the number of episodes that D4PG takes to achieve the final performance of PlaNet to estimate the data efficiency gain. We further include CEM planning ( _H_ = 12 _, I_ = 10 _, J_ = 1000 _, K_ = 100) with the true simulator instead of learned dynamics as an estimated upper bound on performance. Numbers indicate mean final performance over 5 seeds and 10 trajectories. 

|**Method**|**Modality**|**Episodes**|**Cartpole**<br>**Swing Up**|**Reacher**<br>**Easy**|**Cheetah**<br>**Run**|**Finger**<br>**Spin**|**Cup**<br>**Catch**|**Walker**<br>**Walk**|
|---|---|---|---|---|---|---|---|---|
|A3C|proprioceptive|100,000|558|285|214|129|105|311|
|D4PG|pixels|100,000|862|967|524|985|980|968|
|PlaNet (ours)|pixels|1,000|821|832|662|700|930|951|
|CEM + true simulator|simulator state|0|850|964|656|825|993|994|
|Data effciency gain PlaNet over D4PG (factor)|||250|40|500+|300|100|90|



Chua et al., 2018), combining the benefits of model-based and model-free approaches (Kalweit & Boedecker, 2017; Nagabandi et al., 2017; Weber et al., 2017; Kurutach et al., 2018; Buckman et al., 2018; Ha & Schmidhuber, 2018; Wayne et al., 2018; Igl et al., 2018; Srinivas et al., 2018), and pure video prediction without planning (Oh et al., 2015; Krishnan et al., 2015; Karl et al., 2016; Chiappa et al., 2017; Babaeizadeh et al., 2017; Gemici et al., 2017; Denton & Fergus, 2018; Buesing et al., 2018; Doerr et al., 2018; Gre- 

gor & Besse, 2018). Appendix G reviews these orthogonal research directions in more detail. 

Relatively few works have demonstrated successful planning from pixels using learned dynamics models. The robotics community focuses on video prediction models for planning (Agrawal et al., 2016; Finn & Levine, 2017; Ebert et al., 2018; Zhang et al., 2018) that deal with the visual complexity of the real world and solve tasks with 

7 

**==> picture [488 x 270] intentionally omitted <==**

**----- Start of picture text -----**<br>
Learning Latent Dynamics for Planning from Pixels<br>Cartpole Swing Up Reacher Easy Cheetah Run<br>1000 1000 1000<br>800 800 800<br>600 600 600<br>400 400 400<br>200 200 200<br>0 0 0<br>5 250 500 750 1000 5 250 500 750 1000 5 250 500 750 1000<br>Finger Spin Cup Catch Walker Walk<br>1000 1000 1000<br>800 800 800<br>600 600 600<br>400 400 400<br>200 200 200<br>0 0 0<br>5 250 500 750 1000 5 250 500 750 1000 5 250 500 750 1000<br>PlaNet Random collection D4PG (100k episodes)<br>Random shooting A3C (100k episodes, proprio)<br>**----- End of picture text -----**<br>


Figure 5: Comparison of agent designs. Plots show test performance over the number of collected episodes. We compare PlaNet, a version that collects data under random actions (random collection), and a version that chooses the best action out of 1000 sequences at each environment step (random shooting) without iteratively refining plans via CEM. The lines show medians and the areas show percentiles 5 to 95 over 5 seeds and 10 trajectories. 

a simple gripper, such as grasping or pushing objects. In comparison, we focus on simulated environments, where we leverage latent planning to scale to larger state and action spaces, longer planning horizons, as well as sparse reward tasks. E2C (Watter et al., 2015) and RCE (Banijamali et al., 2017) embed images into a latent space, where they learn local-linear latent transitions and plan for actions using LQR. These methods balance simulated cartpoles and control 2-link arms from images, but have been difficult to scale up. We lift the Markov assumption of these models, making our method applicable under partial observability, and present results on more challenging environments that include longer planning horizons, contact dynamics, and sparse rewards. 

## **7. Discussion** 

We present PlaNet, a model-based agent that learns a latent dynamics model from image observations and chooses actions by fast planning in latent space. To enable accurate long-term predictions, we design a model with both stochastic and deterministic paths. We show that our agent succeeds at several continuous control tasks from image observations, reaching performance that is comparable to the best model-free algorithms while using 200 _×_ fewer episodes and similar or less computation time. The results 

show that learning latent dynamics models for planning in image domains is a promising approach. 

Directions for future work include learning temporal abstraction instead of using a fixed action repeat, possibly through hierarchical models. To further improve final performance, one could learn a value function to approximate the sum of rewards beyond the planning horizon. Moreover, gradient-based planning could increase the computational efficiency of the agent and learning representations without reconstruction could help to solve tasks with higher visual diversity. Our work provides a starting point for multi-task control by sharing the dynamics model. 

**Acknowledgements** We thank Jacob Buckman, Nicolas Heess, John Schulman, Rishabh Agarwal, Silviu Pitis, Mohammad Norouzi, George Tucker, David Duvenaud, Shane Gu, Chelsea Finn, Steven Bohez, Jimmy Ba, Stephanie Chan, and Jenny Liu for helpful discussions. 

8 

**Learning Latent Dynamics for Planning from Pixels** 

## **References** 

- Agrawal, P., Nair, A. V., Abbeel, P., Malik, J., and Levine, S. Learning to poke by poking: Experiential learning of intuitive physics. In _Advances in Neural Information Processing Systems_ , pp. 5074–5082, 2016. 

- Amos, B., Dinh, L., Cabi, S., Rothörl, T., Muldal, A., Erez, T., Tassa, Y., de Freitas, N., and Denil, M. Learning awareness models. In _International Conference on Learning Representations_ , 2018. 

- Babaeizadeh, M., Finn, C., Erhan, D., Campbell, R. H., and Levine, S. Stochastic variational video prediction. _arXiv preprint arXiv:1710.11252_ , 2017. 

- Banijamali, E., Shu, R., Ghavamzadeh, M., Bui, H., and Ghodsi, A. Robust locally-linear controllable embedding. _arXiv preprint arXiv:1710.05373_ , 2017. 

- Barth-Maron, G., Hoffman, M. W., Budden, D., Dabney, W., Horgan, D., Muldal, A., Heess, N., and Lillicrap, T. Distributed distributional deterministic policy gradients. _arXiv preprint arXiv:1804.08617_ , 2018. 

- Bengio, S., Vinyals, O., Jaitly, N., and Shazeer, N. Scheduled sampling for sequence prediction with recurrent neural networks. In _Advances in Neural Information Processing Systems_ , pp. 1171–1179, 2015. 

- Buckman, J., Hafner, D., Tucker, G., Brevdo, E., and Lee, H. Sample-efficient reinforcement learning with stochastic ensemble value expansion. _arXiv preprint arXiv:1807.01675_ , 2018. 

- Buesing, L., Weber, T., Racaniere, S., Eslami, S., Rezende, D., Reichert, D. P., Viola, F., Besse, F., Gregor, K., Hassabis, D., et al. Learning and querying fast generative models for reinforcement learning. _arXiv preprint arXiv:1802.03006_ , 2018. 

- Chiappa, S., Racaniere, S., Wierstra, D., and Mohamed, S. Recurrent environment simulators. _arXiv preprint arXiv:1704.02254_ , 2017. 

- Cho, K., Van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., and Bengio, Y. Learning phrase representations using rnn encoder-decoder for statistical machine translation. _arXiv preprint arXiv:1406.1078_ , 2014. 

- Chua, K., Calandra, R., McAllister, R., and Levine, S. Deep reinforcement learning in a handful of trials using probabilistic dynamics models. _arXiv preprint arXiv:1805.12114_ , 2018. 

- Chung, J., Kastner, K., Dinh, L., Goel, K., Courville, A. C., and Bengio, Y. A recurrent latent variable model for sequential data. In _Advances in neural information processing systems_ , pp. 2980–2988, 2015. 

- Clevert, D.-A., Unterthiner, T., and Hochreiter, S. Fast and accurate deep network learning by exponential linear units (elus). _arXiv preprint arXiv:1511.07289_ , 2015. 

- Deisenroth, M. and Rasmussen, C. E. Pilco: A model-based and data-efficient approach to policy search. In _Proceedings of the 28th International Conference on machine learning (ICML-11)_ , pp. 465–472, 2011. 

- Denton, E. and Fergus, R. Stochastic video generation with a learned prior. _arXiv preprint arXiv:1802.07687_ , 2018. 

- Dillon, J. V., Langmore, I., Tran, D., Brevdo, E., Vasudevan, S., Moore, D., Patton, B., Alemi, A., Hoffman, M., and Saurous, R. A. Tensorflow distributions. _arXiv preprint arXiv:1711.10604_ , 2017. 

- Doerr, A., Daniel, C., Schiegg, M., Nguyen-Tuong, D., Schaal, S., Toussaint, M., and Trimpe, S. Probabilistic recurrent state-space models. _arXiv preprint arXiv:1801.10395_ , 2018. 

- Ebert, F., Finn, C., Dasari, S., Xie, A., Lee, A., and Levine, S. Visual foresight: Model-based deep reinforcement learning for vision-based robotic control. _arXiv preprint arXiv:1812.00568_ , 2018. 

- Finn, C. and Levine, S. Deep visual foresight for planning robot motion. In _Robotics and Automation (ICRA), 2017 IEEE International Conference on_ , pp. 2786–2793. IEEE, 2017. 

- Gal, Y., McAllister, R., and Rasmussen, C. E. Improving pilco with bayesian neural network dynamics models. In _Data-Efficient Machine Learning workshop, ICML_ , 2016. 

- Gemici, M., Hung, C.-C., Santoro, A., Wayne, G., Mohamed, S., Rezende, D. J., Amos, D., and Lillicrap, T. Generative temporal models with memory. _arXiv preprint arXiv:1702.04649_ , 2017. 

- Gregor, K. and Besse, F. Temporal difference variational auto-encoder. _arXiv preprint arXiv:1806.03107_ , 2018. 

- Ha, D. and Schmidhuber, J. World models. _arXiv preprint arXiv:1803.10122_ , 2018. 

- Henaff, M., Whitney, W. F., and LeCun, Y. Model-based planning with discrete and continuous actions. _arXiv preprint arXiv:1705.07177_ , 2018. 

9 

**Learning Latent Dynamics for Planning from Pixels** 

- Higgins, I., Matthey, L., Pal, A., Burgess, C., Glorot, X., Botvinick, M., Mohamed, S., and Lerchner, A. betavae: Learning basic visual concepts with a constrained variational framework. In _International Conference on Learning Representations_ , 2016. 

- Higuera, J. C. G., Meger, D., and Dudek, G. Synthesizing neural network controllers with probabilistic model based reinforcement learning. _arXiv preprint arXiv:1803.02291_ , 2018. 

- Igl, M., Zintgraf, L., Le, T. A., Wood, F., and Whiteson, S. Deep variational reinforcement learning for pomdps. _arXiv preprint arXiv:1806.02426_ , 2018. 

- Kalchbrenner, N., Oord, A. v. d., Simonyan, K., Danihelka, I., Vinyals, O., Graves, A., and Kavukcuoglu, K. Video pixel networks. _arXiv preprint arXiv:1610.00527_ , 2016. 

- Kalweit, G. and Boedecker, J. Uncertainty-driven imagination for continuous deep reinforcement learning. In _Conference on Robot Learning_ , pp. 195–206, 2017. 

- Karl, M., Soelch, M., Bayer, J., and van der Smagt, P. Deep variational bayes filters: Unsupervised learning of state space models from raw data. _arXiv preprint arXiv:1605.06432_ , 2016. 

- Kingma, D. P. and Ba, J. Adam: A method for stochastic optimization. _arXiv preprint arXiv:1412.6980_ , 2014. 

- Kingma, D. P. and Dhariwal, P. Glow: Generative flow with invertible 1x1 convolutions. _arXiv preprint arXiv:1807.03039_ , 2018. 

- Kingma, D. P. and Welling, M. Auto-encoding variational bayes. _arXiv preprint arXiv:1312.6114_ , 2013. 

- Krishnan, R. G., Shalit, U., and Sontag, D. Deep kalman filters. _arXiv preprint arXiv:1511.05121_ , 2015. 

- Krishnan, R. G., Shalit, U., and Sontag, D. Structured inference networks for nonlinear state space models. In _AAAI_ , pp. 2101–2109, 2017. 

- Kurutach, T., Clavera, I., Duan, Y., Tamar, A., and Abbeel, P. Model-ensemble trust-region policy optimization. _arXiv preprint arXiv:1802.10592_ , 2018. 

- Lamb, A. M., GOYAL, A. G. A. P., Zhang, Y., Zhang, S., Courville, A. C., and Bengio, Y. Professor forcing: A new algorithm for training recurrent networks. In _Advances In Neural Information Processing Systems_ , pp. 4601–4609, 2016. 

- Mathieu, M., Couprie, C., and LeCun, Y. Deep multiscale video prediction beyond mean square error. _arXiv preprint arXiv:1511.05440_ , 2015. 

- Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., Graves, A., Riedmiller, M., Fidjeland, A. K., Ostrovski, G., et al. Human-level control through deep reinforcement learning. _Nature_ , 518(7540): 529, 2015. 

- Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., Silver, D., and Kavukcuoglu, K. Asynchronous methods for deep reinforcement learning. In _International Conference on Machine Learning_ , pp. 1928– 1937, 2016. 

- Moerland, T. M., Broekens, J., and Jonker, C. M. Learning multimodal transition dynamics for model-based reinforcement learning. _arXiv preprint arXiv:1705.00470_ , 2017. 

- Moravˇcík, M., Schmid, M., Burch, N., Lisy, V., Morrill, D.,` Bard, N., Davis, T., Waugh, K., Johanson, M., and Bowling, M. Deepstack: Expert-level artificial intelligence in heads-up no-limit poker. _Science_ , 356(6337):508–513, 2017. 

- Nagabandi, A., Kahn, G., Fearing, R. S., and Levine, S. Neural network dynamics for model-based deep reinforcement learning with model-free fine-tuning. _arXiv preprint arXiv:1708.02596_ , 2017. 

- Nair, V. and Hinton, G. E. Rectified linear units improve restricted boltzmann machines. In _Proceedings of the 27th international conference on machine learning (ICML-10)_ , pp. 807–814, 2010. 

- Oh, J., Guo, X., Lee, H., Lewis, R. L., and Singh, S. Actionconditional video prediction using deep networks in atari games. In _Advances in Neural Information Processing Systems_ , pp. 2863–2871, 2015. 

- Rezende, D. J., Mohamed, S., and Wierstra, D. Stochastic backpropagation and approximate inference in deep generative models. _arXiv preprint arXiv:1401.4082_ , 2014. 

- Richards, A. G. _Robust constrained model predictive control_ . PhD thesis, Massachusetts Institute of Technology, 2005. 

- Rubinstein, R. Y. Optimization of computer simulation models with rare events. _European Journal of Operational Research_ , 99(1):89–112, 1997. 

- Silver, D., Schrittwieser, J., Simonyan, K., Antonoglou, I., Huang, A., Guez, A., Hubert, T., Baker, L., Lai, M., Bolton, A., et al. Mastering the game of go without human knowledge. _Nature_ , 550(7676):354, 2017. 

- Srinivas, A., Jabri, A., Abbeel, P., Levine, S., and Finn, C. Universal planning networks. _arXiv preprint arXiv:1804.00645_ , 2018. 

10 

**Learning Latent Dynamics for Planning from Pixels** 

- Talvitie, E. Model regularization for stable sample rollouts. In _UAI_ , pp. 780–789, 2014. 

- Tassa, Y., Erez, T., and Todorov, E. Synthesis and stabilization of complex behaviors through online trajectory optimization. In _Intelligent Robots and Systems (IROS), 2012 IEEE/RSJ International Conference on_ , pp. 4906– 4913. IEEE, 2012. 

- Tassa, Y., Doron, Y., Muldal, A., Erez, T., Li, Y., Casas, D. d. L., Budden, D., Abdolmaleki, A., Merel, J., Lefrancq, A., et al. Deepmind control suite. _arXiv preprint arXiv:1801.00690_ , 2018. 

- van den Oord, A., Vinyals, O., et al. Neural discrete representation learning. In _Advances in Neural Information Processing Systems_ , pp. 6309–6318, 2017. 

- Venkatraman, A., Hebert, M., and Bagnell, J. A. Improving multi-step prediction of learned time series models. In _AAAI_ , pp. 3024–3030, 2015. 

- Vondrick, C., Pirsiavash, H., and Torralba, A. Generating videos with scene dynamics. In _Advances In Neural Information Processing Systems_ , 2016. 

- Watter, M., Springenberg, J., Boedecker, J., and Riedmiller, M. Embed to control: A locally linear latent dynamics model for control from raw images. In _Advances in neural information processing systems_ , pp. 2746–2754, 2015. 

- Wayne, G., Hung, C.-C., Amos, D., Mirza, M., Ahuja, A., Grabska-Barwinska, A., Rae, J., Mirowski, P., Leibo, J. Z., Santoro, A., et al. Unsupervised predictive memory in a goal-directed agent. _arXiv preprint arXiv:1803.10760_ , 2018. 

- Weber, T., Racanière, S., Reichert, D. P., Buesing, L., Guez, A., Rezende, D. J., Badia, A. P., Vinyals, O., Heess, N., Li, Y., et al. Imagination-augmented agents for deep reinforcement learning. _arXiv preprint arXiv:1707.06203_ , 2017. 

- Zhang, M., Vikram, S., Smith, L., Abbeel, P., Johnson, M., and Levine, S. SOLAR: deep structured representations for model-based reinforcement learning. _arXiv preprint arXiv:1808.09105_ , 2018. 

11 

**Learning Latent Dynamics for Planning from Pixels** 

## **A. Hyper Parameters** 

We use the convolutional and deconvolutional networks from Ha & Schmidhuber (2018), a GRU (Cho et al., 2014) with 200 units as deterministic path in the dynamics model, and implement all other functions as two fully connected layers of size 200 with ReLU activations (Nair & Hinton, 2010). Distributions in latent space are 30-dimensional diagonal Gaussians with predicted mean and standard deviation. 

We pre-process images by reducing the bit depth to 5 bits as in Kingma & Dhariwal (2018). The model is trained using the Adam optimizer (Kingma & Ba, 2014) with a learning rate of 10 _[−]_[3] , _ϵ_ = 10 _[−]_[4] , and gradient clipping norm of 1000 on batches of _B_ = 50 sequence chunks of length _L_ = 50. We do not scale the KL divergence terms relatively to the reconstruction terms but grant the model 3 free nats by clipping the divergence loss below this value. In a previous version of the agent, we used latent overshooting and an additional fixed global prior, but we found this to not be necessary. 

For planning, we use CEM with horizon length _H_ = 12, optimization iterations _I_ = 10, candidate samples _J_ = 1000, and refitting to the best _K_ = 100. We start from _S_ = 5 seed episodes with random actions and collect another episode every _C_ = 100 update steps under _ϵ ∼_ Normal(0 _,_ 0 _._ 3) action noise. The action repeat differs between domains: cartpole ( _R_ = 8), reacher ( _R_ = 4), cheetah ( _R_ = 4), finger ( _R_ = 2), cup ( _R_ = 4), walker ( _R_ = 2). We found important hyper parameters to be the action repeat, the KL-divergence scales _β_ , and the learning rate. 

## **B. Planning Algorithm** 

**Algorithm 2:** Latent planning with CEM 

**Input :** _H_ Planning horizon distance _q_ ( _st | o≤t, a<t_ ) Current state belief _I_ Optimization iterations _p_ ( _st | st−_ 1 _, at−_ 1) Transition model _J_ Candidates per iteration _p_ ( _rt | st_ ) Reward model _K_ Number of top candidates to fit 

**1** Initialize factorized belief over action sequences _q_ ( _at_ : _t_ + _H_ ) _←_ Normal(0 _,_ I). 

**2 for** _optimization iteration i_ = 1 _..I_ **do** // Evaluate _J_ action sequences from the current belief. **3 for** _candidate action sequence j_ = 1 _..J_ **do 4** _at_[(] : _[j] t_[)] + _H[∼][q]_[(] _[a][t]_[:] _[t]_[+] _[H]_[)] **5** _st_[(] : _[j] t_[)] + _H_ +1 _[∼][q]_[(] _[s][t][|][ o]_[1:] _[t][, a]_[1:] _[t][−]_[1][)][ �] _τ[t]_[+] = _[H] t_ +1[+1] _[p]_[(] _[s][τ][|][ s][τ][−]_[1] _[, a]_[(] _τ[j] −_[)] 1[)] **6** _R_[(] _[j]_[)] =[�] _τ[t]_[+] = _[H] t_ +1[+1][E[] _[p]_[(] _[r][τ][|][ s] τ_[(] _[j]_[)][)]] // Re-fit belief to the _K_ best action sequences. **7** _K ←_ argsort( _{R_[(] _[j]_[)] _}[J] j_ =1[)][1:] _[K]_ **8** _µt_ : _t_ + _H_ = _K_[1] � _k∈K[a]_[(] _t_ : _[k] t_[)] + _H[,] σt_ : _t_ + _H_ = _K_ 1 _−_ 1 � _k∈K[|][a]_[(] _t_ : _[k] t_[)] + _H[−][µ][t]_[:] _[t]_[+] _[H][|]_[.] **9** _q_ ( _at_ : _t_ + _H_ ) _←_ Normal( _µt_ : _t_ + _H , σt_[2] : _t_ + _H_[I][)] 

**10 return** _first action mean µt._ 

12 

**Learning Latent Dynamics for Planning from Pixels** 

## **C. Multi-Task Learning** 

**==> picture [172 x 137] intentionally omitted <==**

**----- Start of picture text -----**<br>
Average over tasks<br>1000<br>800<br>600<br>400<br>200<br>0<br>5 250 500 750 1000<br>**----- End of picture text -----**<br>


Figure 6: We compare a single PlaNet agent trained on all tasks to individual PlaNet agents. The plot shows test performance over the number of episodes collected for each task. The single agent learns to solve all the tasks while learning more slowly compared to the individual agents. The lines show mean and one standard deviation over 6 tasks, 5 seeds, and 10 trajectories. 

**==> picture [474 x 282] intentionally omitted <==**

**----- Start of picture text -----**<br>
Cartpole Swing Up Reacher Easy Cheetah Run<br>1000 1000 1000<br>800 800 800<br>600 600 600<br>400 400 400<br>200 200 200<br>0 0 0<br>5 250 500 750 1000 5 250 500 750 1000 5 250 500 750 1000<br>Finger Spin Cup Catch Walker Walk<br>1000 1000 1000<br>800 800 800<br>600 600 600<br>400 400 400<br>200 200 200<br>0 0 0<br>5 250 500 750 1000 5 250 500 750 1000 5 250 500 750 1000<br>Separate agents D4PG (100k episodes)<br>Single agent A3C (100k episodes, proprio)<br>**----- End of picture text -----**<br>


Figure 7: Per-task performance of a single PlaNet agent trained on the six tasks. Plots show test performance over the number of episodes collected per task. The agent is not told which task it is solving and it needs to infer this from the image observations. The agent learns to distinguish the tasks and solve them with just a moderate slowdown in learning. The lines show medians and the areas show percentiles 5 to 95 over 4 seeds and 10 trajectories. 

13 

**Learning Latent Dynamics for Planning from Pixels** 

## **D. Latent Overshooting** 

**==> picture [474 x 282] intentionally omitted <==**

**----- Start of picture text -----**<br>
Cartpole Swing Up Reacher Easy Cheetah Run<br>1000 1000 1000<br>800 800 800<br>600 600 600<br>400 400 400<br>200 200 200<br>0 0 0<br>5 250 500 750 1000 5 250 500 750 1000 5 250 500 750 1000<br>Finger Spin Cup Catch Walker Walk<br>1000 1000 1000<br>800 800 800<br>600 600 600<br>400 400 400<br>200 200 200<br>0 0 0<br>5 250 500 750 1000 5 250 500 750 1000 5 250 500 750 1000<br>RSSM DRNN D4PG (100k episodes)<br>RSSM + Overshooting DRNN + Overshooting A3C (100k episodes, proprio)<br>**----- End of picture text -----**<br>


Figure 8: We compare the standard variational objective with latent overshooting on our proposed RSSM and another model called DRNN that uses two RNNs as encoder and decoder with a stochastic state sequence in between. Latent overshooting can substantially improve the performance of the DRNN and other models we have experimented with (not shown), but slightly reduces performance of our RSSM. The lines show medians and the areas show percentiles 5 to 95 over 5 seeds and 10 trajectories. 

14 

**Learning Latent Dynamics for Planning from Pixels** 

## **E. Activation Function** 

**==> picture [474 x 283] intentionally omitted <==**

**----- Start of picture text -----**<br>
Cartpole Swing Up Reacher Easy Cheetah Run<br>1000 1000 1000<br>800 800 800<br>600 600 600<br>400 400 400<br>200 200 200<br>0 0 0<br>5 250 500 750 1000 5 250 500 750 1000 5 250 500 750 1000<br>Finger Spin Cup Catch Walker Walk<br>1000 1000 1000<br>800 800 800<br>600 600 600<br>400 400 400<br>200 200 200<br>0 0 0<br>5 250 500 750 1000 5 250 500 750 1000 5 250 500 750 1000<br>PlaNet (ReLU) SSM (ReLU) D4PG (100k episodes)<br>PlaNet (ELU) SSM (ELU) A3C (100k episodes, proprio)<br>**----- End of picture text -----**<br>


Figure 9: Comparison of hard ReLU (Nair & Hinton, 2010) and smooth ELU (Clevert et al., 2015) activation functions. We find that smooth activations help improve performance of the purely stochastic model (and the purely deterministic model; not shown) while our proposed RSSM is robust to the choice of activation function. The lines show medians and the areas show percentiles 5 to 95 over 5 seeds and 10 trajectories. 

15 

**Learning Latent Dynamics for Planning from Pixels** 

## **F. Bound Derivations** 

**One-step predictive distribution** The variational bound for latent dynamics models _p_ ( _o_ 1: _T , s_ 1: _T | a_ 1: _T_ ) =[�] _t[p]_[(] _[s][t][|] st−_ 1 _, at−_ 1) _p_ ( _ot | st_ ) and a variational posterior _q_ ( _s_ 1: _T | o_ 1: _T , a_ 1: _T_ ) =[�] _t[q]_[(] _[s][t][|][o][≤][t][, a][<t]_[)][follows][from][importance] weighting and Jensen’s inequality as shown, 

**==> picture [443 x 146] intentionally omitted <==**

> **Multi-step predictive distribution** The variational bound on the _d_ -step predictive distribution _pd_ ( _o_ 1: _T , s_ 1: _T | a_ 1: _T_ ) = � _t[p]_[(] _[s][t][|][s][t][−][d][, a][t][−]_[1][)] _[p]_[(] _[o][t][|][s][t]_[)][and][a][variational][posterior] _[q]_[(] _[s]_[1:] _[T][|][o]_[1:] _[T][ , a]_[1:] _[T]_[ )][=][�] _t[q]_[(] _[s][t][|][o][≤][t][, a][<t]_[)][follows][anal-] ogously. The second bound comes from moving the log inside the multi-step priors, which satisfy the recursion _p_ ( _st | st−d, at−d−_ 1: _t−_ 1) = E _p_ ( _st−_ 1 _|st−d,at−d−_ 1: _t−_ 2)[ _p_ ( _st | st−_ 1 _, at−_ 1)]. 

**==> picture [459 x 179] intentionally omitted <==**

Since all expectations are on the outside of the objective, we can easily obtain an unbiased estimator of this bound by changing expectations to sample averages. 

**Relation between one-step and multi-step predictive distributions** We conjecture that the multi-step predictive distribution _pd_ ( _o_ 1: _T_ ) lower bounds the one-step predictive distribution _p_ ( _o_ 1: _T_ ) of the same latent sequence model model in expectation over the data set. Since the latent state sequence is Markovian, for _d ≥_ 1 we have the data processing inequality 

**==> picture [330 x 56] intentionally omitted <==**

Therefore, any bound on the multi-step predictive distribution, including Equation 9 and Equation 7, is also a bound on the one-step predictive distribution. 

16 

**Learning Latent Dynamics for Planning from Pixels** 

## **G. Additional Related Work** 

**Planning in state space** When low-dimensional states of the environment are available to the agent, it is possible to learn the dynamics directly in state space. In the regime of control tasks with only a few state variables, such as the cart pole and mountain car tasks, PILCO (Deisenroth & Rasmussen, 2011) achieves remarkable sample efficiency using Gaussian processes to model the dynamics. Similar approaches using neural networks dynamics models can solve two-link balancing problems (Gal et al., 2016; Higuera et al., 2018) and implement planning via gradients (Henaff et al., 2018). Chua et al. (2018) use ensembles of neural networks, scaling up to the cheetah running task. The limitation of these methods is that they access the low-dimensional Markovian state of the underlying system and sometimes the reward function. Amos et al. (2018) train a deterministic model using overshooting in observation space for active exploration with a robotics hand. We move beyond low-dimensional state representations and use a latent dynamics model to solve control tasks from images. 

**Hybrid agents** The challenges of model-based RL have motivated the research community to develop hybrid agents that accelerate policy learning by training on imagined experience (Kalweit & Boedecker, 2017; Nagabandi et al., 2017; Kurutach et al., 2018; Buckman et al., 2018; Ha & Schmidhuber, 2018), improving feature representations (Wayne et al., 2018; Igl et al., 2018), or leveraging the information content of the model directly (Weber et al., 2017). Srinivas et al. (2018) learn a policy network with integrated planning computation using reinforcement learning and without prediction loss, yet require expert demonstrations for training. 

**Multi-step predictions** Training sequence models on multi-step predictions has been explored for several years. Scheduled sampling (Bengio et al., 2015) changes the rollout distance of the sequence model over the course of training. Hallucinated replay (Talvitie, 2014) mixes predictions into the data set to indirectly train multi-step predictions. Venkatraman et al. (2015) take an imitation learning approach. Recently, Amos et al. (2018) train a dynamics model on all multi-step predictions at once. We generalize this idea to latent sequence models trained via variational inference. 

**Latent sequence models** Classic work has explored models for non-Markovian observation sequences, including recurrent neural networks (RNNs) with deterministic hidden state and probabilistic state-space models (SSMs). The ideas behind variational autoencoders (Kingma & Welling, 2013; Rezende et al., 2014) have enabled non-linear SSMs that are trained via variational inference (Krishnan et al., 2015). The VRNN (Chung et al., 2015) combines RNNs and SSMs and is trained via variational inference. In contrast to our RSSM, it feeds generated observations back into the model which makes forward predictions expensive. Karl et al. (2016) address mode collapse to a single future by restricting the transition function, (Moerland et al., 2017) focus on multi-modal transitions, and Doerr et al. (2018) stabilize training of purely stochastic models. Buesing et al. (2018) propose a model similar to ours but use in a hybrid agent instead for explicit planning. 

**Video prediction** Video prediction is an active area of research in deep learning. Oh et al. (2015) and Chiappa et al. (2017) achieve visually plausible predictions on Atari games using deterministic models. Kalchbrenner et al. (2016) introduce an autoregressive video prediction model using gated CNNs and LSTMs. Recent approaches introduce stochasticity to the model to capture multiple futures (Babaeizadeh et al., 2017; Denton & Fergus, 2018). To obtain realistic predictions, Mathieu et al. (2015) and Vondrick et al. (2016) use adversarial losses. In simulated environments, Gemici et al. (2017) augment dynamics models with an external memory to remember long-time contexts. van den Oord et al. (2017) propose a variational model that avoids sampling using a nearest neighbor look-up, yielding high fidelity image predictions. These models are complimentary to our approach. 

17 

**Learning Latent Dynamics for Planning from Pixels** 

## **H. Video Predictions** 

**==> picture [320 x 560] intentionally omitted <==**

**----- Start of picture text -----**<br>
Context 6 10 15 20 25 30 35 40 45 50<br>Context 6 10 15 20 25 30 35 40 45 50<br>Context 6 10 15 20 25 30 35 40 45 50<br>Context 6 10 15 20 25 30 35 40 45 50<br>True<br>Model<br>True<br>PlaNet Model<br>True<br>Model<br>True<br>Model<br>True<br>Model<br>True<br>PlaNet + Overshooting<br>Model<br>True<br>Model<br>True<br>Model<br>True<br>Deterministic (GRU)<br>Model<br>True<br>Model<br>True<br>Model<br>Stochastic (SSM) True<br>Model<br>**----- End of picture text -----**<br>


Figure 10: Open-loop video predictions for test episodes. The columns 1–5 show reconstructed context frames and the remaining images are generated open-loop. Our RSSM achieves pixel-accurate predictions for 50 steps into the future in the cheetah environment. We randomly selected action sequences from test episodes collected with action noise alongside the training episodes. 

18 

**Learning Latent Dynamics for Planning from Pixels** 

## **I. State Diagnostics** 

**==> picture [487 x 496] intentionally omitted <==**

Figure 11: Open-loop state diagnostics. We freeze the dynamics model of a PlaNet agent and learn small neural networks to predict the true positions, velocities, and reward of the simulator. The open-loop predictions of these quantities show that most information about the underlying system is present in the learned latent space and can be accurately predicted forward further than the planning horizons used in this work. 

19 

**Learning Latent Dynamics for Planning from Pixels** 

## **J. Planning Parameters** 

**==> picture [487 x 414] intentionally omitted <==**

**----- Start of picture text -----**<br>
fraction=0.05 horizon=6.0 fraction=0.05 horizon=8.0 fraction=0.05 horizon=10.0 fraction=0.05 horizon=12.0 fraction=0.05 horizon=14.0<br>3.0<br>5.0<br>10.0<br>15.0<br>fraction=0.1 horizon=6.0 fraction=0.1 horizon=8.0 fraction=0.1 horizon=10.0 fraction=0.1 horizon=12.0 fraction=0.1 horizon=14.0<br>3.0<br>5.0<br>10.0<br>15.0<br>fraction=0.3 horizon=6.0 fraction=0.3 horizon=8.0 fraction=0.3 horizon=10.0 fraction=0.3 horizon=12.0 fraction=0.3 horizon=14.0<br>3.0<br>5.0<br>10.0<br>15.0<br>fraction=0.5 horizon=6.0 fraction=0.5 horizon=8.0 fraction=0.5 horizon=10.0 fraction=0.5 horizon=12.0 fraction=0.5 horizon=14.0<br>3.0<br>5.0<br>10.0<br>15.0<br>100.0 300.0 500.0 1000.0 100.0 300.0 500.0 1000.0 100.0 300.0 500.0 1000.0 100.0 300.0 500.0 1000.0 100.0 300.0 500.0 1000.0<br>proposals proposals proposals proposals proposals<br>iterations<br>iterations<br>iterations<br>iterations<br>**----- End of picture text -----**<br>


Figure 12: Planning performance on the cheetah running task with the true simulator using different planner settings. Performance ranges from 132 (blue) to 837 (yellow). Evaluating more action sequences, optimizing for more iterations, and re-fitting to fewer of the best proposals tend to improve performance. A planning horizon length of 6 is not sufficient and results in poor performance. Much longer planning horizons hurt performance because of the increased search space. For this environment, best planning horizon length is near 8 steps. 

20 

