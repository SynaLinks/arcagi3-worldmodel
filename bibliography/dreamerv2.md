# Mastering Atari with Discrete World Models (DreamerV2)

- **Authors:** Hafner, Lillicrap, Norouzi, Ba
- **arXiv:** 2010.02193 (2021)
- **Source PDF:** bibliography/dreamerv2.pdf
- **Extracted:** 2026-06-20 via `pymupdf4llm.to_markdown`
- **Why here:** Origin of the categorical latent + KL balancing we implement.

---

Published as a conference paper at ICLR 2021 

# MASTERING ATARI WITH DISCRETE WORLD MODELS 

**Danijar Hafner** _[∗]_ **Timothy Lillicrap Mohammad Norouzi Jimmy Ba** Google Research DeepMind Google Research University of Toronto 

## ABSTRACT 

Intelligent agents need to generalize from past experience to achieve goals in complex environments. World models facilitate such generalization and allow learning behaviors from imagined outcomes to increase sample-efficiency. While learning world models from image inputs has recently become feasible for some tasks, modeling Atari games accurately enough to derive successful behaviors has remained an open challenge for many years. We introduce DreamerV2, a reinforcement learning agent that learns behaviors purely from predictions in the compact latent space of a powerful world model. The world model uses discrete representations and is trained separately from the policy. DreamerV2 constitutes the first agent that achieves human-level performance on the Atari benchmark of 55 tasks by learning behaviors inside a separately trained world model. With the same computational budget and wall-clock time, Dreamer V2 reaches 200M frames and surpasses the final performance of the top single-GPU agents IQN and Rainbow. DreamerV2 is also applicable to tasks with continuous actions, where it learns an accurate world model of a complex humanoid robot and solves stand-up and walking from only pixel inputs. 

## 1 INTRODUCTION 

## Atari Performance 

To successfully operate in unknown environments, reinforcement learning agents need to learn about their environments over time. World models are an explicit way to represent an agent’s knowledge about its environment. Compared to model-free reinforcement learning that learns through trial and error, world models facilitate generalization and can predict the outcomes of potential actions to enable planning (Sutton, 1991). Capturing general aspects of the environment, world models have been shown to be effective for transfer to novel tasks (Byravan et al., 2019), directed exploration (Sekar et al., 2020), and generalization from offline datasets (Yu et al., 2020). When the inputs are high-dimensional images, latent dynamics models predict ahead in an abstract latent space (Watter et al., 2015; Ha and Schmidhuber, 2018; Hafner et al., 2018; Zhang et al., 2019). Predicting compact representations instead of images has been hypothesized to reduce accumulating errors and their small memory footprint enables thousands of parallel predictions on a single GPU (Hafner et al., 2018; 2019). Leveraging this approach, the recent Dreamer agent (Hafner et al., 2019) has solved a wide range of continuous control tasks from image inputs. 

Despite their intriguing properties, world models have so far not been accurate enough to compete with the stateof-the-art model-free algorithms on the most competitive benchmarks. The well-established Atari benchmark 

> _∗_ Correspondence to: Danijar Hafner <mail@danijar.com>. 

**==> picture [159 x 134] intentionally omitted <==**

**----- Start of picture text -----**<br>
2.0 Model-based<br>Model-free<br>1.6<br>1.2 Human Gamer<br>0.8<br>0.4<br>0.0<br>DreamerV2Rainbow IQN DQNDreamerV1SimPLe<br>**----- End of picture text -----**<br>


Figure 1: Gamer normalized median score on the Atari benchmark of 55 games with sticky actions at 200M steps. DreamerV2 is the first agent that learns purely within a world model to achieve human-level Atari performance, demonstrating the high accuracy of its learned world model. DreamerV2 further outperforms the top single-GPU agents Rainbow and IQN, whose scores are provided by Dopamine (Castro et al., 2018). According to its authors, SimPLe (Kaiser et al., 2019) was only evaluated on an easier subset of 36 games and trained for fewer steps and additional training does not further increase its performance. 

1 

Published as a conference paper at ICLR 2021 

(Bellemare et al., 2013) historically required model-free algorithms to achieve human-level performance, such as DQN (Mnih et al., 2015), A3C (Mnih et al., 2016), or Rainbow (Hessel et al., 2018). Several attempts at learning accurate world models of Atari games have been made, without achieving competitive performance (Oh et al., 2015; Chiappa et al., 2017; Kaiser et al., 2019). On the other hand, the recently proposed MuZero agent (Schrittwieser et al., 2019) shows that planning can achieve impressive performance on board games and deterministic Atari games given extensive engineering effort and a vast computational budget. However, its implementation is not available to the public and it would require over 2 months of computation to train even one agent on a GPU, rendering it impractical for most research groups. 

In this paper, we introduce DreamerV2, the first reinforcement learning agent that achieves humanlevel performance on the Atari benchmark by learning behaviors purely within a separately trained world model, as shown in Figure 1. Learning successful behaviors purely within the world model demonstrates that the world model learns to accurately represent the environment. To achieve this, we apply small modifications to the Dreamer agent (Hafner et al., 2019), such as using discrete latents and balancing terms within the KL loss. Using a single GPU and a single environment instance, DreamerV2 outperforms top single-GPU Atari agents Rainbow (Hessel et al., 2018) and IQN (Dabney et al., 2018), which rest upon years of model-free reinforcement learning research (Van Hasselt et al., 2015; Schaul et al., 2015; Wang et al., 2016; Bellemare et al., 2017; Fortunato et al., 2017). Moreover, aspects of these algorithms are complementary to our world model and could be integrated into the Dreamer framework in the future. To rigorously compare the algorithms, we report scores normalized by both a human gamer (Mnih et al., 2015) and the human world record (Toromanoff et al., 2019) and make a suggestion for reporting scores going forward. 

## 2 DREAMERV2 

We present DreamerV2, an evolution of the Dreamer agent (Hafner et al., 2019). We refer to the original Dreamer agent as DreamerV1 throughout this paper. This section describes the complete DreamerV2 algorithm, consisting of the three typical components of a model-based agent (Sutton, 1991). We learn the world model from a dataset of past experience, learn an actor and critic from imagined sequences of compact model states, and execute the actor in the environment to grow the experience dataset. In Appendix C, we include a list of changes that we applied to DreamerV1 and which of them we found to increase empirical performance. 

## 2.1 WORLD MODEL LEARNING 

World models summarize an agent’s experience into a predictive model that can be used in place of the environment to learn behaviors. When inputs are high-dimensional images, it is beneficial to learn compact state representations of the inputs to predict ahead in this learned latent space (Watter et al., 2015; Karl et al., 2016; Ha and Schmidhuber, 2018). These models are called latent dynamics models. Predicting ahead in latent space not only facilitates long-term predictions, it also allows to efficiently predict thousands of compact state sequences in parallel in a single batch, without having to generate images. DreamerV2 builds upon the world model that was introduced by PlaNet (Hafner et al., 2018) and used in DreamerV1, by replacing its Gaussian latents with categorical variables. 

**Experience dataset** The world model is trained from the agent’s growing dataset of past experience that contains sequences of images _x_ 1: _T_ , actions _a_ 1: _T_ , rewards _r_ 1: _T_ , and discount factors _γ_ 1: _T_ . The discount factors equal a fixed hyper parameter _γ_ = 0 _._ 999 for time steps within an episode and are set to zero for terminal time steps. For training, we use batches of _B_ = 50 sequences of fixed length _L_ = 50 that are sampled randomly within the stored episodes. To observe enough episode ends during training, we sample the start index of each training sequence uniformly within the episode and then clip it to not exceed the episode length minus the training sequence length. 

**Model components** The world model consists of an image encoder, a Recurrent State-Space Model (RSSM; Hafner et al., 2018) to learn the dynamics, and predictors for the image, reward, and discount factor. The world model is summarized in Figure 2. The RSSM uses a sequence of deterministic recurrent states _ht_ , from which it computes two distributions over stochastic states at each step. The posterior state _zt_ incorporates information about the current image _xt_ , while the prior state _z_ ˆ _t_ aims to predict the posterior without access to the current image. The concatenation of deterministic and 

2 

Published as a conference paper at ICLR 2021 

**==> picture [396 x 211] intentionally omitted <==**

**----- Start of picture text -----**<br>
^r1 x^1 ^r2 x^2 ^r3 x^3<br>a1 a2<br>h h h<br>1 2 3<br>^ ^ ^<br>z1 z1 z2 z2 z3 z3<br>min KL min KL min KL<br>x1 x2 x3 32 classes each<br>32 categoricals<br>**----- End of picture text -----**<br>


Figure 2: World Model Learning. The training sequence of images _xt_ is encoded using the CNN. The RSSM uses a sequence of deterministic recurrent states _ht_ . At each step, it computes a posterior stochastic state _zt_ that incorporates information about the current image _xt_ , as well as a prior stochastic state _z_ ˆ _t_ that tries to predict the posterior without access to the current image. Unlike in PlaNet and DreamerV1, the stochastic state of DreamerV2 is a vector of multiple categorical variables. The learned prior is used for imagination, as shown in Figure 3. The KL loss both trains the prior and regularizes how much information the posterior incorporates from the image. The regularization increases robustness to novel inputs. It also encourages reusing existing information from past steps to predict rewards and reconstruct images, thus learning long-term dependencies. 

stochastic states forms the compact model state. From the posterior model state, we reconstruct the current image _xt_ and predict the reward _rt_ and discount factor _γt_ . The model components are: 

 Recurrent model: _ht_ = _fφ_ ( _ht−_ 1 _, zt−_ 1 _, at−_ 1) RSSM Representation model: _zt ∼ qφ_ ( _zt | ht, xt_ )  ˆ  Transition predictor: _zt ∼ pφ_ (ˆ _zt | ht_ ) ˆ (1) Image predictor: _xt ∼ pφ_ (ˆ _xt | ht, zt_ ) ˆ Reward predictor: _rt ∼ pφ_ (ˆ _rt | ht, zt_ ) ˆ Discount predictor: _γt ∼ pφ_ (ˆ _γt | ht, zt_ ) _._ 

All components are implemented as neural networks and _φ_ describes their combined parameter vector. The transition predictor guesses the next model state only from the current model state and the action but without using the next image, so that we can later learn behaviors by predicting sequences of model states without having to observe or generate images. The discount predictor lets us estimate the probability of an episode ending when learning behaviors from model predictions. 

**Neural networks** The representation model is implemented as a Convolutional Neural Network (CNN; LeCun et al., 1989) followed by a Multi-Layer Perceptron (MLP) that receives the image embedding and the deterministic recurrent state. The RSSM uses a Gated Recurrent Unit (GRU; Cho et al., 2014) to compute the deterministic recurrent states. The model state is the concatenation of deterministic GRU state and a sample of the stochastic state. The image predictor is a transposed CNN and the transition, reward, and discount predictors are MLPs. We down-scale the 84 _×_ 84 grayscale images to 64 _×_ 64 pixels so that we can apply the convolutional architecture of DreamerV1. 

**Algorithm 1:** Straight-Through Gradients with Automatic Differentiation 

sample **= one_hot(draw(** logits **))** # sample has no gradient probs **= softmax(** logits **)** # want gradient of this sample **=** sample **+** probs **- stop_grad(** probs **)** # has gradient of probs 

3 

Published as a conference paper at ICLR 2021 

We use the ELU activation function for all components of the model (Clevert et al., 2015). The world model uses a total of 20M trainable parameters. 

**Distributions** The image predictor outputs the mean of a diagonal Gaussian likelihood with unit variance, the reward predictor outputs a univariate Gaussian with unit variance, and the discount predictor outputs a Bernoulli likelihood. In prior work, the latent variable in the model state was a diagonal Gaussian that used reparameterization gradients during backpropagation (Kingma and Welling, 2013; Rezende et al., 2014). In DreamerV2, we instead use a vector of several categorical variables and optimize them using straight-through gradients (Bengio et al., 2013), which are easy to implement using automatic differentiation as shown in Algorithm 1. We discuss possible benefits of categorical over Gaussian latents in the experiments section. 

**Loss function** All components of the world model are optimized jointly. The distributions produced by the image predictor, reward predictor, discount predictor, and transition predictor are trained to maximize the log-likelihood of their corresponding targets. The representation model is trained to produce model states that facilitates these prediction tasks, through the expectation below. Moreover, it is regularized to produce model states with high entropy, such that the model becomes robust to many different model states during training. The loss function for learning the world model is: 

**==> picture [389 x 55] intentionally omitted <==**

We jointly minimize the loss function with respect to the vector _φ_ that contains all parameters of the world model using the Adam optimizer (Kingma and Ba, 2014). We scale the KL loss by _β_ = 0 _._ 1 for Atari and by _β_ = 1 _._ 0 for continuous control (Higgins et al., 2016). 

**KL balancing** The world model loss function in Equation 2 is the ELBO or variational free energy of a hidden Markov model that is conditioned on the action sequence. The world model can thus be interpreted as a sequential VAE, where the representation model is the approximate posterior and the transition predictor is the temporal prior. In the ELBO objective, the KL loss serves two purposes: it trains the prior toward the representations, and it regularizes the representations toward the prior. However, learning the transition function is difficult and we want to avoid regularizing the representations toward a poorly trained prior. To solve this problem, we minimize the KL loss faster with respect to the prior than the representations by using different learning rates, _α_ = 0 _._ 8 for the prior and 1 _− α_ for the approximate posterior. We implement this technique as shown in Algorithm 2 and refer to it as KL balancing. KL balancing encourages learning an accurate prior over increasing posterior entropy, so that the prior better approximates the aggregate posterior. KL balancing is different from and orthogonal to beta-VAEs (Higgins et al., 2016). 

## 2.2 BEHAVIOR LEARNING 

DreamerV2 learns long-horizon behaviors purely within its world model using an actor and a critic. The actor chooses actions for predicting imagined sequences of compact model states. The critic accumulates the future predicted rewards to take into account rewards beyond the planning horizon. Both the actor and critic operate on top of the learned model states and thus benefit from the representations learned by the world model. The world model is fixed during behavior learning, so the actor and value gradients do not affect its representations. Not predicting images during behavior learning lets us efficiently simulate 2500 latent trajectories in parallel on a single GPU. 

**Imagination MDP** To learn behaviors within the latent space of the world model, we define the imagination MPD as follows. The distribution of initial states _z_ ˆ0 in the imagination MDP is the distribution of compact model states encountered during world model training. From there, the ˆ ˆ transition predictor _pφ_ (ˆ _zt | zt−_ 1 _,_ ˆ _at−_ 1) outputs sequences _z_ 1: _H_ of compact model states up to the 

## **Algorithm 2:** KL Balancing with Automatic Differentiation 

**==> picture [398 x 21] intentionally omitted <==**

4 

Published as a conference paper at ICLR 2021 

**==> picture [358 x 198] intentionally omitted <==**

**----- Start of picture text -----**<br>
^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^<br>r1 v1 a1 r2 v2 a2 r3 v3 a3 r4 v4 a4<br>h1 h^ 2 h^ 3 h^ 4<br>^ ^ ^<br>z1 z2 z3 z4<br>x<br>1<br>**----- End of picture text -----**<br>


Figure 3: Actor Critic Learning. The world model learned in Figure 2 is used for learning a policy from trajectories imagined in the compact latent space. The trajectories start from posterior states computed during model training and predict forward by sampling actions from the actor network. The critic network predicts the expected sum of future rewards for each state. The critic uses temporal difference learning on the imagined rewards. The actor is trained to maximize the critic prediction, via reinforce gradients, straight-through gradients of the world model, or a combination of them. 

ˆ imaginationsequence ˆ _r_ 1: _H_ horizon. The discount predictor _H_ = 15. The mean _pφ_ (ˆ _γt_ of _| z_ ˆthe _t_ ) outputs the discount sequencereward predictor _pφ_ (ˆ _rt | zt_ ) ˆ _γ_ is1: _H_ usedthat is used toas reward down-weight rewards. Moreover, we weigh the loss terms of the actor and critic by the cumulative predicted discount factors to softly account for the possibility of episode ends. 

**Model components** To learn long-horizon behaviors in the imagination MDP, we leverage a stochastic actor that chooses actions and a deterministic critic. The actor and critic are trained cooperatively, where the actor aims to output actions that lead to states that maximize the critic output, while the critic aims to accurately estimate the sum of future rewards achieved by the actor from each imagined state. The actor and critic use the parameter vectors _ψ_ and _ξ_ , respectively: 

**==> picture [298 x 33] intentionally omitted <==**

In contrast to the actual environment, the latent state sequence is Markovian, so that there is no need for the actor and critic to condition on more than the current model state. The actor and critic are both MLPs with ELU activations (Clevert et al., 2015) and use 1M trainable parameters each. The actor outputs a categorical distribution over actions and the critic has a deterministic output. The two components are trained from the same imagined trajectories but optimize separate loss functions. 

**Critic loss function** The critic aims to predict the discounted sum of future rewards that the actor achieves in a given model state, known as the state value. For this, we leverage temporal-difference learning, where the critic is trained toward a value target that is constructed from intermediate rewards and critic outputs for later states. A common choice is the 1-step target that sums the current reward and the critic output for the following state. However, the imagination MDP lets us generate on-policy trajectories of multiple steps, suggesting the use of n-step targets that incorporate reward information into the critic more quickly. We follow DreamerV1 in using the more general _λ_ -target (Sutton and Barto, 2018; Schulman et al., 2015) that is defined recursively as follows: 

**==> picture [311 x 26] intentionally omitted <==**

Intuitively, the _λ_ -target is a weighted average of n-step returns for different horizons, where longer horizons are weighted exponentially less. We set _λ_ = 0 _._ 95 in practice, to focus more on long horizon 

5 

Published as a conference paper at ICLR 2021 

targets than on short horizon targets. Given a trajectory of model states, rewards, and discount factors, we train the critic to regress the _λ_ -return using a squared loss: 

**==> picture [296 x 19] intentionally omitted <==**

We optimize the critic loss with respect to the critic parameters _ξ_ using the Adam optimizer. There is no loss term for the last time step because the target equals the critic at that step. We stop the gradients around the targets, denoted by the sg( _·_ ) function, as typical in the literature. We stabilize value learning using a target network (Mnih et al., 2015), namely, we compute the targets using a copy of the critic that is updated every 100 gradient steps. 

**Actor loss function** The actor aims to output actions that maximize the prediction of long-term future rewards made by the critic. To incorporate intermediate rewards more directly, we train the actor to maximize the same _λ_ -return that was computed for training the critic. There are different gradient estimators for maximizing the targets with respect to the actor parameters. DreamerV2 combines unbiased but high-variance Reinforce gradients with biased but low-variance straightthrough gradients. Moreover, we regularize the entropy of the actor to encourage exploration where feasible while allowing the actor to choose precise actions when necessary. 

Learning by Reinforce (Williams, 1992) maximizes the actor’s probability of its own sampled actions weighted by the values of those actions. The variance of this estimator can be reduced by subtracting the state value as baseline, which does not depend on the current action. Intuitively, subtracting the baseline centers the weights and leads to faster learning. The benefit of Reinforce is that it produced unbiased gradients and the downside is that it can have high variance, even with baseline. 

DreamerV1 relied entirely on reparameterization gradients (Kingma and Welling, 2013; Rezende et al., 2014) to train the actor directly by backpropagating value gradients through the sequence of sampled model states and actions. DreamerV2 uses both discrete latents and discrete actions. To backpropagate through the sampled actions and state sequences, we leverage straight-through gradients (Bengio et al., 2013). This results in a biased gradient estimate with low variance. The combined actor loss function is: 

**==> picture [390 x 32] intentionally omitted <==**

We optimize the actor loss with respect to the actor parameters _ψ_ using the Adam optimizer. We consider both Reinforce gradients and straight-through gradients, which backpropagate directly through the learned dynamics. Intuitively, the low-variance but biased dynamics backpropagation could learn faster initially and the unbiased but high-variance could to converge to a better solution. For Atari, we find Reinforce gradients to work substantially better and use _ρ_ = 1 and _η_ = 10 _[−]_[3] . For continuous control, we find dynamics backpropagation to work substantially better and use _ρ_ = 0 and _η_ = 10 _[−]_[4] . Annealing these hyper parameters can improve performance slightly but to avoid the added complexity we report the scores without annealing. 

## 3 EXPERIMENTS 

We evaluate DreamerV2 on the well-established Atari benchmark with sticky actions, comparing to four strong model-free algorithms. DreamerV2 outperforms the four model-free algorithms in all scenarios. For an extensive comparison, we report four scores according to four aggregation protocols and give a recommendation for meaningfully aggregating scores across games going forward. We also ablate the importance of discrete representations in the world model. Our implementation of DreamerV2 reaches 200M environment steps in under 10 days, while using only a single NVIDIA V100 GPU and a single environment instance. During the 200M environment steps, DreamerV2 learns its policy from 468B compact states imagined under the model, which is 10,000 _×_ more than the 50M inputs received from the real environment after action repeat. Refer to the project website for videos, the source code, and training curves in JSON format.[1] 

> 1https://danijar.com/dreamerv2 

6 

Published as a conference paper at ICLR 2021 

**==> picture [398 x 95] intentionally omitted <==**

**----- Start of picture text -----**<br>
Gamer Median Gamer Mean Record Mean Clipped Record Mean<br>2.4 12 0.45<br>0.24<br>1.8 9<br>0.30<br>0.16<br>1.2 6<br>0.15 0.08<br>0.6 3<br>0.0 0 0.00 0.00<br>0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0<br>DreamerV2 IQN Rainbow C51 DQN 1e8<br>**----- End of picture text -----**<br>


Figure 4: Atari performance over 200M steps. See Table 1 for numeric scores. The standards in the literature to aggregate over tasks are shown in the left two plots. These normalize scores by a professional gamer and compute the median or mean over tasks (Mnih et al., 2015; 2016). In Section 3, we point out limitations of this methodology. As a robust measure of performance, we recommend the metric in the right-most plot. We normalize scores by the human world record (Toromanoff et al., 2019) and then clip them, such that exceeding the record does not further increase the score, before averaging over tasks. 

**Experimental setup** We select the 55 games that prior works in the literature from different research labs tend to agree on (Mnih et al., 2016; Brockman et al., 2016; Hessel et al., 2018; Castro et al., 2018; Badia et al., 2020) and recommend this set of games for evaluation going forward. We follow the evaluation protocol of Machado et al. (2018) with 200M environment steps, action repeat of 4, a time limit of 108,000 steps per episode that correspond to 30 minutes of game play, no access to life information, full action space, and sticky actions. Because the world model integrates information over time, DreamerV2 does not use frame stacking. The experiments use a single-task setup where a separate agent is trained for each game. Moreover, each agent uses only a single environment instance. We compare the algorithms based on both human gamer and human world record normalization (Toromanoff et al., 2019). 

**Model-free baselines** We compare the learning curves and final scores of DreamerV2 to four model-free algorithms, IQN (Dabney et al., 2018), Rainbow (Hessel et al., 2018), C51 (Bellemare et al., 2017), and DQN (Mnih et al., 2015). We use the scores of these agents provided by the Dopamine framework (Castro et al., 2018) that use sticky actions. These may differ from the reported results in the papers that introduce these algorithms in the deterministic Atari setup. The training time of Rainbow was reported at 10 days on a single GPU and using one environment instance. 

## 3.1 ATARI PERFORMANCE 

The performance curves of DreamerV2 and four standard model-free algorithms are visualized in Figure 4. The final scores at 200M environment steps are shown in Table 1 and the scores on individual games are included in Table K.1. There are different approaches for aggregating the scores across the 55 games and we show that this choice can have a substantial impact on the relative performance between algorithms. To extensively compare DreamerV2 to the model-free algorithms, we consider the following four aggregation approaches: 

|**Agent**||**Gamer Median**|**Gamer Mean**|**Record Mean**|**Clipped Record Mean**|
|---|---|---|---|---|---|
|DreamerV2||2.15|**11.33**|**0.44**|**0.28**|
|DreamerV2|(schedules)|**2.64**|10.45|0.43|**0.28**|
|IQN||1.29|8.85|0.21|0.21|
|Rainbow||1.47|9.12|0.17|0.17|
|C51||1.09|7.70|0.15|0.15|
|DQN||0.65|2.84|0.12|0.12|



Table 1: Atari performance at 200M steps. The scores of the 55 games are aggregated using the four different protocols described in Section 3. To overcome limitations of the previous metrics, we recommend the task mean of clipped record normalized scores as a robust measure of algorithm performance, shown in the right-most column. DreamerV2 outperforms previous single-GPU agents across all metrics. The baseline scores are taken from Dopamine Baselines (Castro et al., 2018). 

7 

Published as a conference paper at ICLR 2021 

**==> picture [397 x 104] intentionally omitted <==**

**----- Start of picture text -----**<br>
Latent Variables KL Balancing Image Gradients Reward Gradients<br>0.24 0.24 0.24 0.24<br>0.18 0.18 0.16 0.18<br>0.12 0.12 0.12<br>0.08<br>0.06 0.06 0.06<br>0.00<br>0.00 0.00 0.00<br>0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0<br>1e8<br>Categorical Enabled Enabled Enabled<br>Gaussian Disabled Disabled Disabled<br>**----- End of picture text -----**<br>


Figure 5: Clipped record normalized scores of various ablations of the DreamerV2 agent. This experiment uses a slightly earlier version of DreamerV2. The score curves for individual tasks are shown in Figure H.1. The ablations highlight the benefit of using categorical over Gaussian latent variables and of using KL balancing. Moreover, they show that the world model relies on image gradients for learning its representations. Stopping reward gradients even improves performance on some tasks, suggesting that representations that are not specifically trained to predict previously experienced rewards may generalize better to new situations. 

- **Gamer Median** Atari scores are commonly normalized based on a random policy and a professional gamer, averaged over seeds, and the median over tasks is reported (Mnih et al., 2015; 2016). However, if almost half of the scores would be zero, the median would not be affected. Thus, we argue that median scores are not reflective of the robustness of an algorithm and results in wasted computational resources for games that will not affect the score. 

- **Gamer Mean** Compared to the task median, the task mean considers all tasks. However, the gamer performed poorly on a small number of games, such as Crazy Climber, James Bond, and Video Pinball. This makes it easy for algorithms to achieve a high normalized score on these few games, which then dominate the task mean so it is not informative of overall performance. 

- **Record Mean** Instead of normalizing based on the professional gamer, Toromanoff et al. (2019) suggest to normalize based on the registered human world record of each game. This partially addresses the outlier problem but the mean is still dominated by games where the algorithms easily achieve superhuman performance. 

• **Clipped Record Mean** To overcome these limitations, we recommend normalizing by the human world record and then clipping the scores to not exceed a value of 1, so that performance above the record does not further increase the score. The result is a robust measure of algorithm performance on the Atari suite that considers performance across all games. From Figure 4 and Table 1, we see that the different aggregation approaches let us examine agent performance from different angles. Interestingly, Rainbow clearly outperforms IQN in the first aggregation method but IQN clearly outperforms Rainbow in the remaining setups. DreamerV2 outperforms the model-free agents in all four metrics, with the largest margin in record normalized mean performance. Despite this, we recommend clipped record normalized mean as the most meaningful aggregation method, as it considers all tasks to a similar degree without being dominated by a small number of outlier scores. In Table 1, we also include DreamerV2 with schedules that anneal the actor entropy loss scale and actor gradient mixing over the course of training, which further increases the gamer median score of DreamerV2. 

**Individual games** The scores on individual Atari games at 200M environment steps are included in Table K.1, alongside the model-free algorithms and the baselines of random play, human gamer, and human world record. We filled in reasonable values for the 2 out of 55 games that have no registered world record. Figure E.1 compares the score differences between DreamerV2 and each model-free algorithm for the individual games. DreamerV2 achieves comparable or higher performance on most games except for Video Pinball. We hypothesize that the reconstruction loss of the world model does not encourage learning a meaningful latent representation because the most important object in the game, the ball, occupies only a single pixel. One the other hand, DreamerV2 achieves the strongest improvements over the model-free agents on the games James Bond, Up N Down, and Assault. 

8 

Published as a conference paper at ICLR 2021 

|**Agent**<br>**Gamer Median**|**Agent**<br>**Gamer Median**|**Gamer Mean**|**Record Mean**|**Clipped Record Mean**|
|---|---|---|---|---|
|DreamerV2|1.64|11.33|0.36|0.25|
|No Layer Norm|1.66|5.95|0.38|0.25|
|No Reward Gradients|1.68|6.18|0.37|0.24|
|No Discrete Latents|1.08|3.71|0.24|0.19|
|No KL Balancing|0.84|3.49|0.19|0.16|
|No Policy Reinforce|0.69|2.74|0.16|0.15|
|No Image Gradients|0.04|0.31|0.01|0.01|



Table 2: Ablations to DreamerV2 measured by their Atari performance at 200M frames, sorted by the last column. The this experiment uses a slightly earlier version of DreamerV2 compared to Table 1. Each ablation only removes one part of the DreamerV2 agent. Discrete latent variables and KL balancing substantially contribute to the success of DreamerV2. Moreover, the world model relies on image gradients to learn general representations that lead to successful behaviors, even if the representations are not specifically learned for predicting past rewards. 

## 3.2 ABLATION STUDY 

To understand which ingredients of DreamerV2 are responsible for its success, we conduct an extensive ablation study. We compare equipping the world model with categorical latents, as in DreamerV2, to Gaussian latents, as in DreamerV1. Moreover, we study the importance of KL balancing. Finally, we investigate the importance of gradients from image reconstruction and reward prediction for learning the model representations, by stopping one of the two gradient signals before entering the model states. The results of the ablation study are summarized in Figure 5 and Table 2. Refer to the appendix for the score curves of the individual tasks. 

**Categorical latents** Categorical latent variables outperform than Gaussian latent variables on 42 tasks, achieve lower performance on 8 tasks, and are tied on 5 tasks. We define a tie as being within 5% of another. While we do not know the reason why the categorical variables are beneficial, we state several hypotheses that can be investigated in future work: 

- A categorical prior can perfectly fit the aggregate posterior, because a mixture of categoricals is again a categorical. In contrast, a Gaussian prior cannot match a mixture of Gaussian posteriors, which could make it difficult to predict multi-modal changes between one image and the next. 

- The level of sparsity enforced by a vector of categorical latent variables could be beneficial for generalization. Flattening the sample from the 32 categorical with 32 classes each results in a sparse binary vector of length 1024 with 32 active bits. 

- Despite common intuition, categorical variables may be easier to optimize than Gaussian variables, possibly because the straight-through gradient estimator ignores a term that would otherwise scale the gradient. This could reduce exploding and vanishing gradients. 

- Categorical variables could be a better inductive bias than unimodal continuous latent variables for modeling the non-smooth aspects of Atari games, such as when entering a new room, or when collected items or defeated enemies disappear from the image. 

**KL balancing** KL balancing outperforms the standard KL regularizer on 44 tasks, achieves lower performance on 6 tasks, and is tied on 5 tasks. Learning accurate prior dynamics of the world model is critical because it is used for imagining latent state trajectories using policy optimization. By scaling up the prior cross entropy relative to the posterior entropy, the world model is encouraged to minimize the KL by improving its prior dynamics toward the more informed posteriors, as opposed to reducing the KL by increasing the posterior entropy. KL balancing may also be beneficial for probabilistic models with learned priors beyond world models. 

**Model gradients** Stopping the image gradients increases performance on 3 tasks, decreases performance on 51 tasks, and is tied on 1 task. The world model of DreamerV2 thus heavily relies on the learning signal provided by the high-dimensional images. Stopping the reward gradients increases performance on 15 tasks, decreases performance on 22 tasks, and is tied on 18 tasks. Figure H.1 further shows that the difference in scores is small. In contrast to MuZero, DreamerV2 thus learns general representations of the environment state from image information alone. Stopping reward gradients improved performance on a number of tasks, suggesting that the representations that are not specific to previously experienced rewards may generalize better to unseen situations. 

9 

Published as a conference paper at ICLR 2021 

|**Algorithm**|**Reward**<br>**Modeling**|**Image**<br>**Modeling**|**Latent**<br>**Transitions**|**Single**<br>**GPU**|**Trainable**<br>**Parameters**|**Atari**<br>**Frames**|**Accelerator**<br>**Days**|
|---|---|---|---|---|---|---|---|
|DreamerV2|||||22M|200M|10|
|SimPLe|||||74M|4M|40|
|MuZero|||||40M|20B|80|
|MuZero Reanalyze<br>|||||40M|200M|80|



Table 3: Conceptual comparison of recent RL algorithms that leverage planning with a learned model. DreamerV2 and SimPLe learn complete models of the environment by leveraging the learning signal provided by the image inputs, while MuZero learns its model through value gradients that are specific to an individual task. The Monte-Carlo tree search used by MuZero is effective but adds complexity and is challenging to parallelize. This component is orthogonal to the world model proposed here. 

**Policy gradients** Using only Reinforce gradients to optimize the policy increases performance on 18 tasks, decreases performance on 24 tasks, and is tied on 13 tasks. This shows that DreamerV2 relies mostly on Reinforce gradients to learn the policy. However, mixing Reinforce and straight-through gradients yields a substantial improvement on James Bond and Seaquest, leading to a higher gamer normalized task mean score. Using only straight-through gradients to optimize the policy increases performance on 5 tasks, decreases performance on 44 tasks, and is tied on 6 tasks. We conjecture that straight-through gradients alone are not well suited for policy optimization because of their bias. 

## 4 RELATED WORK 

**Model-free Atari** The majority of agents applied to the Atari benchmark have been trained using model-free algorithms. DQN (Mnih et al., 2015) showed that deep neural network policies can be trained using Q-learning by incorporating experience replay and target networks. Several works have extended DQN to incorporate bias correction as in DDQN (Van Hasselt et al., 2015), prioritized experience replay (Schaul et al., 2015), architectural improvements (Wang et al., 2016), and distributional value learning (Bellemare et al., 2017; Dabney et al., 2017; 2018). Besides value learning, agents based on policy gradients have targeted the Atari benchmark, such as ACER (Schulman et al., 2017a), PPO (Schulman et al., 2017a), ACKTR (Wu et al., 2017), and Reactor (Gruslys et al., 2017). Another line of work has focused on improving performance by distributing data collection, often while increasing the budget of environment steps beyond 200M (Mnih et al., 2016; Schulman et al., 2017b; Horgan et al., 2018; Kapturowski et al., 2018; Badia et al., 2020). 

**World models** Several model-based agents focus on proprioceptive inputs (Watter et al., 2015; Gal et al., 2016; Higuera et al., 2018; Henaff et al., 2018; Chua et al., 2018; Wang et al., 2019; Wang and Ba, 2019), model images without using them for planning (Oh et al., 2015; Krishnan et al., 2015; Karl et al., 2016; Chiappa et al., 2017; Babaeizadeh et al., 2017; Gemici et al., 2017; Denton and Fergus, 2018; Buesing et al., 2018; Doerr et al., 2018; Gregor and Besse, 2018), or combine the benefits of model-based and model-free approaches (Kalweit and Boedecker, 2017; Nagabandi et al., 2017; Weber et al., 2017; Kurutach et al., 2018; Buckman et al., 2018; Ha and Schmidhuber, 2018; Wayne et al., 2018; Igl et al., 2018; Srinivas et al., 2018; Lee et al., 2019). Risi and Stanley (2019) optimize discrete latents using evolutionary search. Parmas et al. (2019) combine reinforce and reparameterization gradients. Most world model agents with image inputs have thus far been limited to relatively simple control tasks (Watter et al., 2015; Ebert et al., 2017; Ha and Schmidhuber, 2018; Hafner et al., 2018; Zhang et al., 2019; Hafner et al., 2019). We explain the two model-based approaches that were applied to Atari in detail below. 

**SimPLe** The SimPLe agent (Kaiser et al., 2019) learns a video prediction model in pixel-space and uses its predictions to train a PPO agent (Schulman et al., 2017a), as shown in Table 3. The model directly predicts each frame from the previous four frames and receives an additional discrete latent variable as input. The authors evaluate SimPLe on a subset of Atari games for 400k and 2M environment steps, after which they report diminishing returns. Some recent model-free methods have followed the comparison at 400k steps (Srinivas et al., 2020; Kostrikov et al., 2020). However, the highest performance achieved in this data-efficient regime is a gamer normalized median score of 0.28 (Kostrikov et al., 2020) that is far from human-level performance. Instead, we focus on the well-established and competitive evaluation after 200M frames, where many successful model-free algorithms are available for comparison. 

10 

Published as a conference paper at ICLR 2021 

**MuZero** The MuZero agent (Schrittwieser et al., 2019) learns a sequence model of rewards and values (Oh et al., 2017) to solve reinforcement learning tasks via Monte-Carlo Tree Search (MCTS; Coulom, 2006; Silver et al., 2017). The sequence model is trained purely by predicting task-specific information and does not incorporate explicit representation learning using the images, as shown in Table 3. MuZero shows that with significant engineering effort and a vast computational budget, planning can achieve impressive performance on several board games and deterministic Atari games. However, MuZero is not publicly available, and it would require over 2 months to train an Atari agent on one GPU. By comparison, DreamerV2 is a simple algorithm that achieves human-level performance on Atari on a single GPU in 10 days, making it reproducible for many researchers. Moreover, the advanced planning components of MuZero are complementary and could be applied to the accurate world models learned by DreamerV2. DreamerV2 leverages the additional learning signal provided by the input images, analogous to recent successes by semi-supervised image classification (Chen et al., 2020; He et al., 2020; Grill et al., 2020). 

## 5 DISCUSSION 

We present DreamerV2, a model-based agent that achieves human-level performance on the Atari 200M benchmark by learning behaviors purely from the latent-space predictions of a separately trained world model. Using a single GPU and a single environment instance, DreamerV2 outperforms top model-free single-GPU agents Rainbow and IQN using the same computational budget and training time. To develop DreamerV2, we apply several small modifications to the Dreamer agent (Hafner et al., 2019). We confirm experimentally that learning a categorical latent space and using KL balancing improves the performance of the agent. Moreover, we find the DreamerV2 relies on image information for learning generally useful representations — its performance is not impacted by whether the representations are especially learned for predicting rewards. 

DreamerV2 serves as proof of concept, showing that model-based RL can outperform top model-free algorithms on the most competitive RL benchmarks, despite the years of research and engineering effort that modern model-free agents rest upon. Beyond achieving strong performance on individual tasks, world models open avenues for efficient transfer and multi-task learning, sample-efficient learning on physical robots, and global exploration based on uncertainty estimates. 

**Acknowledgements** We thank our anonymous reviewers for their feedback and Nick Rhinehart for an insightful discussion about the potential benefits of categorical latent variables. 

11 

Published as a conference paper at ICLR 2021 

## REFERENCES 

- M Babaeizadeh, C Finn, D Erhan, RH Campbell, S Levine. Stochastic Variational Video Prediction. _ArXiv Preprint ArXiv:1710.11252_ , 2017. 

- AP Badia, B Piot, S Kapturowski, P Sprechmann, A Vitvitskyi, D Guo, C Blundell. Agent57: Outperforming the Atari Human Benchmark. _ArXiv Preprint ArXiv:2003.13350_ , 2020. 

- MG Bellemare, Y Naddaf, J Veness, M Bowling. The Arcade Learning Environment: An Evaluation Platform for General Agents. _Journal of Artificial Intelligence Research_ , 47, 2013. 

- MG Bellemare, W Dabney, R Munos. A Distributional Perspective on Reinforcement Learning. _ArXiv Preprint ArXiv:1707.06887_ , 2017. 

- Y Bengio, N Léonard, A Courville. Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation. _ArXiv Preprint ArXiv:1308.3432_ , 2013. 

- G Brockman, V Cheung, L Pettersson, J Schneider, J Schulman, J Tang, W Zaremba. Openai Gym, 2016. 

- J Buckman, D Hafner, G Tucker, E Brevdo, H Lee. Sample-Efficient Reinforcement Learning With Stochastic Ensemble Value Expansion. _Advances in Neural Information Processing Systems_ , 2018. 

- L Buesing, T Weber, S Racaniere, S Eslami, D Rezende, DP Reichert, F Viola, F Besse, K Gregor, D Hassabis, et al. Learning and Querying Fast Generative Models for Reinforcement Learning. _ArXiv Preprint ArXiv:1802.03006_ , 2018. 

- A Byravan, JT Springenberg, A Abdolmaleki, R Hafner, M Neunert, T Lampe, N Siegel, N Heess, M Riedmiller. Imagined Value Gradients: Model-Based Policy Optimization With Transferable Latent Dynamics Models. _ArXiv Preprint ArXiv:1910.04142_ , 2019. 

- PS Castro, S Moitra, C Gelada, S Kumar, MG Bellemare. Dopamine: A Research Framework for Deep Reinforcement Learning. _ArXiv Preprint ArXiv:1812.06110_ , 2018. 

- T Chen, S Kornblith, M Norouzi, G Hinton. A Simple Framework for Contrastive Learning of Visual Representations. _ArXiv Preprint ArXiv:2002.05709_ , 2020. 

- S Chiappa, S Racaniere, D Wierstra, S Mohamed. Recurrent Environment Simulators. _ArXiv Preprint ArXiv:1704.02254_ , 2017. 

- K Cho, B Van Merriënboer, C Gulcehre, D Bahdanau, F Bougares, H Schwenk, Y Bengio. Learning Phrase Representations Using Rnn Encoder-Decoder for Statistical Machine Translation. _ArXiv Preprint ArXiv:1406.1078_ , 2014. 

- K Chua, R Calandra, R McAllister, S Levine. Deep Reinforcement Learning in a Handful of Trials Using Probabilistic Dynamics Models. _Advances in Neural Information Processing Systems_ , 2018. 

- DA Clevert, T Unterthiner, S Hochreiter. Fast and Accurate Deep Network Learning by Exponential Linear Units (Elus). _ArXiv Preprint ArXiv:1511.07289_ , 2015. 

- R Coulom. Efficient Selectivity and Backup Operators in Monte-Carlo Tree Search. _International Conference on Computers and Games_ . Springer, 2006. 

- W Dabney, M Rowland, MG Bellemare, R Munos. Distributional Reinforcement Learning With Quantile Regression. _ArXiv Preprint ArXiv:1710.10044_ , 2017. 

- W Dabney, G Ostrovski, D Silver, R Munos. Implicit Quantile Networks for Distributional Reinforcement Learning. _ArXiv Preprint ArXiv:1806.06923_ , 2018. 

- E Denton R Fergus. Stochastic Video Generation With a Learned Prior. _ArXiv Preprint ArXiv:1802.07687_ , 2018. 

- A Doerr, C Daniel, M Schiegg, D Nguyen-Tuong, S Schaal, M Toussaint, S Trimpe. Probabilistic Recurrent State-Space Models. _ArXiv Preprint ArXiv:1801.10395_ , 2018. 

12 

Published as a conference paper at ICLR 2021 

- F Ebert, C Finn, AX Lee, S Levine. Self-Supervised Visual Planning With Temporal Skip Connections. _ArXiv Preprint ArXiv:1710.05268_ , 2017. 

- M Fortunato, MG Azar, B Piot, J Menick, I Osband, A Graves, V Mnih, R Munos, D Hassabis, O Pietquin, et al. Noisy Networks for Exploration. _ArXiv Preprint ArXiv:1706.10295_ , 2017. 

- Y Gal, R McAllister, CE Rasmussen. Improving Pilco With Bayesian Neural Network Dynamics Models. _Data-Efficient Machine Learning Workshop, ICML_ , 2016. 

- M Gemici, CC Hung, A Santoro, G Wayne, S Mohamed, DJ Rezende, D Amos, T Lillicrap. Generative Temporal Models With Memory. _ArXiv Preprint ArXiv:1702.04649_ , 2017. 

- K Gregor F Besse. Temporal Difference Variational Auto-Encoder. _ArXiv Preprint ArXiv:1806.03107_ , 2018. 

- JB Grill, F Strub, F Altché, C Tallec, PH Richemond, E Buchatskaya, C Doersch, BA Pires, ZD Guo, MG Azar, et al. Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning. _ArXiv Preprint ArXiv:2006.07733_ , 2020. 

- A Gruslys, W Dabney, MG Azar, B Piot, M Bellemare, R Munos. The Reactor: A Fast and SampleEfficient Actor-Critic Agent for Reinforcement Learning. _ArXiv Preprint ArXiv:1704.04651_ , 2017. 

D Ha J Schmidhuber. World Models. _ArXiv Preprint ArXiv:1803.10122_ , 2018. 

- D Hafner, T Lillicrap, I Fischer, R Villegas, D Ha, H Lee, J Davidson. Learning Latent Dynamics for Planning From Pixels. _ArXiv Preprint ArXiv:1811.04551_ , 2018. 

- D Hafner, T Lillicrap, J Ba, M Norouzi. Dream to Control: Learning Behaviors by Latent Imagination. _ArXiv Preprint ArXiv:1912.01603_ , 2019. 

- K He, H Fan, Y Wu, S Xie, R Girshick. Momentum Contrast for Unsupervised Visual Representation Learning. _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , 2020. 

- M Henaff, WF Whitney, Y LeCun. Model-Based Planning With Discrete and Continuous Actions. _ArXiv Preprint ArXiv:1705.07177_ , 2018. 

- M Hessel, J Modayil, H Van Hasselt, T Schaul, G Ostrovski, W Dabney, D Horgan, B Piot, M Azar, D Silver. Rainbow: Combining Improvements in Deep Reinforcement Learning. _Thirty-Second AAAI Conference on Artificial Intelligence_ , 2018. 

- I Higgins, L Matthey, A Pal, C Burgess, X Glorot, M Botvinick, S Mohamed, A Lerchner. BetaVae: Learning Basic Visual Concepts With a Constrained Variational Framework. _International Conference on Learning Representations_ , 2016. 

- JCG Higuera, D Meger, G Dudek. Synthesizing Neural Network Controllers With Probabilistic Model Based Reinforcement Learning. _ArXiv Preprint ArXiv:1803.02291_ , 2018. 

- D Horgan, J Quan, D Budden, G Barth-Maron, M Hessel, H Van Hasselt, D Silver. Distributed Prioritized Experience Replay. _ArXiv Preprint ArXiv:1803.00933_ , 2018. 

- M Igl, L Zintgraf, TA Le, F Wood, S Whiteson. Deep Variational Reinforcement Learning for Pomdps. _ArXiv Preprint ArXiv:1806.02426_ , 2018. 

- L Kaiser, M Babaeizadeh, P Milos, B Osinski, RH Campbell, K Czechowski, D Erhan, C Finn, P Kozakowski, S Levine, et al. Model-Based Reinforcement Learning for Atari. _ArXiv Preprint ArXiv:1903.00374_ , 2019. 

- G Kalweit J Boedecker. Uncertainty-Driven Imagination for Continuous Deep Reinforcement Learning. _Conference on Robot Learning_ , 2017. 

- S Kapturowski, G Ostrovski, J Quan, R Munos, W Dabney. Recurrent Experience Replay in Distributed Reinforcement Learning. _International Conference on Learning Representations_ , 2018. 

13 

Published as a conference paper at ICLR 2021 

- M Karl, M Soelch, J Bayer, P van der Smagt. Deep Variational Bayes Filters: Unsupervised Learning of State Space Models From Raw Data. _ArXiv Preprint ArXiv:1605.06432_ , 2016. 

- DP Kingma J Ba. Adam: A Method for Stochastic Optimization. _ArXiv Preprint ArXiv:1412.6980_ , 2014. 

- DP Kingma M Welling. Auto-Encoding Variational Bayes. _ArXiv Preprint ArXiv:1312.6114_ , 2013. 

- I Kostrikov, D Yarats, R Fergus. Image Augmentation Is All You Need: Regularizing Deep Reinforcement Learning From Pixels. _ArXiv Preprint ArXiv:2004.13649_ , 2020. 

- RG Krishnan, U Shalit, D Sontag. Deep Kalman Filters. _ArXiv Preprint ArXiv:1511.05121_ , 2015. 

- T Kurutach, I Clavera, Y Duan, A Tamar, P Abbeel. Model-Ensemble Trust-Region Policy Optimization. _ArXiv Preprint ArXiv:1802.10592_ , 2018. 

- Y LeCun, B Boser, JS Denker, D Henderson, RE Howard, W Hubbard, LD Jackel. Backpropagation Applied to Handwritten Zip Code Recognition. _Neural Computation_ , 1(4), 1989. 

- AX Lee, A Nagabandi, P Abbeel, S Levine. Stochastic Latent Actor-Critic: Deep Reinforcement Learning With a Latent Variable Model. _ArXiv Preprint ArXiv:1907.00953_ , 2019. 

- MC Machado, MG Bellemare, E Talvitie, J Veness, M Hausknecht, M Bowling. Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents. _Journal of Artificial Intelligence Research_ , 61, 2018. 

- V Mnih, K Kavukcuoglu, D Silver, AA Rusu, J Veness, MG Bellemare, A Graves, M Riedmiller, AK Fidjeland, G Ostrovski, et al. Human-Level Control Through Deep Reinforcement Learning. _Nature_ , 518(7540), 2015. 

- V Mnih, AP Badia, M Mirza, A Graves, T Lillicrap, T Harley, D Silver, K Kavukcuoglu. Asynchronous Methods for Deep Reinforcement Learning. _International Conference on Machine Learning_ , 2016. 

- A Nagabandi, G Kahn, RS Fearing, S Levine. Neural Network Dynamics for Model-Based Deep Reinforcement Learning With Model-Free Fine-Tuning. _ArXiv Preprint ArXiv:1708.02596_ , 2017. 

- J Oh, X Guo, H Lee, RL Lewis, S Singh. Action-Conditional Video Prediction Using Deep Networks in Atari Games. _Advances in Neural Information Processing Systems_ , 2015. 

- J Oh, S Singh, H Lee. Value Prediction Network. _Advances in Neural Information Processing Systems_ , 2017. 

- P Parmas, CE Rasmussen, J Peters, K Doya. Pipps: Flexible Model-Based Policy Search Robust to the Curse of Chaos. _ArXiv Preprint ArXiv:1902.01240_ , 2019. 

- D Pathak, P Agrawal, AA Efros, T Darrell. Curiosity-Driven Exploration by Self-Supervised Prediction. _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops_ , 2017. 

- DJ Rezende, S Mohamed, D Wierstra. Stochastic Backpropagation and Approximate Inference in Deep Generative Models. _ArXiv Preprint ArXiv:1401.4082_ , 2014. 

- S Risi KO Stanley. Deep Neuroevolution of Recurrent and Discrete World Models. _Proceedings of the Genetic and Evolutionary Computation Conference_ , 2019. 

- T Schaul, J Quan, I Antonoglou, D Silver. Prioritized Experience Replay. _ArXiv Preprint ArXiv:1511.05952_ , 2015. 

- J Schrittwieser, I Antonoglou, T Hubert, K Simonyan, L Sifre, S Schmitt, A Guez, E Lockhart, D Hassabis, T Graepel, et al. Mastering Atari, Go, Chess and Shogi by Planning With a Learned Model. _ArXiv Preprint ArXiv:1911.08265_ , 2019. 

- J Schulman, P Moritz, S Levine, M Jordan, P Abbeel. High-Dimensional Continuous Control Using Generalized Advantage Estimation. _ArXiv Preprint ArXiv:1506.02438_ , 2015. 

14 

Published as a conference paper at ICLR 2021 

- J Schulman, F Wolski, P Dhariwal, A Radford, O Klimov. Proximal Policy Optimization Algorithms. _ArXiv Preprint ArXiv:1707.06347_ , 2017a. 

- J Schulman, F Wolski, P Dhariwal, A Radford, O Klimov. Proximal Policy Optimization Algorithms. _ArXiv Preprint ArXiv:1707.06347_ , 2017b. 

- R Sekar, O Rybkin, K Daniilidis, P Abbeel, D Hafner, D Pathak. Planning to Explore via SelfSupervised World Models. _ArXiv Preprint ArXiv:2005.05960_ , 2020. 

- D Silver, J Schrittwieser, K Simonyan, I Antonoglou, A Huang, A Guez, T Hubert, L Baker, M Lai, A Bolton, et al. Mastering the Game of Go Without Human Knowledge. _Nature_ , 550(7676), 2017. 

- A Srinivas, A Jabri, P Abbeel, S Levine, C Finn. Universal Planning Networks. _ArXiv Preprint ArXiv:1804.00645_ , 2018. 

- A Srinivas, M Laskin, P Abbeel. Curl: Contrastive Unsupervised Representations for Reinforcement Learning. _ArXiv Preprint ArXiv:2004.04136_ , 2020. 

- RS Sutton. Dyna, an Integrated Architecture for Learning, Planning, and Reacting. _ACM SIGART Bulletin_ , 2(4), 1991. 

- RS Sutton AG Barto. _Reinforcement Learning: An Introduction_ . MIT press, 2018. 

- AA Taiga, W Fedus, MC Machado, A Courville, MG Bellemare. On Bonus Based Exploration Methods in the Arcade Learning Environment. _International Conference on Learning Representations_ , 2019. 

- M Toromanoff, E Wirbel, F Moutarde. Is Deep Reinforcement Learning Really Superhuman on Atari? Leveling the Playing Field. _ArXiv Preprint ArXiv:1908.04683_ , 2019. 

- H Van Hasselt, A Guez, D Silver. Deep Reinforcement Learning With Double Q-Learning. _ArXiv Preprint ArXiv:1509.06461_ , 2015. 

- T Wang J Ba. Exploring Model-Based Planning With Policy Networks. _ArXiv Preprint ArXiv:1906.08649_ , 2019. 

- T Wang, X Bao, I Clavera, J Hoang, Y Wen, E Langlois, S Zhang, G Zhang, P Abbeel, J Ba. Benchmarking Model-Based Reinforcement Learning. _CoRR_ , abs/1907.02057, 2019. 

- Z Wang, T Schaul, M Hessel, H Hasselt, M Lanctot, N Freitas. Dueling Network Architectures for Deep Reinforcement Learning. _International Conference on Machine Learning_ , 2016. 

- M Watter, J Springenberg, J Boedecker, M Riedmiller. Embed to Control: A Locally Linear Latent Dynamics Model for Control From Raw Images. _Advances in Neural Information Processing Systems_ , 2015. 

- G Wayne, CC Hung, D Amos, M Mirza, A Ahuja, A Grabska-Barwinska, J Rae, P Mirowski, JZ Leibo, A Santoro, et al. Unsupervised Predictive Memory in a Goal-Directed Agent. _ArXiv Preprint ArXiv:1803.10760_ , 2018. 

- T Weber, S Racanière, DP Reichert, L Buesing, A Guez, DJ Rezende, AP Badia, O Vinyals, N Heess, Y Li, et al. Imagination-Augmented Agents for Deep Reinforcement Learning. _ArXiv Preprint ArXiv:1707.06203_ , 2017. 

- RJ Williams. Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning. _Machine Learning_ , 8(3-4), 1992. 

- Y Wu, E Mansimov, RB Grosse, S Liao, J Ba. Scalable Trust-Region Method for Deep Reinforcement Learning Using Kronecker-Factored Approximation. _Advances in Neural Information Processing Systems_ , 2017. 

- T Yu, G Thomas, L Yu, S Ermon, J Zou, S Levine, C Finn, T Ma. Mopo: Model-Based Offline Policy Optimization. _ArXiv Preprint ArXiv:2005.13239_ , 2020. 

- M Zhang, S Vikram, L Smith, P Abbeel, M Johnson, S Levine. Solar: Deep Structured Representations for Model-Based Reinforcement Learning. _International Conference on Machine Learning_ , 2019. 

15 

Published as a conference paper at ICLR 2021 

## A HUMANOID FROM PIXELS 

**==> picture [65 x 64] intentionally omitted <==**

**==> picture [65 x 65] intentionally omitted <==**

**==> picture [64 x 64] intentionally omitted <==**

**==> picture [64 x 65] intentionally omitted <==**

**==> picture [64 x 64] intentionally omitted <==**

**==> picture [64 x 65] intentionally omitted <==**

**==> picture [64 x 64] intentionally omitted <==**

**==> picture [64 x 65] intentionally omitted <==**

**==> picture [64 x 64] intentionally omitted <==**

**==> picture [64 x 65] intentionally omitted <==**

**==> picture [65 x 64] intentionally omitted <==**

**==> picture [65 x 65] intentionally omitted <==**

Figure A.1: Behavior learned by DreamerV2 on the Humanoid Walk task from pixel inputs only. The task is provided by the DeepMind Control Suite and uses a continuous action space with 21 dimensions. The frames show the agent inputs. 

While the main experiments of this paper focus on the Atari benchmark with discrete actions, DreamerV2 is also applicable to control tasks with continuous actions. For this, we the actor outputs a truncated normal distribution instead of a categorical distribution. To demonstrate the abilities of DreamerV2 for continuous control, we choose the challenging humanoid environment with only image inputs, shown in Figure A.1. We find that for continuous control tasks, dynamics backpropagation substantially outperforms reinforce gradients and thus set _ρ_ = 0. We also set _η_ = 10 _[−]_[5] and _β_ = 2 to further accelerate learning. We find that DreamerV2 reliably solves both the stand-up motion required at the beginning of the episode and the subsequent walking. The score is shown in Figure A.2. To the best of our knowledge, this constitutes the first published result of solving the humanoid environment from only pixel inputs. 

**==> picture [161 x 124] intentionally omitted <==**

**----- Start of picture text -----**<br>
Humanoid Walk<br>800<br>600<br>400<br>200<br>DreamerV2<br>0<br>0 1 2 3 4<br>1e7<br>**----- End of picture text -----**<br>


Figure A.2: Performance on the humanoid walking task from only pixel inputs. 

16 

Published as a conference paper at ICLR 2021 

## B MONTEZUMA’S REVENGE 

**==> picture [65 x 64] intentionally omitted <==**

**==> picture [65 x 65] intentionally omitted <==**

**==> picture [64 x 64] intentionally omitted <==**

**==> picture [64 x 65] intentionally omitted <==**

**==> picture [64 x 64] intentionally omitted <==**

**==> picture [64 x 65] intentionally omitted <==**

**==> picture [64 x 64] intentionally omitted <==**

**==> picture [64 x 65] intentionally omitted <==**

**==> picture [64 x 64] intentionally omitted <==**

**==> picture [64 x 65] intentionally omitted <==**

**==> picture [65 x 64] intentionally omitted <==**

**==> picture [65 x 65] intentionally omitted <==**

Figure B.1: Behavior learned by DreamerV2 on the Atari game Montezuma’s Revenge, that poses a hard exploration challenge. Without any explicit exploration mechanism, DreamerV2 reaches about the same performance as the exploration method ICM. 

While our main experiments use the same hyper parameters across all tasks, we find that DreamerV2 achieves higher performance on Montezuma’s Revenge by using a lower discount factor of _γ_ = 0 _._ 99, possibly to stabilize value learning under sparse rewards. Figure B.2 shows the resulting performance, with all other hyper parameters left at their defaults. DreamerV2 outperforms existing modelfree approaches on the hard-exploration game Montezuma’s Revenge and matches the performance of the explicit exploration algorithm ICM (Pathak et al., 2017) that was applied on top of Rainbow by Taiga et al. (2019). This suggests that the world model may help with solving sparse reward tasks, for example due to improved generalization, efficient policy optimization in the compact latent space enabling more actor critic updates, or because the reward predictor generalizes and thus smooths out the sparse rewards. 

**==> picture [162 x 167] intentionally omitted <==**

**----- Start of picture text -----**<br>
Montezuma Revenge<br>3000<br>2000<br>1000<br>0<br>0.0 0.5 1.0 1.5 2.0<br>1e8<br>DreamerV2 ( =0.99) IQN<br>Rainbow + Curiosity C51<br>Rainbow DQN<br>**----- End of picture text -----**<br>


Figure B.2: Performance on the Atari game Montezuma’s Revenge. 

17 

Published as a conference paper at ICLR 2021 

## C SUMMARY OF MODIFICATIONS 

To develop DreamerV2, we used the Dreamer agent (Hafner et al., 2019) as a starting point. This subsection describes the changes that we applied to the agent to achieve high performance on the Atari benchmark, as well as the changes that were tried but not found to increase performance and thus were not not included in DreamerV2. 

Summary of changes that were tried and were found to help: 

- **Categorical latents** Using categorical latent states using straight-through gradients in the world model instead of Gaussian latents with reparameterized gradients. 

- **KL balancing** Separately scaling the prior cross entropy and the posterior entropy in the KL loss to encourage learning an accurate temporal prior, instead of using free nats. 

- **Reinforce only** Reinforce gradients worked substantially better for Atari than dynamics backpropagation. For continuous control, dynamics backpropagation worked substantially better. 

- **Model size** Increasing the number of units or feature maps per layer of all model components, resulting in a change from 13M parameters to 22M parameters. 

- **Policy entropy** Regularizing the policy entropy for exploration both in imagination and during data collection, instead of using external action noise during data collection. 

Summary of changes that were tried but were found to not help substantially: 

- **Binary latents** Using a larger number of binary latents for the world model instead of categorical latents, which could have encouraged a more disentangled representation, was worse. 

- **Long-term entropy** Including the policy entropy into temporal-difference loss of the value function, so that the actor seeks out states with high action entropy beyond the planning horizon. 

- **Mixed actor gradients** Combining Reinforce and dynamics backpropagation gradients for learning the actor instead of Reinforce provided marginal or no benefits. 

- **Scheduling** Scheduling the learning rates, KL scale, actor entropy loss scale, and actor gradient mixing (from 0.1 to 0) provided marginal or no benefits. 

- **Layer norm** Using layer normalization in the GRU that is used as part of the RSSM latent transition model, instead of no normalization, provided no or marginal benefits. 

Due to the large computational requirements, a comprehensive ablation study on this list of all changes is unfortunately infeasible for us. This would require 55 tasks times 5 seeds for 10 days per change to run, resulting in over 60,000 GPU hours per change. However, we include ablations for the most important design choices in the main text of the paper. 

18 

Published as a conference paper at ICLR 2021 

## D HYPER PARAMETERS 

|**Name**|**Symbol**|**Value**|
|---|---|---|
|World Model|||
|Dataset size (FIFO)|—|2_·_106|
|Batch size|_B_|50|
|Sequence length|_L_|50|
|Discrete latent dimensions|—|32|
|Discrete latent classes|—|32|
|RSSM number of units|—|600|
|KL loss scale|_β_|0.1|
|KL balancing|_α_|0.8|
|World model learning rate|—|2_·_10_−_4|
|Reward transformation|—|tanh|
|Behavior|||
|Imagination horizon|_H_|15|
|Discount|_γ_|0.995|
|_λ_-target parameter|_λ_|0.95|
|Actor gradient mixing|_ρ_|1|
|Actor entropy loss scale|_η_|1_·_10_−_3|
|Actor learning rate|—|4_·_10_−_5|
|Critic learning rate|—|1_·_10_−_4|
|Slow critic update interval|—|100|
|Common|||
|Policy steps per gradient step|—|4|
|MPL number of layers|—|4|
|MPL number of units|—|400|
|Gradient clipping|—|100|
|Adam epsilon|_ϵ_|10_−_5|
|Weight decay (decoupled)|—|10_−_6|



Table D.1: Atari hyper parameters of DreamerV2. When tuning the agent for a new task, we recommend searching over the KL loss scale _β ∈{_ 0 _._ 1 _,_ 0 _._ 3 _,_ 1 _,_ 3 _}_ , actor entropy loss scale _η ∈ {_ 3 _·_ 10 _[−]_[5] _,_ 10 _[−]_[4] _,_ 3 _·_ 10 _[−]_[4] _,_ 10 _[−]_[3] _}_ , and the discount factor _γ ∈{_ 0 _._ 99 _,_ 0 _._ 999 _}_ . The training frequency update should be increased when aiming for higher data-efficiency. 

19 

Published as a conference paper at ICLR 2021 

## E AGENT COMPARISON 

**==> picture [396 x 550] intentionally omitted <==**

**----- Start of picture text -----**<br>
100<br>DreamerV2 vs IQN<br>10<br>0<br>10<br>100<br>100<br>DreamerV2 vs Rainbow<br>10<br>0<br>10<br>100<br>100<br>DreamerV2 vs C51<br>10<br>0<br>10<br>100<br>100<br>DreamerV2 vs DQN<br>10<br>0<br>10<br>100<br>James Bond Up N Down Krull Gopher Demon Attack Assault Road Runner Time Pilot Breakout Asterix Phoenix Qbert Atlantis Zaxxon Ice Hockey Wizard Of Wor Yars Revenge Name This Game Kung Fu Master Robotank Crazy Climber Asteroids Frostbite Centipede Gravitar Beam Rider Kangaroo Fishing Derby Skiing Amidar Bank Heist Tutankham Riverraid Bowling Ms Pacman Berzerk Pitfall Freeway Pong Battle Zone Private Eye Solaris Montezuma Rev. Alien Seaquest Boxing Hero Tennis Enduro Chopper Com. Venture Space Invaders Double Dunk Star Gunner Video Pinball<br>James Bond Up N Down Krull Assault Gopher Demon Attack Road Runner Time Pilot Atlantis Breakout Asterix Phoenix Qbert Zaxxon Ice Hockey Yars Revenge Kung Fu Master Skiing Robotank Wizard Of Wor Name This Game Tennis Asteroids Gravitar Beam Rider Frostbite Crazy Climber Centipede Kangaroo Fishing Derby Ms Pacman Tutankham Alien Bank Heist Bowling Amidar Battle Zone Pitfall Freeway Berzerk Pong Seaquest Solaris Montezuma Rev. Private Eye Riverraid Boxing Enduro Hero Space Invaders Venture Chopper Com. Double Dunk Star Gunner Video Pinball<br>James Bond Up N Down Assault Demon Attack Krull Gopher Road Runner Time Pilot Atlantis Double Dunk Asterix Phoenix Qbert Zaxxon Breakout Yars Revenge Ice Hockey Wizard Of Wor Kangaroo Robotank Kung Fu Master Frostbite Crazy Climber Fishing Derby Skiing Gravitar Boxing Asteroids Beam Rider Amidar Bank Heist Enduro Centipede Battle Zone Name This Game Ms Pacman Alien Riverraid Tutankham Bowling Berzerk Pong Pitfall Freeway Private Eye Solaris Montezuma Rev. Hero Chopper Com. Tennis Seaquest Space Invaders Venture Star Gunner Video Pinball<br>James Bond Up N Down Assault Demon Attack Krull Gopher Road Runner Time Pilot Double Dunk Asterix Atlantis Breakout Phoenix Qbert Zaxxon Ice Hockey Frostbite Yars Revenge Wizard Of Wor Crazy Climber Robotank Kung Fu Master Name This Game Fishing Derby Enduro Boxing Gravitar Tutankham Tennis Centipede Asteroids Kangaroo Amidar Beam Rider Bank Heist Battle Zone Skiing Space Invaders Ms Pacman Riverraid Freeway Alien Hero Seaquest Bowling Berzerk Pong Private Eye Chopper Com. Montezuma Rev. Pitfall Venture Solaris Star Gunner Video Pinball<br>**----- End of picture text -----**<br>


Figure E.1: Atari agent comparison. The bars show the difference in gamer normalized scores at 200M steps. DreamerV2 outperforms the four model-free algorithms IQN, Rainbow, C51, and DQN while learning behaviors purely by planning within a separately learned world model. DreamerV2 achieves higher or similar performance on all tasks besides Video Pinball, where we hypothesize that the reconstruction loss does not focus on the ball that makes up only one pixel on the screen. 

20 

Published as a conference paper at ICLR 2021 

## F MODEL-FREE COMPARISON 

**==> picture [413 x 580] intentionally omitted <==**

**----- Start of picture text -----**<br>
Alien Amidar Assault Asterix Asteroids 1e6Atlantis<br>8000 4000 40000 120000 1.00<br>6000 3000 30000 90000 80000 0.75<br>4000 2000 20000 60000 40000 0.50<br>2000 1000 10000 30000 0 0.25<br>0 0 0 0 40000 0.00<br>0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0<br>Bank Heist Battle Zone Beam Rider Berzerk Bowling Boxing<br>1200900 45000 2000015000 800 60 80<br>600 30000 10000 600 45 40<br>300 15000 5000 400 30 0<br>0 0 200 15<br>0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0<br>Breakout Centipede Chopper Com. Crazy Climber Demon Attack Double Dunk<br>450300150 1500012000900060003000 12000900060003000 1600001200008000040000 1600001200008000040000 15150<br>0 0 0<br>0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0<br>2400 Enduro 8040Fishing Derby 4030 Freeway 2400018000 Frostbite 12000090000 Gopher 4500 Gravitar<br>1600 0 20 12000 60000 3000<br>800 40 10 6000 30000 1500<br>0 80 0 0 0 0<br>0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0<br>Hero Ice Hockey James Bond Kangaroo Krull Kung Fu Master<br>45000 30 60000 16000 100000 80000<br>30000 15 4500030000 120008000 7500050000 6000040000<br>15000 0 15000 4000 25000 20000<br>0 15 0 0 0 0<br>0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0<br>1500Montezuma Rev.8000Ms Pacman 16000Name This Game80000 Phoenix 0 Pitfall 20 Pong<br>1000 6000 12000 60000 80 10<br>500 4000 8000 40000 160 0<br>5000 2000 4000 200000 240 1020<br>0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0<br>Private Eye Qbert Riverraid Road Runner Robotank Seaquest<br>30000 300000200000 2000015000 450000300000 8060 2400018000<br>20000100000 1000001000000 1000050000 1500000 40200 1200060000<br>0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0<br>Skiing Solaris Space Invaders Star Gunner Tennis Time Pilot<br>6000 30<br>12000 2400 6000 75000 15 45000<br>18000 1600 4000 50000 0 30000<br>24000 800 2000 25000 15000<br>30000 0 0 0 15 0<br>0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0<br>Tutankham Up N Down Venture Video Pinball Wizard Of Wor Yars Revenge<br>600000 1500 600000 24000 200000<br>240 450000 1000 450000 18000 150000<br>16080 300000150000 500 300000150000 120006000 10000050000<br>0 0 0 0 0<br>0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0<br>Zaxxon Gamer Median Gamer Mean Record Mean Clip Record Mean<br>60000 2.4 12 0.45 0.24<br>45000 1.8 9 0.30<br>30000 1.2 6 0.16<br>15000 0.6 3 0.15 0.08<br>0 0.0 0 0.00 0.00<br>0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0<br>DreamerV2 IQN Rainbow<br>**----- End of picture text -----**<br>


Figure F.1: Comparison of DreamerV2 to the top model-free RL methods IQN and Rainbow. The curves show mean and standard deviation over 5 seeds. IQN and Rainbow additionally average each point over 10 evaluation episodes, explaining the smoother curves. DreamerV2 outperforms IQN and Rainbow in all four aggregated scores. While IQN and Rainbow tend to succeed on the same tasks, DreamerV2 shows a different performance profile. 

21 

Published as a conference paper at ICLR 2021 

## G LATENTS AND KL BALANCING ABLATIONS 

**==> picture [413 x 579] intentionally omitted <==**

**----- Start of picture text -----**<br>
Alien Amidar Assault Asterix Asteroids 1e6Atlantis<br>4500 3200 12000 32000 16000 1.00<br>2400 9000 24000 12000 0.75<br>3000<br>1600 6000 16000 8000 0.50<br>1500 800 3000 8000 4000 0.25<br>0 0 0 0 0.00<br>0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0<br>1600 Bank Heist 40000Battle Zone Beam Rider 1000 Berzerk Bowling Boxing<br>1200 30000 12000 800 80 90<br>800400 2000010000 80004000 600400 604020 60300<br>0 0 200 30<br>0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0<br>Breakout Centipede Chopper Com. Crazy Climber Demon Attack Double Dunk<br>450 10000 160000 20<br>300 75005000 45003000 12000080000 120006000 100<br>150 2500 1500 40000 0 10<br>0 0 6000 20<br>0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0<br>Enduro Fishing Derby Freeway Frostbite Gopher Gravitar<br>100000<br>2400 40 32 80000 75000 4500<br>18001200600 40800 24168 600004000020000 5000025000 30001500<br>0 0 0 0<br>0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0<br>Hero Ice Hockey James Bond Kangaroo Krull Kung Fu Master<br>3200024000160008000 30150 180001200060000 1200080004000 10000800060004000 100000750005000025000<br>0 15 0 2000<br>0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0<br>Montezuma Rev. Ms Pacman Name This Game Phoenix Pitfall Pong<br>16000 20<br>2400 6000 12000 30000 0 10<br>1600800 45003000 8000 2000010000 16080 100<br>0 1500 4000 0 240 20<br>0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0<br>6000Private Eye 450000 Qbert Riverraid Road Runner Robotank Seaquest<br>45003000 300000 1600012000 160000120000 6040 16000080000<br>15000 1500000 80004000 80000400000 200 800000<br>0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0<br>Skiing Solaris Space Invaders Star Gunner Tennis Time Pilot<br>6000 6000 4000 30 45000<br>12000 4500 3000 30000 15 30000<br>18000 3000 2000 20000 0<br>24000 1500 1000 10000 15 15000<br>30000 0 0 0<br>0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0<br>Tutankham Up N Down Venture Video Pinball Wizard Of Wor Yars Revenge<br>60<br>300 600000 40 45000 24000 60000<br>240 18000<br>180 400000 20 30000 12000 40000<br>120 200000 0 15000 6000 20000<br>0 20 0 0<br>0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0<br>Zaxxon Gamer Median Gamer Mean Record Mean Clip Record Mean<br>20000 2.0 6.0 0.24<br>15000 1.5 4.5 0.3 0.18<br>10000 1.0 3.0 0.2 0.12<br>5000 0.5 1.5 0.1 0.06<br>0 0.0 0.0 0.0 0.00<br>0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0<br>DreamerV2 Gaussian Latents No KL Balance<br>**----- End of picture text -----**<br>


Figure G.1: Comparison of DreamerV2, Gaussian instead of categorical latent variables, and no KL balancing. The ablation experiments use a slightly earlier version of the agent. The curves show mean and standard deviation across two seeds. Categorical latent variables and KL balancing both substantially improve performance across many of the tasks. The importance of the two techniques is reflected in all four aggregated scores. 

22 

Published as a conference paper at ICLR 2021 

## H REPRESENTATION LEARNING ABLATIONS 

**==> picture [416 x 580] intentionally omitted <==**

**----- Start of picture text -----**<br>
Alien Amidar Assault Asterix Asteroids 1e6Atlantis<br>6000 3200 12000 30000 6000 1.00<br>2400 4500 0.75<br>4000 1600 8000 20000 3000 0.50<br>2000 800 4000 10000 1500 0.25<br>0 0 0 0 0 0.00<br>0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0<br>1200 Bank Heist 40000Battle Zone Beam Rider 1000 Berzerk Bowling 100 Boxing<br>900 30000 12000 800 60 50<br>600 20000 8000 600 40 0<br>300 10000 4000 400 20 50<br>200<br>0 0 0 0 100<br>0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0<br>Breakout Centipede Chopper Com. Crazy Climber Demon Attack Double Dunk<br>450 100008000 80006000 160000120000 1800012000 15<br>300 6000 4000 80000 6000 0<br>150 4000 2000 40000 0 15<br>0 2000 0 0 6000<br>0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0<br>Enduro Fishing Derby Freeway Frostbite Gopher Gravitar<br>24001800 400 3224 3200024000 10000075000 45003000<br>1200 16 16000 50000<br>600 40 8 8000 25000 1500<br>80<br>0 0 0 0 0<br>0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0<br>320002400016000 Hero 30150Ice Hockey 18000120006000James Bond 16000120008000 Kangaroo 1000075005000 Krull 1000007500050000Kung Fu Master<br>8000 15 0 4000 2500 25000<br>0 0 0<br>0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0<br>Montezuma Rev. Ms Pacman Name This Game Phoenix Pitfall Pong<br>8000 16000 0 20<br>2400 6000 12000 30000 150 10<br>1600 4000 8000 20000 300 0<br>800 2000 4000 10000 450 10<br>0 0 0 0 600 20<br>0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0<br>Private Eye Qbert Riverraid Road Runner Robotank Seaquest<br>450000 20000 320000 80<br>2000 300000 15000 240000 60 160000<br>1000 150000 10000 160000 40 80000<br>0 5000 80000 20 0<br>0<br>1000 0 0 0 80000<br>0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0<br>Skiing Solaris Space Invaders Star Gunner Tennis Time Pilot<br>12000180006000 60004000 400030002000 3000020000 30150 4500030000<br>24000 2000 1000 10000 15 15000<br>30000 0 0 0 0<br>0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0<br>Tutankham Up N Down Venture Video Pinball Wizard Of Wor Yars Revenge<br>60 100000<br>320240 600000 40 8000060000 2400018000 75000<br>160 400000 20 40000 12000 50000<br>80 200000 0 20000 6000 25000<br>0 0 20 0 0 0<br>0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0<br>Zaxxon Gamer Median Gamer Mean Record Mean Clip Record Mean<br>20000 2.0 6.0 0.24<br>0.3<br>15000 1.5 4.5 0.16<br>10000 1.0 3.0 0.2<br>5000 0.5 1.5 0.1 0.08<br>0 0.0 0.0 0.0 0.00<br>0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0<br>DreamerV2 No Reward Gradients No Image Gradients<br>**----- End of picture text -----**<br>


Figure H.1: Comparison of leveraging image prediction, reward prediction, or both for learning the model representations. While image gradients are crucial, reward gradients are not necessary for our world model to succeed and their gradients can be stopped. Representations learned purely from images are not biased toward previously encountered rewards and outperform reward-specific representations on a number of tasks, suggesting that they may generalize better to unseen situations. 

23 

Published as a conference paper at ICLR 2021 

**==> picture [413 x 594] intentionally omitted <==**

**----- Start of picture text -----**<br>
I POLICY LEARNING ABLATIONS<br>Alien Amidar Assault Asterix Asteroids 1e6Atlantis<br>3200 12000 32000 6000 1.00<br>7500 2400 9000 24000 4500 0.75<br>5000 1600 6000 16000 3000 0.50<br>2500 800 3000 8000 1500 0.25<br>0 0 0 0 0 0.00<br>0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0<br>Bank Heist Battle Zone Beam Rider Berzerk Bowling Boxing<br>1200900600300 40000300002000010000 160001200080004000 12501000750500 75604530 906030<br>0 0 250 15 0<br>0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0<br>Breakout Centipede Chopper Com. Crazy Climber Demon Attack Double Dunk<br>450 10000 6000 160000 20<br>300 7500 4500 120000 16000 10<br>5000 3000 80000 8000 0<br>150 2500 1500 40000 0 10<br>0 0 8000 20<br>0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0<br>Enduro Fishing Derby Freeway Frostbite Gopher Gravitar<br>240018001200 400 322416 320002400016000 1000007500050000 45003000<br>600 40 8 8000 25000 1500<br>80<br>0 0 0 0 0<br>0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0<br>Hero Ice Hockey James Bond Kangaroo Krull Kung Fu Master<br>3200024000160008000 30150 180001200060000 1200080004000 1000075005000 100000750005000025000<br>0 15 0 2500 0<br>0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0<br>Montezuma Rev. Ms Pacman Name This Game Phoenix Pitfall Pong<br>16000 20<br>24001600800 600045003000 120008000 300002000010000 160800 10100<br>0 1500 4000 0 240 20<br>0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0<br>Private Eye Qbert Riverraid Road Runner Robotank Seaquest<br>750500250 450000300000150000 200001500010000 450000300000 604530 16000080000<br>0 0 5000 150000 15 0<br>250 0 0 80000<br>0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0<br>Skiing Solaris Space Invaders Star Gunner Tennis Time Pilot<br>12000180006000 45003000 600045003000 400003000020000 30150 4500030000<br>2400030000 15000 15000 100000 15 150000<br>0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0<br>Tutankham Up N Down Venture Video Pinball Wizard Of Wor Yars Revenge<br>320240160 600000400000 604020 600004500030000 240001800012000 800006000040000<br>80 200000 0 15000 6000 20000<br>0 0 20 0 0<br>0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0<br>Zaxxon Gamer Median Gamer Mean Record Mean Clip Record Mean<br>32000 2.0 6.0 0.24<br>24000 1.5 4.5 0.3 0.18<br>16000 1.0 3.0 0.2 0.12<br>8000 0.5 1.5 0.1 0.06<br>0 0.0 0.0 0.0 0.00<br>0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0<br>DreamerV2 No Straight-Through No Reinforce<br>**----- End of picture text -----**<br>


Figure I.1: Comparison of leveraging Reinforce gradients, straight-through gradients, or both for training the actor. While Reinforce gradients are crucial, straight-through gradients are not important for most of the tasks. Nonetheless, combining both gradients yields substantial improvements on a small number of games, most notably on Seaquest. We conjecture that straight-through gradients have low variance and thus help the agent start learning, whereas Reinforce gradients are unbiased and help converging to a better solution. 

24 

Published as a conference paper at ICLR 2021 

## J ADDITIONAL ABLATIONS 

**==> picture [413 x 580] intentionally omitted <==**

**----- Start of picture text -----**<br>
Alien Amidar Assault Asterix Asteroids 1e6Atlantis<br>4500 3200 12000 32000 6000 1.00<br>3000 2400 9000 24000 4500 0.75<br>1600 6000 16000 3000 0.50<br>1500 800 3000 8000 1500 0.25<br>0 0 0 0 0.00<br>0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0<br>Bank Heist Battle Zone Beam Rider Berzerk Bowling Boxing<br>1200 1250 75<br>900 45000 12000 1000 60 90<br>600 30000 8000 750 45 60<br>300 15000 4000 500 30 30<br>0 0 0 250 15 0<br>0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0<br>Breakout 10000 Centipede 5000Chopper Com. Crazy Climber 24000Demon Attack Double Dunk<br>450 160000<br>300 7500 40003000 120000 16000 15<br>150 50002500 20001000 8000040000 80000 150<br>0 0 8000<br>0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0<br>Enduro Fishing Derby Freeway Frostbite Gopher Gravitar<br>2400 80 32 32000 100000 4500<br>1800 40 24 24000 75000<br>1200 0 16 16000 50000 3000<br>600 40 8 8000 25000 1500<br>80<br>0 0 0 0 0<br>0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0<br>Hero Ice Hockey James Bond Kangaroo Krull Kung Fu Master<br>45 32000 120000<br>10000<br>30000 30 24000 12000 90000<br>8000<br>20000 15 16000 8000 60000<br>10000 0 8000 4000 6000 30000<br>0 15 0 0 4000 0<br>0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0<br>Montezuma Rev. Ms Pacman Name This Game Phoenix Pitfall Pong<br>16000 20<br>2400 6000 12000 30000 0 10<br>1600 45003000 8000 20000 80 0<br>800 1500 4000 10000 160 10<br>0 0 0 240 20<br>0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0<br>2000015000100005000Private Eye 4500003000001500000 Qbert 160001200080004000 Riverraid 24000018000012000060000Road Runner 60453015 Robotank 160000800000 Seaquest<br>0 0 0 80000<br>0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0<br>Skiing Solaris Space Invaders Star Gunner Tennis Time Pilot<br>6000 4000 40000 30 45000<br>12000 4500 3000 30000 15 30000<br>18000 3000 2000 20000 0<br>24000 1500 1000 10000 15 15000<br>30000 0 0 0 0<br>0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0<br>Tutankham Up N Down Venture Video Pinball Wizard Of Wor Yars Revenge<br>320 60 45000 24000 32000<br>240160 600000400000 4020 30000 1800012000 2400016000<br>80 200000 0 15000 6000 8000<br>0 20 0 0 0<br>0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0<br>Zaxxon Gamer Median Gamer Mean Record Mean Clip Record Mean<br>20000 2.0 6.0 0.24<br>15000 1.5 4.5 0.3 0.18<br>10000 1.0 3.0 0.2 0.12<br>5000 0.5 1.5 0.1 0.06<br>0 0.0 0.0 0.0 0.00<br>0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0 0.0 0.5 1.0 1.5 2.0<br>DreamerV2 No Layer Norm Random Data<br>**----- End of picture text -----**<br>


Figure J.1: Comparison of DreamerV2 to a version without layer norm in the GRU and to training from experience collected over time by a uniform random policy. We find that the benefit of layer norm depends on the task at hand, increasing and decreasing performance on a roughly equal number of tasks. The comparison to random data collection highlights which of the tasks require non-trivial exploration, which can help guide future work on directed exploration using world models. 

25 

Published as a conference paper at ICLR 2021 

## K ATARI TASK SCORES 

|**Task**|Baselines<br>**Random**<br>**Gamer**<br>**Record**|Algorithms<br>**Rainbow**<br>**IQN**<br>**DreamerV2**|
|---|---|---|
|Alien<br>Amidar<br>Assault<br>Asterix<br>Asteroids<br>Atlantis<br>Bank Heist<br>Battle Zone<br>Beam Rider<br>Berzerk<br>Bowling<br>Boxing<br>Breakout<br>Centipede<br>Chopper Command<br>Crazy Climber<br>Demon Attack<br>Double Dunk<br>Enduro<br>Fishing Derby<br>Freeway<br>Frostbite<br>Gopher<br>Gravitar<br>Hero<br>Ice Hockey<br>James Bond<br>Kangaroo<br>Krull<br>Kung Fu Master<br>Montezuma Revenge<br>Ms Pacman<br>Name This Game<br>Phoenix<br>Pitfall<br>Pong<br>Private Eye<br>Qbert<br>Riverraid<br>Road Runner<br>Robotank<br>Seaquest<br>Skiing<br>Solaris<br>Space Invaders<br>Star Gunner<br>Tennis<br>Time Pilot<br>Tutankham<br>Up N Down<br>Venture<br>Video Pinball<br>Wizard Of Wor<br>Yars Revenge<br>Zaxxon|229<br>7128<br>251916<br>6<br>1720<br>104159<br>222<br>742<br>8647<br>210<br>8503<br>1000000<br>719<br>47389<br>10506650<br>12850<br>29028<br>10604840<br>14<br>753<br>82058<br>2360<br>37188<br>801000<br>364<br>16926<br>999999<br>124<br>2630<br>1057940<br>23<br>161<br>300<br>0<br>12<br>100<br>2<br>30<br>864<br>2091<br>12017<br>1301709<br>811<br>7388<br>999999<br>10780<br>35829<br>219900<br>152<br>1971<br>1556345<br>-19<br>-16<br>22<br>0<br>860<br>9500<br>-92<br>-39<br>71<br>0<br>30<br>38<br>65<br>4335<br>454830<br>258<br>2412<br>355040<br>173<br>3351<br>162850<br>1027<br>30826<br>1000000<br>-11<br>1<br>36<br>7<br>303<br>45550<br>52<br>3035<br>1424600<br>1598<br>2666<br>104100<br>258<br>22736<br>1000000<br>0<br>4753<br>1219200<br>307<br>6952<br>290090<br>2292<br>8049<br>25220<br>761<br>7243<br>4014440<br>-229<br>6464<br>114000<br>-21<br>15<br>21<br>25<br>69571<br>101800<br>164<br>13455<br>2400000<br>1338<br>17118<br>1000000<br>12<br>7845<br>2038100<br>2<br>12<br>76<br>68<br>42055<br>999999<br>-17098<br>-4337<br>-3272<br>1236<br>12327<br>111420<br>148<br>1669<br>621535<br>664<br>10250<br>77400<br>-24<br>-8<br>21<br>3568<br>5229<br>65300<br>11<br>168<br>5384<br>533<br>11693<br>82840<br>0<br>1188<br>38900<br>16257<br>17668<br>89218328<br>564<br>4756<br>395300<br>3093<br>54577<br>15000105<br>32<br>9173<br>83700|3457<br>**4961**<br>3967<br>**2529**<br>2393<br>**2577**<br>3229<br>4885<br>**23625**<br>18367<br>10374<br>**72311**<br>1484<br>1585<br>**41526**<br>802548<br>890214<br>**978778**<br>**1075**<br>1052<br>**1126**<br>**40061**<br>**40953**<br>**40325**<br>6290<br>7130<br>**18646**<br>**833**<br>648<br>**810**<br>43<br>39<br>**49**<br>**99**<br>**98**<br>92<br>120<br>79<br>**312**<br>6510<br>3728<br>**11883**<br>**12338**<br>9282<br>2861<br>145389<br>132738<br>**161839**<br>17071<br>15350<br>**82263**<br>**22**<br>**21**<br>17<br>**2200**<br>**2203**<br>1656<br>42<br>45<br>**65**<br>**34**<br>**34**<br>**33**<br>8208<br>7812<br>**11384**<br>10641<br>12108<br>**92282**<br>1272<br>1347<br>**3789**<br>**46675**<br>36058<br>21868<br>0<br>-5<br>**26**<br>1097<br>3166<br>**40445**<br>12748<br>12602<br>**14064**<br>4066<br>8844<br>**50061**<br>26475<br>31653<br>**62741**<br>**500**<br>**500**<br>81<br>3861<br>5218<br>**5652**<br>9026<br>6639<br>**14649**<br>8545<br>5102<br>**49375**<br>-20<br>-13<br>**0**<br>**20**<br>**20**<br>**20**<br>**21334**<br>4181<br>2198<br>17383<br>16730<br>**94688**<br>**20756**<br>15183<br>16351<br>54662<br>58966<br>**203576**<br>66<br>66<br>**78**<br>9903<br>**17039**<br>7480<br>-28708<br>-11162<br>**-9299**<br>1583<br>**1684**<br>922<br>4131<br>**4530**<br>2474<br>57909<br>**80003**<br>7800<br>0<br>**23**<br>14<br>12051<br>11666<br>**37945**<br>239<br>**251**<br>**264**<br>34888<br>59944<br>**653662**<br>**1529**<br>1313<br>2<br>**466895**<br>415833<br>41860<br>7879<br>5671<br>**12851**<br>45542<br>84144<br>**156748**<br>14603<br>11023<br>**50699**|



Table K.1: Atari individual scores. We select the 55 games that are common among most papers in the literature. We compare the algorithms DreamerV2, IQN, and Rainbow to the baselines of random actions, DeepMind’s human gamer, and the human world record. Algorithm scores are highlighted in bold when they fall within 5% of the best algorithm. Note that these scores are already averaged across seeds, whereas any aggregated scores must be computed before averaging across seeds. 

26 

