# ML Lab Viva Questions & Answers

---

## 1. Exploratory Data Analysis (EDA)

**Q: What is EDA?**
> EDA is the process of analyzing datasets to summarize their main characteristics using statistics and visualizations before applying ML models.

**Q: Why is EDA important before building ML models?**
> It helps understand data distribution, detect outliers, find missing values, identify correlations, and choose appropriate preprocessing/modeling techniques.

**Q: What is an outlier?**
> A data point that significantly differs from other observations. Can be errors or genuine extreme values.

**Q: How do you detect outliers using IQR method?**
> Values below Q1 - 1.5×IQR or above Q3 + 1.5×IQR are outliers, where IQR = Q3 - Q1.

**Q: What is the difference between mean and median?**
> Mean is the average (affected by outliers); median is the middle value (robust to outliers).

**Q: What does a skewed distribution indicate?**
> Data is not symmetric. Right-skewed: mean > median (long tail right). Left-skewed: mean < median.

**Q: How do histograms help in EDA?**
> They show frequency distribution, help identify skewness, modality (peaks), and spread of data.

---

## 2. Correlation Analysis

**Q: What is correlation?**
> A statistical measure of the linear relationship between two variables, ranging from -1 to +1.

**Q: What do correlation values indicate?**
> +1: Perfect positive correlation, -1: Perfect negative correlation, 0: No linear relationship.

**Q: What is the difference between correlation and causation?**
> Correlation shows variables move together; causation means one variable causes change in another. Correlation ≠ Causation.

**Q: What is multicollinearity?**
> When independent variables are highly correlated with each other, causing issues in regression models.

**Q: Why is multicollinearity a problem?**
> It makes it difficult to determine individual feature importance and can make model coefficients unstable.

**Q: What is Pearson correlation coefficient?**
> Measures linear correlation between two continuous variables. Formula: cov(X,Y) / (σx × σy).

**Q: When would you use Spearman correlation instead of Pearson?**
> When data is ordinal, non-linear, or has outliers. Spearman measures monotonic relationships.

---

## 3. Principal Component Analysis (PCA)

**Q: What is dimensionality reduction?**
> Reducing the number of features while preserving important information. Helps with visualization, speed, and overfitting.

**Q: What is PCA?**
> An unsupervised technique that transforms data into orthogonal principal components ordered by variance captured.

**Q: Why is scaling necessary before PCA?**
> PCA is variance-based; features with larger scales would dominate. Standardization ensures equal contribution.

**Q: What is explained variance ratio?**
> The proportion of total variance captured by each principal component. Helps decide how many components to keep.

**Q: What are eigenvectors and eigenvalues in PCA?**
> Eigenvectors define direction of principal components; eigenvalues indicate variance along each direction.

**Q: How do you choose the number of components?**
> Keep components that capture 95% of variance, or use elbow method on cumulative explained variance plot.

**Q: What are the limitations of PCA?**
> Assumes linear relationships, sensitive to outliers, components may not be interpretable, loses some information.

---

## 4. Candidate Elimination Algorithm

**Q: What is concept learning?**
> Learning a boolean-valued function (concept) from positive and negative training examples.

**Q: What is a hypothesis in ML?**
> A candidate function that maps inputs to outputs. In concept learning, it defines conditions for classification.

**Q: What is the hypothesis space?**
> The set of all possible hypotheses that a learning algorithm can consider.

**Q: What is the Candidate Elimination algorithm?**
> It finds all hypotheses consistent with training data by maintaining specific (S) and general (G) boundaries.

**Q: What does the specific boundary (S) represent?**
> The most specific hypothesis that covers all positive examples seen so far.

**Q: What does the general boundary (G) represent?**
> The most general hypotheses that exclude all negative examples seen so far.

**Q: What is the version space?**
> The set of all hypotheses between S and G that are consistent with all training examples.

**Q: How does the algorithm handle positive vs negative examples?**
> Positive: Generalize S to include it. Negative: Specialize G to exclude it.

**Q: What is inductive bias?**
> The set of assumptions a learner uses to predict outputs for unseen inputs.

---

## 5. Linear Regression

**Q: What is regression?**
> Predicting a continuous output variable based on input features.

**Q: What is the linear regression equation?**
> y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε (intercept + weighted sum of features + error)

**Q: What are the assumptions of linear regression?**
> Linearity, independence, homoscedasticity (constant variance), normality of residuals, no multicollinearity.

**Q: What is R² score?**
> Coefficient of determination. Measures proportion of variance explained by the model. Range: 0 to 1 (1 is perfect).

**Q: What is RMSE?**
> Root Mean Squared Error = √(Σ(actual - predicted)²/n). Measures average prediction error in original units.

**Q: What is the difference between R² and Adjusted R²?**
> Adjusted R² penalizes adding irrelevant features. R² always increases with more features; adjusted R² may decrease.

**Q: What is the cost function in linear regression?**
> Mean Squared Error (MSE). The algorithm minimizes MSE to find optimal coefficients.

**Q: What is gradient descent?**
> An optimization algorithm that iteratively updates parameters in the direction that minimizes the cost function.

---

## 6. Polynomial Regression

**Q: When do you use polynomial regression?**
> When the relationship between X and y is non-linear and cannot be captured by a straight line.

**Q: How does polynomial regression work?**
> It adds polynomial terms (x², x³, etc.) as new features, then applies linear regression.

**Q: What is the bias-variance tradeoff?**
> Low complexity (high bias): underfitting. High complexity (high variance): overfitting. Need balance.

**Q: What is overfitting?**
> Model fits training data too well (including noise) but performs poorly on new data.

**Q: What is underfitting?**
> Model is too simple to capture underlying patterns; performs poorly on both training and test data.

**Q: How do you detect overfitting?**
> High training accuracy but low test accuracy. Large gap between training and validation performance.

**Q: How do you prevent overfitting?**
> Cross-validation, regularization (L1/L2), reduce model complexity, get more data, early stopping.

**Q: What is regularization?**
> Adding a penalty term to the cost function to discourage complex models. L1 (Lasso), L2 (Ridge).

---

## 7. Logistic Regression

**Q: What is classification?**
> Predicting a categorical/discrete output (class label) from input features.

**Q: What is logistic regression?**
> A classification algorithm that predicts probability of a binary outcome using the sigmoid function.

**Q: What is the sigmoid function?**
> σ(z) = 1/(1 + e^(-z)). Maps any real number to range [0, 1] for probability interpretation.

**Q: Why can't we use linear regression for classification?**
> Linear regression outputs can exceed [0,1] range and doesn't model probability properly.

**Q: What is the decision boundary?**
> The threshold (usually 0.5) that separates classes based on predicted probability.

**Q: What is a confusion matrix?**
> A table showing True Positives (TP), True Negatives (TN), False Positives (FP), False Negatives (FN).

**Q: What is precision?**
> TP / (TP + FP). Of all positive predictions, how many were correct.

**Q: What is recall (sensitivity)?**
> TP / (TP + FN). Of all actual positives, how many were correctly identified.

**Q: What is the F1 score?**
> Harmonic mean of precision and recall: 2 × (Precision × Recall) / (Precision + Recall).

**Q: What is the ROC curve?**
> Plot of True Positive Rate vs False Positive Rate at various classification thresholds.

**Q: What is AUC?**
> Area Under the ROC Curve. 1.0 = perfect, 0.5 = random. Measures overall model performance.

---

## 8. K-Nearest Neighbors (KNN)

**Q: What is KNN?**
> A lazy learning algorithm that classifies based on majority vote of K nearest neighbors.

**Q: Why is KNN called a lazy learner?**
> It doesn't learn a model during training; it stores data and computes distances at prediction time.

**Q: What distance metric does KNN typically use?**
> Euclidean distance: √Σ(xᵢ - yᵢ)². Others: Manhattan, Minkowski, Cosine.

**Q: Why is feature scaling important for KNN?**
> KNN uses distances; features with larger scales would dominate the distance calculation.

**Q: How does the value of K affect the model?**
> Small K: Sensitive to noise (overfitting). Large K: Over-smoothing (underfitting).

**Q: How do you choose the optimal K?**
> Use cross-validation. Try odd values to avoid ties. Typically √n as starting point.

**Q: What are the advantages of KNN?**
> Simple, no training time, works for multi-class, no assumptions about data distribution.

**Q: What are the disadvantages of KNN?**
> Slow prediction (distance to all points), sensitive to irrelevant features, memory intensive.

**Q: What is the curse of dimensionality?**
> In high dimensions, distances become similar, making nearest neighbors less meaningful.

---

## 9. Decision Tree

**Q: What is a decision tree?**
> A tree-structured model that makes decisions by splitting data based on feature values.

**Q: What are the components of a decision tree?**
> Root node (first split), internal nodes (decisions), branches (outcomes), leaf nodes (predictions).

**Q: What is Gini impurity?**
> Gini = 1 - Σ(pᵢ²). Measures probability of misclassifying a random sample. Range: 0 (pure) to 0.5 (impure).

**Q: What is entropy?**
> Entropy = -Σ(pᵢ × log₂(pᵢ)). Measures disorder/uncertainty. Range: 0 (pure) to 1 (impure).

**Q: What is information gain?**
> Reduction in entropy after splitting on a feature. Used to select best split.

**Q: How does a decision tree choose which feature to split on?**
> Selects feature that maximizes information gain (or minimizes Gini impurity).

**Q: What is pruning?**
> Removing branches that don't improve generalization to prevent overfitting.

**Q: What is the difference between pre-pruning and post-pruning?**
> Pre-pruning: Stop growing early (max_depth, min_samples). Post-pruning: Grow full tree, then remove branches.

**Q: What are advantages of decision trees?**
> Easy to interpret, no scaling needed, handles both numerical and categorical data.

**Q: What are disadvantages of decision trees?**
> Prone to overfitting, unstable (small changes cause different trees), biased toward features with more levels.

---

## 10. Expectation-Maximization (EM) Algorithm

**Q: What is clustering?**
> Unsupervised learning that groups similar data points without predefined labels.

**Q: What is the EM algorithm?**
> An iterative algorithm for finding maximum likelihood estimates when data has latent (hidden) variables.

**Q: What are the two steps in EM?**
> E-step: Compute expected values of hidden variables. M-step: Maximize parameters given those expectations.

**Q: What is a Gaussian Mixture Model (GMM)?**
> A probabilistic model that assumes data is generated from a mixture of K Gaussian distributions.

**Q: What are the parameters of a GMM?**
> Means (μ), covariances (Σ), and mixing weights (π) for each component.

**Q: What is the difference between K-Means and GMM?**
> K-Means: Hard assignment, spherical clusters. GMM: Soft assignment (probabilities), elliptical clusters.

**Q: What are responsibilities in EM?**
> The probability that each data point belongs to each cluster (soft assignment).

**Q: How do you choose the number of clusters?**
> Elbow method, silhouette score, BIC/AIC for GMM.

**Q: What is a local optimum?**
> EM may converge to different solutions based on initialization, not necessarily the global best.

---

## 11. Ensemble Learning

**Q: What is ensemble learning?**
> Combining multiple models to achieve better performance than individual models.

**Q: What is the wisdom of crowds principle?**
> Aggregating multiple opinions often gives better results than any single expert.

**Q: What is bagging?**
> Bootstrap Aggregating: Train models on random subsets (with replacement), then average/vote predictions.

**Q: What is Random Forest?**
> Bagging with decision trees + random feature selection at each split.

**Q: Why does Random Forest work well?**
> Reduces variance by averaging many uncorrelated trees, each trained on different data/features.

**Q: What is boosting?**
> Sequentially training models where each focuses on correcting errors of previous models.

**Q: What is the difference between bagging and boosting?**
> Bagging: Parallel, reduces variance. Boosting: Sequential, reduces bias.

**Q: What is AdaBoost?**
> Adjusts sample weights; misclassified samples get higher weights for next model.

**Q: What is Gradient Boosting?**
> Each model fits the residuals (errors) of the previous model directly.

**Q: What is a voting classifier?**
> Combines different model types and predicts by majority vote (hard) or averaged probabilities (soft).

---

## 12. Reinforcement Learning

**Q: What is reinforcement learning?**
> Learning through interaction with an environment to maximize cumulative reward via trial and error.

**Q: What are the key components of RL?**
> Agent, Environment, State, Action, Reward, Policy.

**Q: What is a policy?**
> A mapping from states to actions. Defines the agent's behavior.

**Q: What is the reward signal?**
> Immediate feedback from the environment indicating how good an action was.

**Q: What is the discount factor (γ)?**
> Determines importance of future rewards. γ=0: Only immediate rewards. γ≈1: Future rewards matter.

**Q: What is the return (G)?**
> Cumulative discounted reward: G = r₀ + γr₁ + γ²r₂ + ...

**Q: What is exploration vs exploitation?**
> Exploration: Try new actions. Exploitation: Use known best actions. Need balance.

**Q: What is the difference between value-based and policy-based methods?**
> Value-based: Learn value function, derive policy. Policy-based: Directly learn the policy.

**Q: What is the policy gradient method?**
> Directly optimizes the policy by computing gradients of expected reward with respect to policy parameters.

---

## General ML Fundamentals

**Q: What is the difference between supervised and unsupervised learning?**
> Supervised: Labeled data (classification, regression). Unsupervised: No labels (clustering, dimensionality reduction).

**Q: Why do we split data into training and test sets?**
> To evaluate model performance on unseen data and detect overfitting.

**Q: What is cross-validation?**
> Splitting data into K folds, training on K-1, testing on 1, repeated K times. Gives robust performance estimate.

**Q: What is feature scaling and why is it important?**
> Normalizing features to similar ranges. Important for distance-based algorithms and gradient descent.

**Q: What is the difference between StandardScaler and MinMaxScaler?**
> StandardScaler: Mean=0, Std=1. MinMaxScaler: Scales to [0,1] range.

**Q: What is one-hot encoding?**
> Converting categorical variables into binary columns (one column per category).

**Q: What is train-test split ratio typically used?**
> 80-20 or 70-30 (training-testing). With validation: 60-20-20.

**Q: What is a hyperparameter?**
> Model configuration set before training (e.g., K in KNN, learning rate, max_depth).

**Q: What is the no free lunch theorem?**
> No single algorithm works best for all problems. Algorithm choice depends on the data.
