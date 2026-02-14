.

📘 Linear Regression – Complete Structured Notes
1️⃣ Introduction

Linear Regression is a supervised learning algorithm used to predict continuous numerical values.
It models the relationship between independent variables (X) and a dependent variable (Y) using a straight line.

It assumes that there is a linear relationship between input features and output.

Common examples:

House price prediction

Salary estimation

Sales forecasting

2️⃣ Mathematical Model
Simple Linear Regression
𝑦
=
𝑏
0
+
𝑏
1
𝑥
y=b
0
	​

+b
1
	​

x

Where:

y = Predicted value

x = Input feature

b₀ = Intercept

b₁ = Slope

Multiple Linear Regression
𝑦
=
𝑏
0
+
𝑏
1
𝑥
1
+
𝑏
2
𝑥
2
+
.
.
.
+
𝑏
𝑛
𝑥
𝑛
y=b
0
	​

+b
1
	​

x
1
	​

+b
2
	​

x
2
	​

+...+b
n
	​

x
n
	​


The goal is to find coefficients that minimize error.

3️⃣ Line of Best Fit

Linear Regression finds the best straight line that minimizes total prediction error.

        Y
        |
        |                    ●
        |               ●
        |          ●
        |     ●
        | ●
        |________________________________ X
                 \
                  \
                   \  Best Fit Line


The slope determines direction of relationship.

4️⃣ Residuals (Error Concept)

Residual is the vertical distance between actual value and predicted value.

𝑅
𝑒
𝑠
𝑖
𝑑
𝑢
𝑎
𝑙
=
𝑦
−
𝑦
^
Residual=y−
y
^
	​

           ●  (Actual)
           |
           |   Residual
           |
-----------+------------------
          Regression Line


Good model → Residuals randomly scattered
Bad model → Residuals show pattern

5️⃣ Cost Function

Linear Regression minimizes Mean Squared Error (MSE).

𝑀
𝑆
𝐸
=
1
𝑛
∑
(
𝑦
−
𝑦
^
)
2
MSE=
n
1
	​

∑(y−
y
^
	​

)
2

Why square errors?

Removes negative sign

Penalizes large errors more

Lower MSE means better model performance.

6️⃣ Gradient Descent

Gradient Descent is used to minimize the cost function.

Update rule:

𝑏
=
𝑏
−
𝛼
×
∂
𝐶
𝑜
𝑠
𝑡
∂
𝑏
b=b−α×
∂b
∂Cost
	​


Where α is learning rate.

Cost
  |
  |\
  | \
  |  \
  |   \
  |    \____
  |
  +---------------- Iterations


Learning rate controls speed of convergence.

7️⃣ Model Evaluation Metrics
MAE

Average absolute difference between actual and predicted values.

MSE

Average squared difference.

RMSE

Square root of MSE. Same unit as target.

R² Score
𝑅
2
=
1
−
𝑆
𝑆
𝑟
𝑒
𝑠
𝑆
𝑆
𝑡
𝑜
𝑡
𝑎
𝑙
R
2
=1−
SS
total
	​

SS
res
	​

	​


Range: 0 to 1
Higher value → Better model

8️⃣ Adjusted R²

R² increases when more features are added, even if they are useless.

Adjusted R² penalizes unnecessary features.

𝐴
𝑑
𝑗
𝑢
𝑠
𝑡
𝑒
𝑑
 
𝑅
2
=
1
−
(
1
−
𝑅
2
)
(
𝑛
−
1
)
(
𝑛
−
𝑘
−
1
)
Adjusted R
2
=1−
(n−k−1)
(1−R
2
)(n−1)
	​


Useful for comparing multiple regression models.

9️⃣ Underfitting vs Overfitting
Underfitting

Model too simple

High bias

Poor performance on train & test

Data:   ●   ●   ●
Model:  ----------

Overfitting

Model too complex

High variance

High train accuracy, low test accuracy

Data:   ●   ●   ●
Model:  /\/\/\/\/\/\


Regularization helps control overfitting.

🔟 Regularization

Regularization adds penalty to large coefficients.

New objective:

Minimize (MSE + Penalty)

1️⃣1️⃣ Ridge Regression (L2)
𝑀
𝑆
𝐸
+
𝜆
∑
𝑏
2
MSE+λ∑b
2

Shrinks coefficients

Reduces variance

Handles multicollinearity

1️⃣2️⃣ Lasso Regression (L1)
𝑀
𝑆
𝐸
+
𝜆
∑
∣
𝑏
∣
MSE+λ∑∣b∣

Shrinks coefficients

Can make some exactly zero

Performs feature selection

1️⃣3️⃣ Bias-Variance Tradeoff

Underfitting → High Bias
Overfitting → High Variance

Goal: Balance bias and variance.

Regularization helps achieve that balance.

Final Summary

Linear Regression predicts continuous values using a best-fit line.
Residuals measure prediction error.
Gradient Descent minimizes cost.
R² evaluates model performance.
Adjusted R² prevents misleading feature addition.
Ridge and Lasso prevent overfitting using regularization.
