.

📘 Linear Regression – Complete Professional Notes
🔹 1. What is Linear Regression?

Linear Regression is a supervised machine learning algorithm used to predict continuous numerical values.

It models the relationship between independent variables (X) and dependent variable (Y) by fitting a straight line.

It assumes a linear relationship between input features and output.

Common use cases:

House price prediction

Salary prediction

Sales forecasting

🔹 2. Mathematical Model

For single feature:

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

For multiple features:

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


Where:

b₀ = Intercept

b₁ = Slope

y = Predicted value

🔹 3. Line of Best Fit (Concept)

The model finds the best straight line that minimizes prediction error.

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

🔹 4. Residuals

Residual = Actual − Predicted

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


Residual is vertical distance between data point and regression line.

Good Model:
Residuals randomly scattered around zero.

Bad Model:
Residuals show pattern → non-linear relationship.

🔹 5. Cost Function (MSE)
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

The goal of Linear Regression is to minimize MSE.

Lower MSE = Better model.

🔹 6. Gradient Descent

Used to minimize cost function.

Update Rule:

𝑏
=
𝑏
−
𝛼
×
𝑔
𝑟
𝑎
𝑑
𝑖
𝑒
𝑛
𝑡
b=b−α×gradient

Learning Rate (α):

Too small → Slow training

Too large → Overshoot

Cost decreases gradually until convergence.

🔹 7. Evaluation Metrics

MAE – Mean Absolute Error
MSE – Mean Squared Error
RMSE – Root Mean Squared Error
R² – Variance explained by model

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

🔹 8. Adjusted R²

R² increases when features are added.

Adjusted R² penalizes unnecessary features.

Used in Multiple Linear Regression.

🔹 9. Underfitting vs Overfitting

Underfitting:
Model too simple → High bias

Overfitting:
Model too complex → High variance

Regularization helps control this.

🔹 10. Ridge Regression (L2)
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

Handles multicollinearity

Reduces overfitting

🔹 11. Lasso Regression (L1)
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

Can make coefficients zero

Performs feature selection

Produces simpler model

🔹 12. Bias-Variance Tradeoff

Underfitting → High Bias
Overfitting → High Variance

Goal → Balance both.
