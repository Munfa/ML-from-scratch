<h1>Implementation of Machine Learning Algorithms from scratch in Python</h1>

<p>This repository contains implementations of various Machine Learning algorithms from scratch. The code is written in Python with comments provided for explanation.</p>

<h3>Dependencies:</h3>
<ul>
  <li><b>Numpy</b> - for numerical computations and splitting the datasets</li>
  <li><b>Scikit-learn</b>
    <ul>
      <li>datasets - for loading sample datasets</li>
      <li>train_test_split - for splitting the datasets</li>
      <li>StandardScaler - for feature scaling</li>
    </ul>  
  </li>
  <li><b>Matplotlib</b> 
    <ul>
      <li>pyplot - for data visualization</li>
    </ul>  
  </li>
</ul>

<h3>Implemented Algorithms</h3>
<table>
  <tr>
    <th><b>Algorithm</b></th>
    <th><b>Description</b></th>
  </tr>
  <tr>
    <td>K-Nearest Neighbors</td>
    <td>
      Euclidean distance of data points is calculated to find the nearest data points. An unknown data point is assigned to the class that has the most points closer to it. <br>
      Euclidean Distance = √(x2 - x1)² + (y2 - y1)²<br>
    </td>
  </tr>
  <tr>
    <td>Linear Regression</td>
    <td>
      Gradients of the slope m and intercept b are calculated from the Mean Squared Error equation. This gradient is used for updating m and b for better predictions. <br>
      Linear equation = mx + b where m is the slope and b is the intercept of a line<br>
      Error = 1/n * ∑(y - (mx + b))²<br>
      m_gradient = -2/n * ∑x(y - (mx+b))<br>
      b_gradient = -2/n * ∑(y - (mx+b))<br>
    </td>
  </tr>
  <tr>
    <td>Logistic Regression</td>
    <td>
      The Sigmoid Function is used for making predictions. Gradients of the weights and bias are calculated from the Loss Function equation. This gradient is used for updating weights and bias for better predictions. <br>
      Sigmoid Function = 1/(1+e<sup>-z</sup>); where z = w.X + b <br>
      Loss Function = (1/n) * ∑(-y*log(y_pred) - (1-y)*log(y_pred))<br>
      Partial derivatives of the Loss Function<br>
      dw = 1/n * ∑x(y_pred - y)<br>
      db = 1/n * ∑(y_pred - y)<br>
    </td>
  </tr>
  <tr>
    <td>Multiclass Logistic Regression</td>
    <td>
      Using the Softmax Function for making predictions of probabilities and choosing the class that has the highest probability for the classification. Gradients of the weights and bias are calculated from the Cross-Entropy Loss Function. This gradient is used for updating weights and bias for better predictions. <br>
      Softmax Function = e<sup>z</sup>/∑(e<sup>z</sup>); where z = w.X + b <br>
      Loss Function = - (1/n) * ∑(y * log(y_pred))<br>
      Partial derivatives of the Loss Function<br>
      dw = 1/n * ∑ x<sup>T</sup>(y_pred - y)<br>
      db = 1/n * ∑(y_pred - y)<br>
    </td>
  </tr>
  <tr>
    <td>Support Vector Machine (Stochastic Gradient Descent)</td>
    <td>
      The linear equation w.X+b is used for making predictions for the classification. The gradients of the Hinge-Loss Function are used for updating the weights and bias.<br>
      Hinge-Loss Function = 1/2 * ||w||^2 + C * ∑max(0, 1 - y * (wx + b))<br>
      Partial derivatives of the Hinge-Loss Function<br>
      if (y * (wx + b) < 1): <br>
      (misclassified) <br>
      dw = C*w - y*x; where C = regularizing parameter<br>
      db = y<br>
      else if (y * (wx + b) >= 1): <br>
      (correctly classified) <br>
      dw = w
    </td>
  </tr>
  <tr>
    <td>Support Vector Machine (Sequential Minimal Optimization)</td>
    <td>
      The same linear equation w.X+b is used for making predictions of this algorithm. The optimum value of the Lagrange multiplier(α) is achieved by solving the Dual Optimization Problem or the Quadratic Programming Problem. This Lagrange multiplier is used to calculate the weight and bias.<br>
      Dual Optimization Problem = ∑αi - 1/2 ∑αiαj*yiyj*(xi.xj)<br>
      where,<br>
      0<=αi<=C; C = regularizing parameter <br>
      ∑yiαi = 0
    </td>
  </tr>
  <tr>
    <td>Decision Tree</td>
    <td>
      This algorithm is implemented using Gini Impurity to find the best feature and best threshold to build the trees. For predictions, the tree is traversed until a leaf node is reached for each sample.<br>
      Gini Impurity = 1 - ∑(p<sub>i</sub>)<sup>2</sup>; where i is the class and p is the probability of each class
    </td>
  </tr>
  <tr>
    <td>Random Forest</td>
    <td>
      This algorithm is implemented using Entropy and Information Gain to find the best feature and best threshold to build the tree. A higher value of information gain is needed for lower node impurity. For predictions, the trees from the forest are traversed to classify a sample.<br>
      Gini Impurity = - ∑p<sub>i</sub>*log<sub>2</sub>(p<sub>i</sub>); where i is the class and p is the probability of each class
    </td>
  </tr>
  <tr>
    <td>Gaussian Naive Bayes</td>
    <td>
      This algorithm used the Gaussian formula to predict the likelihood of each feature belonging to a class. <br>
      The formula = 1/(√2*π*σ<sup>2</sup>) * e<sup>-(x<sub>i</sub> - μ)<sup>2</sup> / (2*π<sup>2</sup>)</sup>
      <br>Here, 
      <br>x<sub>i</sub> = feature value 
      <br>σ<sup>2</sup> = variance
      <br>μ = mean
      <br>The prediction of the class is determined by the maximum probability<br>
      P(c|x<sub>i</sub>) = log(P(c) * ∑P(x<sub>i</sub>|c))
    </td>
  </tr>
</table>

<h3>References</h3>
<ul>
  <li><b>Youtube Channels</b>
    <ul>
      <li>sentdex - Machine Learning with Python (Playlist)</li>
      <li>AssemblyAI - Machine Learning From Scratch (Playlist)</li>
      <li>NeuralNine</li>
      <li>StatQuest with Josh Starmer</li>
    </ul>
  </li>
  <li><b>ChatGPT</b> - Used for conceptual understanding and code improvements</li>
  <li><b>Kaggle Notebooks</b></li>
</ul>

<h3>Contributing</h3>
Contributions are welcome. Feel free to submit issues or pull requests.







