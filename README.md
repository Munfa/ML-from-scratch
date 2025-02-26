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
      <b>Euclidean Distance</b> = √(x2 - x1)² + (y2 - y1)²<br>
    </td>
  </tr>
  <tr>
    <td>Linear Regression</td>
    <td>
      Gradients of the slope <i>m</i> and intercept <i>b</i> are calculated from the <b>Mean Squared Error</b> equation. This gradient is used for updating <i>m</i> and <i>b</i> 
      for better predictions. <br>
      Linear equation = mx + b where m is the slope and b is the intercept of a line<br>
      <b>Error</b> = 1/n * ∑(y - (mx + b))²<br>
      m_gradient = -2/n * ∑x(y - (mx+b))<br>
      b_gradient = -2/n * ∑(y - (mx+b))<br>
    </td>
  </tr>
  <tr>
    <td>Logistic Regression</td>
    <td>
      <b>Sigmoid Function</b> is used for making predictions. Gradients of the <i>weights</i> and <i>bias</i> are calculated from the <b>Loss Function</b> equation. This gradient is used for updating <i>weights</i> and <i>bias</i> 
      for better predictions. <br>
      Sigmoid Function = 1/(1+e<sup>-z</sup>); where z = w.X + b
      Loss Function = (1/n) * ∑(-y*log(y_pred) - (1-y)*log(y_pred))<br>
      Partial derivatives of the Loss Function<br>
      dw = 1/n * ∑x(y_pred - y)<br>
      db = 1/n * ∑(y_pred - y)<br>
    </td>
  </tr>
  <tr>
    <td>Multiclass Logistic Regression</td>
    <td>
      Using <b>Softmax Function</b> for making predictions of probabilities and choosing the class that has the highest probability for the classification. Gradients of the <i>weights</i> and <i>bias</i> are calculated from the <b>Cross-Entropy Loss Function</b>. This gradient is used for updating <i>weights</i> and <i>bias</i> 
      for better predictions. <br>
      Softmax Function = e<sup>z</sup>/∑(e<sup>z</sup>); where z = w.X + b
      Loss Function = - (1/n) * ∑(y * log(y_pred))<br>
      Partial derivatives of the Loss Function<br>
      dw = 1/n * ∑ x<sup>T</sup>(y_pred - y)<br>
      db = 1/n * ∑(y_pred - y)<br>
    </td>
  </tr>
  <tr>
    <td>Support Vector Machine (Stochastic Gradient Descent)</td>
    <td>
      The linear equation <i>w.X+b</i> is used for making predictions for the classification. The gradient of the <b>Hinge-Loss Function</b> is used for updating the weights and bias.<br>
      <b>Hinge-Loss Function</b> = 1/2 * ||w||^2 + C * ∑max(0, 1 - y * (wx + b))<br>
      Partial derivatives of the Hinge-Loss Function<br>
      if (y * (wx + b) < 1):<br>
      dw = C*w - y*x; where C = regularizing parameter<br>
      db = y<br>
      else if (y * (wx + b) >= 1):
      dw = w
    </td>
  </tr>
  <tr>
    <td>Support Vector Machine (Sequential Minimal Optimization)</td>
    <td>
      The same linear equation <i>w.X+b</i> is used for making predictions of this algorithm. The optimum value of the Lagrange multiplier( is achieved by solving the Dual Optimization Problem or the Quadratic Programming Problem. This Lagrange multiplier is used for calculating the weight and bias.<br>
      Dual Optimization Problem = ∑αi - 1/2 ∑αiαj*yiyj*(xi.xj)<br>
      where,<br>
      0<=αi<=C; C = regularizing parameter<br>
      ∑yiαi = 0
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
    </ul>
  </li>
  <li><b>ChatGPT</b> - Used for conceptual understanding and code improvements</li>
  <li><b>Kaggle Notebooks</b></li>
</ul>

<h3>Contributing</h3>
Contributions are welcome. Feel free to submit issues or pull requests.







