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
    <td>Linear Regression</td>
    <td>
      Gradients of the slope <i>m</i> and intercept <i>b</i> is calculated from the <b>Mean Sqaured Error</b> equation. This gradient is used for updating <i>m</i> and <i>b</i> 
      for better predictions. <br>
      Linear equation = mx + b where m is the slope and b is the intercept of a line<br>
      <b>Error</b> = 1/n * ∑ (y - (mx + b))²<br>
      m_gradient = -2/n * ∑ x(y - (mx+b))<br>
      b_gradient = -2/n * ∑ (y - (mx+b))<br>
    </td>
  </tr>
</table>











