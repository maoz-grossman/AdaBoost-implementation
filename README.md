# AdaBoost-implementation
## Implementation of an adaboost algorithm on the dataset  HC_Body_Temperature

<h4>Ariel University</h4>
<h5>Machine Learning</h5>
<h5>Homework: AdaBoost</h5>

<p>
  In this homework you will implement AdaBoost.<br>
  We will use the HC Temperature data set found on the course’s webpage.<br>
  It contains 130 data points. The label (1 and -1) will be the gender,<br>
  and the temperature and heartrate define the 2-dimensional point.  
  </p>
  
<p>
The hypothesis class is the set of axis-parallel rectangles for which the inside is positive and
the outside is negative.<br> 
Note that a rectangle can defined by 2, 3 or 4 points.<br>
Write a function called Rectangle, which given a set of labelled and weighted points,<br>
finds a rectangle which minimizes the weighted error on the points,
that is the sum of weights of wrongly placed points.<br>
  </p>

<p>
  Now write the AdaBoost algorithm on a training set of size <b>n</b>. The pseudocode is as follows: 
</p>

```python
Initialize each point weight to be 1/n: D0(x_i) = 1/n
For round t in range(r):
  1.Use Rectangle to find a rectangle with minimum weighted error ε
    and call this rectangle h_t
  2.Compute the weight 
  3.Compute new weights for the points:
      i. For an error on point x_i: D_t(x_i) = D_t-1(x_i) exp(ά_t)
      ii. Not an error on point x_i: D_t(x_i) = D_t-1(x_i) exp(-ά_t)
  4.Normalize these weights: D_t(x_i) = D_t(x_i) / ∑_j(D_t(x_j))
  
```
