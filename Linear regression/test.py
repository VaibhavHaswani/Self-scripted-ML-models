import LinearRegression as lr
l=lr.LinearRegression()
data=l.load('ex1data1.txt')
x=[i[0] for i in data]
y=[i[1] for i in data]
theta=[0,0]
print(l.costFunction(x,y,theta))
l.plot()
l.gradientDescent(0.01,1500)
print(l.theta)
l.fitplot()
print(l.accuracy())
