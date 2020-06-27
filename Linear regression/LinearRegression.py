import matplotlib.pyplot as plt
import numpy as np
class LinearRegression:

	def __init__(self):
		pass

	def load(self,path):
		try:
			data=open(path,'r')
			data=list(map(lambda x:x.replace('\n',''),data.readlines()))
			li=[[j for j in i.split(',')] for i in data]
			li=list(map(lambda x:list(map(float,x)),li))
			return li
		except:
			print('Invalid Data')


	def plot(self,xlabel='',ylabel=''):
		try:
			plt.plot(self.x,self.y,'rx')
			plt.xlabel=xlabel
			plt.ylabel=ylabel
			plt.show()
		except:
			print("Create costFunction to specify x,y values")

	def costFunction(self,x,y,theta):
		self.theta=np.array(theta).astype(np.float)
		self.x=np.array(x)
		self.y=np.array(y)
		self.m=len(y)
		self.J=0
		self.J=(1/(2*self.m)*np.sum(np.power((self.theta[0]+self.theta[1]*self.x)-self.y,2)))
		return self.J

	def _compCost(self,theta1,theta2):
		(1/(2*self.m)*np.sum(np.power((theta1+theta2*self.x)-self.y,2)))

	def gradientDescent(self,alpha,iter):
		self.cachedCostFunction=np.empty(shape=(iter,1),dtype=float)
		self.cachedtheta=np.empty(shape=(iter,2),dtype=float)
		n=0
		for i in range(1,iter+1):
			t1=np.sum((self.theta[0]+self.theta[1]*self.x)-self.y)
			t2=np.sum(((self.theta[0]+self.theta[1]*self.x)-self.y)*self.x)
			self.theta[0]=self.theta[0]-(alpha/self.m)*t1
			self.theta[1]=self.theta[1]-(alpha/self.m)*t2
			self.cachedCostFunction[n]=self._compCost(self.theta[0],self.theta[1])
			self.cachedtheta[n,0]=self.theta[0]
			self.cachedtheta[n,1]=self.theta[1]
			n+=1
		self.__predict()
	

	''' ---Under Working---	
	def contourPlot(self):
		theta0 = np.linspace(-10,10,100)
		theta1 = np.linspace(-1,4,100)
		X,Y=np.meshgrid(theta0,theta1)
		Z=np.empty(shape=X.shape,dtype=float)
		for i in range(theta0.size):
			for j in range(theta1.size):
				Z[i,j]=self._compCost(X[i,j],Y[i,j])
		plt.contour(X,Y,Z)
		plt.show()
	'''

	def __predict(self):
		self.h=self.theta[0]+self.theta[1]*self.x

	def predict(self):
		return self.h

	def accuracy(self):
		error=abs(self.y-self.h)
		error_percentage=(error/self.y).mean()
		return f"accuracy: {100-error_percentage}%"

	def fitplot(self,xlabel='',ylabel='',title=''):
		fig,ax=plt.subplots()
		ax.plot(self.x,self.y,'rx')
		ax.plot(self.x,self.h)
		ax.set(xlabel=xlabel,ylabel=ylabel,title=title)
		plt.show()





