#Data analysis recipes: Fitting a Model to Data
#Hogg, Bovy ,Lang,(2010)
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

def listToVector(l):
	return np.atleast_2d(np.array(l))

def fitData(x, y, sx, sy):
	x = listToVector(x).T
	y = listToVector(y).T
	sx = listToVector(sx).T
	sy = listToVector(sy).T
	I = np.identity(np.shape(sy)[0])


	Y = np.matrix(y)
	A = np.matrix(np.hstack((np.ones_like(x), x)))
	C = np.matrix((np.identity(np.shape(sy)[0])*sy)**2)

	X = (A.T*C.I*A).I*(A.T*C.I*Y)			#Wooooo numpy matrices rock!

	return X, (A.T*C.I*A).I

def fitDataQuadratic(x, y, sx, sy):
	x = listToVector(x).T
	y = listToVector(y).T
	sx = listToVector(sx).T
	sy = listToVector(sy).T
	I = np.identity(np.shape(sy)[0])


	Y = np.matrix(y)
	A = np.matrix(np.hstack((np.ones_like(x), x, x**2)))
	C = np.matrix((np.identity(np.shape(sy)[0])*sy)**2)

	X = (A.T*C.I*A).I*(A.T*C.I*Y)			#Wooooo numpy matrices rock!

	return X, (A.T*C.I*A).I

def makeLine(x, parameters):
	b = parameters[0,0]
	m = parameters[1,0]
	return b+m*x

def makeQuadratic(x, parameters):
	b = parameters[0,0]
	m = parameters[1,0]
	q = parameters[2,0]
	return b+m*x+q*x**2

def e1Data():
	data = [1, 201, 592, 61, 9, -0.84,
			2, 244, 401, 25, 4, 0.31,
			3, 47, 583, 38, 11, 0.64,
			4, 287, 402, 15, 7, -0.27,
			5, 203, 495, 21, 5, -0.33,
			6, 58, 173, 15, 9, 0.67,
			7, 210, 479, 27, 4, -0.02,
			8, 202, 504, 14, 4, -0.05,
			9, 198, 510, 30, 11, -0.84,
			10, 158, 416, 16, 7, -0.69,
			11, 165, 393, 14, 5, 0.30,
			12, 201, 442, 25, 5, -0.46,
			13, 157, 317, 52, 5, -0.03,
			14, 131, 311, 16, 6, 0.50,
			15, 166, 400, 34, 6, 0.73,
			16, 160, 337, 31, 5, -0.52,
			17, 186, 423, 42, 9, 0.90,
			18, 125, 334, 26, 8, 0.40,
			19, 218, 533, 16, 6, -0.78,
			20, 146, 344, 22, 5, -0.56]

	ID = np.array(data[0::6])
	x = np.array(data[1::6])
	y = np.array(data[2::6])
	sy = np.array(data[3::6])
	sx = np.array(data[4::6])
	pxy = np.array(data[5::6])

	return ID, x, y, sx, sy, pxy

def e1():
	ID, x, y, sx, sy, pxy = e1Data()
	fitparams, cov = fitData(x[4:], y[4:], sx[4:], sy[4:])

	fig = plt.figure()
	plt.errorbar(x[4:], y[4:], xerr=sx[4:], yerr=sy[4:], linestyle="none")
	plt.plot(x, makeLine(x, fitparams))
	plt.show()

def e2():
	ID, x, y, sx, sy, pxy = e1Data()
	fitparams, cov = fitData(x, y, sx, sy)

	fitX = np.linspace(np.min(x), np.max(x), 100)

	fig = plt.figure()
	plt.errorbar(x, y, xerr=sx, yerr=sy, linestyle="none")
	plt.plot(fitX, makeLine(fitX, fitparams))
	plt.show()

def e3():
	ID, x, y, sx, sy, pxy = e1Data()
	fitparams, cov = fitDataQuadratic(x[4:], y[4:], sx[4:], sy[4:])

	fitX = np.linspace(np.min(x[4:]), np.max(x[4:]), 100)

	fig = plt.figure()
	plt.errorbar(x[4:], y[4:], xerr=sx[4:], yerr=sy[4:], linestyle="none")
	plt.plot(fitX, makeQuadratic(fitX, fitparams))
	plt.title("({:2.2E}±{:2.2E}) + ({:2.2E}±{:2.2E})x + ({:2.2E}±{:2.2E})x**2".format(fitparams[0,0], np.sqrt(cov[0,0]), fitparams[1,0], np.sqrt(cov[1,1]), fitparams[2,0], np.sqrt(cov[2,2])))
	plt.show()


if __name__ == "__main__":

	# e1()
	# e2()
	# e3()