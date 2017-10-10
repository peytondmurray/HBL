#Data analysis recipes: Fitting a Model to Data
#Hogg, Bovy ,Lang,(2010)
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import seaborn as sb


def e1():
	data = [1,201,592,61,9,-0.84,
2,244,401,25,4,0.31,
3,47,583,38,11,0.64,
4,287,402,15,7,-0.27,
5,203,495,21,5,-0.33,
6,58,173,15,9,0.67,
7,210,479,27,4,-0.02,
8,202,504,14,4,-0.05,
9,198,510,30,11,-0.84,
10,158,416,16,7,-0.69,
11,165,393,14,5,0.30,
12,201,442,25,5,-0.46,
13,157,317,52,5,-0.03,
14,131,311,16,6,0.50,
15,166,400,34,6,0.73,
16,160,337,31,5,-0.52,
17,186,423,42,9,0.90,
18,125,334,26,8,0.40,
19,218,533,16,6,-0.78,
20,146,344,22,5,-0.56]

	ID = data[0::6]
	x = data[1::6]
	y = data[2::6]
	sy = data[3::6]
	sx = data[4::6]
	pxy = data[5::6]
	
	fitData(x[4:], y[4:], sx[4:], sy[4:])

	fig = plt.figure()
	plt.errorbar(x, y, xerr=sx, yerr=sy)
	plt.show()

if __name__ == "__main__":
