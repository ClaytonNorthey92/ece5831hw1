import math
import numpy

INDENTITY = numpy.matrix('1, 0 ; 0, 1')

w1 = {
	'mean': numpy.matrix('0;0'),
	'variance': INDENTITY, # 2D identity matrix
}

w2 = {
	'mean': numpy.matrix('1;1'),
	'variance': INDENTITY, # 2D identity matrix
}

w3 = {
	'mean': 0.5*(numpy.matrix('0.5;0.5') + numpy.matrix('-0.5;0.5')),
	'variance': 0.5*(INDENTITY + INDENTITY), # 2D identity matrix
}

def pxw(class_id, x):
	value = numpy.matrix(2 * math.pi * numpy.sqrt(numpy.absolute(class_id['variance']))).I
	power = (-0.5 * (x - class_id['mean']).T * class_id['variance'].I * (x - class_id['mean'])).item(0, 0)
	return value * numpy.exp(power)

def pxw_pw(pwx, pw):
	return (pwx * pw).item(0,0)

if __name__=='__main__':
	pw = 0.333 # <- same for all w
	x = numpy.matrix('0.3 ; 0.3')
	pw1 = pxw_pw(pxw(w1, x), pw)
	pw2 = pxw_pw(pxw(w2, x), pw)
	pw3 = pxw_pw(pxw(w3, x), pw)
	print("w1: ", pw1)
	print("w2: ", pw2)
	print("w3: ", pw3)
	p_max = None
	for index, prob in enumerate([pw1, pw2, pw3]):
		if p_max is None or p_max['value'] < prob:
			p_max = {
				'value': prob,
				'class': 'w' + str(index + 1)
			}

	print('We classify as: ', p_max['class'], ' at ', p_max['value'])