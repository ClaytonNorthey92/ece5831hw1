import math
import numpy

INDENTITY = numpy.matrix('1, 0 ; 0, 1') #2D Identity matrix

w1 = {
	'mean': numpy.matrix('0;0'),
	'variance': INDENTITY
}

w2 = {
	'mean': numpy.matrix('1;1'),
	'variance': INDENTITY
}

w3 = {
	'mean': 0.5*(numpy.matrix('0.5;0.5') + numpy.matrix('-0.5;0.5')),
	'variance': 0.5*(INDENTITY + INDENTITY)
}

def pxw(class_id, x):
	value = numpy.matrix(2 * math.pi * numpy.sqrt(numpy.absolute(class_id['variance']))).I
	power = (-0.5 * (x - class_id['mean']).T * class_id['variance'].I * (x - class_id['mean'])).item(0, 0)
	return value * numpy.exp(power)

def pxw_pw(pwx, pw):
	return (pwx * pw).item(0,0)

def classify(pw1, pw2, pw3):
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


if __name__=='__main__':
	p1a_doc_string = """
		--- Problem #48 Part A: ---
	"""
	pw = 0.333 # <- same for all w
	x = numpy.matrix('0.3 ; 0.3')
	pw1 = pxw_pw(pxw(w1, x), pw)
	pw2 = pxw_pw(pxw(w2, x), pw)
	pw3 = pxw_pw(pxw(w3, x), pw)
	print(p1a_doc_string)
	classify(pw1, pw2, pw3)

	p1b_doc_string = """
		--- Problem #48 Part B: ---
		We need to integrate over the unknown variable, x1...
	"""

	print(p1b_doc_string)
	# we should go from -infinity to +infinity but we physically can't, so we will
	# go from -100 to +100 because at those points our probability will be 0
	pw_x1_range = [numpy.matrix([[x1], [0.3]]) for x1 in range(-100, 100)]
	pw1 = sum([pxw_pw(pxw(w1, x1), pw) for x1 in pw_x1_range])
	pw2 = sum([pxw_pw(pxw(w2, x1), pw) for x1 in pw_x1_range])
	pw3 = sum([pxw_pw(pxw(w3, x1), pw) for x1 in pw_x1_range])
	classify(pw1, pw2, pw3)


	p1c_doc_string = """
		--- Problem #48 Part C: ---
		We need to integrate over the unknown variable, x2...
	"""

	print(p1c_doc_string)
	# we should go from -infinity to +infinity but we physically can't, so we will
	# go from -100 to +100 because at those points our probability will be 0
	pw_x2_range = [numpy.matrix([[0.3], [x2]]) for x2 in range(-100, 100)]
	pw1 = sum([pxw_pw(pxw(w1, x2), pw) for x2 in pw_x2_range])
	pw2 = sum([pxw_pw(pxw(w2, x2), pw) for x2 in pw_x2_range])
	pw3 = sum([pxw_pw(pxw(w3, x2), pw) for x2 in pw_x2_range])
	classify(pw1, pw2, pw3)

