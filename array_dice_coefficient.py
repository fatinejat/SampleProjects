def diceCoefficient(image_array1, image_array2):
    if len(image_array1) != len(image_array2):
	print 'Lengths must be same'
        return -1
    
    result = 0
    n1 = 0
    n2 = 0
    for i in range(0, len(image_array1)):
        result = result + image_array1[i] * image_array2[i]
        n1 = n1 + image_array1[i]*image_array1[i]
	n2 = n2 + image_array2[i]*image_array2[i]
    result = 2*result/(n1+n2)
    print result
    return


