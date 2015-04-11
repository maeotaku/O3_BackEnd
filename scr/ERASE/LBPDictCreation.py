def isUniform(pixel_values):
    prev = pixel_values[-1]
    t = 0
    for p in range(0, len(pixel_values)):
        cur = pixel_values[p]
        if cur != prev:
            t += 1
            if t>2:
                return False
        prev = cur
    return True

def shiftList(seq, n):
    n = n % len(seq)
    return seq[n:] + seq[:n]

def binListToInt(L):
    res=0
    P = len(L)
    for a in L:
        P-=1
        if a == 1:
            res=res+(2**P)
    return res

def intToBinList(num, length):
    binNum = [int(x) for x in bin(num)[2:]]
    if len(binNum) < length:
        #populate the rest with zeroes
        binNum =  [0 for x in range(0, length - len(binNum))] + binNum
    return binNum

def genDict(P):
    rotationInv = {}
    total  = 2 ** P
    for num in range(0, total):
        #calculate the binary list for the current number
        binList = intToBinList(num,P)
        if isUniform(binList):
            #do P shifts and get the lowest
            candidates = [binListToInt(binList)]
            for nshift in range(0,P-1):    
                binList = shiftList(binList, 1)
                candidate = binListToInt( binList)
                candidates.append(candidate)
            rotationInv[tuple(binList)] = min(candidates)
            #print(binNum)
        else:
            rotationInv[tuple(binList)] = -1
    print("LBP Dict Created")
    return rotationInv