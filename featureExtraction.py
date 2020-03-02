import librosa, os, csv, time

# pip install librosa

# sampleFile is path to file
# sampleLabel is label for sample type to be placed in the last column of csv
class Sample:
    def __init__(self, sampleFile, sampleLabel):
        self.name = str(sampleLabel)
        self.y, self.sr = librosa.load(sampleFile)
        self.chromaCens = librosa.feature.chroma_stft(y=self.y, sr=self.sr, n_chroma=4)
        self.melSpect = librosa.feature.melspectrogram(y=self.y, sr=self.sr)                # probably not useful and returns 20,000 data points, excluded
        self.mfcc = librosa.feature.mfcc(y=self.y, sr=14000, n_mfcc=4)                      # https://ieeexplore.ieee.org/document/647282
        self.rms = librosa.feature.rms(y=self.y)
        self.spectralCentroid = librosa.feature.spectral_centroid(y=self.y, sr=self.sr)
        self.spectralBandwidth = librosa.feature.spectral_bandwidth(y=self.y, sr=self.sr)
        self.spectralContrast = librosa.feature.spectral_contrast(y=self.y, sr=self.sr)     # leaving this at ~1500 vals since it will probably correlate to replays
        self.spectralFlatness = librosa.feature.spectral_flatness(y=self.y)
        self.spectralRolloff = librosa.feature.spectral_rolloff(y=self.y, sr=self.sr)
        self.tonalCentroid = librosa.feature.tonnetz(y=self.y, sr=self.sr)
        self.zeroCrossingRate = librosa.feature.zero_crossing_rate(y=self.y)


    def getChromaCens(self):
        return self.chromaCens

    def getMelSpect(self):
        return self.melSpect

    def getMfcc(self):
        return self.mfcc

    def getRms(self):
        return self.rms

    def getSpectralCentroid(self):
        return self.spectralCentroid

    def getSpectralBandwidth(self):
        return self.spectralBandwidth

    def getSpectralContrast(self):
        return self.spectralContrast

    def getSpectralFlatness(self):
        return self.spectralFlatness

    def getSpectralRolloff(self):
        return self.spectralRolloff

    def getTonalCentroid(self):
        return self.tonalCentroid

    def getZeroCrossingRate(self):
        return self.zeroCrossingRate


# functions that generate labels for csv header
# returns a list of strings
    def getChromaCensLabels(self):
        m = 0
        n = 0
        chromaCensLabelsList = []
        for x in self.chromaCens:
            for y in x:
                chromaCensLabelsList.append("chromC(" + str(m) + ":" + str(n) +")")
                n += 1
            m += 1
            n = 0
        return chromaCensLabelsList

    def getMfccLabels(self):
        m = 0
        n = 0
        mfccLabelsList = []
        for x in self.mfcc:
            for y in x:
                mfccLabelsList.append("mfcc(" + str(m) + ":" + str(n) + ")")
            n += 1
        m += 1
        n = 0
        return mfccLabelsList

    def getRmsLabels(self):
        i = 0
        rmsLabelsList = []
        for x in self.rms:
            for y in x:
                i += 1
        for n in range(i):
            rmsLabelsList.append("rms" + str(n))
        return rmsLabelsList

    def getSpecCLabels(self):
        i = 0
        specCLabelsList = []
        for x in self.spectralCentroid:
            for y in x:
                i += 1
        for n in range(i):
            specCLabelsList.append("specC" + str(n))
        return specCLabelsList

    def getSpecBLabels(self):
        i = 0
        specBLabelsList = []
        for x in self.spectralBandwidth:
            for y in x:
                i += 1
        for n in range(i):
            specBLabelsList.append("specB" + str(n))
        return specBLabelsList

    def getSpecCoLabels(self):
        m = 0
        n = 0
        specCoLabelsList = []
        for x in self.spectralContrast:
            for y in x:
                specCoLabelsList.append("specCo(" + str(m) + ":" + str(n) + ")")
                n += 1
            m += 1
            n = 0
        return specCoLabelsList

    def getSpecFlLabels(self):
        i = 0
        specFlLabelsList = []
        for x in self.spectralFlatness:
            for y in x:
                i += 1
        for n in range(i):
            specFlLabelsList.append("specFl" + str(n))
        return specFlLabelsList

    def getSpecRoLabels(self):
        i = 0
        specRoLabelsList = []
        for x in self.spectralRolloff:
            for y in x:
                i += 1
        for n in range(i):
            specRoLabelsList.append("specRo" + str(n))
        return specRoLabelsList

    def getTonalCentroidLabels(self):
        m = 0
        n = 0
        tonCLabelsList = []
        for x in self.tonalCentroid:
            for y in x:
                tonCLabelsList.append("tonC(" + str(m) + ":" + str(n) + ")")
                n += 1
            m += 1
            n = 0
        return tonCLabelsList

    def getZeroCrossLabels(self):
        i = 0
        zeroCrLabelsList = []
        for x in self.zeroCrossingRate:
            for y in x:
                i += 1
        for n in range(i):
            zeroCrLabelsList.append("zeroCr" + str(n))
        return zeroCrLabelsList


# simple way to deal with arrays within arrays and create a master list of values
# annoyingArray is a nested array returned by the get feature functions
# returns a list of values
def listValuesPretty(annoyingArray):
    valuesList = []
    for x in annoyingArray:
        for y in x:
            valuesList.append(y)
    return valuesList

# creates row of values for csv
# returns list of values from all features, last column is sample ID
def constructCsvRow(sample):
    row = []
    for x in listValuesPretty(sample.getChromaCens()):
        row.append(x)
    for x in listValuesPretty(sample.getMfcc()):
        row.append(x)
    for x in listValuesPretty(sample.getRms()):
        row.append(x)
    for x in listValuesPretty(sample.getSpectralCentroid()):
        row.append(x)
    for x in listValuesPretty(sample.getSpectralBandwidth()):
        row.append(x)
    for x in listValuesPretty(sample.getSpectralContrast()):
        row.append(x)
    for x in listValuesPretty(sample.getSpectralFlatness()):
        row.append(x)
    for x in listValuesPretty(sample.getSpectralRolloff()):
        row.append(x)
    for x in listValuesPretty(sample.getTonalCentroid()):
        row.append(x)
    for x in listValuesPretty(sample.getZeroCrossingRate()):
        row.append(x)
    row.append(sample.name)
    return row

# creates the header for the csv from the get label functions
# returns list of strings
def constructCsvHeader(sample):
    header = []
    header.append("FileNum")
    header.append("FileName")
    for x in sample.getChromaCensLabels():
        header.append(x)
    for x in sample.getMfccLabels():
        header.append(x)
    for x in sample.getRmsLabels():
        header.append(x)
    for x in sample.getSpecCLabels():
        header.append(x)
    for x in sample.getSpecBLabels():
        header.append(x)
    for x in sample.getSpecCoLabels():
        header.append(x)
    for x in sample.getSpecFlLabels():
        header.append(x)
    for x in sample.getSpecRoLabels():
        header.append(x)
    for x in sample.getTonalCentroidLabels():
        header.append(x)
    for x in sample.getZeroCrossLabels():
        header.append(x)
    header.append("SampleID")
    return header


# call this with the path to a folder containing audio recordings on your computer as sampleFileFolderPath
# second parameter is a string sampleTypeID that designates whether the samples are testing, training, 0PR, 1PR, 2PR
# sampleNumStart is the line/row index of the first file to process according to position in the folder, starting at 1
# all listed files between sampleNumStart, sampleNumEnd inclusive will be processed and data output in order
# sampleNumEnd is the last file to be processed according to position in the folder
# to process all files in a folder: sampleNumStart=1, sampleNumEnd = num of files the folder contains (Properties)
# creates a csv containing feature data
def writeSpecificFilesToCsv(sampleFileFolderPath, sampleTypeID, sampleNumStart, sampleNumEnd):
    fileNameList = os.listdir(sampleFileFolderPath)
    outputFilename = "featureData" + sampleTypeID + ".csv"
    dataOutput = open(outputFilename, "w+", encoding="utf8")
    writer = csv.writer(dataOutput, delimiter=',', lineterminator="\r", quoting=csv.QUOTE_NONE)
    headerSampleName = fileNameList[0]
    headerSample = Sample((sampleFileFolderPath + "\\" + headerSampleName), sampleTypeID)
    writer.writerow(constructCsvHeader(headerSample))
    i = 1
    for sampleFile in fileNameList:
        if i>= sampleNumStart and i<= sampleNumEnd:
            samplePath = sampleFileFolderPath + "\\" + sampleFile
            audioSample = Sample(samplePath, sampleTypeID)
            row = []
            row.append(str(i))
            row.append(sampleFile)
            featureList = constructCsvRow(audioSample)
            for y in featureList:
                row.append(y)
            print("Recording row number " + str(i))
            writer.writerow(row)
        i += 1


# call functions with the path to the Replay-Recordings recordings folder on your computer as sampleParentFolderPath
# the second parameter numOfSamples is an integer number of samples to process in each of the 3 folders
# total number of samples to process is 3 * numOfSamples
# latest runtime was about 1 sample per second


# creates a csv files one containing training data, the other containing testing data
def writeTrainingDataToCsv(sampleParentFolderPath, numOfSamples):
    startTime = time.time()
    trainingFolders = sampleParentFolderPath + "\\training"
    trainingFolderSet = [(trainingFolders + "\\0PR"), (trainingFolders + "\\1PR"), (trainingFolders + "\\2PR")]
    trainingDataOutput = open("trainingData.csv", "w+", encoding="utf8")
    c = 0
    writer = csv.writer(trainingDataOutput, delimiter=',', lineterminator="\r", quoting=csv.QUOTE_NONE)
    for x in trainingFolderSet:
        sampleTypeID = "training" + str(c) + "PR"
        fileNameList = os.listdir(x)
        if c == 0:
            headerSampleName = fileNameList[0]
            headerSample = Sample((x + "\\" + headerSampleName), sampleTypeID)
            writer.writerow(constructCsvHeader(headerSample))
        i = 1
        print(x)
        for sampleFile in fileNameList:
            if i <= numOfSamples:
                samplePath = x + "\\" + sampleFile
                audioSample = Sample(samplePath, sampleTypeID)
                row = []
                row.append(str(i))
                row.append(sampleFile)
                featureList = constructCsvRow(audioSample)
                for y in featureList:
                    row.append(y)
                print("Recording sample number " + str(i) + " of " + str(numOfSamples) + " for " + str(c) + "PR training data set.")
                writer.writerow(row)
            i += 1
        c += 1
    runTime = time.time() - startTime
    print("RUNTIME:" + str(runTime) + " seconds")
    print("TOTAL SAMPLES PROCESSED: " + str(numOfSamples*3))


# creates a csv files one containing training data, the other containing testing data
def writeTestingDataToCsv(sampleParentFolderPath, numOfSamples):
    startTime = time.time()
    testingFolders = sampleParentFolderPath + "\\testing"
    testingFolderSet = [(testingFolders + "\\0PR"), (testingFolders + "\\1PR"), (testingFolders + "\\2PR")]
    testingDataOutput = open("testingData.csv", "w+", encoding="utf8")
    c = 0
    writer = csv.writer(testingDataOutput, delimiter=',', lineterminator="\r", quoting=csv.QUOTE_NONE)
    for x in testingFolderSet:
        sampleTypeID = "testing" + str(c) + "PR"
        fileNameList = os.listdir(x)
        if c == 0:
            headerSampleName = fileNameList[0]
            headerSample = Sample((x + "\\" + headerSampleName), sampleTypeID)
            writer.writerow(constructCsvHeader(headerSample))
        i = 1
        print(x)
        for sampleFile in fileNameList:
            if i <= numOfSamples:
                samplePath = x + "\\" + sampleFile
                audioSample = Sample(samplePath, sampleTypeID)
                row = []
                row.append(str(i))
                row.append(sampleFile)
                featureList = constructCsvRow(audioSample)
                for y in featureList:
                    row.append(y)
                print("Recording sample number " + str(i) + " of " + str(numOfSamples) + " for " + str(c) + "PR testing data set.")
                writer.writerow(row)
            i += 1
        c += 1
    runTime = time.time() - startTime
    print("RUNTIME:" + str(runTime) + " seconds")
    print("TOTAL SAMPLES PROCESSED: " + str(numOfSamples*3))



writeTrainingDataToCsv("C:\\Users\\sydney\\Documents\\Recordings\\Replay-Recordings", 2)
writeTestingDataToCsv("C:\\Users\\sydney\\Documents\\Recordings\\Replay-Recordings", 2)