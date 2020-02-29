import librosa, os, csv

# pip install librosa
class Sample:
    def __init__(self, sampleFile, sampleLabel):
        self.name = str(sampleLabel)
        self.y, self.sr = librosa.load(sampleFile)
        self.chromaCens = librosa.feature.chroma_stft(y=self.y, sr=self.sr)
        self.melSpect = librosa.feature.melspectrogram(y=self.y, sr=self.sr)
        self.mfcc = librosa.feature.mfcc(y=self.y, sr=self.sr)
        self.rms = librosa.feature.rms(y=self.y)
        self.spectralCentroid = librosa.feature.spectral_centroid(y=self.y, sr=self.sr)
        self.spectralBandwidth = librosa.feature.spectral_bandwidth(y=self.y, sr=self.sr)
        self.spectralContrast = librosa.feature.spectral_contrast(y=self.y, sr=self.sr)
        self.spectralFlatness = librosa.feature.spectral_flatness(y=self.y)
        self.spectralRolloff = librosa.feature.spectral_rolloff(y=self.y, sr=self.sr)
        self.tonalCentroid = librosa.feature.tonnetz(y=self.y, sr=self.sr)
        self.zeroCrossingRate = librosa.feature.zero_crossing_rate(y=self.y)

# this method is acting wacky and not returning a string so just use sample.name instead of sample.getID
    def getID(self):
        return self.name

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

    def getChromaCensLabels(self):
        i = 0
        chromaCensLabelsList = []
        for x in self.chromaCens:
            for y in x:
                i += 1
        for n in range(i):
            chromaCensLabelsList.append("chromC" + str(n))
        return chromaCensLabelsList

    def getMelSpectLabels(self):
        i = 0
        melSpectLabelsList = []
        for x in self.melSpect:
            for y in x:
                i += 1
        for n in range(i):
            melSpectLabelsList.append("melS" + str(n))
        return melSpectLabelsList

    def getMfccLabels(self):
        i = 0
        mfccLabelsList = []
        for x in self.mfcc:
            for y in x:
                i += 1
        for n in range(i):
            mfccLabelsList.append("mfcc" + str(n))
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
        i = 0
        specCoLabelsList = []
        for x in self.spectralContrast:
            for y in x:
                i += 1
        for n in range(i):
            specCoLabelsList.append("specCo" + str(n))
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
        i = 0
        tonCLabelsList = []
        for x in self.tonalCentroid:
            for y in x:
                i += 1
        for n in range(i):
            tonCLabelsList.append("tonC" + str(n))
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


def listValuesPretty(annoyingArray):
    valuesList = []
    for x in annoyingArray:
        for y in x:
            valuesList.append(y)
    return valuesList

# melspectrogram was returning like 20,000 data points
def constructCsvRow(sample):
    row = []
    for x in listValuesPretty(sample.getChromaCens()):
        row.append(x)
    #for x in listValuesPretty(sample.getMelSpect()):
        #row.append(x)
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

def constructCsvHeader(sample):
    header = []
    for x in sample.getChromaCensLabels():
        header.append(x)
    #for x in sample.getMelSpectLabels():
        #header.append(x)
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

def writeAllToCsv(sampleFileFolderPath, sampleTypeID):
    fileNameList = os.listdir(sampleFileFolderPath)
    outputFilename = "featureData" + sampleTypeID + ".csv"
    dataOutput = open(outputFilename, "w+", encoding="utf8")
    writer = csv.writer(dataOutput, delimiter=',')
    headerSampleName = fileNameList[0]
    headerSample = Sample((sampleFileFolderPath + "\\" + headerSampleName), sampleTypeID)
    writer.writerow(constructCsvHeader(headerSample))
    for sampleFile in fileNameList:
        audioSample = Sample((sampleFileFolderPath + "\\" + sampleFile), sampleTypeID)
        row = constructCsvRow(audioSample)
        writer.writerow(row)


# call this with the path to the trial recordings folder as the path

writeAllToCsv("C:\\Users\\sydney\\Documents\\Recordings\\trial recordings", "test")