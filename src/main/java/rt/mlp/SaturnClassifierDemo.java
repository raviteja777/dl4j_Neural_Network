package rt.mlp;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public class SaturnClassifierDemo {

    public static void main (String args[]){

        int batchSize = 50;
        //Number of epochs (full passes of the data)
        int nEpochs = 30;
        //hyper parameters
        double learningRate = 0.005;
        int numInputs = 2;
        int numOutputs = 2;
        int numHiddenNodes = 20;

        //init classifier
        SaturnClassifier classifier = new SaturnClassifier();

        try {
            //download and initilize train and test iterators
            DataSetIterator[]  dataSets = classifier.initData(batchSize);
            DataSetIterator trainIter = dataSets[0];
            DataSetIterator testIter = dataSets[1];

            //build model
            MultiLayerConfiguration config = classifier.buildModel(learningRate,numInputs,numOutputs,numHiddenNodes);
            //train model
            MultiLayerNetwork model = classifier.trainModel(config,trainIter,nEpochs);
            //evaluate
            classifier.displayEvaluation(model,testIter);
            //display plot
            classifier.generateVisuals(model,trainIter,testIter);

        } catch (Exception e) {
            e.printStackTrace();
        }


    }

}
